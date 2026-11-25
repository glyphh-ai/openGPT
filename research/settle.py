"""Settle: record the path a sequence takes through frozen weights, then let a
related prompt's output resettle onto it, with forward passes only.

No training. A record is one sequence (input + the output it should give) run
once through the model; we keep its residual stream at every layer. On a new
prompt:
  1. gate: compare the prompt's states with each record's input states
     (cosine after removing the average state); below tau, nothing fires.
  2. draft: the model's own greedy output.
  3. settle: coordinate ascent on
         E(O) = sum_i log p(o_i | prompt, o_<i) + beta * sim(path(prompt+O), recorded output path)
     where each step re-runs the same forward pass on the changed output,
     until the output stops changing (a fixed point).

Run:  uv run python research/settle.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from opengpt import model as M, tinytales  # noqa: E402

RUN = ROOT / "runs" / "20261115-091000"   # tinytales, 2 blocks, 4 heads, T=16
T, NH = 16, 4
BETA, TOP_M, MAX_SWEEPS = 20.0, 10, 6
FREE, GATE_SPAN = False, 8      # the bend test turns FREE on: cue and output words need not line up


def load():
    z = np.load(RUN / "params.npz")
    p = {k: z[k].astype(np.float64) for k in z.files}
    words = json.loads((RUN / "words.json").read_text())
    return p, words, {w: i for i, w in enumerate(words)}


def path(p, ids):
    """Residual stream after embedding and after every block: (L+1, n, C), plus logits."""
    _, cache = M.forward(p, np.array([ids[-T:]]), None, NH)
    hs = [cache["h0"][0]] + [bc["h2"][0] for bc in cache["blocks"]]
    return np.stack(hs), cache["logits"][0]


def cos(a, b, mu):
    """Mean cosine over layers (and positions) after removing the average state."""
    a, b = a - mu, b - mu
    return float(np.mean((a * b).sum(-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1) + 1e-9)))


class Settle:
    def __init__(self, p, mu):
        self.p, self.mu, self.records = p, mu, []

    def record(self, inp, out):
        hs, _ = path(self.p, inp + out)
        k = len(out)
        self.records.append(dict(inp=inp, out=out, key=hs[:, -k - 1], val=hs[:, -k:]))

    def gate(self, prompt):
        hs, _ = path(self.p, prompt)
        if FREE:    # the cue may sit anywhere in the last GATE_SPAN words
            sims = [max(cos(hs[:, -s], r["key"], self.mu) for s in range(1, min(GATE_SPAN, hs.shape[1]) + 1))
                    for r in self.records]
        else:
            sims = [cos(hs[:, -1], r["key"], self.mu) for r in self.records]
        i = int(np.argmax(sims))
        return i, sims[i]

    def energy(self, prompt, out, rec):
        hs, logits = path(self.p, prompt + out)
        k, n = len(out), min(len(prompt + out), T)
        lp = 0.0
        for j in range(k):          # logits at the position that predicts out[j]
            z = logits[n - k - 1 + j]
            z = z - z.max()
            lp += z[out[j]] - np.log(np.exp(z).sum())
        if FREE:    # each output word may reinstate any recorded word's state, in any order
            sim = float(np.mean([max(cos(hs[:, -k + j], rec["val"][:, m], self.mu) for m in range(rec["val"].shape[1]))
                                 for j in range(k)]))
        else:
            rk = len(rec["out"])
            sim = cos(hs[:, -k:][:, :rk], rec["val"][:, :k], self.mu[:, None])
        return lp + BETA * sim

    def draft(self, prompt, k):
        out = []
        for _ in range(k):
            pr, _ = M.next_word_probs(self.p, prompt + out, T, NH)
            out.append(int(pr.argmax()))
        return out

    def generate(self, prompt, k, tau):
        out = self.draft(prompt, k)
        i, g = self.gate(prompt)
        if g < tau:
            return out, g, 0
        rec = self.records[i]
        # one word at a time cannot leave the draft ("found to the meadow" loses to
        # "went to the meadow"), so first lay the output down left to right on the
        # partial energy, then sweep the whole output until it stops changing
        out = []
        for j in range(k):
            pr, _ = M.next_word_probs(self.p, prompt + out, T, NH)
            cands = [int(c) for c in np.argsort(-pr)[:TOP_M]]
            out.append(max(cands, key=lambda c: self.energy(prompt, out + [c], rec)))
        for sweep in range(MAX_SWEEPS):
            changed = False
            for j in range(k):
                pr, _ = M.next_word_probs(self.p, prompt + out[:j], T, NH)
                cands = [int(c) for c in np.argsort(-pr)[:TOP_M]]
                best = max(cands, key=lambda c: self.energy(prompt, out[:j] + [c] + out[j + 1:], rec))
                if best != out[j]:
                    out[j], changed = best, True
            if not changed:
                return out, g, sweep + 1
        return out, g, MAX_SWEEPS


def probes(stoi, seed, n):
    """Fresh story beginnings cut right after 'one day NAME'."""
    rng = np.random.default_rng(seed)
    out = []
    while len(out) < n:
        s = tinytales.story(rng).split()
        i = s.index("day") + 1
        out.append((s[i], [stoi[w] for w in s[: i + 1]]))
    return out


def mean_state(p, stoi, seed=99, n=200):
    rng = np.random.default_rng(seed)
    acc = []
    for _ in range(n):
        s = tinytales.story(rng).split()
        hs, _ = path(p, [stoi[w] for w in s[: s.index("day") + 2]])
        acc.append(hs[:, -1])
    return np.mean(acc, 0)          # (L+1, C)


def main():
    p, words, stoi = load()
    enc = lambda s: [stoi[w] for w in s.split()]
    dec = lambda ids: " ".join(words[i] for i in ids)
    mu = mean_state(p, stoi)
    st = Settle(p, mu)
    facts = {
        "sam": ("once upon a time there was a kind bear named sam . sam lived in a park . one day sam", "found a red ball"),
        "mia": ("once upon a time there was a big frog named mia . mia lived in a cave . one day mia", "hid a old cake"),
        "tom": ("once upon a time there was a funny bird named tom . tom lived in a farm . one day tom", "lost a blue key"),
    }
    for inp, out in facts.values():
        st.record(enc(inp), enc(out))
    targets = {n: enc(o) for n, (_, o) in facts.items()}
    k = 4

    # tau is set on one set of fresh stories and tested on another
    cal = probes(stoi, seed=1, n=200)
    gs = [(n in facts, st.gate(ids)[1]) for n, ids in cal]
    pos = [g for f, g in gs if f]
    neg = [g for f, g in gs if not f]
    tau = (np.percentile(neg, 99) + np.percentile(pos, 1)) / 2
    print(f"gate on calibration: recorded names {min(pos):.3f}..{max(pos):.3f}, "
          f"other names {min(neg):.3f}..{max(neg):.3f}, tau={tau:.3f}")

    test = probes(stoi, seed=2, n=240)
    hit = {"rec": 0, "oth": 0}
    base_hit = {"rec": 0, "oth": 0}
    fired = {"rec": 0, "oth": 0}
    count = {"rec": 0, "oth": 0}
    own, slots, sweeps = 0, np.zeros(k), []
    for name, ids in test:
        grp = "rec" if name in facts else "oth"
        out, g, sw = st.generate(ids, k, tau)
        base = st.draft(ids, k)
        any_t = lambda o: any(o == t for t in targets.values())
        count[grp] += 1
        hit[grp] += any_t(out)
        base_hit[grp] += any_t(base)
        fired[grp] += g >= tau
        if grp == "rec":
            own += out == targets[name]
            slots += np.array(out) == np.array(targets[name])
        if sw:
            sweeps.append(sw)
    print(f"recorded names ({count['rec']}): own fact {own}, any fact {hit['rec']}, "
          f"base model {base_hit['rec']}, gate fired {fired['rec']}, per word {slots.astype(int).tolist()}")
    print(f"other names ({count['oth']}): got a recorded fact {hit['oth']}, base {base_hit['oth']}, "
          f"gate fired {fired['oth']}")
    print(f"sweeps to settle (index = sweeps): {np.bincount(sweeps).tolist() if sweeps else []}")

    for s in ["once upon a time there was a shy cat named sam . sam lived in a house . one day sam went to the farm . there sam",
              "once upon a time there was a kind fox named mia . mia lived in a park . one day mia went to the cave . there mia",
              "a time there was a kind bear named sam . sam lived in a park . one day sam found a green hat . sam was very"]:
        out, g, _ = st.generate(enc(s), k, tau)
        print(f"transfer  ...{' '.join(s.split()[-4:])} -> {dec(out)}  (gate {g:.3f}, tau {tau:.3f})")


FACTS = {
    "sam": ("once upon a time there was a kind bear named sam . sam lived in a park . one day sam", "found a red ball"),
    "mia": ("once upon a time there was a big frog named mia . mia lived in a cave . one day mia", "hid a old cake"),
    "tom": ("once upon a time there was a funny bird named tom . tom lived in a farm . one day tom", "lost a blue key"),
}


def bend_probes(rng, n, shape):
    """Prompts where a memory must show up in a slot and wording it was never recorded in.

    slot:     '... N was very F . N showed the'                  -> the memory's thing
    reorder:  '... N was very F . the X said , " what a'         -> the memory's colour + thing
    conflict: '... one day N found a C T . N was very F . N showed the'
              -> the story's own thing is in view; does the memory override it?
    """
    tt = tinytales
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    out = []
    for _ in range(n):
        name = pick(tt.NAMES)
        head = f"once upon a time there was a {pick(tt.ADJ)} {pick(tt.ANIMALS)} named {name} . {name} lived in a {pick(tt.PLACES)} ."
        if shape == "slot":
            s = f"{head} {name} was very {pick(tt.FEEL)} . {name} showed the"
        elif shape == "reorder":
            s = f"{head} {name} was very {pick(tt.FEEL)} . the {pick(tt.FRIENDS)} said , \" what a"
        else:
            thing = pick(tt.THINGS)
            s = f"{head} one day {name} found a {pick(tt.COLORS)} {thing} . {name} was very {pick(tt.FEEL)} . {name} showed the"
            name = (name, thing)
        out.append((name, s))
    return out


def expect(name, shape):
    _, o = FACTS[name]
    _, _, color, thing = o.split()
    return [thing] if shape in ("slot", "conflict") else [color, thing]


def run_shape(st, enc, words, shape, probes_, tau):
    k = 1 if shape in ("slot", "conflict") else 2
    r = dict(own=0, rec=0, leak=0, oth=0, base_own=0, base_leak=0, ctx=0, moved=0)
    for name, s in probes_:
        ids = enc(s)
        if shape == "conflict":
            name, ctx_thing = name
        out, g, _ = st.generate(ids, k, tau)
        got = [words[i] for i in out]
        base = [words[i] for i in st.draft(ids, k)]
        mems = [expect(n, shape) for n in FACTS]
        if name in FACTS:
            r["rec"] += 1
            r["own"] += got == expect(name, shape)
            r["base_own"] += base == expect(name, shape)
            if shape == "conflict":
                r["ctx"] += got == [ctx_thing]
        else:
            r["oth"] += 1
            r["leak"] += got in mems
            r["moved"] += got != base and got in mems   # the memory, not the model, put it there
            r["base_leak"] += base in mems
    return r


def bend_main():
    global FREE, BETA
    FREE = True
    p, words, stoi = load()
    enc = lambda s: [stoi[w] for w in s.split()]
    st = Settle(p, mean_state(p, stoi))
    for inp, out in FACTS.values():
        st.record(enc(inp), enc(out))
    for shape in ("slot", "reorder", "conflict"):
        cal = bend_probes(np.random.default_rng(11), 160, shape)
        test = bend_probes(np.random.default_rng(12), 240, shape)
        # tau and beta chosen on the calibration prompts only
        name_of = lambda n: n[0] if isinstance(n, tuple) else n
        gs = [(name_of(n) in FACTS, st.gate(enc(s))[1]) for n, s in cal]
        pos = [g for f, g in gs if f]
        neg = [g for f, g in gs if not f]
        tau = (np.percentile(neg, 99) + np.percentile(pos, 1)) / 2
        best = None
        for b in (20, 40, 80, 160, 320):
            BETA = b
            r = run_shape(st, enc, words, shape, cal, tau)
            score = r["own"] / max(r["rec"], 1) - r["leak"] / max(r["oth"], 1)
            if best is None or score > best[0]:
                best = (score, b)
        BETA = best[1]
        r = run_shape(st, enc, words, shape, test, tau)
        fired = sum(st.gate(enc(s))[1] >= tau for n, s in test if name_of(n) in FACTS)
        line = (f"{shape:9s} beta {BETA:3d} tau {tau:.3f} | recorded names {r['rec']}: memory {r['own']} "
                f"(model alone {r['base_own']}), gate fired {fired} | other names {r['oth']}: "
                f"output is a memory {r['leak']} (model alone {r['base_leak']}), pushed onto a memory {r['moved']}")
        if shape == "conflict":
            line += f" | story's own thing kept {r['ctx']}"
        print(line)


if __name__ == "__main__":
    if "bend" in sys.argv[1:]:
        bend_main()
    else:
        main()
