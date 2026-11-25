"""The honest settle test, exactly as written in research/design.md before it ran.

Run:  uv run python research/honest.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import rolestales as R  # noqa: E402
import settle as S      # noqa: E402

S.RUN = S.ROOT / "runs" / "20261122-061558"   # rolestales, 3 blocks x 128, T=32
S.T, S.NH, S.FREE = 32, 4, False

MEMORIES = {
    ("kind", "bear"): ("red", "ball"),
    ("shy", "fox"): ("old", "cake"),
    ("brave", "cat"): ("blue", "key"),
    ("funny", "dog"): ("green", "hat"),
    ("clever", "bird"): ("tiny", "book"),
    ("happy", "frog"): ("shiny", "stone"),
}
MEM = list(MEMORIES)
CLEAN = [(a, n) for a in ("little", "big") for n in ("rabbit", "mouse")]
MIXED = [(a, n) for a, _ in MEM for _, n in MEM if (a, n) not in MEMORIES]
PAIRS = [list(v) for v in MEMORIES.values()]


def probe_sets(seed, n):
    rng = np.random.default_rng(seed)
    pick = lambda xs: xs[int(rng.integers(len(xs)))]
    sets = {}
    for name in "ABCD":
        rows = []
        for _ in range(n):
            if name == "A":
                actor, other = pick(MEM), pick(CLEAN)
            elif name == "B":
                actor, other = pick(CLEAN), pick(MEM)
            elif name == "C":
                actor = pick(MEM)
                other = pick([m for m in MEM if m != actor])
            else:
                actor, other = pick(MIXED), pick(CLEAN)
            a, b = (actor, other) if rng.random() < 0.5 else (other, actor)
            text = R.story(rng, a=a, b=b, actor=0 if a == actor else 1, upto=True)
            rows.append((actor, text))
        sets[name] = rows
    return sets


def bow(words_in_view, recency):
    """Bag-of-words retrieval over each memory's description words."""
    scores = []
    for m in MEM:
        hits = [i for i, w in enumerate(words_in_view) if w in m]
        scores.append((len(set(words_in_view) & set(m)), max(hits) if (recency and hits) else 0))
    i = max(range(len(MEM)), key=lambda j: scores[j])
    return i, scores[i][0]


def run_bow(p, words, stoi, sets, thresh, recency):
    out = {}
    for name, rows in sets.items():
        got = []
        for actor, text in rows:
            ids = [stoi[w] for w in text.split()]
            view = text.split()[-S.T:]
            i, sc = bow(view, recency)
            o = []
            for _ in range(2):
                pr, _ = S.M.next_word_probs(p, ids + o, S.T, S.NH)
                lp = np.log(pr + 1e-12)
                if sc >= thresh:
                    for w in MEMORIES[MEM[i]]:
                        lp[stoi[w]] += 10
                o.append(int(lp.argmax()))
            got.append((actor, [words[t] for t in o]))
        out[name] = got
    return out


def run_settle(st, words, stoi, sets, tau):
    out = {}
    for name, rows in sets.items():
        got = []
        for actor, text in rows:
            o, _, _ = st.generate([stoi[w] for w in text.split()], 2, tau)
            got.append((actor, [words[t] for t in o]))
        out[name] = got
    return out


def run_alone(st, words, stoi, sets):
    return {name: [(a, [words[t] for t in st.draft([stoi[w] for w in x.split()], 2)]) for a, x in rows]
            for name, rows in sets.items()}


def score(res):
    acc = {n: np.mean([g == list(MEMORIES[a]) for a, g in res[n]]) for n in "AC"}
    false = {n: np.mean([g in PAIRS for a, g in res[n]]) for n in "BD"}
    total = (acc["A"] + acc["C"]) / 2 - (false["B"] + false["D"]) / 2
    return total, acc, false


def show(label, res):
    total, acc, false = score(res)
    print(f"{label:28s} score {total:+.1%} | A {acc['A']:.0%}  C {acc['C']:.0%} | "
          f"memory given wrongly: B {false['B']:.0%}  D {false['D']:.0%}")
    return total


def main():
    p, words, stoi = S.load()
    enc = lambda s: [stoi[w] for w in s.split()]

    # average state before the output, from ordinary corpus stories (not probes)
    rng = np.random.default_rng(900)
    mu = np.mean([S.path(p, enc(R.story(rng, upto=True)))[0][:, -1] for _ in range(200)], 0)
    st = S.Settle(p, mu)
    rng = np.random.default_rng(901)
    for m, find in MEMORIES.items():
        other = CLEAN[int(rng.integers(len(CLEAN)))]
        st.record(enc(R.story(rng, a=m, b=other, actor=0, upto=True)), enc(" ".join(find)))

    cal, test = probe_sets(1001, 60), probe_sets(2002, 100)

    # every knob on calibration only
    best = {}
    for recency in (False, True):
        best[recency] = max((score(run_bow(p, words, stoi, cal, th, recency))[0], th) for th in (1, 2))
    gates = [st.gate(enc(x))[1] for rows in cal.values() for _, x in rows]
    taus = sorted(set(np.round(np.percentile(gates, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]), 4)))
    best_s = None
    for b in (20, 40, 80, 160, 320):
        S.BETA = b
        for tau in taus:
            sc = score(run_settle(st, words, stoi, cal, tau))[0]
            if best_s is None or sc > best_s[0]:
                best_s = (sc, b, tau)
    print(f"calibration picks: bag-of-words threshold {best[False][1]}, with recency {best[True][1]}, "
          f"settle beta {best_s[1]} tau {best_s[2]}")

    # the test, once
    print(f"test: {sum(len(r) for r in test.values())} fresh probes")
    s_alone = show("model alone", run_alone(st, words, stoi, test))
    s_bow = show("bag-of-words", run_bow(p, words, stoi, test, best[False][1], False))
    s_rec = show("bag-of-words + recency", run_bow(p, words, stoi, test, best[True][1], True))
    S.BETA = best_s[1]
    s_set = show("settle", run_settle(st, words, stoi, test, best_s[2]))
    bar = max(s_bow, s_rec) + 0.10
    print(f"pass bar: beat the best bag-of-words by 10 points ({bar:+.1%}) and the model alone: "
          f"{'PASS' if s_set >= bar and s_set > s_alone else 'FAIL'}")


if __name__ == "__main__":
    main()
