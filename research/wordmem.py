"""Change 2 from design.md: the memory as its own word model, always on, with
attention in memory taught by the model's own generations.

Run:  uv run python research/wordmem.py      (trains W, u, b; then the locked test, seed 4004, once)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest as H  # noqa: E402
import settle as S  # noqa: E402

EPS, LR, PASSES, W0 = 1e-3, 1e-3, 3, 5.0
N_STORIES, N_EPISODES, N_MEM = 3000, 20000, 6
TEST_CHARS = set(H.MEMORIES)


def feats(hs, mu, pos):
    """State at one position: every layer, average removed, each layer at length 1."""
    h = hs[:, pos] - mu
    return (h / (np.linalg.norm(h, axis=-1, keepdims=True) + 1e-9)).reshape(-1)


def model_logp(logits_row):
    z = logits_row - logits_row.max()
    return z - np.log(np.exp(z).sum())


class WordMemory:
    def __init__(self, D, V):
        self.W = W0 * np.eye(D)
        self.u = np.zeros(D)
        self.b = 0.0
        self.V = V
        self.m = {k: 0.0 for k in ("W", "u", "b")}
        self.v = {k: 0.0 for k in ("W", "u", "b")}
        self.t = 0

    def mem_dist(self, q, K, vals):
        s = K @ (self.W.T @ q)                  # q^T W k_e for every entry
        a = np.exp(s - s.max())
        a /= a.sum()
        pm = np.full(self.V, EPS / self.V)
        np.add.at(pm, vals, (1 - EPS) * a)
        return a, pm

    def lam(self, q):
        z = self.u @ q + self.b
        return np.logaddexp(0, z), 1 / (1 + np.exp(-z))

    def combine(self, lp_model, q, K, vals):
        a, pm = self.mem_dist(q, K, vals)
        lam, sig = self.lam(q)
        lc = lp_model + lam * np.log(pm)
        lc = lc - lc.max()
        pc = np.exp(lc)
        pc /= pc.sum()
        return pc, a, pm, lam, sig

    def grads(self, lp_model, q, K, vals, t):
        pc, a, pm, lam, sig = self.combine(lp_model, q, K, vals)
        loss = -np.log(pc[t] + 1e-12)
        dpm = lam * (pc / pm)                   # dL/dp_mem(w)
        dpm[t] -= lam / pm[t]
        g = (1 - EPS) * dpm[vals]               # dL/da_e
        ds = a * (g - (a * g).sum())            # through the softmax
        dW = np.outer(q, ds @ K)                # sum_e ds_e q k_e^T
        dlam = -(np.log(pm[t]) - (pc * np.log(pm)).sum())
        return loss, dict(W=dW, u=dlam * sig * q, b=dlam * sig)

    def step(self, g):
        self.t += 1
        for k in g:
            self.m[k] = 0.9 * self.m[k] + 0.1 * g[k]
            self.v[k] = 0.999 * self.v[k] + 0.001 * g[k] ** 2
            upd = LR * (self.m[k] / (1 - 0.9 ** self.t)) / (np.sqrt(self.v[k] / (1 - 0.999 ** self.t)) + 1e-8)
            setattr(self, k, getattr(self, k) - upd)


def generate_stories(p, words, stoi, rng):
    """Stories the base model writes itself, cut at its own 'found a X Y'."""
    adj, animals = set(H.R.tt.ADJ), set(H.R.tt.ANIMALS)
    out, tries = [], 0
    while len(out) < N_STORIES:
        tries += 1
        ids = [stoi["<eot>"]]
        for _ in range(40):
            pr, _ = S.M.next_word_probs(p, ids, S.T, S.NH)
            ids.append(int(rng.choice(len(pr), p=pr)))
            w = [words[i] for i in ids]
            if len(w) >= 4 and w[-4] == "found" and w[-3] == "a":
                break
        w = [words[i] for i in ids]
        if not (len(w) >= 7 and w[-4] == "found" and w[-3] == "a" and w[-7] == "the"
                and w[-6] in adj and w[-5] in animals):
            continue
        chars = {(w[i], w[i + 1]) for i in range(len(w) - 1) if w[i] in adj and w[i + 1] in animals}
        if chars & TEST_CHARS:
            continue
        out.append(dict(ids=ids[1:], actor=(w[-6], w[-5])))
    return out, tries


def entries(p, mu, prompt, find):
    """Memory entries of one recorded story: the state before each output word, and the word."""
    hs, _ = S.path(p, prompt + find)
    n = hs.shape[1]
    return [feats(hs, mu, n - len(find) - 1 + j) for j in range(len(find))], list(find)


def answer(p, mu, mem, prompt, K, vals, k=2):
    out = []
    for _ in range(k):
        hs, logits = S.path(p, prompt + out)
        q = feats(hs, mu, hs.shape[1] - 1)
        pc, *_ = mem.combine(model_logp(logits[-1]), q, K, vals)
        out.append(int(pc.argmax()))
    return out


def main():
    p, words, stoi = S.load()
    enc = lambda s: [stoi[w] for w in s.split()]
    R = H.R
    rng = np.random.default_rng(900)
    corpus = [S.path(p, enc(R.story(rng)))[0] for _ in range(200)]
    mu = np.mean(np.concatenate([h for h in corpus], axis=1), axis=1)      # (L+1, C), over all positions

    rng = np.random.default_rng(5005)
    stories, tries = generate_stories(p, words, stoi, rng)
    print(f"the model wrote {tries} stories; {len(stories)} usable (end in its own 'found a X Y', no test character)")
    for s in stories:
        prompt, find = s["ids"][:-2], s["ids"][-2:]
        s["prompt"], s["find"] = prompt, find
        s["K"], s["vals"] = entries(p, mu, prompt, find)
    by_actor = {}
    for i, s in enumerate(stories):
        by_actor.setdefault(s["actor"], []).append(i)

    def episode():
        qi = int(rng.integers(len(stories)))
        qa = stories[qi]["actor"]
        others = [i for i in range(len(stories)) if stories[i]["actor"] != qa]
        mems, picked_actors = [], {qa}
        if rng.random() < 0.5 and len(by_actor[qa]) > 1:
            pos = int(rng.choice([i for i in by_actor[qa] if i != qi]))
            mems.append(pos)
            target = stories[pos]["find"]
        else:
            target = stories[qi]["find"]
        while len(mems) < N_MEM:
            j = others[int(rng.integers(len(others)))]
            if stories[j]["actor"] not in picked_actors:
                mems.append(j)
                picked_actors.add(stories[j]["actor"])
        return qi, mems, target

    mem = WordMemory(D=stories[0]["K"][0].shape[0], V=len(words))
    eps_all = [episode() for _ in range(N_EPISODES)]
    split = int(0.9 * len(eps_all))

    def run_episode(e, learn):
        qi, mems, target = e
        K = np.stack([k for j in mems for k in stories[j]["K"]])
        vals = np.array([v for j in mems for v in stories[j]["vals"]])
        prompt = stories[qi]["prompt"]
        hs, logits = S.path(p, prompt + target[:1])
        n = hs.shape[1]
        total, right = 0.0, True
        for j in range(2):
            q = feats(hs, mu, n - 2 + j)
            lp = model_logp(logits[n - 2 + j])
            if learn:
                loss, g = mem.grads(lp, q, K, vals, target[j])
                mem.step(g)
            else:
                pc, *_ = mem.combine(lp, q, K, vals)
                loss = -np.log(pc[target[j]] + 1e-12)
                right &= int(pc.argmax()) == target[j]
            total += loss
        return total, right

    for ep in range(PASSES):
        order = rng.permutation(split)
        tl = np.mean([run_episode(eps_all[i], True)[0] for i in order])
        vl = [run_episode(e, False) for e in eps_all[split:]]
        print(f"pass {ep + 1}: train loss {tl:.3f} | held-out generated episodes: loss "
              f"{np.mean([x[0] for x in vl]):.3f}, both words right {np.mean([x[1] for x in vl]):.1%}")

    # the six test memories, recorded exactly as in honest.py
    rng = np.random.default_rng(901)
    K, vals = [], []
    for m, find in H.MEMORIES.items():
        other = H.CLEAN[int(rng.integers(len(H.CLEAN)))]
        k, v = entries(p, mu, enc(R.story(rng, a=m, b=other, actor=0, upto=True)), enc(" ".join(find)))
        K += k
        vals += v
    K, vals = np.stack(K), np.array(vals)

    cal, test = H.probe_sets(1001, 60), H.probe_sets(4004, 100)
    bow = {r: max((H.score(H.run_bow(p, words, stoi, cal, th, r))[0], th) for th in (1, 2))[1] for r in (False, True)}
    st = S.Settle(p, mu)
    print("locked test, seed 4004, run once:")
    H.show("model alone", H.run_alone(st, words, stoi, test))
    s1 = H.show("bag-of-words", H.run_bow(p, words, stoi, test, bow[False], False))
    s2 = H.show("bag-of-words + recency", H.run_bow(p, words, stoi, test, bow[True], True))
    res = {n: [(a, [words[t] for t in answer(p, mu, mem, enc(x), K, vals)]) for a, x in rows] for n, rows in test.items()}
    s = H.show("word memory", res)
    print(f"pass bar {max(s1, s2) + 0.10:+.1%}: {'PASS' if s >= max(s1, s2) + 0.10 else 'FAIL'}")
    for n in "ABCD":
        a, x = test[n][0]
        print(f"  e.g. {n}: ...{' '.join(x.split()[-12:])} -> {' '.join(res[n][0][1])}")


if __name__ == "__main__":
    main()
