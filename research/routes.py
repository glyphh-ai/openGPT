"""Change 3 from design.md: memory of the route (which weights fired), recalled
once at the start of generation, Hopfield-style, then pushing which weights fire.

Run:  uv run python research/routes.py      (calibration seed 1001, locked test seed 6006, once)
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest as H  # noqa: E402
import settle as S  # noqa: E402

M = S.M
BETAS, GAMMAS, ALPHAS = (5, 20, 80), (0.5, 0.8, 1.0), (1, 2, 4)


def forward(p, ids, push=None):
    """The model's own forward pass; returns last-position logits, the route at
    every position (MLP units on/off, all blocks) and the pre-activations.
    push: (n_blocks, H) added to the last position's MLP pre-activations."""
    x = np.array([ids[-S.T:]])
    h = M.add_position(p, M.token_embedding(p, x))
    route, pre = [], []
    for l in range(M.n_layers(p)):
        lp = M.layer(p, l)
        ln1, _ = M.layernorm_fwd(h, lp["g1"], lp["b1"])
        q, k, v = M.project_qkv(ln1, lp, S.NH)
        scores, _ = M.attention_scores(q, k)
        attn, _ = M.attention_output(M.softmax(scores), v, lp)
        h = h + attn
        ln2, _ = M.layernorm_fwd(h, lp["g2"], lp["b2"])
        z = ln2 @ lp["W1"] + lp["b1m"]
        if push is not None:
            z[0, -1] += push[l]
        pre.append(z[0])
        act = z * (z > 0)
        h = h + act @ lp["W2"] + lp["b2m"]
        route.append(z[0] > 0)
    lnf, _ = M.layernorm_fwd(h, p["gf"], p["bf"])
    logits = lnf @ p["Wh"] + p["bh"]
    return logits[0, -1], np.stack(route, 1), np.stack(pre, 1)   # route: (n, blocks, H)


class Routes:
    def __init__(self, p, rate, sd):
        self.p, self.rate, self.sd, self.mem = p, rate, sd, []

    def write(self, ids, n_out):
        """Keep the route of a pass: input part and the positions before each output word."""
        _, route, _ = forward(self.p, ids)
        n = route.shape[0]
        self.mem.append(dict(inp=route[: n - n_out], out=route[n - n_out - 1: n - 1]))

    def recall(self, prompt, beta, gamma):
        _, route, _ = forward(self.p, prompt)
        sims = []
        for m in self.mem:
            L = min(len(route), len(m["inp"]))
            s = 0.0
            for d in range(L):
                a = (route[-1 - d] - self.rate).reshape(-1)
                b = (m["inp"][-1 - d] - self.rate).reshape(-1)
                s += gamma ** d * a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)
            sims.append(s / sum(gamma ** d for d in range(L)))
        sims = np.array(sims)
        a = np.exp(beta * (sims - sims.max()))
        a /= a.sum()
        return np.einsum("g,gjbh->jbh", a, np.stack([m["out"].astype(float) for m in self.mem]))

    def generate(self, prompt, k, beta, gamma, alpha):
        R = self.recall(prompt, beta, gamma)           # once, at the start
        out = []
        for j in range(k):
            push = alpha * self.sd * (R[j] - self.rate)
            logits, _, _ = forward(self.p, prompt + out, push)
            out.append(int(logits.argmax()))
        return out


def main():
    p, words, stoi = S.load()
    enc = lambda s: [stoi[w] for w in s.split()]
    R = H.R
    rng = np.random.default_rng(900)
    routes, pres = [], []
    for _ in range(200):
        _, r, z = forward(p, enc(R.story(rng)))
        routes.append(r)
        pres.append(z)
    rate = np.concatenate(routes).mean(0)           # (blocks, H) firing rate per unit
    sd = np.concatenate(pres).std(0)

    mem = Routes(p, rate, sd)
    rng = np.random.default_rng(901)                # the six memories, as in honest.py
    for m, find in H.MEMORIES.items():
        other = H.CLEAN[int(rng.integers(len(H.CLEAN)))]
        mem.write(enc(R.story(rng, a=m, b=other, actor=0, upto=True)) + enc(" ".join(find)), 2)

    def run(sets, b, g, a):
        return {n: [(act, [words[t] for t in mem.generate(enc(x), 2, b, g, a)]) for act, x in rows]
                for n, rows in sets.items()}

    cal, test = H.probe_sets(1001, 60), H.probe_sets(6006, 100)
    best = max(((H.score(run(cal, b, g, a))[0], b, g, a) for b, g, a in itertools.product(BETAS, GAMMAS, ALPHAS)),
               key=lambda z: z[0])
    bow = {r: max((H.score(H.run_bow(p, words, stoi, cal, th, r))[0], th) for th in (1, 2))[1] for r in (False, True)}
    print(f"calibration picks: beta {best[1]}, gamma {best[2]}, alpha {best[3]} (calibration score {best[0]:+.1%})")

    st = S.Settle(p, None)
    print("locked test, seed 6006, run once:")
    H.show("model alone", H.run_alone(st, words, stoi, test))
    s1 = H.show("bag-of-words", H.run_bow(p, words, stoi, test, bow[False], False))
    s2 = H.show("bag-of-words + recency", H.run_bow(p, words, stoi, test, bow[True], True))
    res = run(test, *best[1:])
    s = H.show("route memory", res)
    print(f"pass bar {max(s1, s2) + 0.10:+.1%}: {'PASS' if s >= max(s1, s2) + 0.10 else 'FAIL'}")
    for n in "ABCD":
        a, x = test[n][0]
        right = " ".join(H.MEMORIES[a]) if a in H.MEMORIES else "(no memory)"
        print(f"  {n}: ...{' '.join(x.split()[-10:])} -> {' '.join(res[n][0][1])}   right: {right}")


if __name__ == "__main__":
    main()
