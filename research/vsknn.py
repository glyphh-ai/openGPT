"""Route memory vs kNN-LM, exactly as written in design.md before it ran.

Run:  uv run python research/vsknn.py
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitw as F    # noqa: E402
import routes as RT  # noqa: E402
import settle as S   # noqa: E402
from opengpt import data  # noqa: E402

p, words, stoi = F.p, F.words, F.stoi
N_MEM, N_PROBE = 50, 200


def heldout():
    ds = data.load("tinystories", 10)
    return [int(i) for i in ds.val]


def make_set(seed, val):
    rng = np.random.default_rng(seed)
    used = set()

    def span(n):
        while True:
            i = int(rng.integers(0, len(val) - n))
            if not any(j in used for j in range(i, i + n)) and 5 not in val[i:i + n]:   # no <eot>
                used.update(range(i, i + n))
                return val[i:i + n]

    mems = []
    for _ in range(N_MEM):
        lead = span(8)
        logits, _, _ = RT.forward(p, lead)
        ranked = np.argsort(-logits)
        told = int(ranked[int(rng.integers(10, 50))])
        mems.append((lead, told))
    partial = []
    for _ in range(N_PROBE):
        lead, told = mems[int(rng.integers(N_MEM))]
        partial.append((span(6) + lead[-3:], told))
    control = [span(10) for _ in range(N_PROBE)]
    return mems, partial, control


def final_state(ids):
    """The model's last state before the output (input to the head), every position."""
    _, cache = M.forward(p, np.array([ids[-S.T:]]), None, S.NH)
    return cache["lnf"][0]


M = S.M


class KNN:
    def __init__(self, mems):
        K, V = [], []
        for lead, told in mems:
            seq = lead + [told]
            h = final_state(seq)
            for j in range(len(seq) - 1):
                K.append(h[j])
                V.append(seq[j + 1])
        self.K, self.V = np.array(K), np.array(V)

    def next(self, ids, lam, tau):
        logits, _, _ = RT.forward(p, ids)
        pm = np.exp(logits - logits.max())
        pm /= pm.sum()
        q = final_state(ids)[-1]
        d = ((self.K - q) ** 2).sum(1)
        a = np.exp(-(d - d.min()) / tau)
        a /= a.sum()
        pk = np.zeros(len(pm))
        np.add.at(pk, self.V, a)
        return int(((1 - lam) * pm + lam * pk).argmax())


def route_next(ids, memories, beta, gamma):
    F.BETA, F.GAMMA = beta, gamma
    logits, routes, _ = RT.forward(p, ids)
    lp = logits - logits.max()
    lp -= np.log(np.exp(lp).sum())
    _, mi, j = F.best_place(routes, memories)
    fit = (F.first_block_routes(ids[-S.T + 1:]) == memories[mi][j][0]).mean(1)
    return int((lp + beta * fit).argmax())


def evaluate(next_fn, partial, control, alone):
    acc = np.mean([next_fn(x) == t for x, t in partial])
    moved = np.mean([next_fn(x) != a for x, a in zip(control, alone)])
    return acc - moved, acc, moved


def main():
    val = heldout()
    sets = {name: make_set(seed, val) for name, seed in (("cal", 11), ("test", 22))}
    prepared = {}
    for name, (mems, partial, control) in sets.items():
        routes_mem = [RT.forward(p, lead + [told])[1] for lead, told in mems]
        alone = [int(RT.forward(p, x)[0].argmax()) for x in control]
        prepared[name] = (KNN(mems), routes_mem, partial, control, alone)

    knn, rm, partial, control, alone = prepared["cal"]
    best_k = max(((evaluate(lambda x: knn.next(x, l, t), partial, control, alone)[0], l, t)
                  for l, t in itertools.product((0.25, 0.5, 0.75, 0.9), (1, 10, 100))), key=lambda z: z[0])
    best_r = max(((evaluate(lambda x: route_next(x, rm, b, g), partial, control, alone)[0], b, g)
                  for b, g in itertools.product((50, 100, 200, 400), (0.5, 0.8))), key=lambda z: z[0])
    print(f"calibration picks: kNN-LM lam {best_k[1]} tau {best_k[2]} ({best_k[0]:+.1%}); "
          f"route beta {best_r[1]} gamma {best_r[2]} ({best_r[0]:+.1%})")

    knn, rm, partial, control, alone = prepared["test"]
    rows = {
        "model alone": evaluate(lambda x: int(RT.forward(p, x)[0].argmax()), partial, control, alone),
        "kNN-LM": evaluate(lambda x: knn.next(x, best_k[1], best_k[2]), partial, control, alone),
        "route memory": evaluate(lambda x: route_next(x, rm, best_r[1], best_r[2]), partial, control, alone),
    }
    print("locked test, seed 22, run once (50 memories, 200 partial, 200 control):")
    print("| system | score | partial: told word recalled | control: next word changed |\n|---|---|---|---|")
    for k, (s, a, m) in rows.items():
        print(f"| {k} | {s:+.1%} | {a:.1%} | {m:.1%} |")
    gap = rows["route memory"][0] - rows["kNN-LM"][0]
    print(f"route memory minus kNN-LM: {gap:+.1%}; bar +10.0%: {'PASS' if gap >= 0.10 else 'FAIL'}")
    x, t = partial[0]
    print(f"\ne.g. Prompt: {F.dec(x)}\n     told word: {words[t]} | route: {words[route_next(x, rm, best_r[1], best_r[2])]}"
          f" | kNN-LM: {words[knn.next(x, best_k[1], best_k[2])]} | model alone: {words[int(RT.forward(p, x)[0].argmax())]}")


if __name__ == "__main__":
    main()
