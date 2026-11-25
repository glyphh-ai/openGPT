"""The two curves (design.md): recall against unrelated prompts changed, for
kNN-LM and the route memory with attention on the deltas, over every setting.

Run:  uv run python research/curves.py      (calibration seed 11, locked test seed 44, once)
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import attn_deltas as A  # noqa: E402
import fitw as F         # noqa: E402
import routes as RT      # noqa: E402
import settle as S       # noqa: E402
import vsknn as K        # noqa: E402

p = F.p
KNN_GRID = list(itertools.product((0.1, 0.25, 0.5, 0.75, 0.9, 0.97), (0.3, 1, 3, 10, 30, 100)))
ROUTE_GRID = list(itertools.product((0, 10, 30, 100, 300), (30, 100, 300, 1000, 3000, 10000, 30000, 100000)))
BUDGETS = (0.05, 0.10, 0.20, 0.40)
TEST_SEED, NEED = 44, 3
if "wide" in sys.argv[1:]:      # the wider range, design.md
    KNN_GRID = list(itertools.product((0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9), (3, 10, 15, 20, 30, 50)))
    ROUTE_GRID = list(itertools.product((50, 70, 100, 150, 200), (1e3, 3e3, 1e4, 3e4, 1e5, 3e5, 1e6, 3e6, 1e7)))
    BUDGETS = (0.05, 0.10, 0.20, 0.40, 0.60, 0.80)
    TEST_SEED, NEED = 55, 4


def knn_words(knn, ids):
    """The next word under every kNN-LM setting, from one pass."""
    logits, _, _ = RT.forward(p, ids)
    pm = np.exp(logits - logits.max())
    pm /= pm.sum()
    d = ((knn.K - K.final_state(ids)[-1]) ** 2).sum(1)
    out = {}
    for lam, tau in KNN_GRID:
        a = np.exp(-(d - d.min()) / tau)
        a /= a.sum()
        pk = np.zeros(len(pm))
        np.add.at(pk, knn.V, a)
        out[(lam, tau)] = int(((1 - lam) * pm + lam * pk).argmax())
    return out


def route_words(places, ids):
    """The next word under every route-memory setting, from one pass."""
    logits, routes, _ = RT.forward(p, ids)
    lp = logits - logits.max()
    lp -= np.log(np.exp(lp).sum())
    deltas = places.deltas(routes)
    R = F.first_block_routes(ids[-S.T + 1:]).astype(float)
    fit = (R @ places.nxt.T + (1 - R) @ (1 - places.nxt).T) / R.shape[1]
    return {(s, b): int((lp + b * fit @ np.exp(-s * deltas)).argmax()) for s, b in ROUTE_GRID}


def curves(seed, val):
    mems, partial, control = K.make_set(seed, val)
    knn = K.KNN(mems)
    places = A.Places([RT.forward(p, lead + [told])[1] for lead, told in mems])
    alone = [int(RT.forward(p, x)[0].argmax()) for x in control]
    res = {}
    for name, fn, grid in (("kNN-LM", lambda x: knn_words(knn, x), KNN_GRID),
                           ("route memory", lambda x: route_words(places, x), ROUTE_GRID)):
        part = [fn(x) for x, _ in partial]
        ctrl = [fn(x) for x in control]
        res[name] = {g: (np.mean([w[g] == t for w, (_, t) in zip(part, partial)]),
                         np.mean([w[g] != a for w, a in zip(ctrl, alone)])) for g in grid}
    return res


def main():
    val = K.heldout()
    cal, test = curves(11, val), curves(TEST_SEED, val)
    print(f"locked test, seed {TEST_SEED}, run once; settings picked on calibration (seed 11) per budget:")
    print("| budget (unrelated changed) | kNN-LM recall | kNN-LM changed | route recall | route changed | recall gap |")
    print("|---|---|---|---|---|---|")
    wins = 0
    for b in BUDGETS:
        row, got = [], {}
        for name in ("kNN-LM", "route memory"):
            ok = [(r, g) for g, (r, c) in cal[name].items() if c <= b]
            if not ok:
                got[name] = None
                row += ["none within budget", ""]
                continue
            g = max(ok, key=lambda z: z[0])[1]
            got[name] = test[name][g]
            row += [f"{got[name][0]:.1%}", f"{got[name][1]:.1%}"]
        k, r = got["kNN-LM"], got["route memory"]
        gap = ""
        if k and r:
            gap = f"{r[0] - k[0]:+.1%}"
            wins += (r[0] - k[0] >= 0.10) and (r[1] - k[1] <= 0.05)
        print(f"| {b:.0%} | {' | '.join(row)} | {gap} |")
    print(f"budgets won by the route memory: {wins} of {len(BUDGETS)}; bar {NEED}: {'PASS' if wins >= NEED else 'FAIL'}")
    for name in ("kNN-LM", "route memory"):
        print(f"\n{name}, every setting on the test set (changed -> recall):")
        for g, (r, c) in sorted(test[name].items(), key=lambda z: (z[1][1], -z[1][0])):
            print(f"  {g}: changed {c:.1%}, recall {r:.1%}")


if __name__ == "__main__":
    main()
