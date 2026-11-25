"""Attention on the deltas with a scale (design.md): every memory place pulls the
next word with weight exp(-scale * delta), not normalised, no switch.

Run:  uv run python research/attn_deltas.py      (calibration seed 11, locked test seed 33, once)
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fitw as F     # noqa: E402
import routes as RT  # noqa: E402
import settle as S   # noqa: E402
import vsknn as K    # noqa: E402

p, words = F.p, F.words
GAMMA = 0.5


class Places:
    """Every place j >= 1 of every memory: its route history and its next route (block 1)."""

    def __init__(self, memories):
        self.hist, self.nxt = [], []
        for m in memories:
            for j in range(1, len(m)):
                self.hist.append(m[:j].reshape(j, -1))      # routes before the place, (j, units)
                self.nxt.append(m[j][0])                    # block-1 route at the place
        self.nxt = np.array(self.nxt, dtype=float)          # (P, H)

    def deltas(self, routes):
        r = routes.reshape(len(routes), -1)
        out = np.empty(len(self.hist))
        for i, h in enumerate(self.hist):
            L = min(len(r), len(h))
            w = GAMMA ** np.arange(L)
            match = np.array([(r[-1 - d] == h[-1 - d]).mean() for d in range(L)])
            out[i] = 1 - (w * match).sum() / w.sum()
        return out


def next_word(ids, places, scale, beta):
    logits, routes, _ = RT.forward(p, ids)
    lp = logits - logits.max()
    lp -= np.log(np.exp(lp).sum())
    w = np.exp(-scale * places.deltas(routes))              # attention on the deltas, not normalised
    R = F.first_block_routes(ids[-S.T + 1:]).astype(float)  # (V, H) every word's block-1 route
    H = R.shape[1]
    fit = (R @ places.nxt.T + (1 - R) @ (1 - places.nxt).T) / H   # (V, P)
    return int((lp + beta * fit @ w).argmax())


def main():
    val = K.heldout()
    out = {}
    for name, seed in (("cal", 11), ("test", 33)):
        mems, partial, control = K.make_set(seed, val)
        places = Places([RT.forward(p, lead + [told])[1] for lead, told in mems])
        alone = [int(RT.forward(p, x)[0].argmax()) for x in control]
        out[name] = (K.KNN(mems), places, partial, control, alone)

    knn, pl, partial, control, alone = out["cal"]
    bk = max(((K.evaluate(lambda x: knn.next(x, l, t), partial, control, alone)[0], l, t)
              for l, t in itertools.product((0.25, 0.5, 0.75, 0.9), (1, 10, 100))), key=lambda z: z[0])
    ba = max(((K.evaluate(lambda x: next_word(x, pl, s, b), partial, control, alone)[0], s, b)
              for s, b in itertools.product((10, 30, 100), (100, 1000, 10000))), key=lambda z: z[0])
    print(f"calibration picks: kNN-LM lam {bk[1]} tau {bk[2]} ({bk[0]:+.1%}); "
          f"deltas scale {ba[1]} beta {ba[2]} ({ba[0]:+.1%})")

    knn, pl, partial, control, alone = out["test"]
    rows = {
        "model alone": K.evaluate(lambda x: int(RT.forward(p, x)[0].argmax()), partial, control, alone),
        "kNN-LM": K.evaluate(lambda x: knn.next(x, bk[1], bk[2]), partial, control, alone),
        "route memory, attention on deltas": K.evaluate(lambda x: next_word(x, pl, ba[1], ba[2]), partial, control, alone),
    }
    print("locked test, seed 33, run once (50 memories, 200 partial, 200 control):")
    print("| system | score | partial: told word recalled | control: next word changed |\n|---|---|---|---|")
    for k, (s, a, m) in rows.items():
        print(f"| {k} | {s:+.1%} | {a:.1%} | {m:.1%} |")
    gap = rows["route memory, attention on deltas"][0] - rows["kNN-LM"][0]
    print(f"minus kNN-LM: {gap:+.1%}; bar +10.0%: {'PASS' if gap >= 0.10 else 'FAIL'}")


if __name__ == "__main__":
    main()
