"""Change 3 with something to remember (design.md): the first prompt tells the
fact, the model generates from it, and the route of that pass is the memory; the
second prompt asks without the fact.

Run:  uv run python research/routes_told.py      (calibration seed 1001, locked test seed 8008, once)
"""

import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest as H   # noqa: E402
import routes as RT  # noqa: E402
import settle as S   # noqa: E402


def main():
    p, words, stoi = S.load()
    enc = lambda s: [stoi[w] for w in s.split()]
    dec = lambda ids: " ".join(words[i] for i in ids)
    R = H.R
    rng = np.random.default_rng(900)
    routes, pres = [], []
    for _ in range(200):
        _, r, z = RT.forward(p, enc(R.story(rng)))
        routes.append(r)
        pres.append(z)
    mem = RT.Routes(p, np.concatenate(routes).mean(0), np.concatenate(pres).std(0))
    st = S.Settle(p, None)

    rng = np.random.default_rng(901)
    for m, find in H.MEMORIES.items():
        other = H.CLEAN[int(rng.integers(len(H.CLEAN)))]
        full = R.story(rng, a=m, b=other, actor=0, find=find)
        told = full[: full.index('" what a') + len('" what a')]
        out = st.draft(enc(told), 2)               # the model generates from what it was told
        mem.write(enc(told) + out, 2)
        print(f"PROMPT 1: {told}\nOUTPUT 1: {dec(out)}\n")

    def run(sets, b, g, a):
        return {n: [(act, [words[t] for t in mem.generate(enc(x), 2, b, g, a)]) for act, x in rows]
                for n, rows in sets.items()}

    cal, test = H.probe_sets(1001, 60), H.probe_sets(8008, 100)
    best = max(((H.score(run(cal, b, g, a))[0], b, g, a)
                for b, g, a in itertools.product(RT.BETAS, RT.GAMMAS, RT.ALPHAS)), key=lambda z: z[0])
    bow = {r: max((H.score(H.run_bow(p, words, stoi, cal, th, r))[0], th) for th in (1, 2))[1] for r in (False, True)}
    res = run(test, *best[1:])
    for n in "AC":
        a, x = test[n][0]
        print(f"PROMPT 2 ({n}): {x}\nOUTPUT 2: {' '.join(res[n][0][1])}   (told: {' '.join(H.MEMORIES[a])})\n")
    print(f"knobs from calibration: beta {best[1]}, gamma {best[2]}, alpha {best[3]}")
    print("locked test, seed 8008, run once:")
    H.show("model alone", H.run_alone(st, words, stoi, test))
    s1 = H.show("bag-of-words", H.run_bow(p, words, stoi, test, bow[False], False))
    s2 = H.show("bag-of-words + recency", H.run_bow(p, words, stoi, test, bow[True], True))
    s = H.show("route memory", res)
    print(f"pass bar {max(s1, s2) + 0.10:+.1%}: {'PASS' if s >= max(s1, s2) + 0.10 else 'FAIL'}")


if __name__ == "__main__":
    main()
