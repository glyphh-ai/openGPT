"""Change 3 as said in design.md: each memory is the model's own generation,
written as the route of weights it used; recalled once at the start of the next
generation and pushing which weights fire.

Run:  uv run python research/routes_own.py      (calibration seed 1001, locked test seed 7007, once)
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

    # first pass: no key; the model generates, and that generation is written
    rng = np.random.default_rng(901)
    gen = np.random.default_rng(7001)
    for m in list(H.MEMORIES):
        other = H.CLEAN[int(rng.integers(len(H.CLEAN)))]
        prompt = enc(R.story(rng, a=m, b=other, actor=0, upto=True))
        out = []
        for _ in range(2):
            pr, _ = S.M.next_word_probs(p, prompt + out, S.T, S.NH, temperature=1.0)
            out.append(int(gen.choice(len(pr), p=pr)))
        mem.write(prompt + out, 2)
        H.MEMORIES[m] = tuple(words[i] for i in out)
        print(f"first pass: ...{dec(prompt[-6:])} -> {dec(out)}   (written into memory)")
    H.PAIRS[:] = [list(v) for v in H.MEMORIES.values()]

    def run(sets, b, g, a):
        return {n: [(act, [words[t] for t in mem.generate(enc(x), 2, b, g, a)]) for act, x in rows]
                for n, rows in sets.items()}

    cal, test = H.probe_sets(1001, 60), H.probe_sets(7007, 100)
    best = max(((H.score(run(cal, b, g, a))[0], b, g, a)
                for b, g, a in itertools.product(RT.BETAS, RT.GAMMAS, RT.ALPHAS)), key=lambda z: z[0])
    bow = {r: max((H.score(H.run_bow(p, words, stoi, cal, th, r))[0], th) for th in (1, 2))[1] for r in (False, True)}
    print(f"calibration picks: beta {best[1]}, gamma {best[2]}, alpha {best[3]} (calibration score {best[0]:+.1%})")

    st = S.Settle(p, None)
    print("locked test, seed 7007, run once:")
    H.show("model alone", H.run_alone(st, words, stoi, test))
    s1 = H.show("bag-of-words", H.run_bow(p, words, stoi, test, bow[False], False))
    s2 = H.show("bag-of-words + recency", H.run_bow(p, words, stoi, test, bow[True], True))
    res = run(test, *best[1:])
    s = H.show("route memory", res)
    print(f"pass bar {max(s1, s2) + 0.10:+.1%}: {'PASS' if s >= max(s1, s2) + 0.10 else 'FAIL'}")
    for n in "ABCD":
        a, x = test[n][0]
        right = " ".join(H.MEMORIES[a]) if a in H.MEMORIES else "(no memory)"
        print(f"  second pass {n}: ...{' '.join(x.split()[-8:])} -> {' '.join(res[n][0][1])}   remembered: {right}")


if __name__ == "__main__":
    main()
