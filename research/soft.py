"""Change 1 from design.md: no gate; every memory always pulls, weighted by overlap.

Run:  uv run python research/soft.py      (calibration seed 1001, locked test seed 3003, once)
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import honest as H  # noqa: E402
import settle as S  # noqa: E402


class Soft(S.Settle):
    def weights(self, prompt, temp):
        hs, _ = S.path(self.p, prompt)
        ov = np.array([S.cos(hs[:, -1], r["key"], self.mu) for r in self.records])
        z = np.exp((ov - ov.max()) / temp)
        return z / z.sum()

    def energy_soft(self, prompt, out, w):
        hs, logits = S.path(self.p, prompt + out)
        k, n = len(out), min(len(prompt + out), S.T)
        lp = 0.0
        for j in range(k):
            z = logits[n - k - 1 + j]
            z = z - z.max()
            lp += z[out[j]] - np.log(np.exp(z).sum())
        pull = sum(wr * S.cos(hs[:, -k:], r["val"][:, :k], self.mu[:, None]) for wr, r in zip(w, self.records))
        return lp + S.BETA * pull

    def generate_soft(self, prompt, k, temp):
        w = self.weights(prompt, temp)
        E = lambda o: self.energy_soft(prompt, o, w)
        out = []
        for j in range(k):                      # lay down left to right, then sweep to a fixed point
            pr, _ = S.M.next_word_probs(self.p, prompt + out, S.T, S.NH)
            out.append(max((int(c) for c in np.argsort(-pr)[:S.TOP_M]), key=lambda c: E(out + [c])))
        for _ in range(S.MAX_SWEEPS):
            changed = False
            for j in range(k):
                pr, _ = S.M.next_word_probs(self.p, prompt + out[:j], S.T, S.NH)
                best = max((int(c) for c in np.argsort(-pr)[:S.TOP_M]), key=lambda c: E(out[:j] + [c] + out[j + 1:]))
                if best != out[j]:
                    out[j], changed = best, True
            if not changed:
                break
        return out


def run(st, words, enc, sets, temp):
    return {n: [(a, [words[t] for t in st.generate_soft(enc(x), 2, temp)]) for a, x in rows] for n, rows in sets.items()}


def main():
    p, words, stoi = S.load()
    enc = lambda s: [stoi[w] for w in s.split()]
    R = H.R
    rng = np.random.default_rng(900)
    mu = np.mean([S.path(p, enc(R.story(rng, upto=True)))[0][:, -1] for _ in range(200)], 0)
    st = Soft(p, mu)
    rng = np.random.default_rng(901)
    for m, find in H.MEMORIES.items():
        other = H.CLEAN[int(rng.integers(len(H.CLEAN)))]
        st.record(enc(R.story(rng, a=m, b=other, actor=0, upto=True)), enc(" ".join(find)))

    cal, test = H.probe_sets(1001, 60), H.probe_sets(3003, 100)
    best = None
    for b in (20, 40, 80, 160, 320):
        S.BETA = b
        for temp in (0.01, 0.03, 0.1, 0.3):
            sc = H.score(run(st, words, enc, cal, temp))[0]
            if best is None or sc > best[0]:
                best = (sc, b, temp)
    bow = {r: max((H.score(H.run_bow(p, words, stoi, cal, th, r))[0], th) for th in (1, 2))[1] for r in (False, True)}
    print(f"calibration picks: beta {best[1]}, temp {best[2]}; bag-of-words thresholds {bow}")

    print("locked test, seed 3003, run once:")
    H.show("model alone", H.run_alone(st, words, stoi, test))
    s1 = H.show("bag-of-words", H.run_bow(p, words, stoi, test, bow[False], False))
    s2 = H.show("bag-of-words + recency", H.run_bow(p, words, stoi, test, bow[True], True))
    S.BETA = best[1]
    s = H.show("settle, no gate", run(st, words, enc, test, best[2]))
    print(f"pass bar {max(s1, s2) + 0.10:+.1%}: {'PASS' if s >= max(s1, s2) + 0.10 else 'FAIL'}")


if __name__ == "__main__":
    main()
