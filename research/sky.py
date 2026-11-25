"""Prompt 1 tells, prompt 2 asks: the route of prompt 1's pass is the memory.

On the TinyStories run (it knows words like "sky"). Routes are lined up from the
start: the two passes begin alike, and the recalled route for each next word is
the memory's route at the same place.

Run:  uv run python research/sky.py "the sky is red" "the sky is"
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routes as RT  # noqa: E402
import settle as S   # noqa: E402

RUNS = {   # run id: (corpus, T, heads)
    "20261115-082630": ("tinystories", 32, 1),
    "20261115-091000": ("tinytales", 16, 4),
    "20261122-071357": ("tinystories", 32, 4),     # 3 blocks x 128, 8,000 steps
}
N_WORDS, ALPHA = 8, 4.0


def main(first, second, run="20261115-082630"):
    corpus_name, S.T, S.NH = RUNS[run]
    S.RUN = S.ROOT / "runs" / run
    p, words, stoi = S.load()
    from opengpt import data
    enc = lambda s: [stoi.get(w, 0) for w in data.tokenize(s)]
    dec = lambda ids: " ".join(words[i] for i in ids)

    corpus = enc(data.corpus_path(corpus_name).read_text()[:400000])
    rng = np.random.default_rng(0)
    routes, pres = [], []
    for i in rng.integers(0, len(corpus) - S.T, 200):
        _, r, z = RT.forward(p, corpus[i:i + S.T])
        routes.append(r)
        pres.append(z)
    rate, sd = np.concatenate(routes).mean(0), np.concatenate(pres).std(0)

    def generate(prompt, memory=None):
        out = []
        for _ in range(N_WORDS):
            push = None
            pos = len(prompt) + len(out) - 1           # this word's place, lined up from the start
            if memory is not None and pos < len(memory):
                push = ALPHA * sd * (memory[pos] - rate)
            logits, _, _ = RT.forward(p, prompt + out, push)
            out.append(int(logits.argmax()))
        return out

    p1 = enc(first)
    r1 = generate(p1)
    _, memory, _ = RT.forward(p, p1 + r1)          # the whole first pass, written as its route
    p2 = enc(second)
    r2 = generate(p2, memory.astype(float))
    print(f"Baseline, prompt 2 with no memory: {second}\nResponse: {dec(generate(p2))}\n")
    print(f"Prompt 1: {first}\nResponse: {dec(r1)}\n\nPrompt 2: {second}\nResponse: {dec(r2)}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
