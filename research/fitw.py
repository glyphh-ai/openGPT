"""Memory = the routes (which units fired) of earlier passes. A new generation,
at every next word: search all memories for the place whose route history best
fits the routes so far, then use the first block's weights to find, in one
calculation over the whole vocabulary, the word that would fire that place's
next route. The model's own probability and that fit both pull; no switch.

Run:  uv run python research/fitw.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import routes as RT  # noqa: E402
import settle as S   # noqa: E402
from opengpt import data  # noqa: E402

M = S.M
S.RUN = S.ROOT / "runs" / "20261122-071357"     # TinyStories, 3 blocks x 128, T=32
S.T, S.NH = 32, 4
BETA, GAMMA, N_WORDS = 200.0, 0.8, 10
p, words, stoi = S.load()
enc = lambda s: [stoi[w] for w in data.tokenize(s)]
dec = lambda ids: " ".join(words[i] for i in ids)


def first_block_routes(prefix):
    """Which first-block units fire at the next position, for every word in the
    vocabulary at once: (V, H) on/off. Exact for block 1 (its attention included)."""
    lp = M.layer(p, 0)
    t = len(prefix)
    xp = p["tok_emb"][prefix] + p["pos_emb"][:t]                 # (t, C)
    xc = p["tok_emb"] + p["pos_emb"][t]                          # (V, C)
    lnp, _ = M.layernorm_fwd(xp, lp["g1"], lp["b1"])
    lnc, _ = M.layernorm_fwd(xc, lp["g1"], lp["b1"])
    hd = xp.shape[1] // S.NH
    heads = lambda x: x.reshape(x.shape[0], S.NH, hd)
    kp, vp = heads(lnp @ lp["Wk"]), heads(lnp @ lp["Wv"])       # (t, nh, hd)
    qc, kc, vc = heads(lnc @ lp["Wq"]), heads(lnc @ lp["Wk"]), heads(lnc @ lp["Wv"])
    sp = np.einsum("vnd,tnd->vnt", qc, kp) / np.sqrt(hd)
    sc = np.einsum("vnd,vnd->vn", qc, kc)[..., None] / np.sqrt(hd)
    s = np.concatenate([sp, sc], -1)
    a = np.exp(s - s.max(-1, keepdims=True))
    a /= a.sum(-1, keepdims=True)
    out = np.einsum("vnt,tnd->vnd", a[..., :t], vp) + a[..., t:] * vc
    h1 = xc + out.reshape(len(xc), -1) @ lp["Wo"]
    ln2, _ = M.layernorm_fwd(h1, lp["g2"], lp["b2"])
    return (ln2 @ lp["W1"] + lp["b1m"]) > 0


def best_place(routes_so_far, memories):
    """The memory place whose route history fits the routes so far best (Hopfield
    recall over every memory, the nearest words weighing most)."""
    best = (-1.0, None, None)
    t = len(routes_so_far)
    for mi, m in enumerate(memories):
        for j in range(1, len(m)):
            L = min(t, j)
            w_ = GAMMA ** np.arange(L)
            f = sum(w_[d] * (routes_so_far[t - 1 - d] == m[j - 1 - d]).mean() for d in range(L)) / w_.sum()
            if f > best[0]:
                best = (f, mi, j)
    return best


def generate(prompt, memories, show=False):
    ids = enc(prompt)
    out = []
    for step in range(N_WORDS):
        logits, routes, _ = RT.forward(p, ids + out)
        lp = logits - logits.max()
        lp -= np.log(np.exp(lp).sum())
        fit_hist, mi, j = best_place(routes, memories)
        target = memories[mi][j][0]                              # first block's route at that place
        fit = (first_block_routes((ids + out)[-S.T + 1:]) == target).mean(1)
        score = lp + BETA * fit
        c = int(score.argmax())
        if show and step == 0:
            print(f"  memory {mi + 1} place {j + 1} fits the routes so far {fit_hist:.1%}")
            print("  | word | fit to that place's next route | model probability |")
            for k in np.argsort(-score)[:4]:
                print(f"  | {words[k]} | {fit[k]:.1%} | {np.exp(lp[k]):.1%} |")
            k = int(lp.argmax())
            print(f"  | {words[k]} (model's choice) | {fit[k]:.1%} | {np.exp(lp[k]):.1%} |")
        out.append(c)
    return dec(out)


def main():
    told = ["once upon a day , there was a little dog named max .",
            "the sky is red and the sun is very hot .",
            "lily went to the moon with her friend tom ."]
    memories = [RT.forward(p, enc(s))[1] for s in told]           # first pass: only written
    for s in told:
        print(f"memory: {s}")
    # check the one-calculation route against the full forward pass
    pre = enc("once upon a")
    _, r, _ = RT.forward(p, pre + [stoi["day"]])
    assert (first_block_routes(pre)[stoi["day"]] == r[-1][0]).all()
    for prompt in ["once upon a", "the sky is", "lily went to the", "one day , a little"]:
        print(f"\nPrompt: {prompt}")
        print(f"Response with memory: {generate(prompt, memories, show=True)}")
        print(f"Response, model alone: {dec(model_alone(prompt))}")


def model_alone(prompt):
    ids = enc(prompt)
    out = []
    for _ in range(N_WORDS):
        out.append(int(RT.forward(p, ids + out)[0].argmax()))
    return out


if __name__ == "__main__":
    main()
