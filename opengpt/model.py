"""A word-level GPT in plain numpy, one function per step.

Forward, hand-written backward and Adam, with any number of blocks and
attention heads. Grown from thinktank/version_0_base (single block, single
head) and version_14/15 (multi-head attention). With n_layer = 1 and
n_head = 1 it is exactly the version_0_base model.

Shapes: B batch, T context length, C width, V vocabulary, H MLP hidden width,
nh heads, hd = C / nh head width.

Everything here is meant to be edited: the app reloads this file at the start
of every run and the Learn tab reads its functions straight from the source.
"""

import math

import numpy as np

EPS = 1e-5      # layernorm epsilon

LAYER_KEYS = ("g1", "b1", "Wq", "Wk", "Wv", "Wo", "g2", "b2", "W1", "b1m", "W2", "b2m")


# ---- parameters -----------------------------------------------------------------
def init_params(rng, V, T, C, H, n_layer=1, dtype=np.float64):
    def n(*shape, scale):
        return rng.normal(0, scale, size=shape).astype(dtype)
    p = {
        "tok_emb": n(V, C, scale=0.02),
        "pos_emb": n(T, C, scale=0.02),
    }
    for l in range(n_layer):
        p.update({
            f"L{l}.g1": np.ones(C, dtype), f"L{l}.b1": np.zeros(C, dtype),
            f"L{l}.Wq": n(C, C, scale=1/math.sqrt(C)),
            f"L{l}.Wk": n(C, C, scale=1/math.sqrt(C)),
            f"L{l}.Wv": n(C, C, scale=1/math.sqrt(C)),
            f"L{l}.Wo": n(C, C, scale=1/math.sqrt(C)),
            f"L{l}.g2": np.ones(C, dtype), f"L{l}.b2": np.zeros(C, dtype),
            f"L{l}.W1": n(C, H, scale=1/math.sqrt(C)), f"L{l}.b1m": np.zeros(H, dtype),
            f"L{l}.W2": n(H, C, scale=1/math.sqrt(H)), f"L{l}.b2m": np.zeros(C, dtype),
        })
    p.update({
        "gf": np.ones(C, dtype), "bf": np.zeros(C, dtype),
        "Wh": n(C, V, scale=1/math.sqrt(C)), "bh": np.zeros(V, dtype),
    })
    return p


def n_layers(p):
    return sum(1 for k in p if k.endswith(".Wq"))


def layer(p, l):
    """The weights of block l under their plain names (views, not copies)."""
    return {k: p[f"L{l}.{k}"] for k in LAYER_KEYS}


# ---- forward --------------------------------------------------------------------
def token_embedding(p, x):
    return p["tok_emb"][x]                       # (B,T) ids -> (B,T,C)


def add_position(p, tok):
    T = tok.shape[1]
    return tok + p["pos_emb"][:T]                # (T,C) broadcasts over B


def layernorm_fwd(x, g, b):
    mu = x.mean(-1, keepdims=True)
    var = x.var(-1, keepdims=True)
    istd = 1.0 / np.sqrt(var + EPS)
    xhat = (x - mu) * istd
    out = g * xhat + b
    return out, (xhat, istd, g)


def split_heads(x, n_head):
    B, T, C = x.shape                            # (B,T,C) -> (B,nh,T,hd)
    return x.reshape(B, T, n_head, C // n_head).transpose(0, 2, 1, 3)


def merge_heads(x):
    B, nh, T, hd = x.shape                       # (B,nh,T,hd) -> (B,T,C)
    return x.transpose(0, 2, 1, 3).reshape(B, T, nh * hd)


def project_qkv(x, lp, n_head=1):
    q = split_heads(x @ lp["Wq"], n_head)        # what am I looking for
    k = split_heads(x @ lp["Wk"], n_head)        # what do I offer
    v = split_heads(x @ lp["Wv"], n_head)        # what I hand over
    return q, k, v


def attention_scores(q, k):
    T, hd = q.shape[-2], q.shape[-1]
    scores = np.matmul(q, k.swapaxes(-1, -2)) / math.sqrt(hd)   # (B,nh,T,T)
    mask = np.tril(np.ones((T, T), dtype=bool))                 # lower-tri = allowed
    return np.where(mask, scores, -1e9), mask


def softmax(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def attention_output(att, v, lp):
    out = merge_heads(np.matmul(att, v))         # (B,T,C) weighted mix of values
    return out @ lp["Wo"], out


def residual_add(stream, branch):
    return stream + branch


def linear_fwd(x, W, b):
    return x @ W + b, (x, W)


def mlp_fwd(x, lp):
    hid, c1 = linear_fwd(x, lp["W1"], lp["b1m"])
    relu_mask = hid > 0
    act = hid * relu_mask
    out, c2 = linear_fwd(act, lp["W2"], lp["b2m"])
    return out, (c1, relu_mask, c2, hid, act)


def block_fwd(h0, lp, n_head=1):
    ln1, c_ln1 = layernorm_fwd(h0, lp["g1"], lp["b1"])
    q, k, v = project_qkv(ln1, lp, n_head)
    scores, mask = attention_scores(q, k)
    att = softmax(scores)
    attn, out = attention_output(att, v, lp)
    h1 = residual_add(h0, attn)

    ln2, c_ln2 = layernorm_fwd(h1, lp["g2"], lp["b2"])
    mlp, c_mlp = mlp_fwd(ln2, lp)
    h2 = residual_add(h1, mlp)
    return h2, dict(h0=h0, ln1=ln1, q=q, k=k, v=v, scores=scores, mask=mask,
                    att=att, out=out, attn=attn, h1=h1, ln2=ln2, mlp=mlp, h2=h2,
                    c_ln1=c_ln1, c_ln2=c_ln2, c_mlp=c_mlp)


def cross_entropy(logits, y):
    V = logits.shape[-1]
    probs = softmax(logits.reshape(-1, V))       # (B*T, V)
    yf = y.reshape(-1)
    N = yf.shape[0]
    loss = -np.log(probs[np.arange(N), yf] + 1e-12).mean()
    return float(loss), probs


def forward(p, x, y=None, n_head=1):
    tok = token_embedding(p, x)
    h = add_position(p, tok)
    h0 = h

    blocks = []
    for l in range(n_layers(p)):
        h, bc = block_fwd(h, layer(p, l), n_head)
        blocks.append(bc)

    lnf, c_lnf = layernorm_fwd(h, p["gf"], p["bf"])
    logits, c_head = linear_fwd(lnf, p["Wh"], p["bh"])

    cache = dict(x=x, tok=tok, h0=h0, blocks=blocks, lnf=lnf, logits=logits,
                 c_lnf=c_lnf, c_head=c_head)
    if y is None:
        return logits, cache
    loss, probs = cross_entropy(logits, y)
    cache.update(y=y, probs=probs)
    return loss, logits, cache


# ---- backward -------------------------------------------------------------------
def cross_entropy_bwd(probs, y, shape):
    yf = y.reshape(-1)
    N = yf.shape[0]
    dlogits = probs.copy()
    dlogits[np.arange(N), yf] -= 1.0
    return (dlogits / N).reshape(shape)


def linear_bwd(dout, cache):
    x, W = cache
    Cin = W.shape[0]
    dW = x.reshape(-1, Cin).T @ dout.reshape(-1, W.shape[1])
    db = dout.reshape(-1, W.shape[1]).sum(0)
    dx = dout @ W.T
    return dx, dW, db


def layernorm_bwd(dout, cache):
    xhat, istd, g = cache
    Cdim = xhat.shape[-1]
    dg = (dout * xhat).reshape(-1, Cdim).sum(0)
    db = dout.reshape(-1, Cdim).sum(0)
    dxhat = dout * g
    dx = istd * (dxhat
                 - dxhat.mean(-1, keepdims=True)
                 - xhat * (dxhat * xhat).mean(-1, keepdims=True))
    return dx, dg, db


def mlp_bwd(dmlp, cache):
    c1, relu_mask, c2, _, _ = cache
    dact, dW2, db2m = linear_bwd(dmlp, c2)
    dhid = dact * relu_mask                      # gradient only where ReLU was on
    dx, dW1, db1m = linear_bwd(dhid, c1)
    return dx, dW1, db1m, dW2, db2m


def attention_bwd(dy, bc, lp):
    x, q, k, v = bc["ln1"], bc["q"], bc["k"], bc["v"]
    att, out, mask = bc["att"], bc["out"], bc["mask"]
    C, nh, hd = x.shape[-1], q.shape[1], q.shape[-1]
    x2 = x.reshape(-1, C)
    dWo = out.reshape(-1, C).T @ dy.reshape(-1, C)
    dout = split_heads(dy @ lp["Wo"].T, nh)                     # (B,nh,T,hd)
    dv = np.matmul(att.swapaxes(-1, -2), dout)
    datt = np.matmul(dout, v.swapaxes(-1, -2))                  # (B,nh,T,T)
    dscores = att * (datt - (datt * att).sum(-1, keepdims=True))
    dscores = np.where(mask, dscores, 0.0) / math.sqrt(hd)
    dq = merge_heads(np.matmul(dscores, k))
    dk = merge_heads(np.matmul(dscores.swapaxes(-1, -2), q))
    dv = merge_heads(dv)
    dWq = x2.T @ dq.reshape(-1, C)
    dWk = x2.T @ dk.reshape(-1, C)
    dWv = x2.T @ dv.reshape(-1, C)
    dx = dq @ lp["Wq"].T + dk @ lp["Wk"].T + dv @ lp["Wv"].T
    return dx, {"Wq": dWq, "Wk": dWk, "Wv": dWv, "Wo": dWo}


def block_bwd(dh2, bc, lp):
    g = {}
    dh1, dmlp = dh2, dh2                         # split: h2 = h1 + mlp
    dln2, g["W1"], g["b1m"], g["W2"], g["b2m"] = mlp_bwd(dmlp, bc["c_mlp"])
    d, g["g2"], g["b2"] = layernorm_bwd(dln2, bc["c_ln2"])
    dh1 = dh1 + d

    dh0, dattn = dh1, dh1                        # split: h1 = h0 + attn
    dln1, gatt = attention_bwd(dattn, bc, lp)
    g.update(gatt)
    d, g["g1"], g["b1"] = layernorm_bwd(dln1, bc["c_ln1"])
    dh0 = dh0 + d
    bc["trace"] = dict(dh2=dh2, dln2=dln2, dh1=dh1, dln1=dln1, dh0=dh0)
    return dh0, g


def embedding_bwd(dh0, x, p):
    dtok = np.zeros_like(p["tok_emb"])
    np.add.at(dtok, x, dh0)                      # scatter-add rows by word id
    dpos = np.zeros_like(p["pos_emb"])
    dpos[:dh0.shape[1]] += dh0.sum(0)
    return dtok, dpos


def backward(p, cache):
    g = {}
    dlogits = cross_entropy_bwd(cache["probs"], cache["y"], cache["logits"].shape)
    dlnf, g["Wh"], g["bh"] = linear_bwd(dlogits, cache["c_head"])
    dh, g["gf"], g["bf"] = layernorm_bwd(dlnf, cache["c_lnf"])
    cache["trace"] = dict(dlogits=dlogits, dlnf=dlnf)

    for l in reversed(range(n_layers(p))):
        dh, gl = block_bwd(dh, cache["blocks"][l], layer(p, l))
        g.update({f"L{l}.{k}": v for k, v in gl.items()})

    g["tok_emb"], g["pos_emb"] = embedding_bwd(dh, cache["x"], p)
    return g


# ---- update ---------------------------------------------------------------------
class Adam:
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.99), eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, betas[0], betas[1], eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params, grads):
        self.t += 1
        for k in params:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * (g * g)
            mhat = self.m[k] / (1 - self.b1 ** self.t)
            vhat = self.v[k] / (1 - self.b2 ** self.t)
            params[k] -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ---- check and use ----------------------------------------------------------------
def grad_check(p, x, y, n_head=1, h=1e-5, max_per_param=None, rng=None):
    """Largest gap between the hand-written gradients and measured slopes."""
    _, _, cache = forward(p, x, y, n_head)
    grads = backward(p, cache)
    worst = 0.0
    for k in p:
        flat = p[k].reshape(-1)
        idx = range(flat.size)
        if max_per_param and flat.size > max_per_param:
            idx = (rng or np.random.default_rng(0)).choice(flat.size, max_per_param, replace=False)
        for i in idx:
            old = flat[i]
            flat[i] = old + h; lp_, _, _ = forward(p, x, y, n_head)
            flat[i] = old - h; lm_, _, _ = forward(p, x, y, n_head)
            flat[i] = old
            num = (lp_ - lm_) / (2 * h)
            worst = max(worst, abs(num - grads[k].reshape(-1)[i]))
    return worst


def evaluate(p, x, y, n_head=1):
    loss, logits, _ = forward(p, x, y, n_head)
    acc = float((logits.argmax(-1) == y).mean())
    return loss, acc


def next_word_probs(p, ids, T, n_head=1, temperature=1.0):
    """Probabilities for the word after ids, plus the attention maps used."""
    x = np.array([ids[-T:]])
    logits, cache = forward(p, x, None, n_head)
    z = logits[0, -1].astype(np.float64) / max(temperature, 1e-6)
    probs = np.exp(z - z.max())
    probs /= probs.sum()
    return probs, [bc["att"][0] for bc in cache["blocks"]]


def sample(p, start_ids, n, T, rng, n_head=1, temperature=0.8):
    out = list(start_ids)
    for _ in range(n):
        probs, _ = next_word_probs(p, out, T, n_head, temperature)
        out.append(int(rng.choice(len(probs), p=probs)))
    return out
