"""The Learn tab: every step of the GPT with its math, its code and one example.

Nothing here is written down in advance. The code shown is read from model.py
and data.py as they stand, and every example number is produced by running
that code on one tiny sentence, so the lesson follows any change you make.
"""

import html
import importlib
import inspect
import re

import numpy as np

from . import data
from . import model as model_module

TEXT = "The cat sat down."
T, C, H, B = 3, 3, 4, 1
LR, STEPS, SEED = 0.05, 30, 74

STAGES = {
    "A": ("Data", "Turn text into batches of numbers"),
    "B": ("Forward pass", "From word ids to one loss number"),
    "C": ("Backward pass", "From the loss back to a gradient for every weight"),
    "D": ("Update", "Move the weights"),
    "E": ("Train and check", "Repeat, then prove it learned"),
    "F": ("Generate", "Use the trained model to write"),
}

GLOSSARY = [
    ("B", "Batch size", "How many windows of text the model works through at once in one training step. Their losses are averaged, so a bigger B gives a steadier gradient."),
    ("T", "Context length", "How many words the model can see at once. Also the number of positions in a window, and so the number of next-word questions each window asks."),
    ("C", "Width", "How many numbers stand for one word at one position. Every vector in the stream has C numbers."),
    ("V", "Vocabulary size", "How many distinct tokens the model knows, including <unk>. The number of rows in tok_emb and of scores the output head gives per position."),
    ("H", "MLP hidden width", "How many numbers the MLP widens each vector to before narrowing it back to C. Usually 4 times C."),
    ("N", "Positions per batch", "N = B times T: how many next-word predictions one batch holds. The loss is the average over these N."),
]


# Every word the lesson leans on, in an order where each one only uses the ones above it.
TERMS = [
    ("Token", "One piece of text the model works with. Here a token is a whole word or a punctuation mark."),
    ("Vocabulary", "The list of every token the model knows. A token's place in that list is its id."),
    ("<unk>", "The stand-in token for any word that is not in the vocabulary."),
    ("Window", "A run of T tokens in a row, cut from the text. The model reads one window at a time."),
    ("Position", "A slot in the window: the first token, the second token, and so on."),
    ("Vector", "A list of numbers. The model stands for each token by a vector of C numbers."),
    ("Weight", "One number the model can change while it learns. Everything the model knows is in its weights. Also called a parameter."),
    ("Matrix", "A grid of weights. Multiplying a vector by a matrix turns it into a new vector."),
    ("Embedding", "The vector a token starts with. It is looked up in a table (tok_emb) that has one row per token in the vocabulary. A second table (pos_emb) has one row per position."),
    ("Stream", "The vector kept for each position as it travels up through the model (h0, then h1, then h2). Each part of the model reads the stream and adds its result back to it."),
    ("Attention", "The part where each position looks back at the earlier positions and pulls information in from them. It is the only place where tokens exchange information."),
    ("Head (attention head)", "One independent copy of attention. A model with 4 heads runs attention 4 times side by side, each on its own slice of the vector, so each head can follow a different kind of link between words."),
    ("Query, key, value", "The three vectors attention makes for every position. The query says what this position is looking for, the key says what it offers, and the value is what it hands over when another position picks it."),
    ("Causal mask", "The rule that a position may look only at itself and at earlier positions, never ahead. Without it the model could simply read the answer."),
    ("ReLU", "A rule that turns every negative number into 0 and leaves the rest alone. It is what lets the model compute more than plain weighted sums."),
    ("MLP", "Multi-layer perceptron, also called the feed-forward layer: two matrices with a ReLU between them. It works on each position alone, after attention, to process what attention gathered."),
    ("LayerNorm", "A rescale of one vector so that its numbers have mean 0 and spread 1. It keeps the numbers in a steady range."),
    ("Residual (skip) connection", "Adding a part's result to the stream instead of replacing the stream with it."),
    ("Block", "One round of LayerNorm, attention, LayerNorm and MLP, with both results added to the stream. A GPT stacks several blocks."),
    ("Output head", "The last matrix (Wh). It turns the stream at each position into one score per token in the vocabulary. Not the same thing as an attention head."),
    ("Logit", "One of those raw scores: higher means the model finds that token more likely to come next."),
    ("Softmax", "Turns a row of scores into probabilities: all positive, adding up to 1."),
    ("Loss", "One number for how wrong the model was: minus the log of the probability it gave the right next token, averaged over all positions. 0 is perfect. Its full name is cross-entropy."),
    ("Forward pass", "Running a window through the model, from tokens to loss."),
    ("Gradient", "For one weight, how fast the loss changes when that weight is nudged up. It says which way to move the weight."),
    ("Backward pass", "Working out the gradient of every weight, starting at the loss and walking back to the embeddings. Also called backpropagation."),
    ("Cache", "The in-between values the forward pass keeps because the backward pass needs them."),
    ("Adam", "The rule that moves each weight a little against its gradient."),
    ("Learning rate (lr)", "How far a weight moves in one step."),
    ("Training step", "One batch through the forward pass, the backward pass and Adam."),
    ("Hold-out (validation, val)", "Text kept away from training and used only to test the model."),
    ("Accuracy", "The share of positions where the token the model scored highest is the right next token."),
    ("Temperature", "A number the scores are divided by before softmax when generating. Below 1 the model plays safe; above 1 it varies more."),
]


def readable_params(m, rng, V, T, C, H):
    """Same shapes as init_params, but wide values rounded to one decimal."""
    p = m.init_params(rng, V, T, C, H)
    for k in ("tok_emb", "pos_emb", "L0.Wq", "L0.Wk", "L0.Wv", "L0.Wo", "L0.W1", "L0.W2", "Wh"):
        p[k] = np.round(rng.normal(0, 0.6, size=p[k].shape), 1)
    return p


def train(m, p, train_data, rng, steps, B, T, lr):
    opt = m.Adam(p, lr=lr)
    history = []
    for step in range(steps):
        x, y = data.get_batch(train_data, rng, B, T)
        loss, _, cache = m.forward(p, x, y)
        grads = m.backward(p, cache)
        opt.step(p, grads)
        history.append(loss)
    return history


# ---- formatting -------------------------------------------------------------------
def f2(v):
    s = f"{float(v):.2f}"
    return "0.00" if s == "-0.00" else s


def mat(a, rows=None, cols=None, w=6):
    a = np.squeeze(np.asarray(a, dtype=float))
    if a.ndim == 1:
        a = a[None, :]
    rw = max((len(r) for r in rows), default=0) if rows else 0
    lines = []
    if cols:
        lines.append(" " * (rw + 2) + "".join(f"{c_:>{w}}" for c_ in cols))
    for i, r in enumerate(a):
        lab = f"{rows[i]:<{rw}} " if rows else ""
        lines.append(lab + "[" + "".join(f"{f2(v):>{w}}" for v in r) + " ]")
    return "\n".join(lines)


def side(*blocks, gap=3):
    blocks = [b.split("\n") for b in blocks]
    h = max(len(b) for b in blocks)
    ws = [max(len(l) for l in b) for b in blocks]
    return "\n".join((" " * gap).join((b[i] if i < len(b) else "").ljust(w_)
                                       for b, w_ in zip(blocks, ws)).rstrip() for i in range(h))


def titled(title, body):
    return title + "\n" + body


def dot(a, b):
    return " + ".join(f"{f2(u)}×{f2(v)}" for u, v in zip(a, b))


KW = r"\b(def|return|for|in|if|else|elif|class|import|from|is|None|not|and|or|while|True|False|lambda|with|as|try|except|raise|break|continue|pass)\b"
TOK = re.compile(r"(#[^\n]*)|(\"\"\"[\s\S]*?\"\"\"|\"(?:[^\"\\\n]|\\.)*\"|'(?:[^'\\\n]|\\.)*')|" + KW +
                 r"|\b(\d+\.?\d*(?:e-?\d+)?)\b")


def highlight(src):
    out, last = [], 0
    for mt in TOK.finditer(src):
        out.append(html.escape(src[last:mt.start()]))
        cls = "cm" if mt.group(1) else "st" if mt.group(2) else "kw" if mt.group(3) else "nu"
        out.append(f'<span class="{cls}">{html.escape(mt.group(0))}</span>')
        last = mt.end()
    out.append(html.escape(src[last:]))
    s = "".join(out)
    return re.sub(r'(<span class="kw">(?:def|class)</span> )(\w+)', r'\1<span class="fn">\2</span>', s)


def code(*objs):
    parts = []
    for o in objs:
        src = o.strip("\n") if isinstance(o, str) else inspect.getsource(o).rstrip()
        parts.append(re.sub(r" {3,}#", "  #", src))
    return "\n\n\n".join(parts)


LN_MATH = ("μ = mean of the C numbers<br>σ² = their variance<br>"
           "x̂ = (x − μ) / √(σ² + ε)<br>out = g · x̂ + b &nbsp; (g = 1, b = 0 at start)")
LNB_MATH = ("d = dout · g<br>dx = (1/σ) · ( d − mean(d) − x̂ · mean(d · x̂) )<br>"
            "dg = Σ dout · x̂ &nbsp;&nbsp; db = Σ dout")


def build():
    """Run the example and return the whole lesson as plain data."""
    m = importlib.reload(model_module)
    S = []

    def step(stage, title, shape, why, math, src, ex, note=None):
        S.append(dict(n=len(S) + 1, stage=stage, title=title, shape=shape, why=why, math=math,
                      code=highlight(src), example=ex, note=note,
                      check=bool(note and " = " in note)))

    tokens = data.tokenize(TEXT)
    words, stoi = data.build_vocab(tokens, min_freq=1)
    V = len(words)
    ids = data.encode(tokens, stoi)
    train_data, val_data = data.split(ids)
    x, y = data.get_batch(train_data, np.random.default_rng(0), B, T)
    p = readable_params(m, np.random.default_rng(SEED), V, T, C, H)
    p0 = {k: v.copy() for k, v in p.items()}
    lp = m.layer(p, 0)
    loss, logits, c = m.forward(p, x, y)
    grads = m.backward(p, c)
    bc = c["blocks"][0]
    tr, btr = c["trace"], bc["trace"]
    gerr = m.grad_check(p, x, y)
    pos = [words[i] for i in x[0]]
    tgt = [words[i] for i in y[0]]
    npar = sum(v.size for v in p.values())

    # ---- A. data
    step("A", "Tokenize: text to words", "text → N words",
         "Lowercase the text and cut it into words; each punctuation mark is its own token.",
         "text → [w<sub>1</sub>, w<sub>2</sub>, …, w<sub>N</sub>]",
         code(data.tokenize, 'TOKEN_RE = re.compile(r"<eot>|[a-z]+\'[a-z]+|[a-z]+|[.,!?;:\\"]")'),
         f"tokenize({TEXT!r})\n= {tokens}\nN = {len(tokens)}")

    step("A", "Build the vocabulary: cut the rare words", "N words → V ids",
         "Only words seen at least <code>min_freq</code> times get an id; id 0 is reserved for "
         "<code>&lt;unk&gt;</code>, the stand-in for every other word. On TinyStories with "
         "<code>min_freq = 10</code> that keeps about 5.4k words out of 4.5M words of text, "
         "covering 99.57% of it.",
         "vocab = [&lt;unk&gt;] + sorted{ w : count(w) ≥ min_freq }<br>id(w) = position of w in vocab<br>V = |vocab|",
         code(data.build_vocab),
         "build_vocab(tokens, min_freq=1)      # 1, or nothing survives\n\n" +
         "\n".join(f"  {i}  {w}" for i, w in enumerate(words)) + f"\n\nV = {V}",
         "Sorted as text, so the full stop comes before the letters.")

    step("A", "Encode: words to ids", "N words → data (N,)",
         "Swap every word for its id. The numbers mean nothing by themselves: an id is only a row "
         "number into the embedding table (step 7).",
         "data<sub>n</sub> = id(w<sub>n</sub>), or 0 if w<sub>n</sub> is not in the vocab",
         code(data.encode),
         f"encode(tokens, stoi)\n= {ids.tolist()}\n\n" +
         "\n".join(f"  {w:<5} → {i}" for w, i in zip(tokens, ids)))

    step("A", "Hold-out: split train and validation", "data → train, val",
         "The first 90% is the only text the model ever trains on. The last 10% is held out to "
         "test it on text it has never seen (step 33).",
         "n = ⌊0.9 · N⌋<br>train = data[0 : n]<br>val = data[n : N]",
         code(data.split),
         f"n = int(0.9 * {len(ids)}) = {len(train_data)}\n"
         f"train = {train_data.tolist()}   ({' '.join(words[i] for i in train_data)})\n"
         f"val   = {val_data.tolist()}            ({' '.join(words[i] for i in val_data)})",
         "With 5 tokens the hold-out is one token: too small to test on. Real runs in the Train tab hold out thousands.")

    step("A", "Draw a batch of windows", "train → x, y (B,T)",
         "Pick B random start positions. x is T words; y is the same window moved one word to the "
         "right, so y[t] is the word that follows x[t]: T next-word questions per window.",
         "x<sub>b</sub> = src[i<sub>b</sub> : i<sub>b</sub>+T]<br>y<sub>b</sub> = src[i<sub>b</sub>+1 : i<sub>b</sub>+T+1]",
         code(data.get_batch),
         f"get_batch(train, rng, B={B}, T={T})   # only start i = 0 fits\n\n"
         f"x = {x.tolist()}   {' '.join(pos)}\n"
         f"y = {y.tolist()}   {' '.join(tgt)}\n\n" +
         "\n".join(f"  after {a!r:<6} the answer is {b!r}" for a, b in zip(pos, tgt)))

    step("A", "Create the weights", "→ one dict of arrays",
         "Everything the model will ever know lives in these arrays. They start as small random "
         f"numbers. Each block adds 12 arrays (named L0.…, L1.…). Example: V = {V}, T = {T}, "
         f"C = {C}, H = {H}, one block, {npar} numbers.",
         "tok_emb (V,C) &nbsp; pos_emb (T,C)<br>per block: Wq, Wk, Wv, Wo (C,C)<br>"
         "W1 (C,H) &nbsp; W2 (H,C)<br>Wh (C,V)<br>gains g = 1, biases b = 0",
         code(m.init_params, readable_params),
         titled("tok_emb (V,C): one row per word", mat(p["tok_emb"], rows=words)) + "\n\n" +
         titled("pos_emb (T,C): one row per position", mat(p["pos_emb"], rows=["0", "1", "2"])) + "\n\n" +
         side(titled("Wq", mat(lp["Wq"])), titled("Wk", mat(lp["Wk"]))) + "\n\n" +
         side(titled("Wv", mat(lp["Wv"])), titled("Wo", mat(lp["Wo"]))) + "\n\n" +
         side(titled("W1 (C,H)", mat(lp["W1"])), titled("W2 (H,C)", mat(lp["W2"]))) + "\n\n" +
         titled("Wh (C,V): one column per word", mat(p["Wh"], cols=words)),
         "The example draws wider values and rounds them to one decimal so the arithmetic can be done by hand.")

    # ---- B. forward
    tok, h0 = c["tok"], c["h0"]
    step("B", "Token embedding: look up each word's row", "x (B,T) → tok (B,T,C)",
         "Each id picks its row out of <code>tok_emb</code>. That row is the word's vector: C numbers "
         "that training will shape to mean something.",
         "tok<sub>t</sub> = tok_emb[ x<sub>t</sub> ]",
         code(m.token_embedding), mat(tok, rows=pos),
         f"'the' is id {x[0][0]}, so its row is tok_emb[{x[0][0]}].")

    step("B", "Add the position", "tok → h0 (B,T,C)",
         "Attention has no sense of order, so a second learned vector, one per slot, is added. "
         "The same word in a different slot now has a different vector.",
         "h0<sub>t</sub> = tok<sub>t</sub> + pos_emb[t]",
         code(m.add_position),
         side(titled("tok", mat(tok, rows=pos)), titled("+ pos_emb", mat(p["pos_emb"])),
              titled("= h0", mat(h0))),
         f"h0[the][0] = {f2(tok[0,0,0])} + {f2(p['pos_emb'][0,0])} = {f2(h0[0,0,0])}")

    def ln_note(src, out):
        r = src[0, 1]
        mu, sd = r.mean(), np.sqrt(r.var() + m.EPS)
        return (f"row 'cat': mean = ({' + '.join(f2(v) for v in r)}) / {len(r)} = {f2(mu)}, "
                f"std = {f2(sd)}; first entry ({f2(r[0])} − {f2(mu)}) / {f2(sd)} = {f2(out[0,1,0])}")

    ln1 = bc["ln1"]
    step("B", "LayerNorm 1", "h0 → ln1 (B,T,C)",
         "Rescale each position's vector to mean 0 and variance 1, so the numbers going into "
         "attention stay in a steady range however large the stream grows.",
         LN_MATH, code(m.layernorm_fwd),
         side(titled("h0", mat(h0, rows=pos)), titled("→ ln1", mat(ln1))), ln_note(h0, ln1))

    q, k, v = bc["q"][0, 0], bc["k"][0, 0], bc["v"][0, 0]
    step("B", "Attention 1/4: query, key, value", "ln1 → q, k, v (B,nh,T,hd)",
         "Three learned matrices give every word three roles: a query (what am I looking for), a "
         "key (what do I offer) and a value (what I hand over if chosen). With several heads the C "
         "numbers are cut into nh slices of hd, and each head attends on its own slice.",
         "q = ln1 · W<sub>q</sub><br>k = ln1 · W<sub>k</sub><br>v = ln1 · W<sub>v</sub><br>"
         "then cut into heads: C = nh · hd",
         code(m.project_qkv, m.split_heads),
         side(titled("q", mat(q, rows=pos)), titled("k", mat(k)), titled("v", mat(v))),
         f"q[the][0] = ln1[the] · Wq[:,0] = {dot(ln1[0,0], lp['Wq'][:,0])} = {f2(q[0,0])}")

    att = bc["att"][0, 0]
    raw = (q @ k.T) / np.sqrt(C)
    sc_show = "\n".join(
        f"{pos[i]:<4}[" + "".join(f"{(f2(raw[i,j]) if j <= i else 'mask'):>7}" for j in range(T)) + " ]"
        for i in range(T))
    step("B", "Attention 2/4: scores and the causal mask", "q, k → scores (B,nh,T,T)",
         "Compare every query with every key. Row t holds how much word t cares about each word s. "
         "Then forbid looking ahead: a word may only see itself and earlier words, otherwise it "
         "could read the answer.",
         "scores<sub>t,s</sub> = (q<sub>t</sub> · k<sub>s</sub>) / √hd<br>scores<sub>t,s</sub> = −10⁹ for s &gt; t",
         code(m.attention_scores),
         "       " + "".join(f"{w:>7}" for w in pos) + "\n" + sc_show,
         f"scores[cat][the] = ({dot(q[1], k[0])}) / √{C} = {f2(raw[1,0])}")

    e0, e1 = np.exp(raw[1, 0]), np.exp(raw[1, 1])
    step("B", "Attention 3/4: softmax into weights", "scores → att (B,nh,T,T)",
         "Turn each row of scores into weights that are positive and sum to 1. The masked cells "
         "become exactly 0.",
         "att<sub>t,s</sub> = e<sup>scores<sub>t,s</sub></sup> / Σ<sub>s′</sub> e<sup>scores<sub>t,s′</sub></sup>",
         code(m.softmax), mat(att, rows=pos, cols=pos, w=7),
         f"row 'cat': e^{f2(raw[1,0])} = {f2(e0)}, e^{f2(raw[1,1])} = {f2(e1)}; "
         f"{f2(e0)} / ({f2(e0)} + {f2(e1)}) = {f2(att[1,0])}. Row 'the' can only see itself, so it is 1.")

    out, attn = bc["out"], bc["attn"]
    step("B", "Attention 4/4: mix the values", "att, v → attn (B,T,C)",
         "Each word collects the values of the words it attends to, weighted by <code>att</code>. "
         "The heads are glued back into one vector of C numbers and go through one more matrix. "
         "This is how information moves between words.",
         "out<sub>t</sub> = Σ<sub>s</sub> att<sub>t,s</sub> · v<sub>s</sub><br>attn = out · W<sub>o</sub>",
         code(m.attention_output, m.merge_heads),
         side(titled("out = att @ v", mat(out, rows=pos)), titled("attn = out @ Wo", mat(attn))),
         f"out[cat][0] = {f2(att[1,0])}×{f2(v[0,0])} + {f2(att[1,1])}×{f2(v[1,0])} = {f2(out[0,1,0])}")

    h1 = bc["h1"]
    step("B", "Residual add", "h0, attn → h1 (B,T,C)",
         "Attention's result is added to the stream, not put in its place. The stream keeps what it "
         "had, and the gradient later gets a straight path back (step 27).",
         "h1 = h0 + attn", code(m.residual_add),
         side(titled("h0", mat(h0, rows=pos)), titled("+ attn", mat(attn)), titled("= h1", mat(h1))),
         f"h1[the][0] = {f2(h0[0,0,0])} + {f2(attn[0,0,0])} = {f2(h1[0,0,0])}")

    ln2 = bc["ln2"]
    step("B", "LayerNorm 2", "h1 → ln2 (B,T,C)",
         "The same rescaling as step 9, with its own gain and bias, before the MLP.",
         LN_MATH, "# layernorm_fwd is the function from step 9\nln2, c_ln2 = layernorm_fwd(h1, lp[\"g2\"], lp[\"b2\"])",
         side(titled("h1", mat(h1, rows=pos)), titled("→ ln2", mat(ln2))), ln_note(h1, ln2))

    _, relu_mask, _, hid, act = bc["c_mlp"]
    mlp = bc["mlp"]
    step("B", "MLP: think about each word alone", "ln2 → mlp (B,T,C)",
         "Each position goes alone through a two-layer network: widen to H numbers, zero the "
         "negative ones (ReLU, the only non-linearity in the model), narrow back to C.",
         "hid = ln2 · W<sub>1</sub> + b<sub>1</sub><br>act = max(hid, 0)<br>mlp = act · W<sub>2</sub> + b<sub>2</sub>",
         code(m.mlp_fwd, m.linear_fwd),
         side(titled("hid", mat(hid, rows=pos)), titled("act", mat(act))) + "\n\n" +
         titled("mlp", mat(mlp, rows=pos)),
         f"hid[the][0] = {dot(ln2[0,0], lp['W1'][:,0])} = {f2(hid[0,0,0])}"
         + (", negative, so act = 0" if hid[0, 0, 0] <= 0 else ", positive, so it passes"))

    h2 = bc["h2"]
    step("B", "Residual add: the block is done", "h1, mlp → h2 (B,T,C)",
         "Again added to the stream. Steps 9 to 17 are one transformer block, "
         "<code>block_fwd</code>. A deeper model runs the block again on h2, with that block's own weights.",
         "h2 = h1 + mlp", code(m.block_fwd),
         side(titled("h1", mat(h1, rows=pos)), titled("+ mlp", mat(mlp)), titled("= h2", mat(h2))),
         f"h2[the][0] = {f2(h1[0,0,0])} + {f2(mlp[0,0,0])} = {f2(h2[0,0,0])}")

    lnf = c["lnf"]
    step("B", "Final LayerNorm", "h2 → lnf (B,T,C)",
         "One last rescale, after the last block, before reading out the answer.",
         LN_MATH, "# layernorm_fwd is the function from step 9\nlnf, c_lnf = layernorm_fwd(h, p[\"gf\"], p[\"bf\"])",
         side(titled("h2", mat(h2, rows=pos)), titled("→ lnf", mat(lnf))), ln_note(h2, lnf))

    lg = c["logits"]
    ci = int(y[0, 0])
    step("B", "Output head: one score per word", "lnf → logits (B,T,V)",
         "Multiply each position's vector by the output matrix: one raw score (logit) for every "
         "word in the vocabulary. Higher means more likely to come next.",
         "logits<sub>t</sub> = lnf<sub>t</sub> · W<sub>h</sub> + b<sub>h</sub>",
         "# linear_fwd is the function from step 16\nlogits, c_head = linear_fwd(lnf, p[\"Wh\"], p[\"bh\"])",
         "score for the next word being...\n" + mat(lg, rows=pos, cols=words, w=7),
         f"logits[the][{words[ci]}] = lnf[the] · Wh[:,{words[ci]}] = {dot(lnf[0,0], p['Wh'][:,ci])} = {f2(lg[0,0,ci])}")

    probs = c["probs"]
    pt = probs[np.arange(T), y[0]]
    best = [words[i] for i in probs.argmax(1)]
    step("B", "Softmax and the loss", "logits, y → loss (one number)",
         "Softmax turns each row of scores into probabilities. The loss looks only at the "
         "probability given to the right answer: −ln(p) is 0 when p = 1 and grows as p shrinks. "
         "Averaged over all positions, this one number is what training pushes down.",
         "p<sub>t,j</sub> = e<sup>logits<sub>t,j</sub></sup> / Σ<sub>j′</sub> e<sup>logits<sub>t,j′</sub></sup><br>"
         "loss = −(1/N) Σ<sub>t</sub> ln p<sub>t, y<sub>t</sub></sub><br>N = B · T",
         code(m.cross_entropy),
         "probability of each next word\n" + mat(probs, rows=pos, cols=words, w=7) + "\n\n" +
         "\n".join(f"  after {a!r:<6} want {b!r:<7} p = {f2(pp)}  −ln p = {f2(-np.log(pp))}"
                   f"   (it bets on {bb!r})" for a, b, pp, bb in zip(pos, tgt, pt, best)) +
         f"\n\nloss = ({' + '.join(f2(-np.log(v_)) for v_ in pt)}) / {T} = {f2(loss)}",
         f"Blind guessing gives every word 1/{V}, a loss of ln({V}), which is {f2(np.log(V))}. "
         f"The untrained model scores {f2(loss)}.")

    step("B", "The whole forward pass", "x, y → loss",
         "Steps 7 to 20 in order, looping over the blocks. It also returns a cache: every "
         "intermediate value, because the backward pass needs them.",
         "loss = cross_entropy( head( LN( blocks( embed(x) ) ) ), y )",
         code(m.forward), f"loss, logits, cache = forward(p, x, y)\nloss = {loss:.4f}")

    # ---- C. backward
    dl = tr["dlogits"]
    step("C", "Start at the loss: gradient of the logits", "probs, y → dlogits (B,T,V)",
         "Now walk backwards and ask of every number: if I nudge you up, does the loss go up or "
         "down, and how fast? For the logits the answer is simple: the probability, minus 1 at the "
         "right answer. Negative means raising this score lowers the loss.",
         "∂loss/∂logits<sub>t,j</sub> = ( p<sub>t,j</sub> − [j = y<sub>t</sub>] ) / N",
         code(m.cross_entropy_bwd), mat(dl, rows=pos, cols=words, w=7),
         f"dlogits[the][{tgt[0]}] = ({f2(probs[0,ci])} − 1) / {T} = {f2(dl[0,0,ci])}: push '{tgt[0]}' up. "
         "Every other entry in the row is p / 3: push those down.")

    gWh, dlnf = grads["Wh"], tr["dlnf"]
    step("C", "Back through the output head", "dlogits → dWh, dbh, dlnf",
         "A linear layer hands back three things: the gradient for its weights, for its bias, and "
         "for its input, which is passed on to the layer before.",
         "logits = lnf · W<sub>h</sub> + b<sub>h</sub><br>dW<sub>h</sub> = lnfᵀ · dlogits<br>"
         "db<sub>h</sub> = Σ<sub>t</sub> dlogits<sub>t</sub><br>dlnf = dlogits · W<sub>h</sub>ᵀ",
         code(m.linear_bwd),
         titled("dWh (C,V)", mat(gWh, cols=words, w=7)) + "\n\n" +
         titled("dbh", mat(grads["bh"], cols=words, w=7)) + "\n\n" +
         titled("dlnf", mat(dlnf, rows=pos)),
         f"dWh[0][{words[ci]}] = Σ lnf[t][0] × dlogits[t][{words[ci]}] = {dot(lnf[0,:,0], dl[0,:,ci])} = {f2(gWh[0,ci])}")

    dh2 = btr["dh2"]
    step("C", "Back through the final LayerNorm", "dlnf → dh2 (B,T,C)",
         "LayerNorm's backward removes the two things it normalised away: the mean and the scale. "
         "It uses <code>xhat</code> and <code>istd</code> (1/σ) saved on the way forward.",
         LNB_MATH, code(m.layernorm_bwd),
         side(titled("dlnf", mat(dlnf, rows=pos)), titled("→ dh2", mat(dh2))) + "\n\n" +
         side(titled("dgf", mat(grads["gf"])), titled("dbf", mat(grads["bf"]))))

    dln2 = btr["dln2"]
    dact, _, _ = m.linear_bwd(dh2, bc["c_mlp"][2])
    dhid = dact * relu_mask
    step("C", "Residual split, then back through the MLP", "dh2 → dln2 and dW1, dW2",
         "h2 = h1 + mlp, and an add passes its gradient unchanged to both inputs: one copy goes "
         "straight down the skip, one into the MLP. Inside the MLP, the gradient passes only where "
         "the ReLU was on.",
         "dmlp = dh2, &nbsp; dh1 = dh2<br>dW<sub>2</sub> = actᵀ · dmlp<br>dact = dmlp · W<sub>2</sub>ᵀ<br>"
         "dhid = dact where hid &gt; 0, else 0<br>dW<sub>1</sub> = ln2ᵀ · dhid<br>dln2 = dhid · W<sub>1</sub>ᵀ",
         code(m.mlp_bwd),
         side(titled("dhid (0 where ReLU was off)", mat(dhid, rows=pos)), titled("dln2", mat(dln2))) + "\n\n" +
         side(titled("dW1 (C,H)", mat(grads["L0.W1"])), titled("dW2 (H,C)", mat(grads["L0.W2"]))))

    dh1 = btr["dh1"]
    step("C", "Back through LayerNorm 2 and join the skip", "dln2, skip → dh1 (B,T,C)",
         "The two paths that left h1 meet again: the copy from the skip plus what came back through "
         "the MLP.",
         "dh1 = dh2 + layernorm_bwd(dln2)",
         "# layernorm_bwd is the function from step 24\nd, g[\"g2\"], g[\"b2\"] = layernorm_bwd(dln2, bc[\"c_ln2\"])\ndh1 = dh1 + d",
         side(titled("skip: dh2", mat(dh2, rows=pos)),
              titled("+ via the MLP", mat(dh1 - dh2)), titled("= dh1", mat(dh1))))

    dout_ = dh1 @ lp["Wo"].T
    datt = dout_[0] @ v.T
    dsc = att * (datt - (datt * att).sum(-1, keepdims=True))
    dsc = np.where(bc["mask"], dsc, 0.0) / np.sqrt(C)
    dln1 = btr["dln1"]
    step("C", "Residual split, then back through attention", "dh1 → dln1 and dWq, dWk, dWv, dWo",
         "Split again (h1 = h0 + attn), then undo attention's four parts in reverse: the output "
         "matrix, the mix, the softmax, the scores. The input fed q, k and v, so three gradients "
         "flow back into it and add up.",
         "dW<sub>o</sub> = outᵀ · dattn, &nbsp; dout = dattn · W<sub>o</sub>ᵀ<br>"
         "dv = attᵀ · dout, &nbsp; datt = dout · vᵀ<br>"
         "ds = att · ( datt − Σ<sub>s</sub> datt · att ) / √hd<br>"
         "dq = ds · k, &nbsp; dk = dsᵀ · q<br>"
         "dW<sub>q</sub> = xᵀ·dq, dW<sub>k</sub> = xᵀ·dk, dW<sub>v</sub> = xᵀ·dv<br>"
         "dln1 = dq·W<sub>q</sub>ᵀ + dk·W<sub>k</sub>ᵀ + dv·W<sub>v</sub>ᵀ",
         code(m.attention_bwd),
         side(titled("datt", mat(datt, rows=pos)), titled("ds (scores)", mat(dsc))) + "\n\n" +
         side(titled("dWq", mat(grads["L0.Wq"])), titled("dWk", mat(grads["L0.Wk"]))) + "\n\n" +
         side(titled("dWv", mat(grads["L0.Wv"])), titled("dWo", mat(grads["L0.Wo"]))) + "\n\n" +
         titled("dln1", mat(dln1, rows=pos)),
         "Row 'the' of ds is 0: it could only attend to itself, so there was nothing to choose.")

    dh0 = btr["dh0"]
    step("C", "Back through LayerNorm 1 and join the skip", "dln1, skip → dh0 (B,T,C)",
         "The same join as step 26. Steps 25 to 28 are <code>block_bwd</code>; a deeper model "
         "repeats it for each block, last block first.",
         "dh0 = dh1 + layernorm_bwd(dln1)", code(m.block_bwd),
         side(titled("skip: dh1", mat(dh1, rows=pos)),
              titled("+ via attention", mat(dh0 - dh1)), titled("= dh0", mat(dh0))))

    step("C", "Land on the embedding tables", "dh0 → dtok_emb (V,C), dpos_emb (T,C)",
         "h0 = tok_emb[x] + pos_emb, so each row of dh0 is added to the row of the word that stood "
         "there and to the row of its position. Words that were not in the batch get no gradient.",
         "dtok_emb[ x<sub>t</sub> ] += dh0<sub>t</sub><br>dpos_emb[t] += Σ<sub>b</sub> dh0<sub>b,t</sub>",
         code(m.embedding_bwd),
         titled("dtok_emb", mat(grads["tok_emb"], rows=words)) + "\n\n" +
         titled("dpos_emb", mat(grads["pos_emb"], rows=["0", "1", "2"])),
         "'down', '.' and <unk> were never an input in this window: their rows stay 0.")

    step("C", "The whole backward pass, and the proof it is right", "cache → a gradient per array",
         "Steps 22 to 29 in order: one gradient array for each weight array. To prove the "
         "hand-written maths, nudge every single weight by ±0.00001, re-run the forward pass, "
         "and compare the measured slope with the computed gradient.",
         "numeric slope = ( loss(w + h) − loss(w − h) ) / 2h<br>must equal ∂loss/∂w",
         code(m.backward, m.grad_check),
         f"grads = backward(p, cache)\n\ngrad_check(p, x, y)\n"
         f"largest difference over all {npar} weights = {gerr:.1e}",
         "The test suite runs the same check on models with several blocks and heads.")

    # ---- D. update
    g_ = grads["Wh"][0, ci]
    m_, v_ = 0.1 * g_, 0.01 * g_ * g_
    mh, vh = m_ / 0.1, v_ / 0.01
    opt = m.Adam(p, lr=LR)
    opt.step(p, grads)
    loss1, _, _ = m.forward(p, x, y)
    step("D", "Adam: move every weight a little", "p, grads → new p",
         "Each weight steps against its gradient. Adam keeps a running mean of the gradient (m) and "
         "of its square (v) and divides one by the root of the other, so every weight gets a "
         "step of about <code>lr</code> whatever the size of its gradient.",
         "m = 0.9·m + 0.1·g<br>v = 0.99·v + 0.01·g²<br>m̂ = m / (1 − 0.9<sup>t</sup>)<br>"
         "v̂ = v / (1 − 0.99<sup>t</sup>)<br>w ← w − lr · m̂ / (√v̂ + 10⁻⁸)",
         code(m.Adam),
         f"one weight, Wh[0][{words[ci]}], step t = 1, lr = {LR}\n"
         f"  g    = {g_:.4f}\n"
         f"  m    = 0.1 × g        = {m_:.4f}\n"
         f"  v    = 0.01 × g²      = {v_:.6f}\n"
         f"  mhat = m / 0.1        = {mh:.4f}\n"
         f"  vhat = v / 0.01       = {vh:.6f}\n"
         f"  step = {LR} × {mh:.4f} / {np.sqrt(vh):.4f} = {LR * mh / np.sqrt(vh):+.4f}\n"
         f"  w    = {f2(p0['Wh'][0,ci])} − ({LR * mh / np.sqrt(vh):+.2f}) = {f2(p['Wh'][0,ci])}\n\n" +
         titled("Wh before", mat(p0["Wh"], cols=words, w=7)) + "\n\n" +
         titled(f"Wh after 1 step: every entry moved by {LR}", mat(p["Wh"], cols=words, w=7)) +
         f"\n\nloss before {loss:.4f}  →  after {loss1:.4f}",
         "On the first step, mhat divided by the root of vhat is exactly +1 or −1, so every weight moves by exactly lr.")

    # ---- E. train and check
    p = {k: v_.copy() for k, v_ in p0.items()}
    hist = train(m, p, train_data, np.random.default_rng(0), STEPS, B, T, LR)
    lossN, _, cN = m.forward(p, x, y)
    ptN = cN["probs"][np.arange(T), y[0]]
    allh = hist + [lossN]
    tl, ta = m.evaluate(p, x, y)
    step("E", "The training loop", "repeat: batch → forward → backward → Adam",
         "That is all training is: draw a fresh batch from the train part, run steps 7 to 31, "
         "repeat. The Train tab runs this same loop (<code>trainer.Run</code>), with pause, stop "
         "and a check on held-out text every so often.",
         "repeat:<br>&nbsp; x, y = get_batch(train)<br>&nbsp; loss = forward(x, y)<br>"
         "&nbsp; grads = backward()<br>&nbsp; Adam.step(grads)",
         code(train),
         f"train(p, train, rng, steps={STEPS}, B={B}, T={T}, lr={LR})\n\n" +
         "\n".join(f"  loss after {i:>2} steps: {allh[i]:.3f}" for i in (0, 1, 2, 5, 10, 20, 30) if i < len(allh)) +
         f"\n\nafter {STEPS} steps:\n" +
         "\n".join(f"  after {a!r:<6} p({b!r}) = {f2(pp)}" for a, b, pp in zip(pos, tgt, ptN)))
    S[-1]["curve"] = [float(v_) for v_ in allh]
    S[-1]["chance"] = float(np.log(V))

    step("E", "How we know it works", "held-out text → val loss, val accuracy",
         "Four checks. <b>1.</b> Gradients match the measured slopes (step 30). "
         "<b>2.</b> Loss beats guessing: an untrained model scores about ln(V), and the loss on "
         "held-out text must fall well below it. "
         "<b>3.</b> Accuracy beats the dumb baseline: always answering with the most common word. "
         "<b>4.</b> Train and held-out accuracy stay close: it learned patterns, not the training "
         "text. The Train tab draws all four on every run.",
         "accuracy = share of positions where<br>argmax<sub>j</sub> logits<sub>t,j</sub> = y<sub>t</sub><br><br>"
         "measured on <b>val</b>, the 10% never trained on",
         code(m.evaluate),
         f"evaluate(p, x, y) on the training window  →  loss {tl:.3f}, accuracy {ta:.0%}\n"
         f"on val  →  not possible: val has {len(val_data)} token,\n"
         f"           a window needs T + 1 = {T+1}\n\n"
         f"guessing would score ln({V}), which is {f2(np.log(V))}; the model scores {tl:.3f}",
         "So this example proves only that the maths runs and can memorise one window. "
         "Whether a model generalises can only be shown on held-out text: that is what the Train tab is for.")

    # ---- F. generate
    srng = np.random.default_rng(0)
    gen = m.sample(p, [int(x[0, 0])], 3, T, srng, temperature=0.8)
    lines = []
    for i in range(1, len(gen)):
        pr, _ = m.next_word_probs(p, gen[:i], T, temperature=0.8)
        top = ", ".join(f"{words[j]} {f2(pr[j])}" for j in np.argsort(-pr)[:3])
        lines.append(f"  {' '.join(words[j] for j in gen[:i][-T:]):<14} →  {top:<32} {words[gen[i]]!r}")
    step("F", "Generate: the forward pass in a loop", "ids so far → one more id",
         "Writing text uses steps 7 to 19 only: no targets, no loss, no backward. Run forward on "
         "the last T words, keep only the last position's scores, turn them into probabilities, "
         "draw one word, append it, repeat. Temperature below 1 sharpens the choice.",
         "z = logits<sub>last</sub> / temperature<br>p = softmax(z)<br>next ~ p<br>append, repeat",
         code(m.sample, m.next_word_probs),
         f"sample(p, start=['{words[int(x[0, 0])]}'], n=3, T={T}, temperature=0.8)\n\n" +
         "context         →  top probabilities                drawn\n" + "\n".join(lines) +
         f"\n\nresult: {' '.join(words[j] for j in gen)}",
         "It reproduces its one training sentence, which is all it has ever seen.")

    return dict(text=TEXT, stages=STAGES, glossary=GLOSSARY, terms=TERMS, steps=S,
                example=dict(V=V, T=T, C=C, H=H, B=B, lr=LR, steps=STEPS, weights=int(npar)))
