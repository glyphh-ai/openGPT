"""Everything the app can ask for, as plain functions: no sockets in here.

dispatch(method, path, query, body) returns (status, data). The server only
turns HTTP into these calls, so the whole app can be tested without a port.
"""

import importlib

import numpy as np

from . import __version__, data, trainer, walkthrough
from . import model as model_module

manager = trainer.Manager()

SOURCE_FILES = ["opengpt/model.py", "opengpt/data.py", "opengpt/trainer.py",
                "opengpt/tinytales.py", "opengpt/walkthrough.py", "opengpt/api.py",
                "opengpt/server.py"]


class Refused(Exception):
    """A request that cannot be done, with a reason the person can act on."""


def state(q, b):
    active = manager.active
    return dict(version=__version__, corpora=data.list_corpora(), defaults=trainer.DEFAULTS,
                help=trainer.HELP, active=active.summary() if active else None,
                runs=manager.list())


def corpus(q, b):
    return data.load(q.get("name", "tinytales"), int(q.get("min_freq", 1))).stats


def tokenize(q, b):
    ds = data.load(b.get("corpus", "tinytales"), int(b.get("min_freq", 1)))
    toks, ids = ds.encode_text(b.get("text", ""))
    T = int(b.get("T", 8))
    window = None
    if len(ids) > T:
        window = dict(x=toks[:T], y=toks[1:T + 1])
    return dict(tokens=toks, ids=ids, unknown=[t for t, i in zip(toks, ids) if i == 0],
                window=window, T=T)


def run_start(q, b):
    return manager.start(b.get("cfg", b)).summary()


def _run(q, b):
    run = manager.get(q.get("id") or b.get("id"))
    if not run:
        raise Refused("no such live run")
    return run


def run_pause(q, b):
    run = _run(q, b)
    run.pause()
    return run.summary()


def run_resume(q, b):
    run = _run(q, b)
    run.resume()
    return run.summary()


def run_stop(q, b):
    run = _run(q, b)
    run.stop()
    run.join(30)
    return run.summary()


def run_delta(q, b):
    return _run(q, b).delta(int(q.get("m", 0)), int(q.get("e", 0)), int(q.get("s", 0)))


def run_record(q, b):
    rec = manager.record(q.get("id", ""))
    if not rec:
        raise Refused("no such run")
    return rec


def runs(q, b):
    return dict(runs=manager.list())


def generate(q, b):
    p, words, cfg, m = manager.weights(b.get("id") or None)
    stoi = {w: i for i, w in enumerate(words)}
    toks = data.tokenize(b.get("prompt", ""))
    ids = [stoi.get(t, 0) for t in toks] or [stoi.get("<eot>", 0)]
    n = max(0, min(int(b.get("n", 30)), 400))
    temp = float(b.get("temperature", 0.8))
    rng = np.random.default_rng(int(b.get("seed", 0)))
    T, nh = cfg["T"], cfg["n_head"]
    steps = []
    for _ in range(n):
        probs, _ = m.next_word_probs(p, ids, T, nh, temp)
        nxt = int(rng.choice(len(probs), p=probs))
        top = np.argsort(-probs)[:5]
        steps.append(dict(top=[[words[j], float(probs[j])] for j in top], drawn=words[nxt],
                          p=float(probs[nxt])))
        ids.append(nxt)
    # what the model looks at for the very next word, per block and head
    probs, atts = m.next_word_probs(p, ids, T, nh, temp)
    ctx = [words[i] for i in ids[-T:]]
    top = np.argsort(-probs)[:10]
    return dict(text=data.decode(ids, words), prompt_tokens=toks,
                unknown=[t for t in toks if t not in stoi], steps=steps, context=ctx,
                next=[[words[j], float(probs[j])] for j in top],
                attention=[np.round(a, 3).tolist() for a in atts])


def lesson(q, b):
    return walkthrough.build()


def source(q, b):
    files = []
    for rel in SOURCE_FILES:
        text = (data.ROOT / rel).read_text()
        files.append(dict(path=rel, lines=text.count("\n") + 1, html=walkthrough.highlight(text)))
    return dict(files=files)


def gradcheck(q, b):
    """Re-check the hand-written gradients against measured slopes, on the code as it is now."""
    m = importlib.reload(model_module)
    n_layer, n_head = int(b.get("n_layer", 2)), int(b.get("n_head", 2))
    rng = np.random.default_rng(0)
    V, T, C = 11, 5, 4 * n_head
    p = m.init_params(rng, V, T, C, 2 * C, n_layer)
    for k in p:
        p[k] = p[k] + rng.normal(0, 0.3, p[k].shape)
    x, y = rng.integers(0, V, (2, T)), rng.integers(0, V, (2, T))
    err = m.grad_check(p, x, y, n_head)
    return dict(error=float(err), ok=bool(err < 1e-6), n_layer=n_layer, n_head=n_head,
                weights=int(sum(v.size for v in p.values())))


def fetch(q, b):
    path = data.fetch_tinystories()
    data.load.cache_clear()
    return dict(corpora=data.list_corpora(), path=str(path))


ROUTES = {
    ("GET", "/api/state"): state,
    ("GET", "/api/corpus"): corpus,
    ("POST", "/api/tokenize"): tokenize,
    ("POST", "/api/run/start"): run_start,
    ("POST", "/api/run/pause"): run_pause,
    ("POST", "/api/run/resume"): run_resume,
    ("POST", "/api/run/stop"): run_stop,
    ("GET", "/api/run/delta"): run_delta,
    ("GET", "/api/run"): run_record,
    ("GET", "/api/runs"): runs,
    ("POST", "/api/generate"): generate,
    ("GET", "/api/lesson"): lesson,
    ("GET", "/api/source"): source,
    ("POST", "/api/gradcheck"): gradcheck,
    ("POST", "/api/fetch"): fetch,
}


def dispatch(method, path, query=None, body=None):
    fn = ROUTES.get((method, path))
    if not fn:
        return 404, {"error": f"no such call: {method} {path}"}
    try:
        return 200, fn(query or {}, body or {})
    except (Refused, ValueError, RuntimeError, FileNotFoundError, KeyError) as e:
        return 400, {"error": str(e).strip("'\"")}
