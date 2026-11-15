"""Training runs you can start, pause, stop and watch.

A Run trains in its own thread and keeps every number it produces in memory
(per-step loss, held-out checks, text samples); the server streams those to
the dashboard. Each run is saved under runs/<id>/ so it can be compared and
used for generation later.
"""

import importlib
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from . import data
from . import model as model_module

RUNS = data.ROOT / "runs"

DEFAULTS = dict(
    name="", corpus="tinytales", min_freq=1,
    T=16, C=64, mlp_mult=4, n_layer=2, n_head=4,
    B=32, lr=3e-3, schedule="constant", warmup=0, clip=0.0,
    steps=1000, eval_every=100, eval_batches=4,
    sample_words=40, temperature=0.8, seed=0, dtype="float32",
)

HELP = dict(
    corpus="Text file in data/ to learn from.",
    min_freq="Words seen fewer times than this become <unk>.",
    T="Context length: how many words the model sees at once.",
    C="Width: how many numbers stand for one word at one position.",
    mlp_mult="MLP hidden width H as a multiple of C.",
    n_layer="How many transformer blocks are stacked.",
    n_head="Attention heads per block. C must divide evenly by it.",
    B="Batch size: windows per training step.",
    lr="Learning rate: how far Adam moves each weight per step.",
    schedule="constant, or cosine (falls smoothly to 10% of lr).",
    warmup="Steps over which lr rises from 0 at the start.",
    clip="Cap on the gradient norm. 0 turns it off.",
    steps="Training steps to run.",
    eval_every="Check on held-out text every this many steps.",
    eval_batches="Batches averaged for each held-out check.",
    sample_words="Words generated at each check.",
    temperature="Sampling temperature for those samples.",
    seed="Random seed for the weights and the batches.",
    dtype="float32 is about twice as fast; float64 is exact.",
)


def clean_config(cfg):
    out = dict(DEFAULTS)
    for k, v in (cfg or {}).items():
        if k not in DEFAULTS:
            continue
        kind = type(DEFAULTS[k])
        out[k] = kind(v) if kind is not str else str(v)
    if out["C"] % out["n_head"]:
        raise ValueError(f"C ({out['C']}) must divide evenly by n_head ({out['n_head']})")
    for k in ("T", "C", "mlp_mult", "n_layer", "n_head", "B", "steps", "eval_every", "eval_batches"):
        if out[k] < 1:
            raise ValueError(f"{k} must be at least 1")
    if out["lr"] <= 0:
        raise ValueError("lr must be above 0")
    if out["schedule"] not in ("constant", "cosine"):
        raise ValueError("schedule must be constant or cosine")
    if out["dtype"] not in ("float32", "float64"):
        raise ValueError("dtype must be float32 or float64")
    return out


def lr_at(cfg, step):
    lr = cfg["lr"]
    if cfg["warmup"] and step <= cfg["warmup"]:
        return lr * step / cfg["warmup"]
    if cfg["schedule"] == "cosine":
        span = max(1, cfg["steps"] - cfg["warmup"])
        frac = min(1.0, (step - cfg["warmup"]) / span)
        return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
    return lr


class Run:
    def __init__(self, cfg, run_id=None):
        self.cfg = clean_config(cfg)
        self.id = run_id or time.strftime("%Y%m%d-%H%M%S")
        self.status = "starting"      # starting running paused stopped done error
        self.error = ""
        self.step = 0
        self.metrics = []             # [step, loss, grad_norm, lr, steps_per_sec]
        self.evals = []               # {step, train_loss, train_acc, val_loss, val_acc}
        self.samples = []             # {step, text}
        self.info = {}                # vocab size, weight count, baselines
        self.params = None
        self.words = None
        self.model = model_module
        self._stop = threading.Event()
        self._go = threading.Event()
        self._go.set()
        self._thread = None

    # ---- control
    def start(self):
        self._thread = threading.Thread(target=self._safe_loop, daemon=True, name=f"run-{self.id}")
        self._thread.start()
        return self

    def pause(self):
        if self.status == "running":
            self._go.clear()
            self.status = "paused"

    def resume(self):
        if self.status == "paused":
            self.status = "running"
            self._go.set()

    def stop(self):
        self._stop.set()
        self._go.set()

    def join(self, timeout=None):
        if self._thread:
            self._thread.join(timeout)

    @property
    def alive(self):
        return self.status in ("starting", "running", "paused")

    # ---- the loop
    def _safe_loop(self):
        try:
            self._loop()
        except Exception as e:      # shown on the dashboard instead of dying silently
            self.error = f"{type(e).__name__}: {e}"
            self.status = "error"
            self.save()

    def _loop(self):
        cfg = self.cfg
        m = self.model = importlib.reload(model_module)   # pick up edits to model.py
        ds = data.load(cfg["corpus"], cfg["min_freq"])
        T, n_head = cfg["T"], cfg["n_head"]
        if len(ds.val) <= T + 1:
            raise ValueError(f"held-out text has {len(ds.val)} tokens; a window needs {T + 1}")
        self.words = ds.words
        rng = np.random.default_rng(cfg["seed"])
        p = m.init_params(rng, ds.V, T, cfg["C"], cfg["mlp_mult"] * cfg["C"],
                          cfg["n_layer"], np.dtype(cfg["dtype"]).type)
        self.params = p
        self.info = dict(V=ds.V, n_params=int(sum(v.size for v in p.values())),
                         chance_loss=ds.stats["chance_loss"],
                         baseline_acc=ds.stats["baseline_acc"],
                         baseline_word=ds.stats["baseline_word"],
                         train_tokens=ds.stats["train_tokens"],
                         val_tokens=ds.stats["val_tokens"],
                         shapes={k: list(v.shape) for k, v in p.items()})
        opt = m.Adam(p, lr=cfg["lr"])
        self.status = "running"
        self._check(ds, 0)

        sps, t0 = 0.0, time.perf_counter()
        for step in range(1, cfg["steps"] + 1):
            self._go.wait()
            if self._stop.is_set():
                break
            opt.lr = lr_at(cfg, step)
            x, y = data.get_batch(ds.train, rng, cfg["B"], T)
            loss, _, cache = m.forward(p, x, y, n_head)
            if not math.isfinite(loss):
                raise FloatingPointError(f"loss became {loss} at step {step}: lower the learning rate")
            grads = m.backward(p, cache)
            gnorm = math.sqrt(sum(float((g * g).sum()) for g in grads.values()))
            if cfg["clip"] and gnorm > cfg["clip"]:
                for g in grads.values():
                    g *= cfg["clip"] / gnorm
            opt.step(p, grads)

            t1 = time.perf_counter()
            inst = 1.0 / max(t1 - t0, 1e-9)
            sps = inst if step == 1 else 0.9 * sps + 0.1 * inst
            t0 = t1
            self.step = step
            self.metrics.append([step, loss, gnorm, opt.lr, sps])
            if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
                self._check(ds, step)
                self.save()
                t0 = time.perf_counter()

        self.status = "stopped" if self._stop.is_set() else "done"
        if not self.evals or self.evals[-1]["step"] != self.step:
            self._check(ds, self.step)
        self.save()

    def _check(self, ds, step):
        """Loss and accuracy on train and on held-out text, plus a text sample."""
        cfg, m, p = self.cfg, self.model, self.params
        erng = np.random.default_rng(123)     # the same batches at every check
        out = {"step": step}
        for part, src in (("train", ds.train), ("val", ds.val)):
            res = [m.evaluate(p, *data.get_batch(src, erng, cfg["B"], cfg["T"]), cfg["n_head"])
                   for _ in range(cfg["eval_batches"])]
            out[f"{part}_loss"] = float(np.mean([r[0] for r in res]))
            out[f"{part}_acc"] = float(np.mean([r[1] for r in res]))
        self.evals.append(out)
        if cfg["sample_words"]:
            start = [ds.stoi.get("<eot>", int(ds.train[0]))]
            ids = m.sample(p, start, cfg["sample_words"], cfg["T"],
                           np.random.default_rng(step), cfg["n_head"], cfg["temperature"])
            self.samples.append({"step": step, "text": data.decode(ids[1:], ds.words)})

    # ---- reading
    def summary(self):
        last = self.evals[-1] if self.evals else {}
        return dict(id=self.id, name=self.cfg["name"], status=self.status, error=self.error,
                    step=self.step, steps=self.cfg["steps"], cfg=self.cfg, info=self.info,
                    last=last, sps=self.metrics[-1][4] if self.metrics else 0.0)

    def delta(self, mi=0, ei=0, si=0):
        """Everything produced since the given positions."""
        return dict(summary=self.summary(), metrics=self.metrics[mi:], evals=self.evals[ei:],
                    samples=self.samples[si:])

    # ---- disk
    @property
    def dir(self):
        return RUNS / self.id

    def save(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        (self.dir / "run.json").write_text(json.dumps(dict(
            summary=self.summary(), metrics=self.metrics, evals=self.evals, samples=self.samples)))
        if self.params is not None:
            np.savez(self.dir / "params.npz", **self.params)
            (self.dir / "words.json").write_text(json.dumps(self.words))


class Manager:
    """Holds the live run and reads finished ones back from disk."""

    def __init__(self):
        self.live = {}
        self.active = None
        self._lock = threading.Lock()

    def start(self, cfg):
        with self._lock:
            if self.active and self.active.alive:
                raise RuntimeError(f"run {self.active.id} is still going: stop it first")
            run = Run(cfg)
            while run.id in self.live or run.dir.exists():
                run.id += "x"
            self.live[run.id] = self.active = run
            return run.start()

    def get(self, run_id=None):
        if not run_id:
            return self.active
        return self.live.get(run_id)

    def saved(self, run_id):
        path = RUNS / run_id / "run.json"
        return json.loads(path.read_text()) if path.exists() else None

    def record(self, run_id):
        """Full numbers of a run, live or saved."""
        run = self.live.get(run_id)
        return run.delta() if run else self.saved(run_id)

    def list(self):
        out = {}
        if RUNS.exists():
            for d in sorted(RUNS.iterdir()):
                rec = self.saved(d.name)
                if rec:
                    out[d.name] = rec["summary"]
        for rid, run in self.live.items():
            out[rid] = run.summary()
        return sorted(out.values(), key=lambda s: s["id"], reverse=True)

    def weights(self, run_id=None):
        """(params, words, cfg, model) for generating from a run."""
        run = self.get(run_id)
        if run and run.params is not None:
            return run.params, run.words, run.cfg, run.model
        d = RUNS / str(run_id)
        if not (d / "params.npz").exists():
            raise FileNotFoundError("no trained weights yet: start a run first")
        with np.load(d / "params.npz") as z:
            p = {k: z[k] for k in z.files}
        words = json.loads((d / "words.json").read_text())
        return p, words, self.saved(run_id)["summary"]["cfg"], model_module
