"""The app's calls end to end, without opening a port."""

import json
import shutil

import pytest

from opengpt import api, server, trainer


@pytest.fixture(autouse=True)
def runs_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(trainer, "RUNS", tmp_path / "runs")
    api.manager.live.clear()
    api.manager.active = None
    yield
    if api.manager.active and api.manager.active.alive:
        api.manager.active.stop()
        api.manager.active.join(10)


def call(method, path, query=None, body=None):
    status, data = api.dispatch(method, path, query, body)
    server.to_json(data)                           # everything must survive JSON
    return status, data


def test_state_and_corpus():
    s, d = call("GET", "/api/state")
    assert s == 200 and "tinytales" in d["corpora"] and d["defaults"]["steps"] and d["active"] is None
    s, c = call("GET", "/api/corpus", {"name": "tinytales", "min_freq": "1"})
    assert s == 200 and c["tokens"] > 50000 and c["val_tokens"] > 5000 and 0 < c["baseline_acc"] < 0.5
    s, t = call("POST", "/api/tokenize", body={"text": "Once upon a time there was a Zorgblat.", "T": 4})
    assert t["tokens"][0] == "once" and t["unknown"] == ["zorgblat"] and t["window"]["y"][0] == "upon"


def test_a_full_run_streams_saves_and_generates():
    cfg = dict(steps=120, eval_every=40, C=32, n_layer=2, n_head=2, T=12, name="t")
    s, d = call("POST", "/api/run/start", body={"cfg": cfg})
    assert s == 200 and d["status"] in ("starting", "running")
    rid = d["id"]
    s, again = call("POST", "/api/run/start", body={"cfg": cfg})
    assert s == 400 and "still going" in again["error"]

    run = api.manager.get(rid)
    got = dict(metrics=[], evals=[], samples=[])
    for delta in server.stream(run, tick=0.02):     # what the dashboard receives
        server.to_json(delta)
        for k in got:
            got[k] += delta[k]
    assert run.status == "done", run.error
    assert [m[0] for m in got["metrics"]] == list(range(1, 121))     # every step, once, in order
    assert [e["step"] for e in got["evals"]] == [0, 40, 80, 120]
    assert got["evals"][-1]["val_loss"] < got["evals"][0]["val_loss"]
    assert len(got["samples"]) == 4

    s, rec = call("GET", "/api/run", {"id": rid})
    assert rec["summary"]["status"] == "done" and len(rec["metrics"]) == 120
    assert json.loads((trainer.RUNS / rid / "run.json").read_text())["summary"]["step"] == 120

    s, g = call("POST", "/api/generate", body={"id": rid, "prompt": "once upon a", "n": 12})
    assert s == 200 and g["text"].startswith("once upon a") and len(g["steps"]) == 12
    assert len(g["attention"]) == 2 and len(g["attention"][0]) == 2            # blocks x heads
    row = g["attention"][0][0][-1]
    assert abs(sum(row) - 1) < 1e-2 and len(g["next"]) == 10

    api.manager.live.clear()                        # as after a restart: read it back from disk
    api.manager.active = None
    s, g2 = call("POST", "/api/generate", body={"id": rid, "prompt": "once upon a", "n": 12})
    assert g2["text"] == g["text"]
    s, lst = call("GET", "/api/runs")
    assert lst["runs"][0]["id"] == rid


def test_pause_resume_stop_through_the_api():
    import time
    s, d = call("POST", "/api/run/start",
                body={"cfg": dict(steps=10**6, eval_every=10**6, C=16, n_layer=1, n_head=1, T=8)})
    rid = d["id"]
    run = api.manager.get(rid)
    while run.step < 10:
        time.sleep(0.01)
    assert call("POST", "/api/run/pause", body={"id": rid})[1]["status"] == "paused"
    a = run.step
    time.sleep(0.15)
    assert run.step - a <= 1
    assert call("POST", "/api/run/resume", body={"id": rid})[1]["status"] == "running"
    s, d = call("POST", "/api/run/stop", body={"id": rid})
    assert d["status"] == "stopped" and (trainer.RUNS / rid / "params.npz").exists()


def test_errors_are_told_not_thrown():
    assert call("POST", "/api/run/start", body={"cfg": {"C": 30, "n_head": 4}})[0] == 400
    assert call("GET", "/api/corpus", {"name": "nope"})[0] == 400
    assert call("POST", "/api/generate", body={"id": "nope"})[0] == 400
    assert call("GET", "/api/nothing")[0] == 404
    s, d = call("POST", "/api/run/start", body={"cfg": dict(lr=1e6, steps=50, C=16, n_head=1, n_layer=1)})
    run = api.manager.get(d["id"])
    run.join(20)
    assert run.status in ("error", "done")          # a blown-up run reports, it does not hang


def test_lesson_source_and_gradcheck():
    s, les = call("GET", "/api/lesson")
    assert s == 200 and len(les["steps"]) == 34
    assert [st["n"] for st in les["steps"]] == list(range(1, 35))
    assert 'class="fn">tokenize<' in les["steps"][0]["code"] and les["steps"][31]["curve"][-1] < 0.05
    assert "result: the cat sat down" in les["steps"][33]["example"]
    words = " ".join(w for w, _ in les["terms"])
    assert all(t in words for t in ("Head", "MLP", "Stream", "Gradient", "Loss", "Token"))
    s, src = call("GET", "/api/source")
    assert any(f["path"] == "opengpt/model.py" and "attention_bwd" in f["html"] for f in src["files"])
    s, gc = call("POST", "/api/gradcheck", body={"n_layer": 2, "n_head": 2})
    assert gc["ok"] and gc["error"] < 1e-6
