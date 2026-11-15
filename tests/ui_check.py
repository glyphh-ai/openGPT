"""Render every tab of the app in headless Chrome, without starting the server.

The page's network calls are answered from real output of api.dispatch (a real
short training run included), so what is drawn is what the server would send.
Writes one screenshot per tab and fails on any JavaScript error.

    uv run python tests/ui_check.py [out_dir]
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from opengpt import api, server, trainer  # noqa: E402

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
STATIC = Path(server.STATIC)

MOCK = """
window.__errors = [];
window.addEventListener('error', e => { __errors.push(e.message); document.title = 'JSERR ' + e.message; });
window.addEventListener('unhandledrejection', e => { __errors.push(String(e.reason)); document.title = 'JSERR ' + e.reason; });
window.fetch = async (url, opts) => {
  const key = (opts ? 'POST ' : 'GET ') + new URL(url, 'http://x').pathname;
  const d = __FIX[key];
  if (!d) { document.title = 'JSERR missing ' + key; }
  return { ok: !!d, statusText: 'missing ' + key, json: async () => d || { error: 'missing ' + key } };
};
window.EventSource = class {
  constructor() { setTimeout(() => { for (const d of __FIX.stream) this.onmessage({ data: JSON.stringify(d) }); if (this._end) this._end(); }, 30); }
  addEventListener(n, f) { if (n === 'end') this._end = f; }
  close() {}
};
"""


def fixtures(tmp):
    trainer.RUNS = Path(tmp) / "runs"
    fix = {}

    def get(path, q=None):
        s, d = api.dispatch("GET", path, q)
        assert s == 200, (path, d)
        return d

    def post(path, b=None):
        s, d = api.dispatch("POST", path, None, b)
        assert s == 200, (path, d)
        return d

    cfg = dict(steps=400, eval_every=50, C=32, n_layer=2, n_head=4, T=16, name="ui check")
    run_id = post("/api/run/start", {"cfg": cfg})["id"]
    run = api.manager.get(run_id)
    fix["stream"] = list(server.stream(run, tick=0.05))
    assert run.status == "done", run.error
    live = dict(run.summary(), status="running")
    fix["GET /api/state"] = dict(get("/api/state"), active=live)
    fix["GET /api/corpus"] = get("/api/corpus", {"name": "tinytales", "min_freq": "1"})
    fix["POST /api/tokenize"] = post("/api/tokenize", {"text": "Once upon a time there was a brave little zebra named Tom.", "T": 8})
    fix["GET /api/runs"] = get("/api/runs")
    fix["GET /api/run"] = get("/api/run", {"id": run_id})
    fix["POST /api/generate"] = post("/api/generate", {"id": run_id, "prompt": "once upon a time there was a", "n": 30})
    fix["GET /api/lesson"] = get("/api/lesson")
    fix["GET /api/source"] = get("/api/source")
    fix["POST /api/gradcheck"] = post("/api/gradcheck", {"n_layer": 2, "n_head": 2})
    return fix


def chrome(*args):
    return subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                           "--allow-file-access-from-files", "--virtual-time-budget=6000", *args],
                          capture_output=True, text=True, timeout=120)


def main(out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        fix = fixtures(tmp)
        page = (STATIC / "index.html").read_text()
        page = page.replace("<head>", f'<head><base href="{STATIC.as_uri()}/">')
        auto = """
window.addEventListener('load', () => setTimeout(() => {
  const t = (location.hash || '#learn').slice(1);
  if (t === 'generate') document.querySelector('#gGo').click();
  if (t === 'code') document.querySelector('#gcGo').click();
}, 400));
"""
        page = page.replace('<script src="vendor/chart.umd.min.js">',
                            f"<script>window.__FIX = {server.to_json(fix).decode()};{MOCK}{auto}</script>"
                            '<script src="vendor/chart.umd.min.js">')
        html = Path(tmp) / "app.html"
        html.write_text(page)
        failed = []
        expect = dict(learn="Check it by hand", data="held out (10%)", train="How we know it works",
                      generate="block 2 · head 4", runs="ui check", code="Correct: largest gap")
        heights = dict(learn=3600, data=1500, train=1500, generate=1500, runs=1100, code=1100)
        for tab, want in expect.items():
            url = f"{html.as_uri()}#{tab}"
            chrome(f"--window-size=1500,{heights[tab]}", f"--screenshot={out / (tab + '.png')}", url)
            dom = chrome("--dump-dom", url).stdout
            title = dom.split("<title>")[1].split("</title>")[0] if "<title>" in dom else "?"
            ok = "JSERR" not in title and want in dom
            print(f"{tab:9} {'ok' if ok else 'FAILED'}  title={title!r}  found {want!r}: {want in dom}")
            if not ok:
                failed.append(tab)
        print("screenshots in", out)
        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/opengpt-ui"))
