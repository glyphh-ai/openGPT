"""A small local web server: static files, the JSON calls in api.py, one live stream."""

import json
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qsl, urlparse

from . import api

STATIC = Path(__file__).resolve().parent / "static"


def to_json(obj):
    return json.dumps(obj, default=float, allow_nan=False).encode()


def stream(run, tick=0.25, sleep=time.sleep):
    """Yield what a run produced since the last tick, until it ends."""
    mi = ei = si = 0
    while True:
        d = run.delta(mi, ei, si)
        mi, ei, si = mi + len(d["metrics"]), ei + len(d["evals"]), si + len(d["samples"])
        yield d
        if not run.alive and mi == len(run.metrics) and ei == len(run.evals):
            return
        sleep(tick)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, status, body, ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _api(self, method):
        url = urlparse(self.path)
        query = dict(parse_qsl(url.query))
        body = {}
        if method == "POST":
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._send(400, to_json({"error": "the request body is not JSON"}))
        status, data = api.dispatch(method, url.path, query, body)
        self._send(status, to_json(data))

    def _stream(self, query):
        run = api.manager.get(query.get("id"))
        if not run:
            return self._send(404, to_json({"error": "no such live run"}))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            for d in stream(run):
                self.wfile.write(b"data: " + to_json(d) + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"event: end\ndata: {}\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        self.close_connection = True

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/api/run/stream":
            return self._stream(dict(parse_qsl(url.query)))
        if url.path.startswith("/api/"):
            return self._api("GET")
        rel = "index.html" if url.path == "/" else url.path.lstrip("/")
        path = (STATIC / rel).resolve()
        if STATIC not in path.parents or not path.is_file():
            return self._send(404, b"not found", "text/plain")
        ctype = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send(200, path.read_bytes(), ctype)

    def do_POST(self):
        self._api("POST")


def serve(host="127.0.0.1", port=8642):
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    if host == "0.0.0.0":
        import socket
        name = socket.gethostname().split(".")[0]
        print(f"openGPT is running for your network at http://{name}.local:{port}  (Ctrl-C to stop)")
    else:
        print(f"openGPT is running at http://{host}:{port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
        if api.manager.active and api.manager.active.alive:
            api.manager.active.stop()
            api.manager.active.join(30)
    finally:
        httpd.server_close()
