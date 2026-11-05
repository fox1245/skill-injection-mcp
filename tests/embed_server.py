"""Local stdlib HTTP embedding server for multi-process and concurrency tests."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from skill_inject_mcp.embed.embedder import FakeEmbedder


class EmbeddingServer:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.delay_s = float(delay_s)
        self._lock = threading.Lock()
        self._inputs: list[str] = []
        self._request_count = 0
        self._inflight = 0
        self._max_inflight = 0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.start()

    @property
    def inputs(self) -> list[str]:
        with self._lock:
            return list(self._inputs)

    @property
    def request_count(self) -> int:
        with self._lock:
            return self._request_count

    @property
    def max_inflight(self) -> int:
        with self._lock:
            return self._max_inflight

    @property
    def base_url(self) -> str:
        assert self._httpd is not None
        host, port = self._httpd.server_address[:2]
        return f"http://{host}:{port}/v1"

    def start(self) -> EmbeddingServer:
        if self._httpd is not None:
            return self
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path.rstrip("/") != "/v1/embeddings":
                    self.send_error(404)
                    return
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                raw_input = payload.get("input", [])
                texts = [raw_input] if isinstance(raw_input, str) else [str(x) for x in raw_input]
                dim = int(payload.get("dimensions", 16))
                with owner._lock:
                    owner._request_count += 1
                    owner._inputs.extend(texts)
                    owner._inflight += 1
                    if owner._inflight > owner._max_inflight:
                        owner._max_inflight = owner._inflight
                try:
                    if owner.delay_s > 0:
                        time.sleep(owner.delay_s)
                    embedder = FakeEmbedder(dim=dim)
                    data = [
                        {"index": idx, "embedding": embedder._embed_one(text).tolist()}
                        for idx, text in enumerate(texts)
                    ]
                    body = json.dumps({"data": data}).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with owner._lock:
                        owner._inflight -= 1

            def log_message(self, format: str, *args: object) -> None:
                return

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            kwargs={"poll_interval": 0.02},
            daemon=True,
        )
        self._thread.start()
        return self

    def close(self) -> None:
        if self._httpd is None:
            return
        httpd, self._httpd = self._httpd, None
        thread, self._thread = self._thread, None
        httpd.shutdown()
        httpd.server_close()
        if thread is not None:
            thread.join(timeout=5)

    def __enter__(self) -> EmbeddingServer:
        return self.start()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
