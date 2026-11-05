"""Local stdlib HTTP embedding server for multi-process and concurrency tests."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from skill_inject_mcp.embed.embedder import FakeEmbedder


class _HighBacklogThreadingHTTPServer(ThreadingHTTPServer):
    request_queue_size = 64


class EmbeddingServer:
    def __init__(
        self,
        delay_s: float = 0.0,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        responses: list[tuple[int, dict[str, str], dict | None]] | None = None,
    ) -> None:
        self.delay_s = float(delay_s)
        self.status_code = status_code
        self.headers = dict(headers or {})
        self._responses = list(responses or [])
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
                    scripted = owner._responses.pop(0) if owner._responses else None
                try:
                    if owner.delay_s > 0:
                        time.sleep(owner.delay_s)
                    if scripted is not None:
                        code, resp_headers, custom_body = scripted
                    else:
                        code, resp_headers, custom_body = owner.status_code, owner.headers, None
                    if custom_body is not None:
                        response_payload = custom_body
                    elif code == 200:
                        embedder = FakeEmbedder(dim=dim)
                        response_payload = {
                            "data": [
                                {"index": idx, "embedding": embedder._embed_one(text).tolist()}
                                for idx, text in enumerate(texts)
                            ]
                        }
                    else:
                        response_payload = {"error": f"status {code}"}
                    body = json.dumps(response_payload).encode("utf-8")
                    self.send_response(code)
                    for key, value in resp_headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                finally:
                    with owner._lock:
                        owner._inflight -= 1

            def log_message(self, format: str, *args: object) -> None:
                return

        self._httpd = _HighBacklogThreadingHTTPServer(("127.0.0.1", 0), Handler)
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
