"""A stand-in OpenAI-compatible server.

Run as a subprocess by the session tests, exactly as vLLM is. That makes the
session manager's real behaviour testable — process supervision, health polling,
adapter hot-load, idle teardown, lease-to-pid — on a host with no GPU free and
no vLLM installed.

    python -m tests.fake_server --port 8001 --model models/x [--ready-after 0.5]
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

STARTED = time.time()
STATE: dict[str, object] = {"model": "unknown", "ready_after": 0.0, "adapters": {}}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args) -> None:  # keep the test output readable
        pass

    def _send(self, status: int, payload: object) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    @property
    def _ready(self) -> bool:
        return time.time() - STARTED >= float(STATE["ready_after"])

    def do_GET(self) -> None:
        if self.path == "/health":
            if self._ready:
                self._send(200, {"status": "ok"})
            else:
                self._send(503, {"status": "starting"})
            return
        if self.path.startswith("/v1/models"):
            names = [STATE["model"], *STATE["adapters"]]
            self._send(200, {"object": "list", "data": [{"id": n, "object": "model"} for n in names]})
            return
        self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            payload = {}

        if self.path == "/v1/load_lora_adapter":
            name = payload.get("lora_name")
            if not name or not payload.get("lora_path"):
                self._send(400, {"error": "lora_name and lora_path are required"})
                return
            STATE["adapters"][name] = payload["lora_path"]  # type: ignore[index]
            self._send(200, {"status": "loaded", "lora_name": name})
            return

        if self.path == "/v1/unload_lora_adapter":
            name = payload.get("lora_name")
            if name not in STATE["adapters"]:  # type: ignore[operator]
                self._send(404, {"error": "not loaded"})
                return
            STATE["adapters"].pop(name)  # type: ignore[union-attr]
            self._send(200, {"status": "unloaded"})
            return

        if self.path == "/v1/chat/completions":
            model = payload.get("model", STATE["model"])
            known = model == STATE["model"] or model in STATE["adapters"]  # type: ignore[operator]
            if not known:
                self._send(404, {"error": f"unknown model {model}"})
                return
            # A prompt of "ECHO <x>" answers "<x>", so tests control predictions.
            content = f"served by {model}"
            messages = payload.get("messages") or []
            if messages:
                last = messages[-1].get("content", "")
                if isinstance(last, list):
                    last = " ".join(p.get("text", "") for p in last if p.get("type") == "text")
                if isinstance(last, str) and last.startswith("ECHO "):
                    content = last[5:]
            self._send(
                200,
                {
                    "id": "chatcmpl-test",
                    "object": "chat.completion",
                    "model": model,
                    "choices": [
                        {"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}
                    ],
                },
            )
            return

        self._send(404, {"error": "not found"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--model", default="unknown")
    parser.add_argument("--ready-after", type=float, default=0.0)
    parser.add_argument("--exit-after", type=float, default=0.0, help="die, to test startup failure")
    args = parser.parse_args()

    STATE["model"] = args.model
    STATE["ready_after"] = args.ready_after

    if args.exit_after:
        time.sleep(args.exit_after)
        print("fake server exiting on purpose", file=sys.stderr)
        raise SystemExit(7)

    server = HTTPServer(("127.0.0.1", args.port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"fake server on {args.port} serving {args.model}", flush=True)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
