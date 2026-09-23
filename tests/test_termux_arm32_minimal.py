from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import ast
from pathlib import Path

import hermes_termux_arm32_minimal as minimal


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_minimal_runtime_does_not_import_native_blockers():
    tree = ast.parse((REPO_ROOT / "hermes_termux_arm32_minimal.py").read_text(encoding="utf-8"))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".", 1)[0])
    assert imported.isdisjoint({"openai", "pydantic", "yaml", "ruamel", "httpx", "requests", "PIL"})


def test_detector_handles_legacy_termux_linux_arm32(monkeypatch):
    monkeypatch.setattr(minimal.sys, "platform", "linux")
    monkeypatch.setattr(minimal.platform, "machine", lambda: "armv8l")
    monkeypatch.setattr(minimal.platform, "release", lambda: "6.1.0-android14")
    monkeypatch.setattr(minimal.sysconfig, "get_platform", lambda: "linux-armv7l")
    monkeypatch.setenv("PREFIX", "/data/data/com.termux/files/usr")

    assert minimal._is_android_arm32() is True


def test_chat_uses_openai_compatible_payload(capsys):
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["payload"] = json.loads(body)
            response = {"choices": [{"message": {"content": "Halo juga dari mock"}}]}
            raw = json.dumps(response).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, _format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        rc = minimal.main([
            "--base-url",
            f"http://127.0.0.1:{server.server_port}/v1",
            "--api-key",
            "secret",
            "--model",
            "mock-model",
            "chat",
            "Halo",
        ])
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert rc == 0
    assert capsys.readouterr().out.strip() == "Halo juga dari mock"
    assert seen["path"] == "/v1/chat/completions"
    assert seen["auth"] == "Bearer secret"
    assert seen["payload"] == {
        "model": "mock-model",
        "messages": [{"role": "user", "content": "Halo"}],
        "temperature": 0.2,
    }


def test_doctor_reports_missing_key(capsys, monkeypatch):
    monkeypatch.delenv("HERMES_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    assert minimal.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "api_key: missing" in out
    assert "disabled_features" in out
