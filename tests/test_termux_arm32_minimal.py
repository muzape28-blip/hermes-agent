from __future__ import annotations

import ast
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
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


def test_opencode_responses_model_uses_responses_route(capsys):
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen["path"] = self.path
            seen["auth"] = self.headers.get("Authorization")
            seen["payload"] = json.loads(body)
            response = {"output_text": "response route ok"}
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
            "--provider",
            "opencode-zen",
            "--base-url",
            f"http://127.0.0.1:{server.server_port}/zen/v1",
            "--api-key",
            "secret",
            "--model",
            "gpt-5.6-luna",
            "chat",
            "Halo",
        ])
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert rc == 0
    assert capsys.readouterr().out.strip() == "response route ok"
    assert seen["path"] == "/zen/v1/responses"
    assert seen["auth"] == "Bearer secret"
    assert seen["payload"] == {
        "model": "gpt-5.6-luna",
        "input": [{"role": "user", "content": [{"type": "input_text", "text": "Halo"}]}],
        "store": False,
    }


def test_opencode_anthropic_model_uses_messages_route(capsys):
    seen: dict[str, object] = {}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - stdlib callback name
            body = self.rfile.read(int(self.headers["Content-Length"]))
            seen["path"] = self.path
            seen["x_api_key"] = self.headers.get("x-api-key")
            seen["version"] = self.headers.get("anthropic-version")
            seen["payload"] = json.loads(body)
            response = {"content": [{"type": "text", "text": "messages route ok"}]}
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
            "--provider",
            "opencode-zen",
            "--base-url",
            f"http://127.0.0.1:{server.server_port}/zen/v1",
            "--api-key",
            "secret",
            "--model",
            "claude-sonnet-5",
            "chat",
            "Halo",
        ])
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert rc == 0
    assert capsys.readouterr().out.strip() == "messages route ok"
    assert seen["path"] == "/zen/v1/messages"
    assert seen["x_api_key"] == "secret"
    assert seen["version"] == "2023-06-01"
    assert seen["payload"] == {
        "model": "claude-sonnet-5",
        "messages": [{"role": "user", "content": "Halo"}],
        "max_tokens": 4096,
    }


def test_models_free_filters_openrouter_catalog(capsys):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802 - stdlib callback name
            assert self.path == "/v1/models"
            response = {
                "data": [
                    {"id": "paid/model"},
                    {"id": "free/model:free"},
                    {"id": "another/free:free"},
                ]
            }
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
            "--provider",
            "openrouter",
            "--base-url",
            f"http://127.0.0.1:{server.server_port}/v1",
            "models",
            "--free",
        ])
    finally:
        server.shutdown()
        thread.join(timeout=5)

    assert rc == 0
    out = capsys.readouterr().out
    assert "free/model:free" in out
    assert "another/free:free" in out
    assert "paid/model" not in out


def test_dotenv_loader_sets_minimal_config(tmp_path, monkeypatch):
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True)
    (hermes_home / ".env").write_text(
        'HERMES_PROVIDER="openrouter"\n'
        'HERMES_BASE_URL="https://openrouter.ai/api/v1"\n'
        'HERMES_MODEL="test/model:free"\n'
        'HERMES_API_KEY="secret"\n',
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    for key in minimal.DOTENV_KEYS:
        monkeypatch.delenv(key, raising=False)

    assert minimal.main(["doctor"]) == 0


def test_providers_command_lists_minimal_safe_presets(capsys):
    assert minimal.main(["providers"]) == 0
    out = capsys.readouterr().out
    assert "openrouter" in out
    assert "deepseek" in out
    assert "nvidia" in out


def test_providers_selection_shows_opencode_detail(capsys):
    assert minimal.main(["providers", "opencode-zen"]) == 0
    out = capsys.readouterr().out
    assert "Provider · opencode-zen" in out
    assert "OpenCode Zen" in out
    assert "OPENCODE_ZEN_API_KEY" in out


def test_opencode_zen_offline_catalog_has_many_routed_models(capsys):
    assert minimal.main(["models", "opencode-zen", "offline", "all"]) == 0
    out = capsys.readouterr().out
    assert "x-preview-f-free" in out
    assert "gpt-5.6-luna" in out
    assert "responses" in out
    assert "claude-sonnet-5" in out
    assert "messages" in out
    assert "gemini-3-flash" in out
    assert "chat" in out


def test_models_terms_can_choose_provider_and_search(capsys):
    assert minimal.main(["models", "opencode-zen", "kimi", "offline", "--limit", "20"]) == 0
    out = capsys.readouterr().out
    assert "kimi-k3" in out
    assert "kimi-k2.5" in out
    assert "claude-sonnet-5" not in out


def test_doctor_reports_missing_key(capsys, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    for key in ("HERMES_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(key, raising=False)

    assert minimal.main(["doctor"]) == 1
    out = capsys.readouterr().out
    assert "api_key: missing" in out
    assert "disabled_features" in out
