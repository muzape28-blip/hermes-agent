#!/usr/bin/env python3
"""Hermes Termux ARM32 minimal runtime.

This module is intentionally stdlib-only.  It is the tiny, explicitly-limited
fallback path for Android/Termux ARM32 devices where the normal Hermes runtime
cannot safely install native dependencies such as pydantic-core, jiter, PyYAML,
Tornado, Pillow, or firecrawl-anydoc.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import sysconfig
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import Any

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float


def _package_version() -> str:
    try:
        return version("hermes-agent")
    except PackageNotFoundError:
        return "source checkout"


def _is_android_arm32() -> bool:
    """Best-effort Termux ARM32 detector, kept in sync with installer logic."""
    release = (platform.release() or "").lower()
    prefix = os.environ.get("PREFIX", "").lower()
    is_android = (
        sys.platform == "android"
        or "android" in release
        or "com.termux" in prefix
        or bool(os.environ.get("ANDROID_ROOT"))
    )
    if not is_android:
        return False
    machine = (platform.machine() or "").lower()
    platform_tag = (sysconfig.get_platform() or "").lower()
    multiarch = (sysconfig.get_config_var("MULTIARCH") or "").lower()
    return (
        machine in {"arm", "armv7l", "armv8l"}
        or "armeabi" in platform_tag
        or "arm-linux-androideabi" in multiarch
    )


def _env_first(*names: str) -> str:
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return ""


def _normalize_base_url(raw: str) -> str:
    base = (raw or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise SystemExit(f"invalid base URL {base!r}: must start with https:// or http://")
    return base


def _default_model_for_base(base_url: str) -> str:
    if "openrouter.ai" in base_url:
        return DEFAULT_OPENROUTER_MODEL
    return DEFAULT_OPENAI_MODEL


def _runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    base_url = _normalize_base_url(args.base_url or _env_first("HERMES_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL)
    api_key = args.api_key or _env_first("HERMES_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY")
    model = args.model or _env_first("HERMES_MODEL", "OPENAI_MODEL", "OPENROUTER_MODEL") or _default_model_for_base(base_url)
    if not api_key:
        raise SystemExit(
            "missing API key. Set HERMES_API_KEY (or OPENAI_API_KEY / OPENROUTER_API_KEY)."
        )
    return RuntimeConfig(api_key=api_key, base_url=base_url, model=model, timeout=float(args.timeout))


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _request_json(url: str, payload: dict[str, Any], cfg: RuntimeConfig) -> dict[str, Any]:
    headers = {
        "Authorization": f"Bearer {cfg.api_key}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "hermes-agent-termux-arm32-minimal/0.1",
    }
    if "openrouter.ai" in cfg.base_url:
        headers.setdefault("HTTP-Referer", "https://github.com/NousResearch/hermes-agent")
        headers.setdefault("X-Title", "Hermes Agent Termux ARM32 Minimal")
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=cfg.timeout) as response:  # noqa: S310 - user-configured API URL
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-2000:]
        raise SystemExit(f"HTTP {exc.code} from provider: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"provider request failed: {exc.reason}") from exc
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"provider returned non-JSON response: {raw[:500]!r}") from exc
    if not isinstance(parsed, dict):
        raise SystemExit("provider returned JSON but not an object")
    return parsed


def _extract_chat_content(response: dict[str, Any]) -> str:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise SystemExit(f"provider response has no choices: {json.dumps(response)[:1000]}")
    first = choices[0]
    if not isinstance(first, dict):
        raise SystemExit("provider response choice is not an object")
    message = first.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "".join(parts)
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise SystemExit(f"could not find assistant text in provider response: {json.dumps(response)[:1000]}")


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"Hermes ARM32 minimal runtime ({_package_version()})")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    base_url = _normalize_base_url(args.base_url or _env_first("HERMES_BASE_URL", "OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL)
    api_key_present = bool(args.api_key or _env_first("HERMES_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"))
    model = args.model or _env_first("HERMES_MODEL", "OPENAI_MODEL", "OPENROUTER_MODEL") or _default_model_for_base(base_url)
    print("Hermes Termux ARM32 minimal doctor")
    print(f"  python: {sys.version.split()[0]} ({sys.platform}, {platform.machine()})")
    print(f"  android_arm32_detected: {_is_android_arm32()}")
    print(f"  base_url: {base_url}")
    print(f"  model: {model}")
    print(f"  api_key: {'set' if api_key_present else 'missing'}")
    print("  disabled_features: dashboard, vision, HEIF, heavy document extraction, voice/STT, wake-word")
    if not api_key_present:
        print("Set HERMES_API_KEY before running `hermes-arm32 chat ...`.")
        return 1
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    prompt = " ".join(args.prompt).strip() if args.prompt else sys.stdin.read().strip()
    if not prompt:
        raise SystemExit("provide a prompt argument or pipe prompt text on stdin")
    cfg = _runtime_config(args)
    messages: list[dict[str, str]] = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})
    payload: dict[str, Any] = {
        "model": cfg.model,
        "messages": messages,
        "temperature": args.temperature,
    }
    if args.max_tokens is not None:
        payload["max_tokens"] = args.max_tokens
    response = _request_json(_chat_url(cfg.base_url), payload, cfg)
    print(_extract_chat_content(response).rstrip())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-arm32",
        description="Stdlib-only Hermes minimal chat path for Termux Android ARM32.",
    )
    parser.add_argument("--api-key", help="API key (else HERMES_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY)")
    parser.add_argument("--base-url", help=f"OpenAI-compatible API base URL (default: {DEFAULT_OPENAI_BASE_URL})")
    parser.add_argument("--model", help="Model name (else HERMES_MODEL / OPENAI_MODEL / OPENROUTER_MODEL)")
    parser.add_argument("--timeout", default="60", help="HTTP timeout in seconds (default: 60)")
    parser.add_argument("--version", action="store_true", help="Show minimal runtime version")
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="Show minimal runtime environment and missing config")
    doctor.set_defaults(func=cmd_doctor)

    chat = sub.add_parser("chat", help="Send one prompt to an OpenAI-compatible /chat/completions API")
    chat.add_argument("prompt", nargs="*", help="Prompt text. If omitted, stdin is read.")
    chat.add_argument("--system", default="", help="Optional system message")
    chat.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    chat.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens")
    chat.set_defaults(func=cmd_chat)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        return cmd_version(args)
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
