#!/usr/bin/env python3
"""Hermes Termux ARM32 minimal runtime.

This module is intentionally stdlib-only.  It is the tiny, explicitly-limited
fallback path for Android/Termux ARM32 devices where the normal Hermes runtime
cannot safely install native dependencies such as pydantic-core, jiter, PyYAML,
Tornado, Pillow, or firecrawl-anydoc.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import platform
import shlex
import shutil
import sys
import sysconfig
import textwrap
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OPENROUTER_MODEL = "openai/gpt-4o-mini"
DOTENV_KEYS = ("HERMES_PROVIDER", "HERMES_BASE_URL", "HERMES_MODEL", "HERMES_API_KEY")


@dataclass(frozen=True)
class ProviderPreset:
    id: str
    label: str
    base_url: str
    env_vars: tuple[str, ...]
    default_model: str = ""
    notes: str = "OpenAI-compatible /chat/completions"
    allow_no_key: bool = False


MINIMAL_PROVIDERS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        "openrouter", "OpenRouter", DEFAULT_OPENROUTER_BASE_URL, ("OPENROUTER_API_KEY",),
        DEFAULT_OPENROUTER_MODEL, "Aggregator; supports live model listing and :free filter."),
    "openai": ProviderPreset(
        "openai", "OpenAI API", DEFAULT_OPENAI_BASE_URL, ("OPENAI_API_KEY",), DEFAULT_OPENAI_MODEL),
    "deepseek": ProviderPreset(
        "deepseek", "DeepSeek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",), "deepseek-chat"),
    "zai": ProviderPreset(
        "zai", "Z.AI / GLM", "https://api.z.ai/api/paas/v4", ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        "glm-4.5-flash"),
    "kimi": ProviderPreset(
        "kimi", "Kimi / Moonshot", "https://api.moonshot.ai/v1", ("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
        "kimi-k2-turbo-preview"),
    "alibaba": ProviderPreset(
        "alibaba", "Alibaba DashScope", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ("DASHSCOPE_API_KEY",), "qwen3.6-flash"),
    "stepfun": ProviderPreset(
        "stepfun", "StepFun Step Plan", "https://api.stepfun.ai/step_plan/v1", ("STEPFUN_API_KEY",),
        "step-3.5-flash"),
    "arcee": ProviderPreset(
        "arcee", "Arcee AI", "https://api.arcee.ai/api/v1", ("ARCEEAI_API_KEY",), "trinity-mini"),
    "gmi": ProviderPreset(
        "gmi", "GMI Cloud", "https://api.gmi-serving.com/v1", ("GMI_API_KEY",),
        "google/gemini-3.1-flash-lite-preview"),
    "actual": ProviderPreset(
        "actual", "Actual Computer", "https://api.actual.inc/v1", ("ACTUAL_API_KEY",), ""),
    "nvidia": ProviderPreset(
        "nvidia", "NVIDIA NIM", "https://integrate.api.nvidia.com/v1", ("NVIDIA_API_KEY",),
        "nvidia/llama-3.1-nemotron-70b-instruct"),
    "ai-gateway": ProviderPreset(
        "ai-gateway", "Vercel AI Gateway", "https://ai-gateway.vercel.sh/v1", ("AI_GATEWAY_API_KEY",),
        "google/gemini-3-flash"),
    "opencode-zen": ProviderPreset(
        "opencode-zen", "OpenCode Zen", "https://opencode.ai/zen/v1", ("OPENCODE_ZEN_API_KEY",),
        "gemini-3-flash"),
    "opencode-go": ProviderPreset(
        "opencode-go", "OpenCode Go", "https://opencode.ai/zen/go/v1", ("OPENCODE_GO_API_KEY",), "glm-5"),
    "kilocode": ProviderPreset(
        "kilocode", "Kilo Code", "https://api.kilo.ai/api/gateway", ("KILOCODE_API_KEY",),
        "google/gemini-3.6-flash"),
    "huggingface": ProviderPreset(
        "huggingface", "Hugging Face", "https://router.huggingface.co/v1", ("HF_TOKEN",),
        "Qwen/Qwen3.5-72B-Instruct"),
    "deepinfra": ProviderPreset(
        "deepinfra", "DeepInfra", "https://api.deepinfra.com/v1/openai", ("DEEPINFRA_API_KEY",),
        "deepseek-ai/DeepSeek-V4-Flash"),
    "fireworks": ProviderPreset(
        "fireworks", "Fireworks AI", "https://api.fireworks.ai/inference/v1", ("FIREWORKS_API_KEY",),
        "accounts/fireworks/models/glm-5p2"),
    "novita": ProviderPreset(
        "novita", "NovitaAI", "https://api.novita.ai/openai/v1", ("NOVITA_API_KEY",),
        "deepseek/deepseek-v3-0324"),
    "upstage": ProviderPreset(
        "upstage", "Upstage Solar", "https://api.upstage.ai/v1", ("UPSTAGE_API_KEY",), "solar-pro3"),
    "nebius": ProviderPreset(
        "nebius", "Nebius Token Factory", "https://api.tokenfactory.nebius.com/v1",
        ("NEBIUS_API_KEY", "NEBIUS_TOKEN_FACTORY_API_KEY"), "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"),
    "ollama-cloud": ProviderPreset(
        "ollama-cloud", "Ollama Cloud", "https://ollama.com/v1", ("OLLAMA_API_KEY",), "nemotron-3-nano:30b"),
    "lmstudio": ProviderPreset(
        "lmstudio", "LM Studio", "http://127.0.0.1:1234/v1", ("LM_API_KEY",), "local-model",
        "Local OpenAI-compatible server; API key optional.", allow_no_key=True),
    "custom": ProviderPreset(
        "custom", "Custom OpenAI-compatible", "http://127.0.0.1:8000/v1", ("CUSTOM_API_KEY",), "",
        "Set HERMES_BASE_URL and HERMES_MODEL for your endpoint; API key optional.", allow_no_key=True),
    "xiaomi": ProviderPreset(
        "xiaomi", "Xiaomi MiMo", "https://api.xiaomimimo.com/v1", ("XIAOMI_API_KEY",), "mimo-v2.5"),
    "commandcode": ProviderPreset(
        "commandcode", "CommandCode", "https://api.commandcode.ai/v1", ("COMMANDCODE_API_KEY",),
        "deepseek/deepseek-v4-flash"),
}

PROVIDER_ALIASES = {
    "or": "openrouter",
    "openai-api": "openai",
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "moonshot": "kimi",
    "kimi-coding": "kimi",
    "vercel": "ai-gateway",
    "vercel-ai-gateway": "ai-gateway",
    "hf": "huggingface",
    "nvidia-nim": "nvidia",
    "nim": "nvidia",
    "nebius-token-factory": "nebius",
    "local": "custom",
    "vllm": "custom",
    "llamacpp": "custom",
    "llama.cpp": "custom",
    "ollama": "custom",
    "mimo": "xiaomi",
}


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float
    provider: str = "custom"


def _package_version() -> str:
    try:
        return version("hermes-agent")
    except PackageNotFoundError:
        return "source checkout"


def _hermes_home() -> Path:
    return Path(os.environ.get("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def _dotenv_candidates() -> list[Path]:
    return [_hermes_home() / ".env", Path.cwd() / ".env"]


def _parse_dotenv_line(line: str) -> tuple[str, str] | None:
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    if stripped.startswith("export "):
        stripped = stripped[len("export "):].lstrip()
    if "=" not in stripped:
        return None
    key, value = stripped.split("=", 1)
    key = key.strip()
    if not key or any(ch.isspace() for ch in key):
        return None
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
        value = value[1:-1]
    return key, value


def load_dotenv_files(*, override: bool = False) -> None:
    """Load simple KEY=VALUE lines from ~/.hermes/.env and ./ .env.

    Existing process environment variables win unless *override* is true. Values
    from the current directory's .env may override ~/.hermes/.env, but this stays
    deliberately small: no interpolation, no shell execution, no dependencies.
    """
    protected = set() if override else set(os.environ)
    for path in _dotenv_candidates():
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            parsed = _parse_dotenv_line(line)
            if parsed is None:
                continue
            key, value = parsed
            if override or key not in protected:
                os.environ[key] = value


def save_minimal_env(cfg: RuntimeConfig, *, include_key: bool = False) -> Path:
    path = _hermes_home() / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    preserved: list[str] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            parsed = _parse_dotenv_line(line)
            if parsed is None:
                preserved.append(line)
                continue
            key, value = parsed
            if key not in DOTENV_KEYS:
                preserved.append(line)
    updates = {
        "HERMES_PROVIDER": cfg.provider,
        "HERMES_BASE_URL": cfg.base_url,
        "HERMES_MODEL": cfg.model,
    }
    if include_key and cfg.api_key:
        updates["HERMES_API_KEY"] = cfg.api_key
    lines = preserved[:]
    if lines and lines[-1].strip():
        lines.append("")
    lines.append("# Hermes ARM32 minimal")
    for key, value in updates.items():
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        lines.append(f'{key}="{escaped}"')
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:
        pass
    return path


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


def _provider_id(raw: str | None) -> str:
    key = (raw or "").strip().lower()
    if not key:
        return ""
    key = PROVIDER_ALIASES.get(key, key)
    if key not in MINIMAL_PROVIDERS:
        raise SystemExit(
            f"unknown minimal provider {raw!r}. Run `hermes-arm32 providers` for supported providers."
        )
    return key


def _provider_from_base(base_url: str) -> str:
    host = base_url.lower()
    for provider_id, preset in MINIMAL_PROVIDERS.items():
        if preset.base_url and preset.base_url.lower().rstrip("/") == base_url.lower().rstrip("/"):
            return provider_id
    if "openrouter.ai" in host:
        return "openrouter"
    if "api.openai.com" in host:
        return "openai"
    if "deepseek.com" in host:
        return "deepseek"
    if "api.z.ai" in host:
        return "zai"
    if "moonshot" in host or "kimi" in host:
        return "kimi"
    if "nvidia.com" in host:
        return "nvidia"
    return "custom"


def _normalize_base_url(raw: str) -> str:
    base = (raw or DEFAULT_OPENAI_BASE_URL).strip().rstrip("/")
    if not base.startswith(("https://", "http://")):
        raise SystemExit(f"invalid base URL {base!r}: must start with https:// or http://")
    return base


def _default_model(provider: str, base_url: str) -> str:
    preset = MINIMAL_PROVIDERS.get(provider)
    if preset and preset.default_model:
        return preset.default_model
    if "openrouter.ai" in base_url:
        return DEFAULT_OPENROUTER_MODEL
    return DEFAULT_OPENAI_MODEL


def _api_key_for_provider(provider: str, args_key: str | None = None) -> str:
    if args_key:
        return args_key
    preset = MINIMAL_PROVIDERS.get(provider)
    names = list(preset.env_vars if preset else ()) + ["HERMES_API_KEY"]
    # Compatibility: previous minimal runtime accepted OPENAI_API_KEY and OPENROUTER_API_KEY globally.
    names.extend(["OPENAI_API_KEY", "OPENROUTER_API_KEY"])
    return _env_first(*dict.fromkeys(names))


def _runtime_config(args: argparse.Namespace, *, require_key: bool = True) -> RuntimeConfig:
    requested_provider = _provider_id(getattr(args, "provider", "") or os.environ.get("HERMES_PROVIDER", ""))
    preset = MINIMAL_PROVIDERS.get(requested_provider) if requested_provider else None
    base_raw = (
        getattr(args, "base_url", None)
        or _env_first("HERMES_BASE_URL", "OPENAI_BASE_URL")
        or (preset.base_url if preset else "")
        or DEFAULT_OPENAI_BASE_URL
    )
    base_url = _normalize_base_url(base_raw)
    provider = requested_provider or _provider_from_base(base_url)
    preset = MINIMAL_PROVIDERS.get(provider)
    api_key = _api_key_for_provider(provider, getattr(args, "api_key", None))
    model = getattr(args, "model", None) or _env_first("HERMES_MODEL", "OPENAI_MODEL", "OPENROUTER_MODEL") or _default_model(provider, base_url)
    if require_key and not api_key and not (preset and preset.allow_no_key):
        env_hint = ", ".join((preset.env_vars if preset else ()) + ("HERMES_API_KEY",))
        raise SystemExit(f"missing API key for {provider}. Set one of: {env_hint}")
    return RuntimeConfig(api_key=api_key, base_url=base_url, model=model, timeout=float(args.timeout), provider=provider)


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _request_headers(cfg: RuntimeConfig) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "hermes-agent-termux-arm32-minimal/0.2",
    }
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if "openrouter.ai" in cfg.base_url:
        headers.setdefault("HTTP-Referer", "https://github.com/NousResearch/hermes-agent")
        headers.setdefault("X-Title", "Hermes Agent Termux ARM32 Minimal")
    return headers


def _request_json(url: str, payload: dict[str, Any], cfg: RuntimeConfig) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=_request_headers(cfg), method="POST")
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


def _get_json(url: str, cfg: RuntimeConfig | None = None) -> dict[str, Any]:
    headers = {"Accept": "application/json", "User-Agent": "hermes-agent-termux-arm32-minimal/0.2"}
    if cfg is not None:
        headers.update(_request_headers(cfg))
    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=(cfg.timeout if cfg else 30)) as response:  # noqa: S310
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[-1000:]
        raise SystemExit(f"HTTP {exc.code} from model catalog: {body or exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise SystemExit(f"model catalog request failed: {exc.reason}") from exc
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"model catalog returned non-JSON response: {raw[:500]!r}") from exc
    if not isinstance(data, dict):
        raise SystemExit("model catalog returned JSON but not an object")
    return data


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


def _chat_completion(messages: list[dict[str, str]], cfg: RuntimeConfig, *, temperature: float = 0.2,
                     max_tokens: int | None = None) -> str:
    payload: dict[str, Any] = {"model": cfg.model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    response = _request_json(_chat_url(cfg.base_url), payload, cfg)
    return _extract_chat_content(response).rstrip()


def _models_url(cfg: RuntimeConfig) -> str:
    return f"{cfg.base_url.rstrip('/')}/models"


def fetch_models(cfg: RuntimeConfig, *, free_only: bool = False) -> list[str]:
    data = _get_json(_models_url(cfg), cfg)
    rows = data.get("data")
    if not isinstance(rows, list):
        raise SystemExit("model catalog response has no data list")
    out: list[str] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            mid = row["id"]
            if free_only and not mid.endswith(":free"):
                continue
            out.append(mid)
    return list(dict.fromkeys(out))


def _print_box(title: str, body: str = "", *, width: int | None = None) -> None:
    width = max(50, min(width or shutil.get_terminal_size((78, 20)).columns, 100))
    title_text = f" {title} "
    top = "┌" + title_text + "─" * max(0, width - len(title_text) - 2) + "┐"
    print(top)
    if body:
        for para in body.splitlines() or [""]:
            wrapped = textwrap.wrap(para, max(10, width - 4), replace_whitespace=False) or [""]
            for line in wrapped:
                print("│ " + line.ljust(width - 4) + " │")
    print("└" + "─" * (width - 2) + "┘")


def _print_chat(role: str, text: str, *, model: str = "") -> None:
    label = role if not model else f"{role} · {model}"
    _print_box(label, text)


def _banner(cfg: RuntimeConfig) -> None:
    body = (
        f"provider: {cfg.provider}    model: {cfg.model}\n"
        f"base_url: {cfg.base_url}\n"
        f"key: {'set' if cfg.api_key else 'not set'}    mode: ARM32 minimal TUI\n"
        "Dashboard, vision, doc extraction, voice/STT, wake-word: disabled."
    )
    _print_box("☤ Hermes Pocket TUI", body)


def _tui_help() -> str:
    return """Commands:
/help                  show this help
/providers             list minimal-safe providers
/models                list models from current provider's /models endpoint
/models free           list OpenRouter/free models when current provider is OpenRouter
/model <id|number>     switch model; number uses last /models result
/provider <name>       switch provider preset
/base-url <url>        set OpenAI-compatible endpoint
/key                   paste API key without echo
/doctor                show runtime status
/clear                 clear conversation memory
/save [--key]          save provider/base/model to ~/.hermes/.env; --key also saves key
/export <file.md>      export transcript
/exit                  quit
""".strip()


def _print_providers() -> None:
    width = max(78, shutil.get_terminal_size((92, 20)).columns)
    _print_box("minimal-safe providers", "OpenAI-compatible providers supported by hermes-arm32 TUI.", width=width)
    print(f"{'provider':18} {'default model':34} env")
    print("─" * min(width, 100))
    for provider_id, preset in MINIMAL_PROVIDERS.items():
        model = (preset.default_model or "<set HERMES_MODEL>")[:34]
        env = ",".join(preset.env_vars) or "optional"
        print(f"{provider_id:18} {model:34} {env}")


def _print_models(models: list[str], current: str = "") -> None:
    if not models:
        print("No models found.")
        return
    _print_box("models", "Use /model <number> or /model <model-id> to switch.")
    for idx, model in enumerate(models, 1):
        mark = " ✓" if model == current else ""
        print(f"{idx:3d}. {model}{mark}")


def _export_transcript(path: str, transcript: list[tuple[str, str]], cfg: RuntimeConfig) -> Path:
    target = Path(path).expanduser()
    if not target.is_absolute():
        target = Path.cwd() / target
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Hermes ARM32 transcript",
        "",
        f"- provider: `{cfg.provider}`",
        f"- model: `{cfg.model}`",
        f"- base_url: `{cfg.base_url}`",
        f"- exported_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
    ]
    for role, text in transcript:
        lines.extend([f"## {role}", "", text.rstrip(), ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def cmd_version(_args: argparse.Namespace) -> int:
    print(f"Hermes ARM32 minimal runtime ({_package_version()})")
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    cfg: RuntimeConfig | None = None
    error = ""
    try:
        cfg = _runtime_config(args, require_key=False)
    except SystemExit as exc:
        error = str(exc)
        provider = _provider_id(getattr(args, "provider", "") or os.environ.get("HERMES_PROVIDER", "")) or "openai"
        preset = MINIMAL_PROVIDERS.get(provider, MINIMAL_PROVIDERS["openai"])
        base_url = _normalize_base_url(getattr(args, "base_url", None) or os.environ.get("HERMES_BASE_URL", "") or preset.base_url)
        cfg = RuntimeConfig(
            api_key="", base_url=base_url,
            model=getattr(args, "model", None) or os.environ.get("HERMES_MODEL", "") or _default_model(provider, base_url),
            timeout=float(args.timeout), provider=provider)
    print("Hermes Termux ARM32 minimal doctor")
    print(f"  python: {sys.version.split()[0]} ({sys.platform}, {platform.machine()})")
    print(f"  android_arm32_detected: {_is_android_arm32()}")
    print(f"  provider: {cfg.provider}")
    print(f"  base_url: {cfg.base_url}")
    print(f"  model: {cfg.model}")
    print(f"  api_key: {'set' if cfg.api_key else 'missing'}")
    print("  disabled_features: dashboard, vision, HEIF, heavy document extraction, voice/STT, wake-word")
    if error:
        print(error)
        return 1
    preset = MINIMAL_PROVIDERS.get(cfg.provider)
    if not cfg.api_key and not (preset and preset.allow_no_key):
        print(f"Set HERMES_API_KEY or a {cfg.provider} provider key before running `hermes-arm32 chat ...`.")
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
    print(_chat_completion(messages, cfg, temperature=args.temperature, max_tokens=args.max_tokens))
    return 0


def cmd_providers(_args: argparse.Namespace) -> int:
    _print_providers()
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    cfg = _runtime_config(args, require_key=False)
    models = fetch_models(cfg, free_only=bool(args.free))
    if args.limit:
        models = models[: args.limit]
    _print_models(models, cfg.model)
    return 0


def _switch_provider(cfg: RuntimeConfig, provider_name: str) -> RuntimeConfig:
    provider_id = _provider_id(provider_name)
    preset = MINIMAL_PROVIDERS[provider_id]
    api_key = _api_key_for_provider(provider_id) or cfg.api_key
    return RuntimeConfig(
        api_key=api_key,
        base_url=preset.base_url,
        model=preset.default_model or cfg.model,
        timeout=cfg.timeout,
        provider=provider_id,
    )


def cmd_tui(args: argparse.Namespace) -> int:
    cfg = _runtime_config(args, require_key=False)
    messages: list[dict[str, str]] = []
    transcript: list[tuple[str, str]] = []
    last_models: list[str] = []
    _banner(cfg)
    print("Ketik pesan langsung, atau /help untuk command. /exit untuk keluar.\n")
    while True:
        try:
            raw = input("› ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if not raw:
            continue
        if raw.startswith("/"):
            try:
                parts = shlex.split(raw[1:])
            except ValueError as exc:
                print(f"Command parse error: {exc}")
                continue
            if not parts:
                continue
            cmd, rest = parts[0].lower(), parts[1:]
            try:
                if cmd in {"exit", "quit", "q"}:
                    print("bye.")
                    return 0
                if cmd == "help":
                    _print_box("commands", _tui_help())
                elif cmd == "doctor":
                    cmd_doctor(argparse.Namespace(**{**vars(args), "provider": cfg.provider, "base_url": cfg.base_url,
                                                     "model": cfg.model, "api_key": cfg.api_key}))
                elif cmd == "providers":
                    _print_providers()
                elif cmd == "provider":
                    if not rest:
                        print(f"current provider: {cfg.provider}")
                    else:
                        cfg = _switch_provider(cfg, rest[0])
                        print(f"provider set: {cfg.provider} ({cfg.base_url})")
                elif cmd in {"base-url", "baseurl"}:
                    if not rest:
                        print(f"current base_url: {cfg.base_url}")
                    else:
                        base_url = _normalize_base_url(rest[0])
                        cfg = replace(cfg, base_url=base_url, provider=_provider_from_base(base_url))
                        print(f"base_url set: {cfg.base_url}")
                elif cmd == "key":
                    key = getpass.getpass("API key: ").strip()
                    cfg = replace(cfg, api_key=key)
                    print("api key set for this session")
                elif cmd == "models":
                    free = bool(rest and rest[0].lower() in {"free", "--free"})
                    last_models = fetch_models(cfg, free_only=free)
                    _print_models(last_models[:80], cfg.model)
                elif cmd == "model":
                    if not rest:
                        print(f"current model: {cfg.model}")
                    else:
                        chosen = rest[0]
                        if chosen.isdigit() and last_models:
                            idx = int(chosen) - 1
                            if idx < 0 or idx >= len(last_models):
                                print(f"model number out of range: {chosen}")
                                continue
                            chosen = last_models[idx]
                        cfg = replace(cfg, model=chosen)
                        print(f"model set: {cfg.model}")
                elif cmd == "clear":
                    messages.clear()
                    transcript.clear()
                    print("conversation cleared")
                elif cmd == "save":
                    include_key = "--key" in rest
                    path = save_minimal_env(cfg, include_key=include_key)
                    print(f"saved {'provider/base/model/key' if include_key else 'provider/base/model'} to {path}")
                elif cmd == "export":
                    if not rest:
                        print("usage: /export <file.md>")
                    else:
                        path = _export_transcript(rest[0], transcript, cfg)
                        print(f"exported transcript: {path}")
                else:
                    print(f"unknown command: /{cmd}. Try /help")
            except SystemExit as exc:
                print(str(exc))
            continue

        _print_chat("you", raw)
        messages.append({"role": "user", "content": raw})
        transcript.append(("you", raw))
        print("☤ thinking...")
        try:
            answer = _chat_completion(messages, cfg, temperature=args.temperature, max_tokens=args.max_tokens)
        except SystemExit as exc:
            print(f"error: {exc}")
            messages.pop()
            transcript.pop()
            continue
        messages.append({"role": "assistant", "content": answer})
        transcript.append(("hermes", answer))
        _print_chat("hermes", answer, model=cfg.model)
        print(f"[{cfg.provider}] [{cfg.model}] [turns: {len(messages)//2}] [ARM32 minimal]\n")



def _add_runtime_overrides(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--provider", default=argparse.SUPPRESS, help="Provider preset override")
    parser.add_argument("--api-key", default=argparse.SUPPRESS, help="API key override")
    parser.add_argument("--base-url", default=argparse.SUPPRESS, help="OpenAI-compatible API base URL override")
    parser.add_argument("--model", default=argparse.SUPPRESS, help="Model override")
    parser.add_argument("--timeout", default=argparse.SUPPRESS, help="HTTP timeout override")

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hermes-arm32",
        description="Stdlib-only Hermes minimal chat path for Termux Android ARM32.",
    )
    parser.add_argument("--provider", help="Provider preset (try: openrouter, openai, deepseek, nvidia, custom)")
    parser.add_argument("--api-key", help="API key (else provider env var or HERMES_API_KEY)")
    parser.add_argument("--base-url", help=f"OpenAI-compatible API base URL (default: {DEFAULT_OPENAI_BASE_URL})")
    parser.add_argument("--model", help="Model name (else HERMES_MODEL / provider default)")
    parser.add_argument("--timeout", default="60", help="HTTP timeout in seconds (default: 60)")
    parser.add_argument("--version", action="store_true", help="Show minimal runtime version")
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="Show minimal runtime environment and missing config")
    _add_runtime_overrides(doctor)
    doctor.set_defaults(func=cmd_doctor)

    providers = sub.add_parser("providers", help="List minimal-safe provider presets")
    providers.set_defaults(func=cmd_providers)

    models = sub.add_parser("models", help="List models from the selected provider's /models endpoint")
    _add_runtime_overrides(models)
    models.add_argument("--free", action="store_true", help="Only show model ids ending in :free (OpenRouter)")
    models.add_argument("--limit", type=int, default=80, help="Max models to print (default: 80)")
    models.set_defaults(func=cmd_models)

    chat = sub.add_parser("chat", help="Send one prompt to an OpenAI-compatible /chat/completions API")
    _add_runtime_overrides(chat)
    chat.add_argument("prompt", nargs="*", help="Prompt text. If omitted, stdin is read.")
    chat.add_argument("--system", default="", help="Optional system message")
    chat.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    chat.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens")
    chat.set_defaults(func=cmd_chat)

    tui = sub.add_parser("tui", aliases=["repl"], help="Interactive Hermes Pocket TUI")
    _add_runtime_overrides(tui)
    tui.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature (default: 0.2)")
    tui.add_argument("--max-tokens", type=int, default=None, help="Optional max_tokens")
    tui.set_defaults(func=cmd_tui)
    return parser


def main(argv: list[str] | None = None) -> int:
    load_dotenv_files()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.version:
        return cmd_version(args)
    if not hasattr(args, "func"):
        # A bare `hermes-arm32` should feel like a TUI app, not a dead help page, on a TTY.
        if sys.stdin.isatty() and sys.stdout.isatty():
            args = parser.parse_args(["tui"])
            return int(args.func(args))
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
