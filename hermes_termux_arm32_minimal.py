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

API_CHAT_COMPLETIONS = "chat_completions"
API_CODEX_RESPONSES = "codex_responses"
API_ANTHROPIC_MESSAGES = "anthropic_messages"


@dataclass(frozen=True)
class ProviderPreset:
    id: str
    label: str
    base_url: str
    env_vars: tuple[str, ...]
    default_model: str = ""
    notes: str = "OpenAI-compatible /chat/completions"
    allow_no_key: bool = False


@dataclass(frozen=True)
class RuntimeConfig:
    api_key: str
    base_url: str
    model: str
    timeout: float
    provider: str = "custom"


@dataclass(frozen=True)
class ModelEntry:
    id: str
    api_mode: str = API_CHAT_COMPLETIONS
    source: str = "live"


MINIMAL_PROVIDERS: dict[str, ProviderPreset] = {
    "openrouter": ProviderPreset(
        "openrouter",
        "OpenRouter",
        DEFAULT_OPENROUTER_BASE_URL,
        ("OPENROUTER_API_KEY",),
        DEFAULT_OPENROUTER_MODEL,
        "Aggregator; live model listing, free-model filtering.",
    ),
    "openai": ProviderPreset("openai", "OpenAI API", DEFAULT_OPENAI_BASE_URL, ("OPENAI_API_KEY",), DEFAULT_OPENAI_MODEL),
    "deepseek": ProviderPreset("deepseek", "DeepSeek", "https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",), "deepseek-chat"),
    "zai": ProviderPreset(
        "zai",
        "Z.AI / GLM",
        "https://api.z.ai/api/paas/v4",
        ("GLM_API_KEY", "ZAI_API_KEY", "Z_AI_API_KEY"),
        "glm-4.5-flash",
    ),
    "kimi": ProviderPreset(
        "kimi",
        "Kimi / Moonshot",
        "https://api.moonshot.ai/v1",
        ("KIMI_API_KEY", "KIMI_CODING_API_KEY"),
        "kimi-k2-turbo-preview",
    ),
    "alibaba": ProviderPreset(
        "alibaba",
        "Alibaba DashScope",
        "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
        ("DASHSCOPE_API_KEY",),
        "qwen3.6-flash",
    ),
    "stepfun": ProviderPreset("stepfun", "StepFun Step Plan", "https://api.stepfun.ai/step_plan/v1", ("STEPFUN_API_KEY",), "step-3.5-flash"),
    "arcee": ProviderPreset("arcee", "Arcee AI", "https://api.arcee.ai/api/v1", ("ARCEEAI_API_KEY",), "trinity-mini"),
    "gmi": ProviderPreset(
        "gmi",
        "GMI Cloud",
        "https://api.gmi-serving.com/v1",
        ("GMI_API_KEY",),
        "google/gemini-3.1-flash-lite-preview",
    ),
    "actual": ProviderPreset("actual", "Actual Computer", "https://api.actual.inc/v1", ("ACTUAL_API_KEY",), ""),
    "nvidia": ProviderPreset(
        "nvidia",
        "NVIDIA NIM",
        "https://integrate.api.nvidia.com/v1",
        ("NVIDIA_API_KEY",),
        "nvidia/llama-3.1-nemotron-70b-instruct",
    ),
    "ai-gateway": ProviderPreset(
        "ai-gateway",
        "Vercel AI Gateway",
        "https://ai-gateway.vercel.sh/v1",
        ("AI_GATEWAY_API_KEY",),
        "google/gemini-3-flash",
    ),
    "opencode-zen": ProviderPreset(
        "opencode-zen",
        "OpenCode Zen",
        "https://opencode.ai/zen/v1",
        ("OPENCODE_ZEN_API_KEY",),
        "gemini-3-flash",
        "Zen pay-as-you-go; Hermes routes model families by wire API.",
    ),
    "opencode-go": ProviderPreset(
        "opencode-go",
        "OpenCode Go",
        "https://opencode.ai/zen/go/v1",
        ("OPENCODE_GO_API_KEY",),
        "glm-5",
        "Go subscription; Hermes routes model families by wire API.",
    ),
    "kilocode": ProviderPreset(
        "kilocode",
        "Kilo Code",
        "https://api.kilo.ai/api/gateway",
        ("KILOCODE_API_KEY",),
        "google/gemini-3.6-flash",
    ),
    "huggingface": ProviderPreset(
        "huggingface",
        "Hugging Face",
        "https://router.huggingface.co/v1",
        ("HF_TOKEN",),
        "Qwen/Qwen3.5-72B-Instruct",
    ),
    "deepinfra": ProviderPreset(
        "deepinfra",
        "DeepInfra",
        "https://api.deepinfra.com/v1/openai",
        ("DEEPINFRA_API_KEY",),
        "deepseek-ai/DeepSeek-V4-Flash",
    ),
    "fireworks": ProviderPreset(
        "fireworks",
        "Fireworks AI",
        "https://api.fireworks.ai/inference/v1",
        ("FIREWORKS_API_KEY",),
        "accounts/fireworks/models/glm-5p2",
    ),
    "novita": ProviderPreset(
        "novita",
        "NovitaAI",
        "https://api.novita.ai/openai/v1",
        ("NOVITA_API_KEY",),
        "deepseek/deepseek-v3-0324",
    ),
    "upstage": ProviderPreset("upstage", "Upstage Solar", "https://api.upstage.ai/v1", ("UPSTAGE_API_KEY",), "solar-pro3"),
    "nebius": ProviderPreset(
        "nebius",
        "Nebius Token Factory",
        "https://api.tokenfactory.nebius.com/v1",
        ("NEBIUS_API_KEY", "NEBIUS_TOKEN_FACTORY_API_KEY"),
        "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B",
    ),
    "ollama-cloud": ProviderPreset("ollama-cloud", "Ollama Cloud", "https://ollama.com/v1", ("OLLAMA_API_KEY",), "nemotron-3-nano:30b"),
    "lmstudio": ProviderPreset(
        "lmstudio",
        "LM Studio",
        "http://127.0.0.1:1234/v1",
        ("LM_API_KEY",),
        "local-model",
        "Local OpenAI-compatible server; API key optional.",
        allow_no_key=True,
    ),
    "custom": ProviderPreset(
        "custom",
        "Custom OpenAI-compatible",
        "http://127.0.0.1:8000/v1",
        ("CUSTOM_API_KEY",),
        "",
        "Set HERMES_BASE_URL and HERMES_MODEL for your endpoint; API key optional.",
        allow_no_key=True,
    ),
    "xiaomi": ProviderPreset("xiaomi", "Xiaomi MiMo", "https://api.xiaomimimo.com/v1", ("XIAOMI_API_KEY",), "mimo-v2.5"),
    "commandcode": ProviderPreset(
        "commandcode",
        "CommandCode",
        "https://api.commandcode.ai/v1",
        ("COMMANDCODE_API_KEY",),
        "deepseek/deepseek-v4-flash",
    ),
}

PROVIDER_ALIASES = {
    "or": "openrouter",
    "openai-api": "openai",
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "moonshot": "kimi",
    "kimi-coding": "kimi",
    "opencode": "opencode-zen",
    "zen": "opencode-zen",
    "opencode_zen": "opencode-zen",
    "go": "opencode-go",
    "opencode_go": "opencode-go",
    "opencode-go-sub": "opencode-go",
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

# Curated floor copied from Hermes' static catalog so the ARM32 TUI does not feel
# empty when a provider requires auth for /models or the phone is offline. Live
# /models still leads when available.
OPENCODE_ZEN_MODELS = (
    "x-preview-f-free",
    "kimi-k3",
    "kimi-k2.5",
    "kimi-k2.6",
    "gpt-5.6-sol",
    "gpt-5.6-terra",
    "gpt-5.6-luna",
    "gpt-5.5",
    "gpt-5.5-pro",
    "gpt-5.4-pro",
    "gpt-5.4",
    "gpt-5.4-mini",
    "gpt-5.4-nano",
    "gpt-5.3-codex",
    "gpt-5.3-codex-spark",
    "gpt-5.2",
    "gpt-5.2-codex",
    "gpt-5.1",
    "gpt-5.1-codex",
    "gpt-5.1-codex-max",
    "gpt-5.1-codex-mini",
    "gpt-5",
    "gpt-5-codex",
    "gpt-5-nano",
    "claude-fable-5",
    "claude-opus-5",
    "claude-sonnet-5",
    "claude-opus-4-8",
    "claude-opus-4-7",
    "claude-opus-4-6",
    "claude-opus-4-5",
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-sonnet-4",
    "claude-haiku-4-5",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-pro",
    "gemini-3-flash",
    "grok-4.6",
    "grok-4.5",
    "grok-build-0.1",
    "muse-spark-1.2",
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.2",
    "glm-5.1",
    "glm-5",
    "kimi-k2.7-code",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "qwen3.6-plus",
    "qwen3.5-plus",
    "big-pickle",
    "mimo-v2.5-free",
    "nemotron-3-ultra-free",
    "nemotron-3.5-lightning-free",
    "muse-spark-1.2-contributor-free",
    "muse-spark-1.3-contributor-free",
)

OPENCODE_GO_MODELS = (
    "kimi-k3",
    "kimi-k2.7-code",
    "kimi-k2.6",
    "kimi-k2.5",
    "gpt-5.6-luna",
    "grok-4.5",
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.2",
    "glm-5.1",
    "glm-5",
    "mimo-v2.5-pro",
    "mimo-v2.5",
    "mimo-v2-pro",
    "mimo-v2-omni",
    "minimax-m3",
    "minimax-m2.7",
    "minimax-m2.5",
    "deepseek-v4-pro",
    "deepseek-v4-flash",
    "qwen3.8-max",
    "qwen3.7-max",
    "qwen3.7-plus",
    "qwen3.6-plus",
    "qwen3.5-plus",
    "hy3",
    "hy3-preview",
    "muse-spark-1.2-contributor",
    "muse-spark-1.3-contributor",
)

STATIC_MODEL_CATALOG: dict[str, tuple[str, ...]] = {
    "opencode-zen": OPENCODE_ZEN_MODELS,
    "opencode-go": OPENCODE_GO_MODELS,
}

PROVIDER_LIST_DEFAULT_LIMIT = 12
MODEL_LIST_DEFAULT_LIMIT = 40


class Ui:
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    cyan = "\033[36m"
    green = "\033[32m"
    yellow = "\033[33m"
    red = "\033[31m"


def _color_enabled() -> bool:
    return bool(sys.stdout.isatty()) and not os.environ.get("NO_COLOR") and os.environ.get("TERM") != "dumb"


def _c(text: str, *codes: str) -> str:
    if not _color_enabled() or not codes:
        return text
    return "".join(codes) + text + Ui.reset


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
        stripped = stripped[len("export ") :].lstrip()
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
            key, _value = parsed
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
    return machine in {"arm", "armv7l", "armv8l"} or "armeabi" in platform_tag or "arm-linux-androideabi" in multiarch


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
        raise SystemExit(f"unknown minimal provider {raw!r}. Run `hermes-arm32 providers` for supported providers.")
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
    if "opencode.ai/zen/go" in host:
        return "opencode-go"
    if "opencode.ai/zen" in host:
        return "opencode-zen"
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


def _flat_model_name(model: str | None) -> str:
    return (model or "").strip().rsplit("/", 1)[-1].lower()


def _opencode_model_api_mode(provider: str, model: str) -> str:
    name = _flat_model_name(model)
    if provider == "opencode-go":
        if name.startswith(("gpt-", "grok-", "muse-spark")):
            return API_CODEX_RESPONSES
        if name.startswith(("minimax-", "qwen", "union-alpha")):
            return API_ANTHROPIC_MESSAGES
        return API_CHAT_COMPLETIONS
    if provider == "opencode-zen":
        if name.startswith(("claude-", "union-alpha", "qwen")):
            return API_ANTHROPIC_MESSAGES
        if name.startswith(("gpt-", "grok-", "muse-spark")):
            return API_CODEX_RESPONSES
        return API_CHAT_COMPLETIONS
    return API_CHAT_COMPLETIONS


def _api_mode_for(provider: str, model: str) -> str:
    if provider in {"opencode-zen", "opencode-go"}:
        return _opencode_model_api_mode(provider, model)
    return API_CHAT_COMPLETIONS


def _mode_label(api_mode: str) -> str:
    return {
        API_CHAT_COMPLETIONS: "chat",
        API_CODEX_RESPONSES: "responses",
        API_ANTHROPIC_MESSAGES: "messages",
    }.get(api_mode, api_mode)


def _request_headers(cfg: RuntimeConfig, *, api_mode: str = API_CHAT_COMPLETIONS) -> dict[str, str]:
    headers = {
        "Accept": "application/json",
        "User-Agent": "hermes-agent-termux-arm32-minimal/0.3",
    }
    if api_mode == API_ANTHROPIC_MESSAGES:
        headers["Content-Type"] = "application/json"
        headers["anthropic-version"] = "2023-06-01"
        if cfg.api_key:
            headers["x-api-key"] = cfg.api_key
        return headers
    headers["Content-Type"] = "application/json"
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    if "openrouter.ai" in cfg.base_url:
        headers.setdefault("HTTP-Referer", "https://github.com/NousResearch/hermes-agent")
        headers.setdefault("X-Title", "Hermes Agent Termux ARM32 Minimal")
    return headers


def _post_json(url: str, payload: dict[str, Any], cfg: RuntimeConfig, *, api_mode: str = API_CHAT_COMPLETIONS) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, headers=_request_headers(cfg, api_mode=api_mode), method="POST")
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
    headers = {"Accept": "application/json", "User-Agent": "hermes-agent-termux-arm32-minimal/0.3"}
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


def _chat_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/chat/completions"


def _responses_url(base_url: str) -> str:
    return f"{base_url.rstrip('/')}/responses"


def _anthropic_messages_url(base_url: str) -> str:
    root = base_url.rstrip("/")
    if root.endswith("/v1"):
        root = root[:-3].rstrip("/")
    return f"{root}/v1/messages"


def _models_url(cfg: RuntimeConfig) -> str:
    return f"{cfg.base_url.rstrip('/')}/models"


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
            parts: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts)
    text = first.get("text")
    if isinstance(text, str):
        return text
    raise SystemExit(f"could not extract assistant text from provider response: {json.dumps(response)[:1000]}")


def _extract_responses_content(response: dict[str, Any]) -> str:
    output_text = response.get("output_text")
    if isinstance(output_text, str) and output_text.strip():
        return output_text
    parts: list[str] = []
    output = response.get("output")
    if isinstance(output, list):
        for item in output:
            if not isinstance(item, dict):
                continue
            content = item.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if isinstance(block, dict):
                    text = block.get("text")
                    if isinstance(text, str):
                        parts.append(text)
    if parts:
        return "\n".join(parts)
    raise SystemExit(f"could not extract text from responses API response: {json.dumps(response)[:1000]}")


def _extract_anthropic_content(response: dict[str, Any]) -> str:
    parts: list[str] = []
    content = response.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
    if parts:
        return "\n".join(parts)
    raise SystemExit(f"could not extract text from messages API response: {json.dumps(response)[:1000]}")


def _chat_completion(
    messages: list[dict[str, str]],
    cfg: RuntimeConfig,
    *,
    temperature: float = 0.2,
    max_tokens: int | None = None,
) -> str:
    api_mode = _api_mode_for(cfg.provider, cfg.model)
    if api_mode == API_CODEX_RESPONSES:
        system = ""
        payload_messages = messages
        if messages and messages[0].get("role") == "system":
            system = messages[0].get("content", "")
            payload_messages = messages[1:]
        input_items: list[dict[str, Any]] = []
        for item in payload_messages:
            role = item.get("role", "user")
            text = item.get("content", "")
            part_type = "output_text" if role == "assistant" else "input_text"
            input_items.append({"role": role, "content": [{"type": part_type, "text": text}]})
        payload: dict[str, Any] = {"model": cfg.model, "input": input_items, "store": False}
        if system:
            payload["instructions"] = system
        if max_tokens is not None:
            payload["max_output_tokens"] = max_tokens
        response = _post_json(_responses_url(cfg.base_url), payload, cfg, api_mode=api_mode)
        return _extract_responses_content(response)

    if api_mode == API_ANTHROPIC_MESSAGES:
        system = ""
        anth_messages: list[dict[str, str]] = []
        for item in messages:
            role = item.get("role", "user")
            text = item.get("content", "")
            if role == "system":
                system = text
                continue
            anth_messages.append({"role": "assistant" if role == "assistant" else "user", "content": text})
        payload = {"model": cfg.model, "messages": anth_messages, "max_tokens": max_tokens or 4096}
        if system:
            payload["system"] = system
        response = _post_json(_anthropic_messages_url(cfg.base_url), payload, cfg, api_mode=api_mode)
        return _extract_anthropic_content(response)

    payload = {"model": cfg.model, "messages": messages, "temperature": temperature}
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    return _extract_chat_content(_post_json(_chat_url(cfg.base_url), payload, cfg))


def _live_model_entries(cfg: RuntimeConfig) -> list[ModelEntry]:
    data = _get_json(_models_url(cfg), cfg)
    rows = data.get("data")
    if not isinstance(rows, list):
        raise SystemExit("model catalog response has no data list")
    out: list[ModelEntry] = []
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("id"), str):
            model_id = row["id"]
            out.append(ModelEntry(model_id, _api_mode_for(cfg.provider, model_id), "live"))
    return out


def _static_model_entries(provider: str) -> list[ModelEntry]:
    return [ModelEntry(model_id, _api_mode_for(provider, model_id), "curated") for model_id in STATIC_MODEL_CATALOG.get(provider, ())]


def _looks_free_model(model_id: str) -> bool:
    lower = model_id.lower()
    return lower.endswith(":free") or lower.endswith("-free") or "/free/" in lower


def _filter_model_entries(entries: list[ModelEntry], *, free_only: bool = False, query: str = "") -> list[ModelEntry]:
    terms = [part for part in query.lower().split() if part]
    filtered: list[ModelEntry] = []
    for entry in entries:
        haystack = entry.id.lower()
        if free_only and not _looks_free_model(entry.id):
            continue
        if terms and not all(term in haystack for term in terms):
            continue
        filtered.append(entry)
    return filtered


def _merge_model_entries(live: list[ModelEntry], curated: list[ModelEntry]) -> list[ModelEntry]:
    curated_modes = {entry.id: entry.api_mode for entry in curated}
    merged: list[ModelEntry] = []
    seen: set[str] = set()
    for entry in live:
        merged.append(ModelEntry(entry.id, curated_modes.get(entry.id, entry.api_mode), "live"))
        seen.add(entry.id)
    for entry in curated:
        if entry.id not in seen:
            merged.append(entry)
            seen.add(entry.id)
    return merged


def fetch_model_entries(
    cfg: RuntimeConfig,
    *,
    free_only: bool = False,
    query: str = "",
    offline: bool = False,
) -> tuple[list[ModelEntry], str]:
    curated = _static_model_entries(cfg.provider)
    live: list[ModelEntry] = []
    note = ""
    if not offline:
        try:
            live = _live_model_entries(cfg)
        except SystemExit as exc:
            if curated:
                note = f"live /models failed, showing Hermes curated catalog ({exc})."
            else:
                raise
    elif curated:
        note = "offline mode: showing Hermes curated catalog."
    entries = _merge_model_entries(live, curated)
    if not entries and curated:
        entries = curated
    return _filter_model_entries(entries, free_only=free_only, query=query), note


def fetch_models(cfg: RuntimeConfig, *, free_only: bool = False) -> list[str]:
    entries, _note = fetch_model_entries(cfg, free_only=free_only)
    return [entry.id for entry in entries]


def _term_width(default: int = 78, maximum: int = 96) -> int:
    return max(48, min(shutil.get_terminal_size((default, 20)).columns, maximum))


def _shorten_middle(text: str, width: int) -> str:
    if width <= 0:
        return ""
    if len(text) <= width:
        return text
    if width <= 3:
        return text[:width]
    left = max(1, (width - 1) // 2)
    right = max(1, width - left - 1)
    return text[:left] + "…" + text[-right:]


def _wrap(text: str, width: int | None = None) -> list[str]:
    width = width or _term_width() - 4
    lines: list[str] = []
    for para in str(text).splitlines() or [""]:
        wrapped = textwrap.wrap(para, max(20, width), replace_whitespace=False, drop_whitespace=True) or [""]
        lines.extend(wrapped)
    return lines


def _section(title: str, subtitle: str = "") -> None:
    print()
    print(_c(title, Ui.bold, Ui.cyan))
    if subtitle:
        print(_c(subtitle, Ui.dim))


def _print_box(title: str, body: str = "", *, width: int | None = None) -> None:
    """Compatibility helper for older tests/call sites; rendered in the new compact style."""
    _section(title)
    if body:
        for line in _wrap(body, (width or _term_width()) - 4):
            print(f"  {line}")


def _status_line(cfg: RuntimeConfig) -> str:
    key = "key set" if cfg.api_key else "no key"
    mode = _mode_label(_api_mode_for(cfg.provider, cfg.model))
    width = _term_width() - 8
    model = _shorten_middle(cfg.model or "<no model>", max(12, width - len(cfg.provider) - len(key) - len(mode) - 9))
    return f"{cfg.provider} · {model} · {mode} · {key}"


def _banner(cfg: RuntimeConfig) -> None:
    width = _term_width()
    print()
    print(_c("Hermes Pocket", Ui.bold, Ui.cyan))
    print(_c("─" * min(width, 72), Ui.dim))
    print(f"  provider  {_c(cfg.provider, Ui.green)}")
    print(f"  model     {_shorten_middle(cfg.model or '<set model>', width - 14)}")
    print(f"  route     {_mode_label(_api_mode_for(cfg.provider, cfg.model))}")
    print(f"  key       {'set' if cfg.api_key else 'not set'}")
    print(f"  base      {_shorten_middle(cfg.base_url, width - 14)}")
    print(_c("  minimal   dashboard / vision / docs / voice disabled on ARM32", Ui.dim))
    print()
    print(_c("Type /help for commands. Use /models opencode-zen to browse the Hermes catalog.", Ui.dim))


def _prompt(cfg: RuntimeConfig) -> str:
    status = _shorten_middle(_status_line(cfg), _term_width() - 5)
    return _c(f"\n╭─ {status}\n╰─› ", Ui.cyan)


def _print_chat(role: str, text: str, *, model: str = "") -> None:
    label = "You" if role.lower() in {"you", "user"} else "Hermes"
    if model:
        label = f"{label} · {_shorten_middle(model, max(14, _term_width() - len(label) - 6))}"
    print()
    print(_c(label, Ui.bold if role.lower() in {"you", "user"} else Ui.green))
    for line in _wrap(text, _term_width() - 4):
        print(f"  {line}")


def _print_notice(text: str) -> None:
    print(_c(f"  {text}", Ui.dim))


def _print_error(text: str) -> None:
    print(_c(f"  error: {text}", Ui.red))


HELP_ROWS = (
    ("/help", "show this command palette"),
    ("/status", "show current provider, model, route, key state"),
    ("/providers", "list provider shortcuts"),
    ("/providers <name|number>", "switch provider; plural works like Gemini-style natural command"),
    ("/providers all", "show the full provider table"),
    ("/provider <name|number>", "switch provider preset"),
    ("/models [provider] [search]", "list live+curated models, e.g. /models opencode-zen kimi"),
    ("/models free", "filter free models (:free or -free)"),
    ("/models offline", "skip network and use Hermes curated catalog"),
    ("/model <id|number>", "switch model; number uses the last /models result"),
    ("/base-url <url>", "set OpenAI-compatible endpoint"),
    ("/key", "paste API key without echo"),
    ("/clear", "clear conversation memory"),
    ("/save [--key]", "save provider/base/model to ~/.hermes/.env; --key also saves key"),
    ("/export <file.md>", "export transcript"),
    ("/exit", "quit"),
)


def _tui_help() -> str:
    return "\n".join(f"{cmd:<28} {desc}" for cmd, desc in HELP_ROWS)


def _print_help() -> None:
    _section("Commands")
    width = _term_width()
    cmd_width = min(28, max(len(cmd) for cmd, _desc in HELP_ROWS) + 2)
    for command, desc in HELP_ROWS:
        prefix = f"  {_c(command.ljust(cmd_width), Ui.cyan)}"
        wrapped = _wrap(desc, width - cmd_width - 5)
        print(prefix + (wrapped[0] if wrapped else ""))
        for line in wrapped[1:]:
            print("  " + " " * cmd_width + line)


def _provider_key_state(provider_id: str, current_key: str = "") -> str:
    if current_key:
        return "set"
    preset = MINIMAL_PROVIDERS[provider_id]
    if _api_key_for_provider(provider_id):
        return "set"
    return "optional" if preset.allow_no_key else "missing"


def _print_provider_detail(provider_id: str, cfg: RuntimeConfig | None = None) -> None:
    preset = MINIMAL_PROVIDERS[provider_id]
    _section(f"Provider · {provider_id}", preset.label)
    current_key = cfg.api_key if cfg and cfg.provider == provider_id else ""
    rows = (
        ("base", preset.base_url),
        ("default", preset.default_model or "<set HERMES_MODEL>"),
        ("env", ", ".join(preset.env_vars) or "optional"),
        ("key", _provider_key_state(provider_id, current_key)),
        ("notes", preset.notes),
    )
    for key, value in rows:
        print(f"  {key.ljust(8)} {value}")
    print(_c("  Try: /models " + provider_id, Ui.dim))


def _print_providers(cfg: RuntimeConfig | None = None, *, show_all: bool = False) -> list[str]:
    provider_ids = list(MINIMAL_PROVIDERS)
    rows = provider_ids if show_all else provider_ids[:PROVIDER_LIST_DEFAULT_LIMIT]
    _section("Providers", "Use /provider <name|number>. Use /providers all for every preset.")
    print(_c("  #   provider          key       default model", Ui.dim))
    for index, provider_id in enumerate(rows, 1):
        preset = MINIMAL_PROVIDERS[provider_id]
        key_state = _provider_key_state(provider_id, cfg.api_key if cfg and cfg.provider == provider_id else "")
        marker = "*" if cfg and cfg.provider == provider_id else " "
        model = _shorten_middle(preset.default_model or "<set HERMES_MODEL>", 30)
        print(f"  {index:>2}{marker} {provider_id:<17} {key_state:<9} {model}")
    if not show_all and len(provider_ids) > len(rows):
        print(_c(f"  … {len(provider_ids) - len(rows)} more. /providers all", Ui.dim))
    return rows


def _print_models(
    models: list[str] | list[ModelEntry],
    current: str = "",
    *,
    provider: str = "",
    note: str = "",
    limit: int | None = None,
) -> None:
    entries: list[ModelEntry] = [m if isinstance(m, ModelEntry) else ModelEntry(str(m)) for m in models]
    if limit is not None and limit > 0:
        display = entries[:limit]
    else:
        display = entries
    title = "Models" + (f" · {provider}" if provider else "")
    _section(title, "Use /model <number|id>. Search with /models <text>; provider with /models <provider>.")
    if note:
        _print_notice(note)
    if not display:
        print("  No models found.")
        return
    print(_c("  #   model                                      route      source", Ui.dim))
    for idx, entry in enumerate(display, 1):
        mark = "*" if entry.id == current else " "
        model = _shorten_middle(entry.id, 42)
        print(f"  {idx:>2}{mark} {model:<42} {_mode_label(entry.api_mode):<10} {entry.source}")
    remaining = len(entries) - len(display)
    if remaining > 0:
        print(_c(f"  … {remaining} more. Use /models all or a search term.", Ui.dim))


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
        f"- route: `{_mode_label(_api_mode_for(cfg.provider, cfg.model))}`",
        f"- base_url: `{cfg.base_url}`",
        f"- exported_at: `{time.strftime('%Y-%m-%d %H:%M:%S')}`",
        "",
    ]
    for role, text in transcript:
        lines.extend([f"## {role}", "", text.rstrip(), ""])
    target.write_text("\n".join(lines), encoding="utf-8")
    return target


def _parse_model_terms(terms: list[str]) -> tuple[str, bool, bool, str]:
    provider = ""
    free_only = False
    offline = False
    query_parts: list[str] = []
    for index, raw in enumerate(terms):
        token = raw.strip()
        lowered = token.lower()
        if lowered in {"free", "--free"}:
            free_only = True
            continue
        if lowered in {"offline", "--offline", "curated", "--curated"}:
            offline = True
            continue
        if lowered in {"all", "--all"}:
            # all is handled by caller through limit=None; do not make it a query.
            query_parts.append("__all__")
            continue
        maybe_provider = ""
        if index == 0:
            try:
                maybe_provider = _provider_id(token)
            except SystemExit:
                maybe_provider = ""
        if maybe_provider and not provider:
            provider = maybe_provider
            continue
        query_parts.append(token)
    query = " ".join(part for part in query_parts if part != "__all__")
    return provider, free_only, offline, query


def _terms_want_all(terms: list[str]) -> bool:
    return any(term.lower() in {"all", "--all"} for term in terms)


def _switch_provider(cfg: RuntimeConfig, provider_name: str) -> RuntimeConfig:
    provider_id = _provider_id(provider_name)
    preset = MINIMAL_PROVIDERS[provider_id]
    api_key = _api_key_for_provider(provider_id)
    return RuntimeConfig(
        api_key=api_key,
        base_url=preset.base_url,
        model=preset.default_model or cfg.model,
        timeout=cfg.timeout,
        provider=provider_id,
    )


def _resolve_numbered_provider(selection: str, last_provider_ids: list[str]) -> str:
    if selection.isdigit() and last_provider_ids:
        idx = int(selection) - 1
        if idx < 0 or idx >= len(last_provider_ids):
            raise SystemExit(f"provider number out of range: {selection}")
        return last_provider_ids[idx]
    return _provider_id(selection)


def _namespace_for_cfg(args: argparse.Namespace, cfg: RuntimeConfig) -> argparse.Namespace:
    return argparse.Namespace(**{**vars(args), "provider": cfg.provider, "base_url": cfg.base_url, "model": cfg.model, "api_key": cfg.api_key})


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
            api_key="",
            base_url=base_url,
            model=getattr(args, "model", None) or os.environ.get("HERMES_MODEL", "") or _default_model(provider, base_url),
            timeout=float(args.timeout),
            provider=provider,
        )
    print("Hermes Termux ARM32 minimal doctor")
    print(f"  python: {sys.version.split()[0]} ({sys.platform}, {platform.machine()})")
    print(f"  android_arm32_detected: {_is_android_arm32()}")
    print(f"  provider: {cfg.provider}")
    print(f"  base_url: {cfg.base_url}")
    print(f"  model: {cfg.model}")
    print(f"  route: {_mode_label(_api_mode_for(cfg.provider, cfg.model))}")
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


def cmd_providers(args: argparse.Namespace) -> int:
    selection = getattr(args, "selection", None)
    if selection and selection.lower() not in {"all", "--all"}:
        _print_provider_detail(_provider_id(selection))
        return 0
    _print_providers(show_all=bool(getattr(args, "all", False) or (selection and selection.lower() in {"all", "--all"})))
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    cfg = _runtime_config(args, require_key=False)
    terms = list(getattr(args, "terms", []) or [])
    provider_override, free_from_terms, offline, query = _parse_model_terms(terms)
    if provider_override:
        cfg = _switch_provider(cfg, provider_override)
    free_only = bool(args.free or free_from_terms)
    if getattr(args, "query", ""):
        query = " ".join(part for part in (query, args.query) if part)
    entries, note = fetch_model_entries(cfg, free_only=free_only, query=query, offline=bool(args.offline or offline))
    limit = None if _terms_want_all(terms) else args.limit
    _print_models(entries, cfg.model, provider=cfg.provider, note=note, limit=limit)
    return 0


def _handle_models_command(
    cfg: RuntimeConfig,
    rest: list[str],
    *,
    last_model_provider: str,
) -> tuple[list[ModelEntry], str]:
    provider_override, free_only, offline, query = _parse_model_terms(rest)
    target_cfg = _switch_provider(cfg, provider_override) if provider_override else cfg
    entries, note = fetch_model_entries(target_cfg, free_only=free_only, query=query, offline=offline)
    limit = None if _terms_want_all(rest) else MODEL_LIST_DEFAULT_LIMIT
    _print_models(entries, target_cfg.model, provider=target_cfg.provider, note=note, limit=limit)
    return entries, target_cfg.provider or last_model_provider


def cmd_tui(args: argparse.Namespace) -> int:
    cfg = _runtime_config(args, require_key=False)
    messages: list[dict[str, str]] = []
    transcript: list[tuple[str, str]] = []
    last_models: list[ModelEntry] = []
    last_model_provider = cfg.provider
    last_provider_ids: list[str] = []
    _banner(cfg)
    while True:
        try:
            raw = input(_prompt(cfg)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            return 0
        if not raw:
            continue
        if raw.startswith("/") or raw == "?":
            command_line = "help" if raw == "?" else raw[1:]
            try:
                parts = shlex.split(command_line)
            except ValueError as exc:
                _print_error(f"command parse error: {exc}")
                continue
            if not parts:
                continue
            cmd, rest = parts[0].lower(), parts[1:]
            try:
                if cmd in {"exit", "quit", "q"}:
                    print("bye.")
                    return 0
                if cmd in {"help", "h"}:
                    _print_help()
                elif cmd == "status":
                    _banner(cfg)
                elif cmd == "doctor":
                    cmd_doctor(_namespace_for_cfg(args, cfg))
                elif cmd == "providers":
                    if rest and rest[0].lower() not in {"all", "--all"}:
                        provider_id = _resolve_numbered_provider(rest[0], last_provider_ids)
                        cfg = _switch_provider(cfg, provider_id)
                        last_model_provider = cfg.provider
                        last_models = []
                        _print_provider_detail(cfg.provider, cfg)
                        _print_notice(f"switched to {cfg.provider}. Browse models with /models {cfg.provider}")
                    else:
                        last_provider_ids = _print_providers(cfg, show_all=bool(rest and rest[0].lower() in {"all", "--all"}))
                elif cmd == "provider":
                    if not rest:
                        _print_provider_detail(cfg.provider, cfg)
                    else:
                        provider_id = _resolve_numbered_provider(rest[0], last_provider_ids)
                        cfg = _switch_provider(cfg, provider_id)
                        last_model_provider = cfg.provider
                        last_models = []
                        _print_provider_detail(cfg.provider, cfg)
                        _print_notice(f"switched to {cfg.provider}. Browse models with /models {cfg.provider}")
                elif cmd in {"base-url", "baseurl"}:
                    if not rest:
                        _print_notice(f"current base_url: {cfg.base_url}")
                    else:
                        base_url = _normalize_base_url(rest[0])
                        cfg = replace(cfg, base_url=base_url, provider=_provider_from_base(base_url))
                        last_models = []
                        _print_notice(f"base_url set: {cfg.base_url}")
                elif cmd == "key":
                    key = getpass.getpass("API key: ").strip()
                    cfg = replace(cfg, api_key=key)
                    _print_notice("api key set for this session")
                elif cmd == "models":
                    last_models, last_model_provider = _handle_models_command(cfg, rest, last_model_provider=last_model_provider)
                elif cmd == "model":
                    if not rest:
                        _print_notice(f"current model: {cfg.model}")
                    else:
                        chosen = rest[0]
                        provider_for_choice = cfg.provider
                        if chosen.isdigit():
                            if not last_models:
                                _print_error("run /models first, then choose a number; or paste the full model id")
                                continue
                            idx = int(chosen) - 1
                            if idx < 0 or idx >= len(last_models):
                                _print_error(f"model number out of range: {chosen}")
                                continue
                            chosen = last_models[idx].id
                            provider_for_choice = last_model_provider
                        if provider_for_choice != cfg.provider:
                            cfg = _switch_provider(cfg, provider_for_choice)
                        cfg = replace(cfg, model=chosen)
                        _print_notice(f"model set: {cfg.model} ({_mode_label(_api_mode_for(cfg.provider, cfg.model))})")
                elif cmd == "clear":
                    messages.clear()
                    transcript.clear()
                    _print_notice("conversation cleared")
                elif cmd == "save":
                    include_key = "--key" in rest
                    path = save_minimal_env(cfg, include_key=include_key)
                    _print_notice(f"saved {'provider/base/model/key' if include_key else 'provider/base/model'} to {path}")
                elif cmd == "export":
                    if not rest:
                        _print_notice("usage: /export <file.md>")
                    else:
                        path = _export_transcript(rest[0], transcript, cfg)
                        _print_notice(f"exported transcript: {path}")
                else:
                    _print_error(f"unknown command: /{cmd}. Try /help")
            except SystemExit as exc:
                _print_error(str(exc))
            continue

        _print_chat("you", raw)
        messages.append({"role": "user", "content": raw})
        transcript.append(("you", raw))
        started = time.monotonic()
        print(_c("  Thinking...", Ui.dim))
        try:
            answer = _chat_completion(messages, cfg, temperature=args.temperature, max_tokens=args.max_tokens)
        except SystemExit as exc:
            _print_error(str(exc))
            messages.pop()
            transcript.pop()
            continue
        elapsed = time.monotonic() - started
        messages.append({"role": "assistant", "content": answer})
        transcript.append(("hermes", answer))
        _print_chat("hermes", answer, model=cfg.model)
        _print_notice(f"{len(messages)//2} turn · {cfg.provider} · {_mode_label(_api_mode_for(cfg.provider, cfg.model))} · {elapsed:.1f}s")


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
    parser.add_argument("--provider", help="Provider preset (try: openrouter, openai, opencode-zen, deepseek, nvidia)")
    parser.add_argument("--api-key", help="API key (else provider env var or HERMES_API_KEY)")
    parser.add_argument("--base-url", help=f"OpenAI-compatible API base URL (default: {DEFAULT_OPENAI_BASE_URL})")
    parser.add_argument("--model", help="Model name (else HERMES_MODEL / provider default)")
    parser.add_argument("--timeout", default="60", help="HTTP timeout in seconds (default: 60)")
    parser.add_argument("--version", action="store_true", help="Show minimal runtime version")
    sub = parser.add_subparsers(dest="command")

    doctor = sub.add_parser("doctor", help="Show minimal runtime environment and missing config")
    _add_runtime_overrides(doctor)
    doctor.set_defaults(func=cmd_doctor)

    providers = sub.add_parser("providers", help="List or inspect minimal-safe provider presets")
    providers.add_argument("selection", nargs="?", help="Provider name/alias or 'all'")
    providers.add_argument("--all", action="store_true", help="Show all provider presets")
    providers.set_defaults(func=cmd_providers)

    models = sub.add_parser("models", help="List live+curated models for a provider")
    _add_runtime_overrides(models)
    models.add_argument("terms", nargs="*", help="Optional provider/search terms, e.g. opencode-zen kimi")
    models.add_argument("--free", action="store_true", help="Only show free model ids (:free or -free)")
    models.add_argument("--offline", action="store_true", help="Skip live /models and use Hermes curated catalog")
    models.add_argument("--query", default="", help="Search model ids")
    models.add_argument("--limit", type=int, default=80, help="Max models to print (default: 80; use 'all' for no limit)")
    models.set_defaults(func=cmd_models)

    chat = sub.add_parser("chat", help="Send one prompt to the selected provider")
    _add_runtime_overrides(chat)
    chat.add_argument("prompt", nargs="*", help="Prompt text. If omitted, stdin is read.")
    chat.add_argument("--system", default="", help="Optional system message")
    chat.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for chat-completions models")
    chat.add_argument("--max-tokens", type=int, default=None, help="Optional max token cap")
    chat.set_defaults(func=cmd_chat)

    tui = sub.add_parser("tui", aliases=["repl"], help="Interactive Hermes Pocket TUI")
    _add_runtime_overrides(tui)
    tui.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature for chat-completions models")
    tui.add_argument("--max-tokens", type=int, default=None, help="Optional max token cap")
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
