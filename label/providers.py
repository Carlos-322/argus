"""Explicit opt-in to DeepSeek direct API; original key routing stays the default.

Set ARGUS_PROVIDER=deepseek before starting Python (including auxiliary imports).
Credentials never belong in annotation policies.
"""
from __future__ import annotations

import json
import math
import os

DEEPSEEK_MODEL = "deepseek/deepseek-flash"
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
# USD per token, peak list prices, checked 2026-10-16. Always use peak rates:
# https://api-docs.deepseek.com/quick_start/pricing/
# Conservative list-price estimates, not the account's billed charges.
DEEPSEEK_PRICES = (0.30 / 1e6, 0.006 / 1e6, 1.20 / 1e6)
DEEPSEEK_PRICE_BASIS = "deepseek-flash peak USD list prices, 2026-10-16; estimate, not invoice"


def deepseek_enabled() -> bool:
    provider = os.environ.get("ARGUS_PROVIDER", "").strip().lower()
    if provider not in ("", "auto", "deepseek"):
        raise ValueError("ARGUS_PROVIDER must be auto or deepseek")
    return provider == "deepseek"


def default_model(original: str) -> str:
    return DEEPSEEK_MODEL if deepseek_enabled() else original


def is_deepseek_model(model: str) -> bool:
    return model in (DEEPSEEK_MODEL, "deepseek-flash")


def validate_model(model: str) -> None:
    if deepseek_enabled() and not is_deepseek_model(model):
        raise ValueError("ARGUS_PROVIDER=deepseek requires --model deepseek/deepseek-flash; "
                         "use a new --out directory when switching models")


def validate_key(key: str) -> None:
    # Do not guess from sk-: multiple providers use that prefix.
    allowed = [k.strip() for k in os.environ.get("DEEPSEEK_API_KEY", "").split(",") if k.strip()]
    if not key or key.startswith("sk-or-") or key not in allowed:
        raise ValueError("DeepSeek direct calls require a key from DEEPSEEK_API_KEY")


def request_body(content: list, model: str, reasoning: str, max_tokens: int) -> dict:
    if not is_deepseek_model(model):
        raise ValueError("Only deepseek-flash is supported by this vision adapter")
    efforts = {"none": "none", "minimal": "low", "low": "low", "medium": "high",
               "high": "high", "xhigh": "high", "max": "max"}
    if reasoning not in efforts:
        raise ValueError("Unsupported DeepSeek reasoning effort")
    if type(max_tokens) is not int or not 1 <= max_tokens <= 384000:
        raise ValueError("DeepSeek max_tokens must be between 1 and 384000")
    effort = efforts[reasoning]
    body = {"model": "deepseek-flash", "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "thinking": {"type": "disabled" if effort == "none" else "enabled"},
            "reasoning_effort": effort, "response_format": {"type": "json_object"}}
    if sum(p.get("type") == "image_url" for p in content) > 600:
        raise ValueError("DeepSeek allows at most 600 images per request")
    # Match the encoding used by harness._post. Reject before dispatch/billing.
    if len(json.dumps(body).encode()) > 48 * 1024 * 1024:
        raise ValueError("DeepSeek request exceeds 48 MiB; prepare a shorter recording/window")
    return body


def normalize_usage(usage: dict) -> None:
    """Preserve raw usage and add fields understood by existing Argus accounting."""
    n_in, n_out = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if any(type(n) is not int or n < 0 for n in (n_in, n_out)):
        return
    details = usage.get("prompt_tokens_details")
    details = dict(details) if isinstance(details, dict) else {}
    hit = usage.get("prompt_cache_hit_tokens", details.get("cached_tokens", 0))
    if type(hit) is not int or not 0 <= hit <= n_in:
        return
    details["cached_tokens"] = hit
    usage["prompt_tokens_details"] = details
    p_in, p_hit, p_out = DEEPSEEK_PRICES
    try:
        cost = (n_in - hit) * p_in + hit * p_hit + n_out * p_out
    except OverflowError:
        return
    if math.isfinite(cost):
        usage["list_cost"] = cost
        usage["price_basis"] = DEEPSEEK_PRICE_BASIS
