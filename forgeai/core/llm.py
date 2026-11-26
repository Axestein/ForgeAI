"""
LLM client abstraction - supports OpenAI, Google Gemini, and Anthropic.
Per-provider models, rate-limit throttling, retries that obey the server's
retry hint, model + provider fallback chain, and token tracking for budgets.
"""
from __future__ import annotations

import os
import re
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "200000"))
REQUEST_TIMEOUT = int(os.getenv("LLM_TIMEOUT_S", "120"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "4"))   # retries per model
MAX_RPM = int(os.getenv("LLM_MAX_RPM", "4"))           # free tier is 5/min; stay under
MAX_WAIT_S = int(os.getenv("LLM_MAX_WAIT_S", "90"))    # longer server waits -> skip model

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-3.5-flash",
    "anthropic": "claude-3-haiku-20240307",
}
_KEY_VARS = {
    "openai": "OPENAI_API_KEY",
    "google": "GOOGLE_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
_PLACEHOLDERS = {"", "sample", "your_key_here", "changeme"}


@dataclass
class LLMResponse:
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_s: float = 0.0


@dataclass
class UsageTracker:
    total_input: int = 0
    total_output: int = 0
    total_latency: float = 0.0
    calls: int = 0
    _budget: int = field(default=TOKEN_BUDGET, repr=False)

    @property
    def total_tokens(self) -> int:
        return self.total_input + self.total_output

    def record(self, resp: LLMResponse) -> None:
        self.total_input += resp.input_tokens
        self.total_output += resp.output_tokens
        self.total_latency += resp.latency_s
        self.calls += 1

    def budget_remaining(self) -> int:
        return max(0, self._budget - self.total_tokens)

    def over_budget(self) -> bool:
        return self.total_tokens >= self._budget

    def summary(self) -> dict:
        return {
            "total_tokens": self.total_tokens,
            "input_tokens": self.total_input,
            "output_tokens": self.total_output,
            "calls": self.calls,
            "total_latency_s": round(self.total_latency, 2),
            "budget_remaining": self.budget_remaining(),
        }


usage = UsageTracker()


# ------------------------------------------------------------ throttling
_last_call = 0.0
_lock = threading.Lock()


def _throttle() -> None:
    """Space calls out so we never exceed MAX_RPM."""
    global _last_call
    if MAX_RPM <= 0:
        return
    gap = 60.0 / MAX_RPM
    with _lock:
        wait = _last_call + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _last_call = time.monotonic()


def _retry_delay(err: Exception, default: float) -> float:
    """Use the server's 'retry in Ns' hint on 429s, else exponential backoff."""
    s = str(err)
    m = re.search(r"retry in ([\d.]+)s", s, re.I) or re.search(r"seconds:\s*(\d+)", s)
    return float(m.group(1)) + 2 if m else default


# ---------------------------------------------------------------- config
def _has_key(provider: str) -> bool:
    val = os.getenv(_KEY_VARS.get(provider, ""), "").strip()
    return val.lower() not in _PLACEHOLDERS


def _models_for(provider: str) -> List[str]:
    """Ordered, de-duplicated model list for one provider (each reads its own env var)."""
    p = provider.upper()
    models = []
    if provider == PROVIDER and os.getenv("LLM_MODEL", "").strip():
        models.append(os.getenv("LLM_MODEL").strip())  # legacy override, primary only
    models.append(os.getenv(f"{p}_MODEL", "").strip() or _DEFAULT_MODELS[provider])
    fb = os.getenv(f"{p}_FALLBACK_MODEL", "").strip()
    if fb:
        models.append(fb)
    seen, out = set(), []
    for m in models:
        if m and m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _build_chain() -> List[Tuple[str, str]]:
    """(provider, model) attempts: primary provider's models first, then other
    providers that actually have a real key configured."""
    chain: List[Tuple[str, str]] = []
    order = [PROVIDER] + [p for p in ("google", "openai", "anthropic") if p != PROVIDER]
    for prov in order:
        if prov not in _DEFAULT_MODELS:
            continue
        if prov != PROVIDER and not _has_key(prov):
            continue
        for m in _models_for(prov):
            chain.append((prov, m))
    return chain


def _is_transient(err: Exception) -> bool:
    s = str(err).lower()
    return any(t in s for t in (
        "504", "503", "502", "429", "500", "deadline", "timeout", "timed out",
        "unavailable", "overloaded", "rate limit", "resource exhausted",
        "quota", "connection",
    ))


# ------------------------------------------------------------------ main
def chat(
    messages: List[dict],
    system: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> LLMResponse:
    """Send a chat request. Throttles, retries transient errors, then walks the fallback chain."""
    t0 = time.perf_counter()
    chain = _build_chain()
    if not chain:
        raise RuntimeError("No LLM provider configured. Check LLM_PROVIDER and keys in .env")

    last_err: Optional[Exception] = None
    for idx, (provider, model) in enumerate(chain):
        for attempt in range(MAX_RETRIES + 1):
            try:
                _throttle()
                if provider == "openai":
                    resp = _openai_chat(model, messages, system, temperature, max_tokens)
                elif provider == "google":
                    resp = _google_chat(model, messages, system, temperature, max_tokens)
                else:
                    resp = _anthropic_chat(model, messages, system, temperature, max_tokens)

                if idx > 0:
                    print(f"[LLM] Fallback to {provider}/{model} succeeded.")
                resp.latency_s = time.perf_counter() - t0
                usage.record(resp)
                return resp

            except Exception as e:  # noqa: BLE001
                last_err = e
                if _is_transient(e) and attempt < MAX_RETRIES:
                    wait = _retry_delay(e, 2 ** (attempt + 1))
                    if wait > MAX_WAIT_S:
                        # Server wants a very long wait (e.g. daily quota): try next model.
                        print(f"[LLM] {provider}/{model} asks for {wait:.0f}s wait; "
                              f"skipping to next model.")
                        break
                    print(f"[LLM] {provider}/{model} transient error ({str(e)[:80]}). "
                          f"Retry {attempt + 1}/{MAX_RETRIES} in {wait:.0f}s...")
                    time.sleep(wait)
                    continue
                print(f"[LLM] {provider}/{model} failed: {str(e)[:200]}")
                break  # move to next model/provider

    raise RuntimeError(
        f"All LLM providers/models failed. Last error: {last_err}\n"
        "Check API keys, model names and quota in .env"
    ) from last_err


# ------------------------------------------------------------- providers
def _openai_chat(model, messages, system, temperature, max_tokens) -> LLMResponse:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=REQUEST_TIMEOUT)
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    r = client.chat.completions.create(
        model=model,
        messages=all_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return LLMResponse(
        text=r.choices[0].message.content or "",
        input_tokens=r.usage.prompt_tokens,
        output_tokens=r.usage.completion_tokens,
    )


def _google_chat(model, messages, system, temperature, max_tokens) -> LLMResponse:
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    gmodel = genai.GenerativeModel(model, system_instruction=system or None)

    gemini_msgs = [
        {"role": "user" if m["role"] == "user" else "model", "parts": [m["content"]]}
        for m in messages
    ]

    # Thinking models count their reasoning tokens against max_output_tokens,
    # so a small cap can leave an empty answer. Give them headroom.
    gen_config = genai.GenerationConfig(
        temperature=temperature,
        max_output_tokens=max(max_tokens, 8192),
    )
    r = gmodel.generate_content(
        gemini_msgs,
        generation_config=gen_config,
        request_options={"timeout": REQUEST_TIMEOUT},
    )

    try:
        text = r.text
    except ValueError:  # blocked or empty/truncated response
        parts = []
        for c in getattr(r, "candidates", []) or []:
            for p in getattr(c.content, "parts", []) or []:
                parts.append(getattr(p, "text", "") or "")
        text = "".join(parts)
    if not text.strip():
        raise RuntimeError("Google returned an empty response (timeout/truncation/safety block)")

    um = getattr(r, "usage_metadata", None)
    in_tok = getattr(um, "prompt_token_count", 0) or 0
    out_tok = (getattr(um, "candidates_token_count", 0) or 0) + (
        getattr(um, "thoughts_token_count", 0) or 0
    )
    return LLMResponse(text=text, input_tokens=in_tok, output_tokens=out_tok)


def _anthropic_chat(model, messages, system, temperature, max_tokens) -> LLMResponse:
    import anthropic as ant  # type: ignore

    client = ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"), timeout=REQUEST_TIMEOUT)
    r = client.messages.create(
        model=model,
        system=system or "",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return LLMResponse(
        text=r.content[0].text if r.content else "",
        input_tokens=r.usage.input_tokens,
        output_tokens=r.usage.output_tokens,
    )