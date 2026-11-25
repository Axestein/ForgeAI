"""
LLM client abstraction – supports OpenAI, Google Gemini, and Anthropic.
Tracks approximate token usage for budget enforcement.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import List, Optional

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.getenv("LLM_PROVIDER", "openai").lower()
MODEL_OVERRIDE = os.getenv("LLM_MODEL", "").strip()
TOKEN_BUDGET = int(os.getenv("TOKEN_BUDGET", "200000"))

_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "google": "gemini-1.5-flash",
    "anthropic": "claude-3-haiku-20240307",
}


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


# Singleton tracker shared across the pipeline run
usage = UsageTracker()


def _model_name() -> str:
    return MODEL_OVERRIDE or _DEFAULT_MODELS.get(PROVIDER, "gpt-4o-mini")


def chat(
    messages: List[dict],
    system: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> LLMResponse:
    """Send a chat request to the configured provider, with auto-fallback."""
    t0 = time.perf_counter()

    primary = PROVIDER
    fallback = "openai" if primary != "openai" else None

    for attempt, provider in enumerate([primary, fallback]):
        if provider is None:
            break
        try:
            if provider == "openai":
                resp = _openai_chat(messages, system, temperature, max_tokens)
            elif provider == "google":
                resp = _google_chat(messages, system, temperature, max_tokens)
            elif provider == "anthropic":
                resp = _anthropic_chat(messages, system, temperature, max_tokens)
            else:
                raise ValueError(f"Unknown LLM_PROVIDER: {provider}")

            if attempt > 0:
                print(f"[LLM] Fallback to '{provider}' succeeded.")

            resp.latency_s = time.perf_counter() - t0
            usage.record(resp)
            return resp

        except Exception as e:
            if attempt == 0 and fallback:
                print(f"[LLM] Provider '{provider}' failed: {e}")
                print(f"[LLM] Trying fallback provider: '{fallback}'...")
            else:
                raise RuntimeError(
                    f"All LLM providers failed. Last error ({provider}): {e}\n"
                    "Check your API keys in .env"
                ) from e

    raise RuntimeError("No LLM provider configured correctly.")


def _openai_chat(messages, system, temperature, max_tokens) -> LLMResponse:
    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    r = client.chat.completions.create(
        model=_model_name(),
        messages=all_messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return LLMResponse(
        text=r.choices[0].message.content or "",
        input_tokens=r.usage.prompt_tokens,
        output_tokens=r.usage.completion_tokens,
    )


def _google_chat(messages, system, temperature, max_tokens) -> LLMResponse:
    import google.generativeai as genai  # type: ignore

    genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
    model = genai.GenerativeModel(
        _model_name(),
        system_instruction=system or "",
    )
    # Convert messages to Gemini format
    gemini_msgs = []
    for m in messages:
        role = "user" if m["role"] == "user" else "model"
        gemini_msgs.append({"role": role, "parts": [m["content"]]})

    gen_config = genai.GenerationConfig(
        temperature=temperature, max_output_tokens=max_tokens
    )
    r = model.generate_content(gemini_msgs, generation_config=gen_config)
    text = r.text if hasattr(r, "text") else ""
    in_tok = getattr(r.usage_metadata, "prompt_token_count", 0) or 0
    out_tok = getattr(r.usage_metadata, "candidates_token_count", 0) or 0
    return LLMResponse(text=text, input_tokens=in_tok, output_tokens=out_tok)


def _anthropic_chat(messages, system, temperature, max_tokens) -> LLMResponse:
    import anthropic as ant  # type: ignore

    client = ant.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    r = client.messages.create(
        model=_model_name(),
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
