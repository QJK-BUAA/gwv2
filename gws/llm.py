"""Thin OpenAI-compatible LLM wrapper with call/token accounting.

GraphWalker-SQL 2.0 is training-free: the LLM is used only for lightweight,
white-box decisions (joinability discovery, source/destination anchoring, path
selection, SQL generation). Any OpenAI-compatible endpoint works; this project
defaults to DeepSeek ``deepseek-chat`` at temperature 0 for determinism.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from . import config
from .response_cache import RequestEnvelope, ResponseCache


@dataclass(frozen=True)
class CompletionResult:
    text: str
    request_hash: str
    response_id: str
    usage: dict[str, Any]
    cached: bool


class LLM:
    def __init__(
        self,
        model: str = config.DEFAULT_MODEL,
        base_url: str = config.DEFAULT_BASE_URL,
        api_key_env: str = "DS_API_KEY",
        temperature: float = config.DEFAULT_TEMPERATURE,
        max_retries: int = 4,
        timeout: float = 90.0,
        seed: int | None = config.DEFAULT_SEED,
        response_cache: ResponseCache | None = None,
    ):
        api_key = (
            os.environ.get(api_key_env)
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "No API key found. Set $DS_API_KEY / $DEEPSEEK_API_KEY "
                "(or $OPENAI_API_KEY for an OpenAI endpoint)."
            )
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_retries = int(os.environ.get("GWS2_LLM_MAX_RETRIES",
                                              str(max_retries)))
        self.retry_base = float(os.environ.get("GWS2_LLM_RETRY_BASE", "1.5"))
        self.retry_cap = float(os.environ.get("GWS2_LLM_RETRY_CAP", "60"))
        self.seed = seed
        self.response_cache = response_cache
        # Zhipu GLM-5.x defaults to deep reasoning (reasoning_effort=max), which
        # is ~10x slower/costlier per call. For this white-box, latency-bounded
        # pipeline we disable it so GLM is comparable to the non-reasoning
        # deepseek-chat baseline. Toggle with $GWS2_GLM_THINKING=enabled.
        self.extra_body: dict = {}
        if "glm" in model.lower():
            mode = os.environ.get("GWS2_GLM_THINKING", "disabled").strip().lower()
            self.extra_body["thinking"] = {"type": mode}
        # DeepSeek V4 models expose an explicit reasoning toggle. This white-box,
        # latency-bounded pipeline turns reasoning OFF by default so V4 Flash is
        # comparable to the non-reasoning deepseek-chat baseline and stays cost/
        # latency-bounded. Toggle with $GWS2_DS_THINKING=enabled.
        elif "deepseek-v4" in model.lower():
            mode = os.environ.get("GWS2_DS_THINKING", "disabled").strip().lower()
            self.extra_body["thinking"] = {"type": mode}
        self.num_calls = 0
        self.in_tokens = 0
        self.out_tokens = 0
        self._lock = threading.Lock()

    def complete(self, system: str, user: str, temperature: float | None = None) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return self.complete_messages(messages, temperature=temperature)

    def complete_messages(self, messages: list[dict],
                          temperature: float | None = None) -> str:
        """Complete an explicit multi-turn conversation.

        AutoLink-style iterative SQL revision needs to preserve prior assistant
        SQL and execution feedback across turns; the legacy ``complete`` helper
        remains a two-message convenience wrapper.
        """
        return self.complete_messages_with_metadata(
            messages,
            temperature=temperature,
        ).text

    def complete_messages_with_metadata(
        self,
        messages: list[dict],
        temperature: float | None = None,
        seed: int | None = None,
    ) -> CompletionResult:
        request = RequestEnvelope(
            model=self.model,
            messages=messages,
            temperature=(
                self.temperature
                if temperature is None
                else temperature
            ),
            seed=self.seed if seed is None else seed,
            extra_body=self.extra_body,
        )
        cached = (
            self.response_cache.get(request)
            if self.response_cache is not None
            else None
        )
        if cached is not None:
            return CompletionResult(
                text=cached.text,
                request_hash=cached.request_hash,
                response_id=cached.response_id,
                usage=dict(cached.usage),
                cached=True,
            )

        result = self._call_provider(request)
        if self.response_cache is not None:
            self.response_cache.put(
                request,
                result.text,
                result.response_id,
                result.usage,
            )
        return result

    def _call_provider(
        self,
        request: RequestEnvelope,
    ) -> CompletionResult:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                kwargs = dict(
                    model=request.model,
                    messages=request.messages,
                    temperature=request.temperature,
                )
                if request.seed is not None:
                    kwargs["seed"] = request.seed
                if request.extra_body:
                    kwargs["extra_body"] = request.extra_body
                resp = self.client.chat.completions.create(**kwargs)
                usage = _usage_dict(resp.usage)
                with self._lock:
                    self.num_calls += 1
                    self.in_tokens += int(
                        usage.get("prompt_tokens") or 0
                    )
                    self.out_tokens += int(
                        usage.get("completion_tokens") or 0
                    )
                return CompletionResult(
                    text=(
                        resp.choices[0].message.content or ""
                    ).strip(),
                    request_hash=request.request_hash,
                    response_id=str(getattr(resp, "id", "") or ""),
                    usage=usage,
                    cached=False,
                )
            except Exception as e:  # noqa: BLE001
                last_err = e
                text = str(e).lower()
                # Authentication/billing failures are definitive, not transient.
                if any(x in text for x in (
                        "insufficient balance", "status code: 401",
                        "error code: 401", "error code: 402")):
                    break
                if attempt + 1 < self.max_retries:
                    delay = min(self.retry_cap,
                                self.retry_base * (2 ** attempt))
                    time.sleep(delay)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def stats(self) -> dict:
        return {
            "num_calls": self.num_calls,
            "input_tokens": self.in_tokens,
            "output_tokens": self.out_tokens,
        }


def _usage_dict(usage: Any) -> dict[str, int]:
    if usage is None:
        return {}
    return {
        key: int(value or 0)
        for key, value in (
            ("prompt_tokens", getattr(usage, "prompt_tokens", 0)),
            (
                "completion_tokens",
                getattr(usage, "completion_tokens", 0),
            ),
            ("total_tokens", getattr(usage, "total_tokens", 0)),
        )
    }
