"""
OpenRouter provider — uses the OpenAI SDK pointed at openrouter.ai.

Supports:
  - Any model string (e.g. "openai/gpt-4o", "anthropic/claude-opus-4")
  - "openrouter/auto" — delegates model selection to OpenRouter's NotDiamond router

Requires: pip install openai
"""

from __future__ import annotations

import time

from .base import BaseProvider
from ..schemas import AgentResponse, ComplexityScore, Task

try:
    from openai import AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False


class OpenRouterProvider(BaseProvider):
    """Routes via openrouter.ai using the OpenAI-compatible API."""

    BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(self, api_key: str, base_url: str | None = None) -> None:
        if not _OPENAI_AVAILABLE:
            raise ImportError(
                "The 'openai' package is required for OpenRouterProvider. "
                "Install it with: pip install openai"
            )
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url or self.BASE_URL,
        )

    async def execute(
        self,
        task: Task,
        model_id: str,
        max_tokens: int,
        complexity: ComplexityScore | None = None,
        **_kw,
    ) -> AgentResponse:
        messages = []
        if task.context:
            messages.append({"role": "system", "content": task.context})
        messages.append({"role": "user", "content": task.prompt})

        start = time.perf_counter()
        response = await self._client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        output_text = choice.message.content or ""
        actual_model = response.model or model_id

        return AgentResponse(
            output=output_text,
            model_used=actual_model,
            provider="openrouter",
            complexity=complexity or ComplexityScore(0, "unknown", "", "none"),
            latency_ms=round(elapsed_ms, 1),
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )
