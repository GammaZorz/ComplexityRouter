"""Anthropic provider — async client with prompt caching."""

from __future__ import annotations

import time

import anthropic

from ..schemas import AgentResponse, ComplexityScore, Task
from .base import BaseProvider


class AnthropicProvider(BaseProvider):
    """Wraps ``anthropic.AsyncAnthropic`` with prompt-caching support."""

    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = client or anthropic.AsyncAnthropic()

    async def execute(
        self,
        task: Task,
        model_id: str,
        max_tokens: int,
        complexity: ComplexityScore | None = None,
    ) -> AgentResponse:
        # Build system prompt with prompt caching
        system_blocks: list[dict] = []
        if task.context:
            system_blocks.append(
                {
                    "type": "text",
                    "text": task.context,
                    "cache_control": {"type": "ephemeral"},
                }
            )

        start = time.perf_counter()
        response = await self._client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            system=system_blocks if system_blocks else [],
            messages=[{"role": "user", "content": task.prompt}],
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        output_text = response.content[0].text if response.content else ""

        return AgentResponse(
            output=output_text,
            model_used=model_id,
            provider="anthropic",
            complexity=complexity or ComplexityScore(0, "unknown", "", "none"),
            latency_ms=round(elapsed_ms, 1),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
