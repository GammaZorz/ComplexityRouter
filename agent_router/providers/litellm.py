"""
LiteLLM provider — two modes:

  proxy  (default) — HTTP calls to a running LiteLLM proxy server.
                     No litellm package required; uses httpx or openai SDK.
  sdk             — Uses the litellm Python package directly.
                     Supports 100+ providers without a proxy.

Proxy mode:
  Start a LiteLLM proxy: litellm --model claude-sonnet-4-6 --port 4000
  Config: {"providers": {"litellm": {"base_url": "http://localhost:4000"}}}

SDK mode:
  pip install litellm
  Config: {"providers": {"litellm": {"mode": "sdk"}}}
"""

from __future__ import annotations

import time

from .base import BaseProvider
from ..schemas import AgentResponse, ComplexityScore, Task


class LiteLLMProvider(BaseProvider):
    """
    Routes via LiteLLM — either a running proxy or the litellm SDK.

    Parameters
    ----------
    base_url :
        URL of the LiteLLM proxy (proxy mode). Default: http://localhost:4000
    use_sdk :
        If True, use ``import litellm`` instead of the proxy. Requires
        ``pip install litellm`` and provider API keys in the environment.
    """

    def __init__(
        self,
        base_url: str = "http://localhost:4000",
        use_sdk: bool = False,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._use_sdk = use_sdk

    # ------------------------------------------------------------------
    # Proxy mode (no litellm package needed)
    # ------------------------------------------------------------------

    async def _execute_proxy(
        self,
        task: Task,
        model_id: str,
        max_tokens: int,
        complexity: ComplexityScore | None,
    ) -> AgentResponse:
        try:
            from openai import AsyncOpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for LiteLLMProvider proxy mode. "
                "Install it with: pip install openai"
            )

        client = AsyncOpenAI(api_key="litellm", base_url=f"{self._base_url}/v1")
        messages = []
        if task.context:
            messages.append({"role": "system", "content": task.context})
        messages.append({"role": "user", "content": task.prompt})

        start = time.perf_counter()
        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        return AgentResponse(
            output=choice.message.content or "",
            model_used=response.model or model_id,
            provider="litellm",
            complexity=complexity or ComplexityScore(0, "unknown", "", "none"),
            latency_ms=round(elapsed_ms, 1),
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    # ------------------------------------------------------------------
    # SDK mode
    # ------------------------------------------------------------------

    async def _execute_sdk(
        self,
        task: Task,
        model_id: str,
        max_tokens: int,
        complexity: ComplexityScore | None,
    ) -> AgentResponse:
        try:
            import litellm
        except ImportError:
            raise ImportError(
                "The 'litellm' package is required for LiteLLMProvider SDK mode. "
                "Install it with: pip install litellm"
            )

        messages = []
        if task.context:
            messages.append({"role": "system", "content": task.context})
        messages.append({"role": "user", "content": task.prompt})

        start = time.perf_counter()
        response = await litellm.acompletion(
            model=model_id,
            messages=messages,
            max_tokens=max_tokens,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000

        choice = response.choices[0]
        return AgentResponse(
            output=choice.message.content or "",
            model_used=response.model or model_id,
            provider="litellm",
            complexity=complexity or ComplexityScore(0, "unknown", "", "none"),
            latency_ms=round(elapsed_ms, 1),
            input_tokens=response.usage.prompt_tokens if response.usage else 0,
            output_tokens=response.usage.completion_tokens if response.usage else 0,
        )

    # ------------------------------------------------------------------

    async def execute(
        self,
        task: Task,
        model_id: str,
        max_tokens: int,
        complexity: ComplexityScore | None = None,
        **_kw,
    ) -> AgentResponse:
        if self._use_sdk:
            return await self._execute_sdk(task, model_id, max_tokens, complexity)
        return await self._execute_proxy(task, model_id, max_tokens, complexity)
