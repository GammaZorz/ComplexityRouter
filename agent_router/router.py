"""Orchestrator — evaluates complexity, selects model, executes task, logs result."""

from __future__ import annotations

import os

from .evaluator import ComplexityEvaluator
from .logger import RouterLogger
from .models import select_model
from .providers.base import BaseProvider
from .providers.anthropic import AnthropicProvider
from .schemas import AgentResponse, Task
from .settings import Settings, get_settings


def _build_default_providers(settings: Settings) -> dict[str, BaseProvider]:
    """Build provider instances from settings, skipping those with missing keys."""
    providers: dict[str, BaseProvider] = {}

    # Anthropic (always attempt)
    providers["anthropic"] = AnthropicProvider()

    # OpenRouter (only if api key is resolvable)
    or_key = settings.resolved_api_key("openrouter")
    if or_key:
        try:
            from .providers.openrouter import OpenRouterProvider
            or_cfg = settings.provider_config("openrouter")
            providers["openrouter"] = OpenRouterProvider(
                api_key=or_key,
                base_url=or_cfg.get("base_url"),
            )
        except ImportError:
            pass  # openai package not installed

    # LiteLLM proxy (no key needed — just a running proxy URL)
    ll_cfg = settings.provider_config("litellm")
    if ll_cfg.get("base_url"):
        try:
            from .providers.litellm import LiteLLMProvider
            providers["litellm"] = LiteLLMProvider(
                base_url=ll_cfg["base_url"],
                use_sdk=ll_cfg.get("mode") == "sdk",
            )
        except ImportError:
            pass

    return providers


class AgentRouter:
    """
    Main entry point for the routing workflow.

    Usage::

        router = AgentRouter()
        response = await router.run(Task(prompt="Explain photosynthesis"))
        print(response.model_used, response.output)
    """

    def __init__(
        self,
        providers: dict[str, BaseProvider] | None = None,
        evaluator: ComplexityEvaluator | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._evaluator = evaluator or ComplexityEvaluator(settings=self._settings)
        self._providers: dict[str, BaseProvider] = (
            providers if providers is not None
            else _build_default_providers(self._settings)
        )
        self._logger = RouterLogger(self._settings.logging)

    # ------------------------------------------------------------------

    async def run(self, task: Task) -> AgentResponse:
        """Evaluate complexity, select a model, execute, log, and return."""
        # 1. Complexity evaluation
        complexity = await self._evaluator.evaluate(task)

        # 2. Model selection
        selection = select_model(complexity, self._settings)

        # 3. Provider dispatch
        provider = self._providers.get(selection.provider)
        if provider is None:
            raise RuntimeError(
                f"No provider registered for '{selection.provider}'. "
                f"Available: {list(self._providers.keys())}"
            )

        # 4. Execute
        response = await provider.execute(
            task=task,
            model_id=selection.model_id,
            max_tokens=selection.max_tokens,
            complexity=complexity,
        )

        # 5. Log
        await self._logger.log(task, response)

        return response

    # ------------------------------------------------------------------

    def register_provider(self, name: str, provider: BaseProvider) -> None:
        """Register (or replace) a provider at runtime."""
        self._providers[name] = provider

    @property
    def settings(self) -> Settings:
        return self._settings
