"""Orchestrator — evaluates complexity, selects model, executes task."""

from __future__ import annotations

from .evaluator import ComplexityEvaluator
from .models import MODEL_REGISTRY, select_model
from .providers.base import BaseProvider
from .providers.anthropic import AnthropicProvider
from .schemas import AgentResponse, Task


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
    ) -> None:
        self._evaluator = evaluator or ComplexityEvaluator()
        self._providers: dict[str, BaseProvider] = providers or {
            "anthropic": AnthropicProvider(),
        }

    # -- Public API --------------------------------------------------------

    async def run(self, task: Task) -> AgentResponse:
        """Evaluate complexity, select a model, and execute the task."""
        # 1. Complexity evaluation
        complexity = await self._evaluator.evaluate(task)

        # 2. Model selection
        selection = select_model(complexity)

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
        return response

    # -- Helpers -----------------------------------------------------------

    def register_provider(self, name: str, provider: BaseProvider) -> None:
        """Register (or replace) a provider at runtime."""
        self._providers[name] = provider

    @property
    def registry(self) -> dict:
        """Read-only view of the current model registry."""
        return dict(MODEL_REGISTRY)
