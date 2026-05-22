"""Abstract base class for LLM provider adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..schemas import AgentResponse, Task


class BaseProvider(ABC):
    """
    Every provider must implement ``execute``.

    To add a new provider (OpenAI, Ollama, etc.):
      1. Subclass BaseProvider.
      2. Implement ``execute``.
      3. Register the provider name in ``agent_router/router.py``.
    """

    @abstractmethod
    async def execute(
        self,
        task: Task,
        model_id: str,
        max_tokens: int,
    ) -> AgentResponse:
        """Send the task to the model and return a structured response."""
        ...
