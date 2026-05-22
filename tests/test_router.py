"""Unit tests for the router — uses a mock provider, no real API calls."""

from __future__ import annotations

import asyncio
import unittest

from agent_router import AgentRouter, Task
from agent_router.evaluator import ComplexityEvaluator
from agent_router.schemas import AgentResponse, ComplexityScore
from agent_router.providers.base import BaseProvider


class MockProvider(BaseProvider):
    """Returns a canned response without making any network call."""

    async def execute(self, task, model_id, max_tokens, **kwargs):
        complexity = kwargs.get("complexity", ComplexityScore(0, "unknown", "", "mock"))
        return AgentResponse(
            output=f"Mock response for: {task.prompt[:50]}",
            model_used=model_id,
            provider="mock",
            complexity=complexity,
            latency_ms=1.0,
            input_tokens=len(task.prompt) // 4,
            output_tokens=20,
        )


class SafeEvaluator(ComplexityEvaluator):
    """Evaluator whose LLM fallback never makes a real API call."""

    async def _score_by_llm(self, task):
        # Return a medium score — safe default for ambiguous prompts
        return 50, "Mock LLM classification (no API call)"


class TestRouterWithMockProvider(unittest.TestCase):
    """End-to-end routing with a mock provider — validates the full pipeline."""

    def setUp(self) -> None:
        # Override both provider and evaluator so zero network calls happen
        self.router = AgentRouter(
            providers={"anthropic": MockProvider()},
            evaluator=SafeEvaluator(),
        )

    def test_simple_routes_to_haiku(self) -> None:
        task = Task(prompt="What is the speed of light?")
        response = asyncio.run(self.router.run(task))
        self.assertIn("haiku", response.model_used.lower())

    def test_complex_routes_to_opus(self) -> None:
        task = Task(
            prompt=(
                "Design a microservice architecture with CI/CD pipeline.\n"
                "1. Service design\n2. Database schema\n3. API gateway\n"
                "4. Monitoring\n5. Deploy strategy step by step."
            ),
            domain="code",
        )
        response = asyncio.run(self.router.run(task))
        self.assertIn("opus", response.model_used.lower())

    def test_speed_priority_caps_at_sonnet(self) -> None:
        task = Task(
            prompt=(
                "Design a microservice architecture with CI/CD pipeline.\n"
                "1. Service design\n2. Database schema\n3. API gateway\n"
                "4. Monitoring\n5. Deploy strategy step by step."
            ),
            domain="code",
            priority="speed",
        )
        response = asyncio.run(self.router.run(task))
        self.assertNotIn("opus", response.model_used.lower())

    def test_quality_priority_avoids_haiku(self) -> None:
        task = Task(prompt="What is gravity?", priority="quality")
        response = asyncio.run(self.router.run(task))
        self.assertNotIn("haiku", response.model_used.lower())

    def test_response_metadata(self) -> None:
        task = Task(prompt="Define entropy.")
        response = asyncio.run(self.router.run(task))
        self.assertIsInstance(response.latency_ms, float)
        self.assertGreater(response.input_tokens, 0)
        self.assertIn(response.complexity.tier, ("simple", "medium", "complex"))
        self.assertIn(response.complexity.method, ("rules", "llm", "hybrid"))

    def test_register_custom_provider(self) -> None:
        mock2 = MockProvider()
        self.router.register_provider("custom", mock2)
        self.assertIn("custom", self.router._providers)


class TestTaskValidation(unittest.TestCase):
    """Input validation on the Task schema."""

    def test_invalid_domain_raises(self) -> None:
        with self.assertRaises(ValueError):
            Task(prompt="test", domain="invalid")

    def test_invalid_priority_raises(self) -> None:
        with self.assertRaises(ValueError):
            Task(prompt="test", priority="urgent")


if __name__ == "__main__":
    unittest.main()
