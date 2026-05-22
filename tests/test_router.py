"""Unit tests for the router — mock provider and evaluator, no API calls."""

from __future__ import annotations

import asyncio
import unittest

from agent_router import AgentRouter, Task, EvaluationMode
from agent_router.evaluator import ComplexityEvaluator
from agent_router.schemas import AgentResponse, ComplexityScore
from agent_router.providers.base import BaseProvider
from agent_router.settings import load_settings


def _settings():
    return load_settings()


class MockProvider(BaseProvider):
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
    async def _score_by_llm(self, task):
        return 50, "Mock LLM classification (no API call)"

    async def _score_by_caller_llm(self, task):
        return None


def _router():
    s = _settings()
    return AgentRouter(
        providers={"anthropic": MockProvider()},
        evaluator=SafeEvaluator(settings=s),
        settings=s,
    )


class TestRouterRouting(unittest.TestCase):

    def test_simple_routes_to_haiku(self):
        response = asyncio.run(_router().run(Task(prompt="What is the speed of light?")))
        self.assertIn("haiku", response.model_used.lower())

    def test_complex_routes_to_opus(self):
        task = Task(
            prompt=(
                "Design a microservice architecture with CI/CD pipeline.\n"
                "1. Service design\n2. Database schema\n3. API gateway\n"
                "4. Monitoring\n5. Deploy strategy step by step."
            ),
            domain="code",
        )
        response = asyncio.run(_router().run(task))
        self.assertIn("opus", response.model_used.lower())

    def test_speed_priority_caps_at_sonnet(self):
        task = Task(
            prompt=(
                "Design a microservice architecture with CI/CD pipeline.\n"
                "1. Service design\n2. Database schema\n3. API gateway\n"
                "4. Monitoring\n5. Deploy strategy step by step."
            ),
            domain="code",
            priority="speed",
        )
        response = asyncio.run(_router().run(task))
        self.assertNotIn("opus", response.model_used.lower())

    def test_quality_priority_avoids_haiku(self):
        response = asyncio.run(_router().run(Task(prompt="What is gravity?", priority="quality")))
        self.assertNotIn("haiku", response.model_used.lower())

    def test_response_metadata(self):
        response = asyncio.run(_router().run(Task(prompt="Define entropy.")))
        self.assertIsInstance(response.latency_ms, float)
        self.assertGreater(response.input_tokens, 0)
        self.assertIn(response.complexity.tier, ("simple", "medium", "complex"))
        self.assertIn(response.complexity.method, ("rules", "llm", "hybrid", "caller_llm"))

    def test_register_custom_provider(self):
        r = _router()
        r.register_provider("custom", MockProvider())
        self.assertIn("custom", r._providers)

    def test_offline_mode_skips_llm(self):
        """Offline mode should never call the LLM, even in ambiguous zones."""
        task = Task(prompt="What is DNA?", mode="offline")
        response = asyncio.run(_router().run(task))
        self.assertEqual(response.complexity.method, "rules")


class TestSettingsLoading(unittest.TestCase):
    def test_defaults_load(self):
        s = load_settings()
        self.assertEqual(s.version, "2.0")
        self.assertIn("simple", s.tiers)
        self.assertIn("anthropic", s.providers)

    def test_tier_resolution(self):
        s = load_settings()
        self.assertEqual(s.tier_for_score(10), "simple")
        self.assertEqual(s.tier_for_score(50), "medium")
        self.assertEqual(s.tier_for_score(80), "complex")

    def test_model_for_tier(self):
        s = load_settings()
        self.assertIn("haiku", s.model_for_tier("simple")["model_id"])
        self.assertIn("sonnet", s.model_for_tier("medium")["model_id"])
        self.assertIn("opus", s.model_for_tier("complex")["model_id"])


class TestTaskValidation(unittest.TestCase):
    def test_invalid_domain_raises(self):
        with self.assertRaises(ValueError):
            Task(prompt="test", domain="invalid")

    def test_invalid_priority_raises(self):
        with self.assertRaises(ValueError):
            Task(prompt="test", priority="urgent")

    def test_evaluation_mode_enum_values(self):
        self.assertEqual(EvaluationMode.OFFLINE, "offline")
        self.assertEqual(EvaluationMode.CALLER_LLM, "caller_llm")


if __name__ == "__main__":
    unittest.main()
