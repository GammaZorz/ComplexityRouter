"""Unit tests for the hybrid complexity evaluator (no API calls)."""

from __future__ import annotations

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from agent_router.evaluator import ComplexityEvaluator
from agent_router.schemas import Task


class TestRuleScoring(unittest.TestCase):
    """Tests the pure rule-based scoring — zero cost, no network."""

    def setUp(self) -> None:
        self.evaluator = ComplexityEvaluator()

    # -- Simple prompts should score low -----------------------------------

    def test_simple_lookup(self) -> None:
        task = Task(prompt="What is photosynthesis?")
        score = self.evaluator.score_by_rules(task)
        self.assertLessEqual(score, 33, f"Simple lookup scored {score}, expected <= 33")

    def test_simple_define(self) -> None:
        task = Task(prompt="Define osmosis.")
        score = self.evaluator.score_by_rules(task)
        self.assertLessEqual(score, 33)

    def test_simple_list(self) -> None:
        task = Task(prompt="List the planets in the solar system.")
        score = self.evaluator.score_by_rules(task)
        self.assertLessEqual(score, 33)

    # -- Medium prompts should score mid -----------------------------------

    def test_medium_multi_step(self) -> None:
        task = Task(
            prompt="First explain how TCP works, then describe how TLS wraps it."
        )
        score = self.evaluator.score_by_rules(task)
        self.assertGreater(score, 33, f"Multi-step scored {score}, expected > 33")

    def test_medium_debug(self) -> None:
        task = Task(
            prompt="Why is my React component not working? It renders blank.",
            domain="code",
        )
        score = self.evaluator.score_by_rules(task)
        self.assertGreater(score, 33)

    # -- Complex prompts should score high ---------------------------------

    def test_complex_architecture(self) -> None:
        task = Task(
            prompt=(
                "Design a microservice architecture for an e-commerce platform.\n"
                "First define the service boundaries, then build the API contracts "
                "step by step.\n"
                "```yaml\n"
                "services:\n"
                "  - order-service\n"
                "  - payment-service\n"
                "```\n"
                "Include:\n"
                "1. Service boundaries\n"
                "2. Database schema per service\n"
                "3. API design and inter-service communication\n"
                "4. CI/CD pipeline\n"
                "5. Deployment strategy with Kubernetes"
            ),
            domain="code",
        )
        score = self.evaluator.score_by_rules(task)
        self.assertGreaterEqual(score, 67, f"Architecture scored {score}, expected >= 67")

    def test_complex_code_with_stacktrace(self) -> None:
        task = Task(
            prompt=(
                "```python\n"
                "def process(data):\n"
                "    return data.transform()\n"
                "```\n\n"
                "Traceback (most recent call last):\n"
                "  File \"app.py\", line 42, in process\n"
                "    AttributeError: 'NoneType' has no attribute 'transform'\n\n"
                "Why does this crash? Then refactor the module to handle edge cases "
                "and add comprehensive error handling step by step."
            ),
            domain="code",
        )
        score = self.evaluator.score_by_rules(task)
        self.assertGreaterEqual(score, 67)


class TestPriorityOverride(unittest.TestCase):
    """Ensure priority flags cap/floor the tier correctly."""

    def test_speed_caps_at_medium(self) -> None:
        result = ComplexityEvaluator._apply_priority("complex", "speed")
        self.assertEqual(result, "medium")

    def test_speed_leaves_simple_alone(self) -> None:
        result = ComplexityEvaluator._apply_priority("simple", "speed")
        self.assertEqual(result, "simple")

    def test_quality_floors_at_medium(self) -> None:
        result = ComplexityEvaluator._apply_priority("simple", "quality")
        self.assertEqual(result, "medium")

    def test_quality_leaves_complex_alone(self) -> None:
        result = ComplexityEvaluator._apply_priority("complex", "quality")
        self.assertEqual(result, "complex")

    def test_normal_changes_nothing(self) -> None:
        for tier in ("simple", "medium", "complex"):
            result = ComplexityEvaluator._apply_priority(tier, "normal")
            self.assertEqual(result, tier)


class SafeEvaluator(ComplexityEvaluator):
    """Evaluator whose LLM fallback returns a canned score (no API call)."""

    async def _score_by_llm(self, task):
        return 50, "Mock LLM classification"


class TestFullEvaluateClearTier(unittest.TestCase):
    """Full evaluate() — uses safe evaluator so ambiguous prompts never hit the API."""

    def test_clear_simple(self) -> None:
        evaluator = SafeEvaluator()
        task = Task(prompt="What is the capital of France?")
        result = asyncio.run(evaluator.evaluate(task))
        self.assertEqual(result.tier, "simple")
        self.assertEqual(result.method, "rules")

    def test_clear_complex(self) -> None:
        evaluator = SafeEvaluator()
        task = Task(
            prompt=(
                "Design a scalable microservice architecture with database schema, "
                "API design, CI/CD pipeline, and deployment strategy.\n"
                "1. Service boundaries\n2. Data model\n3. Auth flow\n"
                "4. Caching layer\n5. Monitoring\n"
                "Then refactor the existing monolith step by step."
            ),
            domain="code",
        )
        result = asyncio.run(evaluator.evaluate(task))
        self.assertEqual(result.tier, "complex")


if __name__ == "__main__":
    unittest.main()
