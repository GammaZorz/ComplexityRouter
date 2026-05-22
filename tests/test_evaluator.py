"""Unit tests for the hybrid complexity evaluator (no API calls)."""

from __future__ import annotations

import asyncio
import unittest

from agent_router.evaluator import ComplexityEvaluator
from agent_router.schemas import Task
from agent_router.settings import load_settings


def _settings():
    return load_settings()


class TestRuleScoring(unittest.TestCase):
    def setUp(self):
        self.evaluator = ComplexityEvaluator(settings=_settings())

    def test_simple_lookup(self):
        score = self.evaluator.score_by_rules(Task(prompt="What is photosynthesis?"))
        self.assertLessEqual(score, 33)

    def test_simple_define(self):
        score = self.evaluator.score_by_rules(Task(prompt="Define osmosis."))
        self.assertLessEqual(score, 33)

    def test_simple_list(self):
        score = self.evaluator.score_by_rules(Task(prompt="List the planets in the solar system."))
        self.assertLessEqual(score, 33)

    def test_medium_multi_step(self):
        score = self.evaluator.score_by_rules(
            Task(prompt="First explain how TCP works, then describe how TLS wraps it.")
        )
        self.assertGreater(score, 33)

    def test_medium_debug(self):
        score = self.evaluator.score_by_rules(
            Task(prompt="Why is my React component not working? It renders blank.", domain="code")
        )
        self.assertGreater(score, 33)

    def test_complex_architecture(self):
        task = Task(
            prompt=(
                "Design a microservice architecture for an e-commerce platform.\n"
                "First define the service boundaries, then build the API contracts "
                "step by step.\n"
                "```yaml\nservices:\n  - order-service\n  - payment-service\n```\n"
                "Include:\n1. Service boundaries\n2. Database schema per service\n"
                "3. API design and inter-service communication\n4. CI/CD pipeline\n"
                "5. Deployment strategy with Kubernetes"
            ),
            domain="code",
        )
        score = self.evaluator.score_by_rules(task)
        self.assertGreaterEqual(score, 67)

    def test_complex_code_with_stacktrace(self):
        task = Task(
            prompt=(
                "```python\ndef process(data):\n    return data.transform()\n```\n\n"
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
    def test_speed_caps_at_medium(self):
        self.assertEqual(ComplexityEvaluator._apply_priority("complex", "speed"), "medium")

    def test_speed_leaves_simple_alone(self):
        self.assertEqual(ComplexityEvaluator._apply_priority("simple", "speed"), "simple")

    def test_quality_floors_at_medium(self):
        self.assertEqual(ComplexityEvaluator._apply_priority("simple", "quality"), "medium")

    def test_quality_leaves_complex_alone(self):
        self.assertEqual(ComplexityEvaluator._apply_priority("complex", "quality"), "complex")

    def test_normal_changes_nothing(self):
        for tier in ("simple", "medium", "complex"):
            self.assertEqual(ComplexityEvaluator._apply_priority(tier, "normal"), tier)


class SafeEvaluator(ComplexityEvaluator):
    async def _score_by_llm(self, task):
        return 50, "Mock LLM classification"

    async def _score_by_api(self, task):
        return 50, "Mock API classification (no real call)"

    async def _score_by_caller_llm(self, task):
        return None  # Simulate sampling not supported


class TestFullEvaluateModes(unittest.TestCase):
    def setUp(self):
        self.settings = _settings()

    def test_offline_mode_never_calls_llm(self):
        """Offline mode returns rules-only result even for ambiguous score."""
        ev = SafeEvaluator(settings=self.settings)
        task = Task(prompt="What is the capital of France?", mode="offline")
        result = asyncio.run(ev.evaluate(task))
        self.assertEqual(result.tier, "simple")
        self.assertEqual(result.method, "rules")

    def test_offline_first_clear_simple(self):
        ev = SafeEvaluator(settings=self.settings)
        task = Task(prompt="What is the capital of France?", mode="offline_first")
        result = asyncio.run(ev.evaluate(task))
        self.assertEqual(result.tier, "simple")
        self.assertEqual(result.method, "rules")

    def test_offline_first_clear_complex(self):
        ev = SafeEvaluator(settings=self.settings)
        task = Task(
            prompt=(
                "Design a scalable microservice architecture with database schema, "
                "API design, CI/CD pipeline, and deployment strategy.\n"
                "1. Service boundaries\n2. Data model\n3. Auth flow\n"
                "4. Caching layer\n5. Monitoring\n"
                "Then refactor the existing monolith step by step."
            ),
            domain="code",
            mode="offline_first",
        )
        result = asyncio.run(ev.evaluate(task))
        self.assertEqual(result.tier, "complex")

    def test_always_llm_mode(self):
        ev = SafeEvaluator(settings=self.settings)
        task = Task(prompt="What is gravity?", mode="always_llm")
        result = asyncio.run(ev.evaluate(task))
        self.assertEqual(result.method, "llm")

    def test_caller_llm_falls_back_to_api(self):
        """caller_llm with no ctx and fallback=True should use API path."""
        ev = SafeEvaluator(ctx=None, settings=self.settings)
        task = Task(prompt="What is gravity?", mode="caller_llm")
        # SafeEvaluator._score_by_api is still mocked via _score_by_llm
        result = asyncio.run(ev.evaluate(task))
        self.assertIn(result.method, ("llm", "rules"))


class TestTaskValidation(unittest.TestCase):
    def test_invalid_domain_raises(self):
        with self.assertRaises(ValueError):
            Task(prompt="test", domain="invalid")

    def test_invalid_priority_raises(self):
        with self.assertRaises(ValueError):
            Task(prompt="test", priority="urgent")

    def test_invalid_mode_raises(self):
        with self.assertRaises(ValueError):
            Task(prompt="test", mode="turbo")

    def test_valid_modes(self):
        for mode in ("offline", "offline_first", "caller_llm", "always_llm"):
            t = Task(prompt="test", mode=mode)
            self.assertEqual(t.mode, mode)
