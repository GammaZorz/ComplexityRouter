"""Hybrid complexity evaluator — rules first, Haiku LLM fallback if ambiguous."""

from __future__ import annotations

import json
import re

import anthropic

from .config import (
    AMBIGUOUS_HIGH,
    AMBIGUOUS_LOW,
    LLM_CLASSIFIER_MODEL,
    LLM_WEIGHT,
    RULE_WEIGHT,
    TIER_BOUNDS,
)
from .schemas import ComplexityScore, Task


# ---------------------------------------------------------------------------
# Rule-based signal detectors
# ---------------------------------------------------------------------------

_CODE_PATTERNS = re.compile(
    r"```|\.py\b|\.js\b|\.ts\b|\.go\b|\.rs\b|\.java\b|\.cpp\b|\.c\b|\.rb\b|\.sh\b"
    r"|def\s+\w+|class\s+\w+|function\s+\w+|import\s+\w+|from\s+\w+\s+import",
    re.IGNORECASE,
)

_STACKTRACE_PATTERNS = re.compile(
    r"Traceback|Exception|Error:|at\s+\w+\.\w+\(|File\s+\"",
    re.IGNORECASE,
)

_MULTI_STEP_KEYWORDS = re.compile(
    r"\b(then|after that|next|finally|step\s+by\s+step|first.*then|subsequently)\b",
    re.IGNORECASE,
)

_ARCHITECTURE_KEYWORDS = re.compile(
    r"\b(architect|design|system\s+design|refactor|scalab|microservice|pipeline"
    r"|infrastructure|deploy|CI/CD|database\s+schema|migration|API\s+design)\b",
    re.IGNORECASE,
)

_DEBUG_KEYWORDS = re.compile(
    r"\b(why\s+does|why\s+is|not\s+working|bug|debug|fix\s+this|broken|crash"
    r"|segfault|undefined|null\s+pointer)\b",
    re.IGNORECASE,
)

_SIMPLE_LOOKUP_KEYWORDS = re.compile(
    r"^\s*(what\s+is|define|list\s+|translate|convert\s+\S+\s+to)\b",
    re.IGNORECASE,
)

_FORMAT_KEYWORDS = re.compile(
    r"\b(format|summarise|summarize|rewrite\s+this\s+sentence|rephrase)\b",
    re.IGNORECASE,
)

_NUMBERED_QUESTIONS = re.compile(r"^\s*\d+[\.\)]\s+", re.MULTILINE)


def _estimate_token_count(text: str) -> int:
    """Rough token estimate — ~4 chars per token for English."""
    return len(text) // 4


# ---------------------------------------------------------------------------
# Public evaluator
# ---------------------------------------------------------------------------

class ComplexityEvaluator:
    """Hybrid evaluator: rule score first, LLM fallback in ambiguous zones."""

    def __init__(self, client: anthropic.AsyncAnthropic | None = None) -> None:
        self._client = client

    # -- Rule-based scoring ------------------------------------------------

    def score_by_rules(self, task: Task) -> int:
        """Return a 0-100 rule-based complexity score."""
        score = 30  # neutral baseline
        positive_signals = 0  # track how many complexity signals fire

        # Token length
        tokens = _estimate_token_count(task.prompt)
        if tokens > 500:
            score += 20
            positive_signals += 1
        elif tokens > 200:
            score += 10
            positive_signals += 1

        # Code presence
        if _CODE_PATTERNS.search(task.prompt):
            score += 15
            positive_signals += 1

        # Stack trace / error message
        if _STACKTRACE_PATTERNS.search(task.prompt):
            score += 10
            positive_signals += 1

        # Multi-step language
        if _MULTI_STEP_KEYWORDS.search(task.prompt):
            score += 10
            positive_signals += 1

        # Architecture / design
        if _ARCHITECTURE_KEYWORDS.search(task.prompt):
            score += 15
            positive_signals += 1

        # Numbered sub-questions (3+)
        numbered = _NUMBERED_QUESTIONS.findall(task.prompt)
        if len(numbered) >= 3:
            score += 10
            positive_signals += 1

        # Debugging markers
        if _DEBUG_KEYWORDS.search(task.prompt):
            score += 8
            positive_signals += 1

        # Simple lookup (subtract)
        if _SIMPLE_LOOKUP_KEYWORDS.search(task.prompt):
            score -= 15

        # Formatting / short rewrite (subtract)
        if _FORMAT_KEYWORDS.search(task.prompt):
            score -= 10

        # Code domain bias
        if task.domain == "code" and score > 40:
            score += 5

        # Compound complexity: when 3+ positive signals fire together
        # the task is clearly multi-dimensional — boost past ambiguity
        if positive_signals >= 3:
            score += 10

        return max(0, min(100, score))

    # -- LLM fallback ------------------------------------------------------

    async def _score_by_llm(self, task: Task) -> tuple[int, str]:
        """Ask Haiku to classify the prompt. Returns (score 0-100, reason)."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()

        truncated_prompt = task.prompt[:500]
        response = await self._client.messages.create(
            model=LLM_CLASSIFIER_MODEL,
            max_tokens=150,
            system="You are a complexity classifier. Respond ONLY with valid JSON.",
            messages=[
                {
                    "role": "user",
                    "content": (
                        "Rate the complexity of the following task on a scale of 1-10.\n"
                        'Respond with JSON: {"score": <int 1-10>, '
                        '"tier": "simple|medium|complex", "reason": "<one sentence>"}\n\n'
                        f"Task: {truncated_prompt}"
                    ),
                }
            ],
        )

        raw = response.content[0].text.strip()
        try:
            parsed = json.loads(raw)
            llm_score = int(parsed.get("score", 5))
            reason = parsed.get("reason", "LLM classification")
        except (json.JSONDecodeError, ValueError, TypeError):
            llm_score = 5
            reason = "LLM response could not be parsed; defaulting to medium"

        # Normalise 1-10 → 0-100
        return max(0, min(100, llm_score * 10)), reason

    # -- Tier resolution ---------------------------------------------------

    @staticmethod
    def _tier_from_score(score: int) -> str:
        for tier, (lo, hi) in TIER_BOUNDS.items():
            if lo <= score <= hi:
                return tier
        return "medium"  # fallback

    # -- Priority overrides ------------------------------------------------

    @staticmethod
    def _apply_priority(tier: str, priority: str) -> str:
        if priority == "speed" and tier == "complex":
            return "medium"
        if priority == "quality" and tier == "simple":
            return "medium"
        return tier

    # -- Main entry --------------------------------------------------------

    async def evaluate(self, task: Task) -> ComplexityScore:
        """Run the full hybrid evaluation pipeline."""
        rule_score = self.score_by_rules(task)

        in_low = AMBIGUOUS_LOW[0] <= rule_score <= AMBIGUOUS_LOW[1]
        in_high = AMBIGUOUS_HIGH[0] <= rule_score <= AMBIGUOUS_HIGH[1]

        if in_low or in_high:
            llm_score, llm_reason = await self._score_by_llm(task)
            blended = int(RULE_WEIGHT * rule_score + LLM_WEIGHT * llm_score)
            blended = max(0, min(100, blended))
            tier = self._tier_from_score(blended)
            tier = self._apply_priority(tier, task.priority)
            return ComplexityScore(
                score=blended,
                tier=tier,
                rationale=f"Rules={rule_score}, LLM={llm_score} -> blended={blended}. {llm_reason}",
                method="hybrid",
            )

        tier = self._tier_from_score(rule_score)
        tier = self._apply_priority(tier, task.priority)
        return ComplexityScore(
            score=rule_score,
            tier=tier,
            rationale=f"Rule score {rule_score} — clear tier, no LLM needed.",
            method="rules",
        )
