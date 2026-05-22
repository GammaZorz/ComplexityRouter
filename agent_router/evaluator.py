"""
Hybrid complexity evaluator — four evaluation modes:

  offline       — rules only, no LLM calls ever
  offline_first — rules first; LLM fallback in ambiguous zones (default)
  caller_llm    — MCP sampling first (no API key); API fallback if unsupported
  always_llm    — always call LLM classifier, skip rules
"""

from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any

import anthropic

from .schemas import ComplexityScore, EvaluationMode, Task
from .settings import Settings, get_settings

if TYPE_CHECKING:
    pass  # mcp Context imported lazily to avoid hard dependency


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
    return len(text) // 4


# ---------------------------------------------------------------------------
# LLM response parser (shared)
# ---------------------------------------------------------------------------

def _parse_llm_json(raw: str) -> tuple[int, str]:
    """Parse JSON from LLM classifier. Returns (score 0-100, reason)."""
    try:
        parsed = json.loads(raw.strip())
        llm_score = int(parsed.get("score", 5))
        reason = str(parsed.get("reason", "LLM classification"))
    except (json.JSONDecodeError, ValueError, TypeError):
        llm_score = 5
        reason = "LLM response could not be parsed; defaulting to mid score"
    return max(0, min(100, llm_score * 10)), reason


_CLASSIFIER_PROMPT = (
    "Rate the complexity of the following task on a scale of 1-10.\n"
    'Respond with JSON only: {{"score": <int 1-10>, '
    '"tier": "simple|medium|complex", "reason": "<one sentence>"}}\n\n'
    "Task: {prompt}"
)

_CLASSIFIER_SYSTEM = "You are a complexity classifier. Respond ONLY with valid JSON."


# ---------------------------------------------------------------------------
# Evaluator
# ---------------------------------------------------------------------------

class ComplexityEvaluator:
    """
    Evaluates the complexity of a Task using the configured mode.

    Parameters
    ----------
    ctx :
        MCP Context object (injected by mcp_server.py when available).
        Used for the ``caller_llm`` mode (MCP sampling).
    settings :
        Settings instance. Defaults to the global singleton from get_settings().
    client :
        Optional pre-configured AsyncAnthropic client (useful for testing).
    """

    def __init__(
        self,
        ctx: Any | None = None,
        settings: Settings | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self._ctx = ctx
        self._settings = settings or get_settings()
        self._client = client

    # ------------------------------------------------------------------
    # Rule-based scoring
    # ------------------------------------------------------------------

    def score_by_rules(self, task: Task) -> int:
        """Return a 0-100 rule-based complexity score."""
        s = self._settings
        score = 30
        positive_signals = 0

        tokens = _estimate_token_count(task.prompt)
        if tokens > 500:
            score += 20; positive_signals += 1
        elif tokens > 200:
            score += 10; positive_signals += 1

        if _CODE_PATTERNS.search(task.prompt):
            score += 15; positive_signals += 1
        if _STACKTRACE_PATTERNS.search(task.prompt):
            score += 10; positive_signals += 1
        if _MULTI_STEP_KEYWORDS.search(task.prompt):
            score += 10; positive_signals += 1
        if _ARCHITECTURE_KEYWORDS.search(task.prompt):
            score += 15; positive_signals += 1

        numbered = _NUMBERED_QUESTIONS.findall(task.prompt)
        if len(numbered) >= 3:
            score += 10; positive_signals += 1

        if _DEBUG_KEYWORDS.search(task.prompt):
            score += 8; positive_signals += 1
        if _SIMPLE_LOOKUP_KEYWORDS.search(task.prompt):
            score -= 15
        if _FORMAT_KEYWORDS.search(task.prompt):
            score -= 10
        if task.domain == "code" and score > 40:
            score += 5
        if positive_signals >= 3:
            score += 10

        return max(0, min(100, score))

    # ------------------------------------------------------------------
    # LLM scoring via MCP caller (no API key)
    # ------------------------------------------------------------------

    async def _score_by_caller_llm(self, task: Task) -> tuple[int, str] | None:
        """
        Ask the calling LLM (Claude Code) to classify via MCP sampling.
        Returns None if sampling is unavailable or fails.
        """
        if self._ctx is None:
            return None
        truncated = task.prompt[:self._settings.evaluator.prompt_truncation_chars]
        prompt_text = _CLASSIFIER_PROMPT.format(prompt=truncated)
        try:
            result = await self._ctx.sample(
                prompt_text,
                system_prompt=_CLASSIFIER_SYSTEM,
                max_tokens=80,
            )
            raw = result.text if hasattr(result, "text") else str(result)
            return _parse_llm_json(raw)
        except Exception:
            # Sampling not supported (Claude Code issue #1785) or any other error
            return None

    # ------------------------------------------------------------------
    # LLM scoring via direct API key
    # ------------------------------------------------------------------

    async def _score_by_api(self, task: Task) -> tuple[int, str]:
        """Call the classifier model directly using an API key."""
        if self._client is None:
            self._client = anthropic.AsyncAnthropic()

        truncated = task.prompt[:self._settings.evaluator.prompt_truncation_chars]
        response = await self._client.messages.create(
            model=self._settings.evaluator.classifier_model,
            max_tokens=150,
            system=_CLASSIFIER_SYSTEM,
            messages=[{
                "role": "user",
                "content": _CLASSIFIER_PROMPT.format(prompt=truncated),
            }],
        )
        raw = response.content[0].text.strip()
        return _parse_llm_json(raw)

    # ------------------------------------------------------------------
    # Combined LLM scoring (tries caller first if configured)
    # ------------------------------------------------------------------

    async def _score_by_llm(self, task: Task) -> tuple[int, str]:
        ev = self._settings.evaluator
        if ev.use_caller_llm:
            result = await self._score_by_caller_llm(task)
            if result is not None:
                return result
            # Sampling failed/unsupported
            if not ev.fallback_to_api_if_sampling_fails:
                return 50, "Caller LLM unavailable; no API fallback configured"
        return await self._score_by_api(task)

    # ------------------------------------------------------------------
    # Tier resolution + priority overrides
    # ------------------------------------------------------------------

    def _tier_from_score(self, score: int) -> str:
        return self._settings.tier_for_score(score)

    @staticmethod
    def _apply_priority(tier: str, priority: str) -> str:
        if priority == "speed" and tier == "complex":
            return "medium"
        if priority == "quality" and tier == "simple":
            return "medium"
        return tier

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def evaluate(self, task: Task) -> ComplexityScore:
        """Run the full evaluation pipeline according to the configured mode."""
        ev = self._settings.evaluator

        # Resolve effective mode: task.mode overrides config default
        effective_mode = task.mode if task.mode != "offline_first" else ev.mode

        # --- offline: rules only, never LLM ---
        if effective_mode == EvaluationMode.OFFLINE:
            rule_score = self.score_by_rules(task)
            tier = self._apply_priority(self._tier_from_score(rule_score), task.priority)
            return ComplexityScore(
                score=rule_score, tier=tier,
                rationale=f"Offline rules score: {rule_score}.",
                method="rules",
            )

        # --- always_llm: skip rules entirely ---
        if effective_mode == EvaluationMode.ALWAYS_LLM:
            llm_score, reason = await self._score_by_llm(task)
            tier = self._apply_priority(self._tier_from_score(llm_score), task.priority)
            return ComplexityScore(
                score=llm_score, tier=tier,
                rationale=f"LLM score: {llm_score}. {reason}",
                method="llm",
            )

        # --- caller_llm: try caller first, fall back to offline if nothing works ---
        if effective_mode == EvaluationMode.CALLER_LLM:
            result = await self._score_by_caller_llm(task)
            if result is not None:
                llm_score, reason = result
                tier = self._apply_priority(self._tier_from_score(llm_score), task.priority)
                return ComplexityScore(
                    score=llm_score, tier=tier,
                    rationale=f"Caller LLM score: {llm_score}. {reason}",
                    method="caller_llm",
                )
            if ev.fallback_to_api_if_sampling_fails:
                llm_score, reason = await self._score_by_api(task)
                tier = self._apply_priority(self._tier_from_score(llm_score), task.priority)
                return ComplexityScore(
                    score=llm_score, tier=tier,
                    rationale=f"API fallback score: {llm_score}. {reason}",
                    method="llm",
                )
            # Last resort: offline rules
            rule_score = self.score_by_rules(task)
            tier = self._apply_priority(self._tier_from_score(rule_score), task.priority)
            return ComplexityScore(
                score=rule_score, tier=tier,
                rationale=f"Offline fallback: {rule_score}.",
                method="rules",
            )

        # --- offline_first (default): rules first, LLM in ambiguous zones ---
        rule_score = self.score_by_rules(task)
        lo_lo, lo_hi = ev.ambiguous_low
        hi_lo, hi_hi = ev.ambiguous_high
        in_ambiguous = (lo_lo <= rule_score <= lo_hi) or (hi_lo <= rule_score <= hi_hi)

        if in_ambiguous:
            llm_score, reason = await self._score_by_llm(task)
            blended = int(ev.rule_weight * rule_score + ev.llm_weight * llm_score)
            blended = max(0, min(100, blended))
            tier = self._apply_priority(self._tier_from_score(blended), task.priority)
            return ComplexityScore(
                score=blended, tier=tier,
                rationale=f"Rules={rule_score}, LLM={llm_score} → blended={blended}. {reason}",
                method="hybrid",
            )

        tier = self._apply_priority(self._tier_from_score(rule_score), task.priority)
        return ComplexityScore(
            score=rule_score, tier=tier,
            rationale=f"Rule score {rule_score} — clear tier, no LLM needed.",
            method="rules",
        )
