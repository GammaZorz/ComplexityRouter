"""Data schemas for the ComplexRoute agentic routing workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class EvaluationMode(str, Enum):
    """Controls how complexity is evaluated."""
    OFFLINE       = "offline"       # rules only, never calls any LLM
    OFFLINE_FIRST = "offline_first" # rules first; LLM fallback in ambiguous zones
    CALLER_LLM    = "caller_llm"    # ask the calling LLM via MCP sampling (no API key)
    ALWAYS_LLM    = "always_llm"    # always call LLM classifier, skip rules


@dataclass
class Task:
    """Incoming work unit to be routed to the appropriate model."""

    prompt: str
    context: str = ""              # system prompt or background info
    domain: str = "general"        # "code" | "general" | "research"
    priority: str = "normal"       # "speed" | "normal" | "quality"
    mode: str = "offline_first"    # EvaluationMode value; overrides config if set
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_domains = {"general", "code", "research"}
        valid_priorities = {"speed", "normal", "quality"}
        valid_modes = {m.value for m in EvaluationMode}
        if self.domain not in valid_domains:
            raise ValueError(f"domain must be one of {valid_domains}, got '{self.domain}'")
        if self.priority not in valid_priorities:
            raise ValueError(f"priority must be one of {valid_priorities}, got '{self.priority}'")
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}, got '{self.mode}'")


@dataclass
class ComplexityScore:
    """Result of the hybrid complexity evaluation."""

    score: int          # 0-100
    tier: str           # "simple" | "medium" | "complex"
    rationale: str
    method: str         # "rules" | "llm" | "hybrid" | "caller_llm"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSelection:
    """Which model was selected and why."""

    model_id: str
    provider: str
    max_tokens: int
    rationale: str


@dataclass
class AgentResponse:
    """Full response returned to the caller."""

    output: str
    model_used: str
    provider: str
    complexity: ComplexityScore
    latency_ms: float
    input_tokens: int
    output_tokens: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
