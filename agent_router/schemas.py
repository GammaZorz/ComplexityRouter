"""Data schemas for the ComplexRoute agentic routing workflow."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class Task:
    """Incoming work unit to be routed to the appropriate model."""

    prompt: str
    context: str = ""            # system prompt or background info
    domain: str = "general"      # "code" | "general" | "research"
    priority: str = "normal"     # "speed" | "normal" | "quality"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_domains = {"general", "code", "research"}
        valid_priorities = {"speed", "normal", "quality"}
        if self.domain not in valid_domains:
            raise ValueError(f"domain must be one of {valid_domains}, got '{self.domain}'")
        if self.priority not in valid_priorities:
            raise ValueError(f"priority must be one of {valid_priorities}, got '{self.priority}'")


@dataclass
class ComplexityScore:
    """Result of the hybrid complexity evaluation."""

    score: int          # 0-100
    tier: str           # "simple" | "medium" | "complex"
    rationale: str      # human-readable explanation
    method: str         # "rules" | "llm" | "hybrid"

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
        d = asdict(self)
        return d
