"""Model registry — loads tier→model mapping from Settings."""

from __future__ import annotations

from .schemas import ComplexityScore, ModelSelection
from .settings import Settings, get_settings


def select_model(
    complexity: ComplexityScore,
    settings: Settings | None = None,
) -> ModelSelection:
    """Look up the model for the given complexity tier from settings."""
    s = settings or get_settings()
    entry = s.model_for_tier(complexity.tier)
    max_tokens = s.max_tokens_for_tier(complexity.tier)

    return ModelSelection(
        model_id=entry["model_id"],
        provider=entry["provider"],
        max_tokens=max_tokens,
        rationale=(
            f"Tier '{complexity.tier}' (score {complexity.score}) "
            f"→ {entry['model_id']} via {entry['provider']}"
        ),
    )
