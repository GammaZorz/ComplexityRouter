"""Model registry — maps complexity tiers to provider + model configurations."""

from __future__ import annotations

from typing import Any

from .config import DEFAULT_MAX_TOKENS
from .schemas import ComplexityScore, ModelSelection


# ---------------------------------------------------------------------------
# Registry: each tier maps to a provider, model_id, and max_tokens.
# Add non-Claude entries here later (e.g. "openai", "ollama").
# ---------------------------------------------------------------------------
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "simple": {
        "provider": "anthropic",
        "model_id": "claude-haiku-4-5-20251001",
        "max_tokens": DEFAULT_MAX_TOKENS["simple"],
    },
    "medium": {
        "provider": "anthropic",
        "model_id": "claude-sonnet-4-6",
        "max_tokens": DEFAULT_MAX_TOKENS["medium"],
    },
    "complex": {
        "provider": "anthropic",
        "model_id": "claude-opus-4-7",
        "max_tokens": DEFAULT_MAX_TOKENS["complex"],
    },
}


def select_model(complexity: ComplexityScore) -> ModelSelection:
    """Look up the registry entry for the given complexity tier."""
    entry = MODEL_REGISTRY[complexity.tier]
    return ModelSelection(
        model_id=entry["model_id"],
        provider=entry["provider"],
        max_tokens=entry["max_tokens"],
        rationale=f"Tier '{complexity.tier}' (score {complexity.score}) "
                  f"routed to {entry['model_id']}",
    )
