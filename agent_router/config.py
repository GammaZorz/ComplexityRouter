"""Central configuration — all tunable values in one place."""

# ---------------------------------------------------------------------------
# Ambiguous-zone bands (rule scores that trigger the LLM fallback)
# ---------------------------------------------------------------------------
AMBIGUOUS_LOW: tuple[int, int] = (28, 38)
AMBIGUOUS_HIGH: tuple[int, int] = (60, 70)

# ---------------------------------------------------------------------------
# Blending weights when the LLM fallback fires
# ---------------------------------------------------------------------------
RULE_WEIGHT: float = 0.4
LLM_WEIGHT: float = 0.6

# ---------------------------------------------------------------------------
# Tier boundaries (inclusive)
# ---------------------------------------------------------------------------
TIER_BOUNDS: dict[str, tuple[int, int]] = {
    "simple":  (0, 33),
    "medium":  (34, 66),
    "complex": (67, 100),
}

# ---------------------------------------------------------------------------
# LLM classifier model (used only in the ambiguous-zone fallback)
# ---------------------------------------------------------------------------
LLM_CLASSIFIER_MODEL: str = "claude-haiku-4-5-20251001"

# ---------------------------------------------------------------------------
# Default max-tokens per tier (overridable in MODEL_REGISTRY)
# ---------------------------------------------------------------------------
DEFAULT_MAX_TOKENS: dict[str, int] = {
    "simple":  2048,
    "medium":  4096,
    "complex": 8192,
}
