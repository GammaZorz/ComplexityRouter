"""
Settings loader — reads complexity_router.json and exposes a Settings dataclass.

All other modules import from here instead of config.py.
config.py is kept as a fallback-defaults reference only.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Hard-coded defaults (used when no JSON config file is found)
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, Any] = {
    "version": "2.0",
    "evaluator": {
        "mode": "offline_first",
        "ambiguous_low": [28, 38],
        "ambiguous_high": [60, 70],
        "rule_weight": 0.4,
        "llm_weight": 0.6,
        "classifier_model": "claude-haiku-4-5-20251001",
        "prompt_truncation_chars": 500,
        "use_caller_llm": True,
        "fallback_to_api_if_sampling_fails": True,
    },
    "tiers": {
        "simple":  {"score_range": [0,  33], "max_tokens": 2048},
        "medium":  {"score_range": [34, 66], "max_tokens": 4096},
        "complex": {"score_range": [67, 100], "max_tokens": 8192},
    },
    "models": {
        "simple":  {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001"},
        "medium":  {"provider": "anthropic", "model_id": "claude-sonnet-4-6"},
        "complex": {"provider": "anthropic", "model_id": "claude-opus-4-7"},
    },
    "providers": {
        "anthropic":  {"api_key_env": "ANTHROPIC_API_KEY"},
        "openrouter": {"api_key_env": "OPENROUTER_API_KEY", "base_url": "https://openrouter.ai/api/v1"},
        "litellm":    {"base_url": "http://localhost:4000"},
    },
    "logging": {
        "enabled": True,
        "file": "complexity_router.log",
        "format": "jsonl",
        "include_prompt": True,
        "include_response": False,
    },
}


# ---------------------------------------------------------------------------
# Settings dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvaluatorSettings:
    mode: str                            # offline | offline_first | caller_llm | always_llm
    ambiguous_low: tuple[int, int]
    ambiguous_high: tuple[int, int]
    rule_weight: float
    llm_weight: float
    classifier_model: str
    prompt_truncation_chars: int
    use_caller_llm: bool
    fallback_to_api_if_sampling_fails: bool


@dataclass
class TierSettings:
    score_range: tuple[int, int]
    max_tokens: int


@dataclass
class LoggingSettings:
    enabled: bool
    file: str
    format: str                          # "jsonl" only in v2
    include_prompt: bool
    include_response: bool


@dataclass
class Settings:
    version: str
    evaluator: EvaluatorSettings
    tiers: dict[str, TierSettings]       # "simple" | "medium" | "complex"
    models: dict[str, dict[str, str]]    # tier -> {provider, model_id}
    providers: dict[str, dict[str, str]] # provider_name -> {api_key_env, base_url, ...}
    logging: LoggingSettings
    _raw: dict[str, Any] = field(default_factory=dict, repr=False)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def tier_for_score(self, score: int) -> str:
        for tier, ts in self.tiers.items():
            lo, hi = ts.score_range
            if lo <= score <= hi:
                return tier
        return "medium"

    def model_for_tier(self, tier: str) -> dict[str, str]:
        return self.models.get(tier, self.models["medium"])

    def max_tokens_for_tier(self, tier: str) -> int:
        return self.tiers.get(tier, self.tiers["medium"]).max_tokens

    def provider_config(self, name: str) -> dict[str, str]:
        return self.providers.get(name, {})

    def resolved_api_key(self, provider_name: str) -> str | None:
        """Resolve api_key_env → actual env var value."""
        cfg = self.provider_config(provider_name)
        env_var = cfg.get("api_key_env", "")
        return os.environ.get(env_var) if env_var else None


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base (override wins)."""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_settings(config_path: str | Path | None = None) -> Settings:
    """
    Load settings from a JSON config file, merged on top of built-in defaults.

    Search order for config file (first found wins):
      1. Explicit ``config_path`` argument
      2. ``COMPLEX_ROUTER_CONFIG`` environment variable
      3. ``complexity_router.json`` in the current working directory
      4. Built-in defaults only
    """
    raw: dict[str, Any] = dict(_DEFAULTS)

    candidates: list[Path | None] = [
        Path(config_path) if config_path else None,
        Path(os.environ["COMPLEX_ROUTER_CONFIG"]) if "COMPLEX_ROUTER_CONFIG" in os.environ else None,
        Path.cwd() / "complexity_router.json",
    ]

    for candidate in candidates:
        if candidate and candidate.is_file():
            with open(candidate, encoding="utf-8") as fh:
                user_cfg = json.load(fh)
            raw = _deep_merge(raw, user_cfg)
            break

    ev = raw["evaluator"]
    return Settings(
        version=raw.get("version", "2.0"),
        evaluator=EvaluatorSettings(
            mode=ev.get("mode", "offline_first"),
            ambiguous_low=tuple(ev["ambiguous_low"]),
            ambiguous_high=tuple(ev["ambiguous_high"]),
            rule_weight=float(ev["rule_weight"]),
            llm_weight=float(ev["llm_weight"]),
            classifier_model=ev["classifier_model"],
            prompt_truncation_chars=int(ev["prompt_truncation_chars"]),
            use_caller_llm=bool(ev.get("use_caller_llm", True)),
            fallback_to_api_if_sampling_fails=bool(ev.get("fallback_to_api_if_sampling_fails", True)),
        ),
        tiers={
            name: TierSettings(
                score_range=tuple(cfg["score_range"]),
                max_tokens=int(cfg["max_tokens"]),
            )
            for name, cfg in raw["tiers"].items()
        },
        models=raw["models"],
        providers=raw["providers"],
        logging=LoggingSettings(**raw["logging"]),
        _raw=raw,
    )


# ---------------------------------------------------------------------------
# Module-level singleton — import this everywhere
# ---------------------------------------------------------------------------
_settings: Settings | None = None


def get_settings(config_path: str | Path | None = None, *, reload: bool = False) -> Settings:
    """Return the cached Settings instance, loading it on first call."""
    global _settings
    if _settings is None or reload:
        _settings = load_settings(config_path)
    return _settings
