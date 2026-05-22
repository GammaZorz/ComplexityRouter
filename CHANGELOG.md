# Changelog

All notable changes to ComplexityRouter are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [2.0.0] - 2026-05-22

### Added

- **JSON config file** (`complexity_router.json`) — all settings in one place; no more hardcoded constants. Supports deep-merge so you only override what you need.
- **Settings loader** (`agent_router/settings.py`) — reads `complexity_router.json`, merges with built-in defaults, exposes a typed `Settings` dataclass. Config path can be overridden via `COMPLEX_ROUTER_CONFIG` env var or `--config` CLI flag.
- **Four evaluation modes** (set in config or per-task):
  - `offline` — rules only, never calls any LLM
  - `offline_first` — rules first; LLM fallback in ambiguous score zones (default)
  - `caller_llm` — asks the calling Claude Code instance via MCP sampling (no separate API key needed); falls back to direct API if sampling is unsupported
  - `always_llm` — always call LLM classifier, skip rules entirely
- **MCP sampling support** (`caller_llm` mode) — the MCP server now passes the MCP `Context` into the evaluator so it can request classification from the calling LLM. Falls back gracefully to a direct API call while Claude Code sampling support is pending ([issue #1785](https://github.com/anthropics/claude-code/issues/1785)).
- **OpenRouter provider** (`agent_router/providers/openrouter.py`) — routes via [openrouter.ai](https://openrouter.ai) using the OpenAI-compatible API. Supports any model string including `openrouter/auto` for fully delegated NotDiamond routing across 300+ models.
- **LiteLLM provider** (`agent_router/providers/litellm.py`) — two sub-modes:
  - *Proxy mode* (default): HTTP calls to a running LiteLLM proxy; no `litellm` package required
  - *SDK mode*: `import litellm` for direct 100+ provider access without a proxy
- **JSONL request logging** (`agent_router/logger.py`) — appends one JSON line per routed request to a configurable log file. Fields: timestamp, score, tier, method, model, provider, latency, tokens, domain, priority, mode. Prompt and response inclusion are individually togglable.
- **`EvaluationMode` enum** in `schemas.py` for type-safe mode references.
- **`mode` field on `Task`** — per-request mode override independent of the global config default.
- **`--config` and `--mode` flags** on `run_router.py` CLI.
- **Auto-provider registration** in `AgentRouter` — providers are built from `settings.providers` at startup; missing API keys are skipped silently.
- **12 new unit tests** covering modes, settings loading, new providers, and caller-LLM fallback.

### Changed

- `agent_router/config.py` is now a defaults-only reference; all runtime values come from `settings.py`.
- `agent_router/models.py` now reads the model registry from `Settings` instead of module-level constants.
- `agent_router/router.py` wires in `RouterLogger` and builds providers from settings automatically.
- `mcp_server.py` passes `ctx: Context` to the evaluator to enable sampling.
- `requirements.txt` adds `openai>=1.0.0` (required for OpenRouter and LiteLLM proxy mode).

### Notes

- MCP sampling (`caller_llm` mode) is specified in the MCP protocol but not yet implemented on the Claude Code client side. The code is ready and will activate automatically when support lands.
- `litellm` package is optional — install separately with `pip install litellm` if you want SDK mode.

---

## [1.0.0] - 2026-05-22

### Added

- Initial release: hybrid rules + Haiku LLM fallback complexity evaluator
- Anthropic provider with async client and prompt caching
- Model registry: Haiku (simple), Sonnet (medium), Opus (complex)
- Priority overrides: `speed` caps at Sonnet, `quality` floors at Sonnet
- MCP server (`mcp_server.py`) exposing two tools to Claude Code:
  - `evaluate_complexity` — score a prompt, get model recommendation
  - `route_and_execute` — route + execute, return full response
- CLI entry point (`run_router.py`) with JSON stdout
- 22 unit tests (all offline, no API calls)
- Registered globally with Claude Code via `claude mcp add -s user`
