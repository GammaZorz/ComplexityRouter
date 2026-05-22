# ComplexityRouter

An agentic complexity-routing workflow that evaluates incoming prompts, scores their difficulty, and automatically dispatches them to the most cost-effective Claude model that can handle the task well.

**Simple question?** Haiku answers in milliseconds for pennies.
**Architecture redesign?** Opus takes the wheel.

## How It Works

```
Incoming prompt
      |
      v
+---------------------+
| Hybrid Evaluator    |  Rules first (zero cost), Haiku LLM fallback if ambiguous
+---------------------+
      |
      v  ComplexityScore (0-100)
+---------------------+
|  0-33   -> Haiku    |  Fast, cheap — lookups, formatting, short Q&A
| 34-66   -> Sonnet   |  Balanced — analysis, drafting, multi-step reasoning
| 67-100  -> Opus     |  Powerful — architecture, deep debugging, code generation
+---------------------+
      |
      v
+---------------------+
| Provider Adapter    |  Anthropic now, pluggable for OpenAI/Ollama/others later
+---------------------+
      |
      v
  Response + metadata (model used, score, tier, latency, tokens)
```

### Complexity Evaluation (Hybrid)

**Step 1 — Rule-based scoring (zero cost):** Eight signal detectors scan the prompt for code blocks, stack traces, multi-step language, architecture keywords, numbered sub-questions, debugging markers, and simple lookup patterns. Each adds or subtracts points from a 30-point baseline.

**Step 2 — LLM fallback (Haiku, cheap):** Only fires when the rule score lands in an ambiguous zone (28-38 or 60-70). Haiku classifies the prompt in ~200ms for fractions of a cent. The final score blends rules (40%) and LLM (60%).

**Step 3 — Priority override:**
- `priority="speed"` caps the tier at Sonnet (never sends to Opus)
- `priority="quality"` floors the tier at Sonnet (never sends to Haiku)

## Installation

```bash
git clone https://github.com/GammaZorz/ComplexityRouter.git
cd ComplexityRouter
pip install -r requirements.txt
```

Create a `.env` file with your API key:

```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
```

## Usage

### Python API

```python
import asyncio
from agent_router import AgentRouter, Task

async def main():
    router = AgentRouter()

    # Simple question -> routes to Haiku
    response = await router.run(Task(prompt="What is photosynthesis?"))
    print(response.model_used)   # claude-haiku-4-5-20251001
    print(response.output)

    # Complex code task -> routes to Opus
    response = await router.run(Task(
        prompt="Design a microservice architecture with CI/CD pipeline...",
        domain="code",
        priority="quality",
        context="You are a senior Python engineer.",
    ))
    print(response.model_used)   # claude-opus-4-7
    print(response.output)

asyncio.run(main())
```

### CLI

```bash
python run_router.py \
  --prompt "Refactor this auth module to use JWT" \
  --domain code \
  --priority normal \
  --context "You are a senior Python engineer."
```

Outputs JSON to stdout:

```json
{
  "output": "Here is the refactored auth module...",
  "model_used": "claude-opus-4-7",
  "provider": "anthropic",
  "complexity": {
    "score": 72,
    "tier": "complex",
    "method": "hybrid",
    "rationale": "Rules=65, LLM=80 -> blended=72."
  },
  "latency_ms": 1840.3,
  "input_tokens": 312,
  "output_tokens": 891
}
```

### MCP Server (Claude Code Integration)

ComplexRoute can run as an MCP server, making it a callable tool inside Claude Code.

**Register:**

```bash
# Available in all projects
claude mcp add -s user --transport stdio complex-router -- python /path/to/ComplexityRouter/mcp_server.py

# Available in current project only
claude mcp add --transport stdio complex-router -- python /path/to/ComplexityRouter/mcp_server.py
```

**Verify:**

```bash
claude mcp list
# complex-router: python /path/to/mcp_server.py - ✓ Connected
```

**Tools exposed:**

| Tool | Description |
|------|-------------|
| `evaluate_complexity` | Score a prompt and get a model recommendation — no LLM execution |
| `route_and_execute` | Score, pick the optimal model, execute, return full response |

In a Claude Code session, Claude can call these tools automatically or you can ask it directly.

## Configuration

All tunable values live in `agent_router/config.py`:

| Setting | Default | Purpose |
|---------|---------|---------|
| `AMBIGUOUS_LOW` | `(28, 38)` | Rule score band that triggers LLM fallback |
| `AMBIGUOUS_HIGH` | `(60, 70)` | Rule score band that triggers LLM fallback |
| `RULE_WEIGHT` | `0.4` | Blending weight for rule score |
| `LLM_WEIGHT` | `0.6` | Blending weight for LLM score |
| `TIER_BOUNDS` | `simple: 0-33, medium: 34-66, complex: 67-100` | Score-to-tier mapping |
| `LLM_CLASSIFIER_MODEL` | `claude-haiku-4-5-20251001` | Model used for ambiguous-zone classification |

### Model Registry

Edit `agent_router/models.py` to change which models map to each tier:

```python
MODEL_REGISTRY = {
    "simple":  {"provider": "anthropic", "model_id": "claude-haiku-4-5-20251001", "max_tokens": 2048},
    "medium":  {"provider": "anthropic", "model_id": "claude-sonnet-4-6",          "max_tokens": 4096},
    "complex": {"provider": "anthropic", "model_id": "claude-opus-4-7",            "max_tokens": 8192},
}
```

### Adding a New Provider

1. Subclass `BaseProvider` in `agent_router/providers/`
2. Implement the `execute` method
3. Register it on the router:

```python
from agent_router import AgentRouter
from my_provider import OpenAIProvider

router = AgentRouter()
router.register_provider("openai", OpenAIProvider())
```

## Task Parameters

| Parameter | Values | Default | Effect |
|-----------|--------|---------|--------|
| `prompt` | any string | *(required)* | The task to route |
| `domain` | `"general"`, `"code"`, `"research"` | `"general"` | Code domain gets a +5 bias for complex tasks |
| `priority` | `"speed"`, `"normal"`, `"quality"` | `"normal"` | Speed caps at Sonnet; quality floors at Sonnet |
| `context` | any string | `""` | System prompt passed to the executing model |

## Project Structure

```
ComplexityRouter/
├── agent_router/
│   ├── __init__.py          # Public API
│   ├── schemas.py           # Task, ComplexityScore, ModelSelection, AgentResponse
│   ├── evaluator.py         # Hybrid complexity evaluator
│   ├── models.py            # Model registry (tier -> model mapping)
│   ├── router.py            # Orchestrator
│   ├── config.py            # All tunable thresholds
│   └── providers/
│       ├── base.py          # Abstract provider interface
│       └── anthropic.py     # Anthropic SDK (async, prompt caching)
├── mcp_server.py            # MCP server for Claude Code
├── run_router.py            # CLI entry point
├── examples/
│   ├── simple_usage.py
│   └── code_task.py
├── tests/
│   ├── test_evaluator.py
│   └── test_router.py
├── requirements.txt
└── .env.example
```

## Testing

```bash
pip install pytest
python -m pytest tests/ -v
```

All tests run offline with mock providers — no API key needed.

## Dependencies

- `anthropic>=0.40.0` — async client with prompt caching
- `python-dotenv>=1.0.0` — load API key from `.env`
- `mcp>=1.20.0` — MCP server for Claude Code integration

## License

MIT
