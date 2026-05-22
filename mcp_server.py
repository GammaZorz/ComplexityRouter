#!/usr/bin/env python3
"""
MCP server that exposes ComplexRoute v2 to Claude Code.

Tools:
  evaluate_complexity  — score a prompt and get a model recommendation (no execution)
  route_and_execute    — score + execute on the optimal model, return full response

The evaluator is given the MCP Context so it can attempt MCP sampling
(asking the calling Claude Code instance to classify complexity without
needing a separate API key). Claude Code does not yet support sampling
(issue #1785) so it falls back to a direct API call automatically.

Register with Claude Code:
  claude mcp add -s user --transport stdio complex-router -- python C:/repositories/ComplexRoute/mcp_server.py
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from mcp.server.fastmcp import Context, FastMCP

from agent_router import AgentRouter, Task
from agent_router.evaluator import ComplexityEvaluator
from agent_router.models import select_model
from agent_router.settings import get_settings

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

mcp = FastMCP(
    "complex-router",
    instructions=(
        "Routes prompts to the optimal Claude model based on hybrid complexity analysis. "
        "Use evaluate_complexity to get a routing recommendation, or route_and_execute "
        "to route and run the prompt in one call."
    ),
)

_settings = get_settings()


@mcp.tool()
async def evaluate_complexity(
    prompt: str,
    ctx: Context,
    domain: str = "general",
    priority: str = "normal",
    mode: str = "offline_first",
) -> str:
    """
    Evaluate the complexity of a prompt and recommend which model to use.
    Does NOT execute the prompt — just returns the routing recommendation.

    The evaluator will attempt to use the calling Claude Code session for
    classification (no API key needed) if mode is 'caller_llm', falling
    back to a direct API call if MCP sampling is unavailable.

    Args:
        prompt:   The task or question to evaluate.
        domain:   "general" | "code" | "research"
        priority: "speed" (cap at Sonnet) | "normal" | "quality" (floor at Sonnet)
        mode:     "offline" | "offline_first" | "caller_llm" | "always_llm"
    """
    task = Task(prompt=prompt, domain=domain, priority=priority, mode=mode)
    # Pass ctx so the evaluator can attempt MCP sampling
    evaluator = ComplexityEvaluator(ctx=ctx, settings=_settings)
    complexity = await evaluator.evaluate(task)
    selection = select_model(complexity, _settings)

    return json.dumps({
        "score": complexity.score,
        "tier": complexity.tier,
        "method": complexity.method,
        "rationale": complexity.rationale,
        "recommended_model": selection.model_id,
        "provider": selection.provider,
        "max_tokens": selection.max_tokens,
    }, indent=2)


@mcp.tool()
async def route_and_execute(
    prompt: str,
    ctx: Context,
    domain: str = "general",
    priority: str = "normal",
    context: str = "",
    mode: str = "offline_first",
) -> str:
    """
    Evaluate complexity, select the optimal model, execute the prompt,
    and return the full response with routing metadata.

    Args:
        prompt:   The task or question to route and execute.
        domain:   "general" | "code" | "research"
        priority: "speed" | "normal" | "quality"
        context:  Optional system prompt or background information.
        mode:     "offline" | "offline_first" | "caller_llm" | "always_llm"
    """
    task = Task(prompt=prompt, domain=domain, priority=priority, context=context, mode=mode)
    evaluator = ComplexityEvaluator(ctx=ctx, settings=_settings)
    router = AgentRouter(evaluator=evaluator, settings=_settings)
    response = await router.run(task)
    return json.dumps(response.to_dict(), indent=2)


if __name__ == "__main__":
    mcp.run(transport="stdio")
