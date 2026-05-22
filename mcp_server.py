#!/usr/bin/env python3
"""
MCP server that exposes ComplexRoute to Claude Code.

Claude Code can call two tools:
  - evaluate_complexity: Score a prompt and get a model recommendation (no LLM execution)
  - route_and_execute:   Score + execute the prompt on the optimal model, return the full response

Register with Claude Code:
  claude mcp add --transport stdio complex-router -- python C:/repositories/ComplexRoute/mcp_server.py
"""

from __future__ import annotations

import asyncio
import json
import sys
import os

# Ensure the agent_router package is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from agent_router import AgentRouter, Task
from agent_router.evaluator import ComplexityEvaluator
from agent_router.models import select_model

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    "complex-router",
    instructions="Routes prompts to the optimal Claude model based on hybrid complexity analysis.",
)

# Shared instances (created once, reused across calls)
_evaluator = ComplexityEvaluator()
_router = AgentRouter()


@mcp.tool()
async def evaluate_complexity(
    prompt: str,
    domain: str = "general",
    priority: str = "normal",
) -> str:
    """
    Evaluate the complexity of a prompt and recommend which model to use.
    Does NOT execute the prompt — just returns the routing recommendation.

    Args:
        prompt: The task or question to evaluate.
        domain: "general", "code", or "research".
        priority: "speed" (cap at Sonnet), "normal", or "quality" (floor at Sonnet).

    Returns:
        JSON with score (0-100), tier (simple/medium/complex),
        recommended model, and rationale.
    """
    task = Task(prompt=prompt, domain=domain, priority=priority)
    complexity = await _evaluator.evaluate(task)
    selection = select_model(complexity)

    return json.dumps(
        {
            "score": complexity.score,
            "tier": complexity.tier,
            "method": complexity.method,
            "rationale": complexity.rationale,
            "recommended_model": selection.model_id,
            "provider": selection.provider,
            "max_tokens": selection.max_tokens,
        },
        indent=2,
    )


@mcp.tool()
async def route_and_execute(
    prompt: str,
    domain: str = "general",
    priority: str = "normal",
    context: str = "",
) -> str:
    """
    Evaluate complexity, select the optimal model, execute the prompt,
    and return the full response with routing metadata.

    Args:
        prompt: The task or question to route and execute.
        domain: "general", "code", or "research".
        priority: "speed" (cap at Sonnet), "normal", or "quality" (floor at Sonnet).
        context: Optional system prompt or background information.

    Returns:
        JSON with the model's response, which model was used,
        complexity score/tier, latency, and token usage.
    """
    task = Task(prompt=prompt, domain=domain, priority=priority, context=context)
    response = await _router.run(task)

    return json.dumps(response.to_dict(), indent=2)


# ---------------------------------------------------------------------------
# Entry point — stdio transport for Claude Code
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    mcp.run(transport="stdio")
