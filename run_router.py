#!/usr/bin/env python3
"""CLI entry point — routes a prompt and prints JSON to stdout."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import os

from dotenv import load_dotenv

from agent_router import AgentRouter, Task
from agent_router.settings import get_settings


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ComplexRoute v2 — route a prompt to the optimal model.",
    )
    p.add_argument("--prompt",   required=True, help="The task or question to route.")
    p.add_argument("--domain",   default="general",      choices=["general", "code", "research"])
    p.add_argument("--priority", default="normal",       choices=["speed", "normal", "quality"])
    p.add_argument("--context",  default="",             help="Optional system prompt.")
    p.add_argument("--mode",     default="offline_first",
                   choices=["offline", "offline_first", "caller_llm", "always_llm"],
                   help="Evaluation mode (overrides config file).")
    p.add_argument("--config",   default=None,           help="Path to complexity_router.json.")
    return p.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    settings = get_settings(config_path=args.config, reload=True)

    task = Task(
        prompt=args.prompt,
        context=args.context,
        domain=args.domain,
        priority=args.priority,
        mode=args.mode,
    )

    router = AgentRouter(settings=settings)
    response = await router.run(task)
    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
