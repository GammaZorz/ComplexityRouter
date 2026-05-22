#!/usr/bin/env python3
"""CLI entry point — routes a prompt and prints JSON to stdout."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys

from dotenv import load_dotenv

from agent_router import AgentRouter, Task


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="ComplexRoute — route a prompt to the optimal Claude model.",
    )
    p.add_argument("--prompt", required=True, help="The task or question to route.")
    p.add_argument("--domain", default="general", choices=["general", "code", "research"])
    p.add_argument("--priority", default="normal", choices=["speed", "normal", "quality"])
    p.add_argument("--context", default="", help="Optional system prompt / background.")
    return p.parse_args()


async def main() -> None:
    load_dotenv()
    args = parse_args()

    task = Task(
        prompt=args.prompt,
        context=args.context,
        domain=args.domain,
        priority=args.priority,
    )

    router = AgentRouter()
    response = await router.run(task)

    # Structured JSON output
    print(json.dumps(response.to_dict(), indent=2))


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
