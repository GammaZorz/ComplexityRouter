#!/usr/bin/env python3
"""Example: simple prompt that should route to Haiku."""

import asyncio

from dotenv import load_dotenv

from agent_router import AgentRouter, Task


async def main() -> None:
    load_dotenv()
    router = AgentRouter()

    task = Task(prompt="What is photosynthesis?")
    response = await router.run(task)

    print(f"Model used : {response.model_used}")
    print(f"Provider   : {response.provider}")
    print(f"Tier       : {response.complexity.tier}")
    print(f"Score      : {response.complexity.score}")
    print(f"Method     : {response.complexity.method}")
    print(f"Rationale  : {response.complexity.rationale}")
    print(f"Latency    : {response.latency_ms:.0f} ms")
    print(f"Tokens in  : {response.input_tokens}")
    print(f"Tokens out : {response.output_tokens}")
    print(f"\n--- Response ---\n{response.output}")


if __name__ == "__main__":
    asyncio.run(main())
