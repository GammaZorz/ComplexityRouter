#!/usr/bin/env python3
"""Example: complex code prompt that should route to Opus."""

import asyncio

from dotenv import load_dotenv

from agent_router import AgentRouter, Task


async def main() -> None:
    load_dotenv()
    router = AgentRouter()

    task = Task(
        prompt=(
            "I have a Django REST API with the following models:\n\n"
            "```python\n"
            "class Order(models.Model):\n"
            "    customer = models.ForeignKey(User, on_delete=models.CASCADE)\n"
            "    total = models.DecimalField(max_digits=10, decimal_places=2)\n"
            "    status = models.CharField(max_length=20)\n"
            "    created_at = models.DateTimeField(auto_now_add=True)\n"
            "```\n\n"
            "The API is running slow on the /orders/ endpoint. There are 2M rows.\n"
            "1. Why is the list endpoint slow?\n"
            "2. How should I add database indexing?\n"
            "3. Design a caching strategy with Redis.\n"
            "4. Refactor the serializer for N+1 query elimination.\n"
            "5. Add pagination and filtering.\n"
        ),
        domain="code",
        priority="quality",
        context="You are a senior Django/Python engineer. Be specific and provide code.",
    )

    response = await router.run(task)

    print(f"Model used : {response.model_used}")
    print(f"Tier       : {response.complexity.tier}")
    print(f"Score      : {response.complexity.score}")
    print(f"Method     : {response.complexity.method}")
    print(f"Latency    : {response.latency_ms:.0f} ms")
    print(f"\n--- Response ---\n{response.output}")


if __name__ == "__main__":
    asyncio.run(main())
