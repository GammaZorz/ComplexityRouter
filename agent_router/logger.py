"""JSONL request logger — one line per routed prompt."""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .schemas import AgentResponse, Task
    from .settings import LoggingSettings


class RouterLogger:
    """Appends one JSONL entry per routed request to a log file."""

    def __init__(self, cfg: "LoggingSettings") -> None:
        self._cfg = cfg
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------

    async def log(self, task: "Task", response: "AgentResponse") -> None:
        """Write a log entry. Silently skips if logging is disabled."""
        if not self._cfg.enabled:
            return

        entry = self._build_entry(task, response)
        line = json.dumps(entry, ensure_ascii=False)

        log_path = Path(self._cfg.file)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path

        async with self._lock:
            await asyncio.get_event_loop().run_in_executor(
                None, self._append, log_path, line
            )

    @staticmethod
    def _append(path: Path, line: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def _build_entry(self, task: "Task", response: "AgentResponse") -> dict:
        entry: dict = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "score": response.complexity.score,
            "tier": response.complexity.tier,
            "method": response.complexity.method,
            "model": response.model_used,
            "provider": response.provider,
            "latency_ms": response.latency_ms,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "domain": task.domain,
            "priority": task.priority,
            "mode": task.mode,
        }
        if self._cfg.include_prompt:
            entry["prompt"] = task.prompt[:500]  # cap at 500 chars
        if self._cfg.include_response:
            entry["response"] = response.output[:1000]
        return entry
