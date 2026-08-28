"""Herd plugin: expose /herd as a native Hermes slash command.

Telegram owner-gate: the gateway only delivers slash commands from the
allowlisted owner chat, same as every other Hermes command.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

HERD_SCRIPT = Path.home() / ".hermes/scripts/herd.py"
PY = Path.home() / ".hermes/hermes-agent/venv/bin/python"
_HERD_TIMEOUT = 60


def _run_list_json() -> list[dict] | None:
    try:
        proc = subprocess.run(
            [str(PY), str(HERD_SCRIPT), "list", "--json"],
            capture_output=True, text=True, timeout=_HERD_TIMEOUT,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        rows = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return None
    return rows if isinstance(rows, list) else None


def _fmt_elapsed(iso_at: str, now: float) -> str:
    try:
        from datetime import datetime
        started = datetime.fromisoformat(iso_at).timestamp()
        mins = max(0, int((now - started) // 60))
        if mins >= 1440:
            return f"{mins // 1440}d"
        if mins >= 60:
            return f"{mins // 60}h{mins % 60:02d}m"
        return f"{mins}m"
    except (ValueError, TypeError, OSError):
        return "?"


def render_card(rows: list[dict]) -> str:
    if not rows:
        return (
            " pasture is empty — 0 active · 0 archived · pool 30/30 free\n"
            "派活：说「派个 agent 去 <任务>」"
        )
    icon = {"working": "●", "idle": "○", "done": "○",
            "blocked": "⚠️", "unknown": "·", "gone": "·", "archived": "✓"}
    now = time.time()
    lines: list[str] = []
    active = archived = 0
    for row in rows:
        status = row.get("status", "unknown")
        emoji = row.get("emoji") or "生生"
        name = row.get("name", "?")
        ident = row.get("identity", "")
        task = (row.get("task") or "")[:36]
        suffix = ""
        if status == "working":
            elapsed = _fmt_elapsed(row.get("spawned_at", ""), now)
            suffix = f" · {elapsed}"
        elif status == "archived":
            suffix = ""
        lines.append(
            f"{emoji} {icon.get(status, '·')} {name} · {ident} — {task} · {status}{suffix}"
        )
        if status == "archived":
            archived += 1
        else:
            active += 1
    pool_free = 30 - (active + archived)
    lines.append(f"──────────────")
    lines.append(f"{active} active · {archived} archived · pool {pool_free}/30 free")
    return "\n".join(lines)


def _handle(argv: list[str]) -> str:
    rows = _run_list_json()
    if rows is None:
        return (
            "herd 不可用：herdr server 未运行或脚本执行失败。\n"
            "修复：systemctl --user start herdr-server"
        )
    return render_card(rows)


def register(ctx) -> None:
    ctx.register_command(
        "herd",
        handler=_handle,
        description="Headless agent fleet status (herdr + pi workers).",
    )
