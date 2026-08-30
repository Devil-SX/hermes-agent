"""Persist the mapping Hermes session_id -> codex app-server thread id.

The codex app-server runtime keeps the whole conversation inside the codex
thread (server-side, in the rollout file under CODEX_HOME). Hermes' cached
AIAgent — and with it the ``CodexAppServerSession`` holding the thread — is
routinely destroyed while the conversation continues: gateway agent-cache
eviction (idle TTL, LRU cap, cross-process coherence guard), gateway
restarts, profile reloads. The chat-completions runtime survives that
because the gateway replays the persisted transcript every turn; the codex
runtime only forwards the current user message, so a fresh ``thread/start``
after every rebuild meant the model saw *only* the latest message.

This tiny JSON registry lets a newly spawned ``CodexAppServerSession``
resume the codex thread the previous agent instance was using
(``thread/resume``), restoring full history. It is deliberately keyed by
Hermes *session_id* (stable across agent rebuilds; changes on /new) and
stored under ``get_hermes_home()`` so every multiplex profile gets its own
file.

Failure mode is fail-open: a missing/corrupt registry or a rejected resume
simply starts a fresh thread, and ``run_codex_app_server_turn`` injects a
bounded recent-history block into that thread's first turn instead.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_SCHEMA = "hermes-codex-thread-registry/1"
_REGISTRY_FILENAME = "codex_app_server_threads.json"
# Keep the file tiny — it exists to serve recent, still-active sessions.
_MAX_ENTRIES = 100

_lock = threading.Lock()


def _registry_path() -> Path:
    from hermes_constants import get_hermes_home

    return get_hermes_home() / _REGISTRY_FILENAME


def _load(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(data, dict) or data.get("schema") != _SCHEMA:
        return {}
    entries = data.get("entries")
    return entries if isinstance(entries, dict) else {}


def lookup_thread_id(session_id: Optional[str]) -> Optional[str]:
    """Return the last codex thread id recorded for this Hermes session."""
    if not session_id:
        return None
    path = _registry_path()
    with _lock:
        entry = _load(path).get(session_id)
    if not isinstance(entry, dict):
        return None
    thread_id = entry.get("thread_id")
    if not isinstance(thread_id, str) or not thread_id.strip():
        return None
    return thread_id.strip()


def record_thread_id(session_id: Optional[str], thread_id: Optional[str]) -> None:
    """Record the codex thread id a Hermes session is currently using.

    Read-merge-write under a process lock with an atomic replace; the
    worst case of a cross-process race is a lost resume pointer, which the
    fresh-thread history fallback in ``run_codex_app_server_turn`` covers.
    """
    if not session_id or not thread_id:
        return
    path = _registry_path()
    with _lock:
        entries = _load(path)
        entries[session_id] = {
            "thread_id": thread_id,
            "updated_at": time.time(),
        }
        if len(entries) > _MAX_ENTRIES:
            ordered = sorted(
                entries.items(),
                key=lambda kv: float(kv[1].get("updated_at") or 0.0),
            )
            entries = dict(ordered[-_MAX_ENTRIES:])
        payload = {"schema": _SCHEMA, "entries": entries}
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            fd, tmp_name = tempfile.mkstemp(
                dir=str(path.parent), prefix=".codex-threads-", suffix=".tmp"
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh)
                os.replace(tmp_name, path)
            except BaseException:
                try:
                    os.unlink(tmp_name)
                except OSError:
                    pass
                raise
        except OSError:
            logger.debug("codex thread registry write failed", exc_info=True)
