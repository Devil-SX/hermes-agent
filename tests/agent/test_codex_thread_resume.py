"""Tests for codex app-server thread resume + fresh-thread history fallback.

Behavior contracts covered:

* ``agent.codex_thread_registry`` round-trips a session_id -> thread_id
  mapping under the active HERMES_HOME and tolerates corrupt state.
* ``CodexAppServerSession.ensure_started`` resumes a recorded thread and
  falls back to ``thread/start`` when the resume is rejected.
* ``_build_fresh_thread_context_prefix`` produces a bounded, tool-free
  history block that excludes the current (last) user turn.
* ``AIAgent.release_clients`` closes the codex app-server session so cache
  eviction cannot leak the subprocess or its thread writer lock.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from agent.codex_thread_registry import lookup_thread_id, record_thread_id
from agent.codex_runtime import _build_fresh_thread_context_prefix
from agent.transports.codex_app_server import CodexAppServerError
from agent.transports.codex_app_server_session import CodexAppServerSession


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_registry_roundtrip(tmp_path):
    record_thread_id("sess-a", "thread-123")
    assert lookup_thread_id("sess-a") == "thread-123"
    assert lookup_thread_id("sess-b") is None
    assert lookup_thread_id(None) is None
    assert lookup_thread_id("") is None


def test_registry_overwrites_and_tolerates_corruption(tmp_path):
    record_thread_id("sess-a", "thread-old")
    record_thread_id("sess-a", "thread-new")
    assert lookup_thread_id("sess-a") == "thread-new"

    from agent.codex_thread_registry import _registry_path

    _registry_path().write_text("not json", encoding="utf-8")
    assert lookup_thread_id("sess-a") is None


# ---------------------------------------------------------------------------
# ensure_started resume behavior
# ---------------------------------------------------------------------------


class _FakeClient:
    """Minimal stand-in for CodexAppServerClient."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.requests: list[tuple[str, dict]] = []
        self.resume_error: Exception | None = None
        self.initialized = False

    def initialize(self, **kwargs: Any) -> None:
        self.initialized = True

    def request(self, method: str, params: dict, timeout: float = 0) -> dict:
        self.requests.append((method, params))
        if method == "thread/resume":
            if self.resume_error is not None:
                raise self.resume_error
            return {"thread": {"id": params["threadId"]}}
        if method == "thread/start":
            return {"thread": {"id": "fresh-thread-id"}}
        raise AssertionError(f"unexpected request {method}")

    def close(self) -> None:  # pragma: no cover - trivial
        pass


def _make_session(client: _FakeClient, **kwargs: Any) -> CodexAppServerSession:
    holder: dict[str, _FakeClient] = {}

    def factory(**factory_kwargs: Any) -> _FakeClient:
        holder["client"] = client
        return client

    session = CodexAppServerSession(
        cwd="/tmp", client_factory=factory, **kwargs
    )
    session._test_client_holder = holder  # type: ignore[attr-defined]
    return session


def test_ensure_started_resumes_recorded_thread():
    client = _FakeClient()
    session = _make_session(client, resume_thread_id="thread-abc")

    thread_id = session.ensure_started()

    assert thread_id == "thread-abc"
    assert session.thread_resumed is True
    methods = [m for m, _ in client.requests]
    assert methods == ["thread/resume"]
    assert client.requests[0][1] == {"threadId": "thread-abc"}


def test_ensure_started_falls_back_to_thread_start_on_resume_reject():
    client = _FakeClient()
    client.resume_error = CodexAppServerError(code=-32600, message="no rollout found")
    session = _make_session(client, resume_thread_id="thread-gone")

    thread_id = session.ensure_started()

    assert thread_id == "fresh-thread-id"
    assert session.thread_resumed is False
    methods = [m for m, _ in client.requests]
    assert methods == ["thread/resume", "thread/start"]


def test_ensure_started_without_resume_pointer_starts_fresh():
    client = _FakeClient()
    session = _make_session(client)

    thread_id = session.ensure_started()

    assert thread_id == "fresh-thread-id"
    assert session.thread_resumed is False
    assert [m for m, _ in client.requests] == ["thread/start"]


def test_resume_pointer_consumed_after_first_attempt():
    """A retired session must not retry a stale resume id on respawn."""
    client = _FakeClient()
    client.resume_error = CodexAppServerError(code=-32600, message="stale")
    session = _make_session(client, resume_thread_id="thread-stale")

    session.ensure_started()
    session.close()
    session._closed = False  # simulate reuse of the adapter object
    client.requests.clear()

    session.ensure_started()
    assert [m for m, _ in client.requests] == ["thread/start"]


# ---------------------------------------------------------------------------
# Fresh-thread history prefix
# ---------------------------------------------------------------------------


def test_fresh_thread_prefix_excludes_current_turn_and_tools():
    messages = [
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "tool", "content": "tool output must not appear"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "UNIQUE_CURRENT_TURN_TEXT"},
    ]
    prefix = _build_fresh_thread_context_prefix(messages)

    assert "first question" in prefix
    assert "first answer" in prefix
    assert "tool output must not appear" not in prefix
    assert "UNIQUE_CURRENT_TURN_TEXT" not in prefix
    assert prefix.endswith("[Current message]\n")
    # Oldest-first ordering.
    assert prefix.index("first question") < prefix.index("first answer")


def test_fresh_thread_prefix_empty_without_history():
    assert _build_fresh_thread_context_prefix([]) == ""
    assert _build_fresh_thread_context_prefix([{"role": "user", "content": "hi"}]) == ""


def test_fresh_thread_prefix_respects_char_budget():
    messages = [
        {"role": "user", "content": "x" * 1000},
        {"role": "assistant", "content": "y" * 9000},
        {"role": "user", "content": "current"},
    ]
    prefix = _build_fresh_thread_context_prefix(messages)
    # Per-message truncation applies.
    assert "y" * 9000 not in prefix
    assert len(prefix) < 7000


# ---------------------------------------------------------------------------
# release_clients closes the codex session
# ---------------------------------------------------------------------------


def test_release_clients_closes_codex_session():
    from run_agent import AIAgent

    agent = AIAgent.__new__(AIAgent)
    agent._active_children_lock = threading.Lock()
    agent._active_children = set()
    agent.client = None
    codex_session = MagicMock()
    agent._codex_session = codex_session

    agent.release_clients()

    codex_session.close.assert_called_once()
    assert agent._codex_session is None
