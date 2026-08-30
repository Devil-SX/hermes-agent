"""Gateway behavior tests for durable Codex thread restoration."""

from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
async def test_run_agent_restores_codex_thread_from_session_metadata():
    runner = object.__new__(GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    backing_store = object()
    runner.session_store = backing_store

    class AsyncStore:
        _store = backing_store

        async def get_session_metadata(self, session_key, key):
            assert session_key == "agent:main:telegram:thread:-1001:42"
            assert key == "codex_thread_id"
            return "thread-from-state-db"

    runner._async_session_store = AsyncStore()
    captured = {}

    async def fake_inner(*args, **kwargs):
        captured.update(kwargs)
        return {"final_response": "ok"}

    runner._run_agent_inner = fake_inner
    source = SessionSource(
        platform=Platform.TELEGRAM,
        chat_id="-1001",
        chat_type="thread",
        thread_id="42",
    )

    result = await runner._run_agent(
        "hello",
        "",
        [],
        source,
        "session-1",
        session_key="agent:main:telegram:thread:-1001:42",
    )

    assert result == {"final_response": "ok"}
    assert captured["codex_resume_thread_id"] == "thread-from-state-db"
