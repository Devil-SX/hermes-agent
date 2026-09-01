"""Codex app-server turns retain the normal Telegram typing lifecycle.

The app-server runtime returns early from the synchronous conversation loop,
but a messaging turn is wrapped by ``BasePlatformAdapter`` before runtime
selection.  These tests keep the real Codex runtime path blocked on a fake
app-server session and assert that the outer Telegram heartbeat remains live
and is cleaned up on every terminal path.
"""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.codex_runtime import run_codex_app_server_turn
from gateway.config import Platform, PlatformConfig
from gateway.platforms.base import MessageEvent, MessageType, SendResult
from gateway.session import SessionSource, build_session_key
from plugins.platforms.telegram.adapter import TelegramAdapter


CHAT_ID = "-1001234567890"


def _turn(*, interrupted: bool = False):
    return SimpleNamespace(
        interrupted=interrupted,
        error=None,
        thread_id="thread-1",
        turn_id="turn-1",
        projected_messages=[],
        tool_iterations=0,
        final_text="" if interrupted else "done",
        should_retire=False,
        error_code=None,
        error_http_status=None,
        error_retryable=None,
    )


def _agent(session, *, interrupted: bool = False):
    agent = MagicMock()
    agent._codex_session = session
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent.session_id = "session-1"
    agent._interrupt_requested = interrupted
    agent._interrupt_message = "stop" if interrupted else None

    def clear_interrupt():
        agent._interrupt_requested = False
        agent._interrupt_message = None

    agent.clear_interrupt.side_effect = clear_interrupt
    return agent


def _event() -> MessageEvent:
    return MessageEvent(
        text="work",
        message_type=MessageType.TEXT,
        source=SessionSource(
            platform=Platform.TELEGRAM,
            chat_id=CHAT_ID,
            chat_type="group",
            profile="telegram-deadbeefdeadbeef",
        ),
    )


def _make_adapter(monkeypatch, outcome: str):
    adapter = TelegramAdapter(PlatformConfig(enabled=True, token="test-token"))
    adapter._bot = AsyncMock()
    adapter._send_with_retry = AsyncMock(
        return_value=SendResult(success=True, message_id="reply-1")
    )

    started = threading.Event()
    release = threading.Event()
    session = MagicMock()

    def run_turn(*, user_input):
        started.set()
        assert release.wait(timeout=2.0)
        if outcome == "failure":
            raise RuntimeError("app-server failed")
        return _turn(interrupted=outcome == "interruption")

    session.run_turn.side_effect = run_turn
    agent = _agent(session, interrupted=outcome == "interruption")
    results = []

    async def handle(_message):
        result = await asyncio.to_thread(
            run_codex_app_server_turn,
            agent,
            user_message="work",
            original_user_message="work",
            messages=[{"role": "user", "content": "work"}],
            effective_task_id="task-1",
        )
        results.append(result)
        return result["final_response"]

    adapter._message_handler = handle

    active_loops = 0
    max_active_loops = 0

    async def fast_keep_typing(chat_id, metadata=None, stop_event=None):
        nonlocal active_loops, max_active_loops
        active_loops += 1
        max_active_loops = max(max_active_loops, active_loops)
        try:
            await TelegramAdapter._keep_typing(
                adapter,
                chat_id,
                interval=0.01,
                metadata=metadata,
                stop_event=stop_event,
            )
        finally:
            active_loops -= 1

    monkeypatch.setattr(adapter, "_keep_typing", fast_keep_typing)
    return (
        adapter,
        started,
        release,
        lambda: (active_loops, max_active_loops),
        results,
    )


async def _wait_for_thread(event: threading.Event) -> None:
    for _ in range(100):
        if event.is_set():
            return
        await asyncio.sleep(0.005)
    pytest.fail("fake Codex app-server turn did not start")


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "failure", "interruption"])
async def test_codex_app_server_turn_keeps_one_telegram_typing_loop(
    monkeypatch, outcome
):
    adapter, started, release, loop_counts, results = _make_adapter(
        monkeypatch, outcome
    )
    event = _event()
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    task = asyncio.create_task(adapter._process_message_background(event, session_key))
    await _wait_for_thread(started)
    await asyncio.sleep(0.04)

    assert adapter._bot.send_chat_action.await_count >= 2
    assert loop_counts() == (1, 1)

    release.set()
    await asyncio.wait_for(task, timeout=2.0)

    assert len(results) == 1
    if outcome == "success":
        assert results[0]["completed"] is True
    elif outcome == "failure":
        assert results[0]["completed"] is False
        assert results[0]["error"] == "app-server failed"
    else:
        assert results[0]["interrupted"] is True

    calls_after_cleanup = adapter._bot.send_chat_action.await_count
    await asyncio.sleep(0.04)
    assert adapter._bot.send_chat_action.await_count == calls_after_cleanup
    assert loop_counts() == (0, 1)


@pytest.mark.asyncio
async def test_codex_app_server_turn_cancellation_stops_telegram_typing(monkeypatch):
    adapter, started, release, loop_counts, _results = _make_adapter(
        monkeypatch, "success"
    )
    event = _event()
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    task = asyncio.create_task(adapter._process_message_background(event, session_key))
    await _wait_for_thread(started)
    await asyncio.sleep(0.03)
    assert adapter._bot.send_chat_action.await_count >= 1

    task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task

    calls_after_cleanup = adapter._bot.send_chat_action.await_count
    await asyncio.sleep(0.04)
    assert adapter._bot.send_chat_action.await_count == calls_after_cleanup
    assert loop_counts() == (0, 1)


@pytest.mark.asyncio
async def test_codex_app_server_typing_reuses_telegram_failure_cooldown(monkeypatch):
    adapter, started, release, _loop_counts, results = _make_adapter(
        monkeypatch, "success"
    )
    adapter._telegram_typing_cooldown_seconds = 30.0
    adapter._bot.send_chat_action = AsyncMock(
        side_effect=OSError("temporary Telegram network failure")
    )
    event = _event()
    session_key = build_session_key(event.source)
    adapter._active_sessions[session_key] = asyncio.Event()

    task = asyncio.create_task(adapter._process_message_background(event, session_key))
    await _wait_for_thread(started)
    await asyncio.sleep(0.05)

    assert adapter._bot.send_chat_action.await_count == 1

    release.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert results[0]["completed"] is True
