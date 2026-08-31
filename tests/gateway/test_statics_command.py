"""Behavioral coverage for the gateway-owned /statics read path."""

from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_statics_uses_async_session_db_facade():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_db = MagicMock()
    runner._session_db.session_usage_breakdown = AsyncMock(
        return_value=[
            {
                "p": "openai-codex",
                "m": "gpt-5.6-sol",
                "t": "Research",
                "chat_type": "group",
                "it": 1200,
                "ot": 300,
                "crt": 400,
                "calls": 3,
                "n": 1,
            }
        ]
    )
    runner._async_session_store = AsyncMock()
    runner._async_session_store.get_or_create_session.return_value = MagicMock(
        session_id="session-1"
    )
    runner._async_session_store.load_transcript.return_value = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]
    event = MagicMock()
    event.get_command_args.return_value = "7"

    result = await runner._handle_statics_command(event)

    runner._session_db.session_usage_breakdown.assert_awaited_once()
    assert "1.5K" in result
    assert "Research" in result
    assert "openai-codex/gpt-5.6-sol" in result


@pytest.mark.asyncio
async def test_statics_empty_result_preserves_requested_window():
    from gateway.run import GatewayRunner

    runner = object.__new__(GatewayRunner)
    runner._session_db = MagicMock()
    runner._session_db.session_usage_breakdown = AsyncMock(return_value=[])
    event = MagicMock()
    event.get_command_args.return_value = "30"

    assert await runner._handle_statics_command(event) == "📊 _近 30 天暂无用量数据_"
