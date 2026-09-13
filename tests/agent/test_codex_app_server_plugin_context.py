"""Regression for the Codex app-server model-identity context boundary.

The standard Hermes runtime appends validated ``pre_llm_call`` plugin context
(authoritative model identity, gateway notes, ...) to the outgoing user text,
while the Codex app-server early-return path in ``run_conversation`` dropped
that context entirely. A Codex-routed turn therefore lost the authoritative
"current model" anchor and the model could misreport its own identity from
stale training priors or old session notes (issue-20260903-024940-988cb825).

The fix threads ``plugin_user_context`` through ``run_conversation`` →
``AIAgent._run_codex_app_server_turn`` → ``run_codex_app_server_turn`` and
composes it as a separate leading text item of the Codex ``turn/start`` input.
The context must reach the model for this turn only and must NOT be written to
any persistent store (the messages list itself stays untouched).
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

from agent.codex_runtime import run_codex_app_server_turn


IDENTITY_CONTEXT = (
    "[system] 当前模型: glm-5.3-flash (provider: zai)；如与模型自述冲突，以此为准。"
)


def _make_turn():
    return SimpleNamespace(
        interrupted=False,
        error=None,
        thread_id="thread-ctx-1",
        turn_id="turn-ctx-1",
        projected_messages=[{"role": "assistant", "content": "CTX_OK"}],
        tool_iterations=0,
        final_text="CTX_OK",
        should_retire=False,
    )


def _make_agent():
    agent = MagicMock()
    agent._codex_session = MagicMock()
    agent._codex_session.run_turn.return_value = _make_turn()
    agent.tool_progress_callback = None
    agent._iters_since_skill = 0
    agent._skill_nudge_interval = 0
    agent.valid_tool_names = set()
    agent._session_db = None
    agent._session_db_created = True
    agent.session_id = "sess-identity"
    return agent


def test_plugin_context_composed_into_codex_turn_input():
    """The plugin context must ride the turn input as a leading text item."""
    agent = _make_agent()
    result = run_codex_app_server_turn(
        agent,
        user_message="你是什么模型？",
        original_user_message="你是什么模型？",
        messages=[{"role": "user", "content": "你是什么模型？"}],
        effective_task_id="task-ctx",
        plugin_user_context=IDENTITY_CONTEXT,
    )

    assert result["completed"] is True
    sent = agent._codex_session.run_turn.call_args.kwargs["user_input"]
    assert isinstance(sent, list) and len(sent) == 2
    assert sent[0] == {"type": "text", "text": IDENTITY_CONTEXT}
    assert sent[1]["type"] == "text"
    assert sent[1]["text"] == "你是什么模型？"
    # The context must not be appended to the persisted messages list.
    contents = [m.get("content") for m in result["messages"]]
    assert not any(
        isinstance(c, str) and IDENTITY_CONTEXT in c for c in contents
    ), "plugin context leaked into persisted messages"


def test_plugin_context_prepended_for_multimodal_turns():
    """Image turns keep their native parts; context is prepended, not merged."""
    agent = _make_agent()
    image_part = {"type": "image", "url": "https://example.invalid/pic.png"}
    run_codex_app_server_turn(
        agent,
        user_message=[image_part, {"type": "text", "text": "图里是什么？"}],
        original_user_message="图里是什么？",
        messages=[{"role": "user", "content": "图里是什么？"}],
        effective_task_id="task-ctx-mm",
        plugin_user_context=IDENTITY_CONTEXT,
    )

    sent = agent._codex_session.run_turn.call_args.kwargs["user_input"]
    assert isinstance(sent, list) and len(sent) == 3
    assert sent[0] == {"type": "text", "text": IDENTITY_CONTEXT}
    assert sent[1] == image_part
    assert sent[2] == {"type": "text", "text": "图里是什么？"}


def test_no_plugin_context_keeps_plain_user_message():
    """Without plugin context the wire format must stay byte-compatible."""
    agent = _make_agent()
    run_codex_app_server_turn(
        agent,
        user_message="plain hello",
        original_user_message="plain hello",
        messages=[{"role": "user", "content": "plain hello"}],
        effective_task_id="task-plain",
    )

    sent = agent._codex_session.run_turn.call_args.kwargs["user_input"]
    assert sent == "plain hello"
