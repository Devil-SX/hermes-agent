"""Direct Codex clients must honor an operator-pinned app-server runtime."""

from unittest.mock import patch

from agent import auxiliary_client as auxiliary


def test_direct_codex_client_is_blocked_before_auth_resolution() -> None:
    config = {
        "model": {
            "openai_runtime": "codex_app_server",
            "openai_runtime_policy": "codex_app_server",
        }
    }

    with patch("hermes_cli.config.load_config_readonly", return_value=config), patch.object(
        auxiliary, "_build_codex_client"
    ) as build:
        client, model = auxiliary.resolve_provider_client(
            "openai-codex", "gpt-5.6-sol"
        )

    assert (client, model) == (None, None)
    build.assert_not_called()


def test_unmanaged_hermes_keeps_backward_compatible_direct_codex_client() -> None:
    sentinel = object()

    with patch("hermes_cli.config.load_config_readonly", return_value={}), patch.object(
        auxiliary,
        "_build_codex_client",
        return_value=(sentinel, "gpt-5.6-sol"),
    ):
        client, model = auxiliary.resolve_provider_client(
            "openai-codex", "gpt-5.6-sol"
        )

    assert client is sentinel
    assert model == "gpt-5.6-sol"
