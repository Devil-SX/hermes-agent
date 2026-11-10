"""Native runtime resolution must not touch the independent HTTP OAuth store."""
from unittest.mock import patch

import pytest


def test_real_resolution_uses_native_owner_without_http_auth(tmp_path, monkeypatch):
    from hermes_cli.config import save_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    save_config({'model': {'provider': 'openai-codex',
                          'openai_runtime': 'codex_app_server'}})
    with patch('hermes_cli.runtime_provider.resolve_codex_runtime_credentials',
               side_effect=AssertionError('HTTP OAuth must not run')):
        result = resolve_runtime_provider(requested='openai-codex')
    assert result['api_mode'] == 'codex_app_server'
    assert result['source'] == 'codex_app_server'
    assert result['base_url'] == 'http://127.0.0.1:1'
    assert not (tmp_path / 'auth.json').exists()


def test_native_selection_still_honors_disabled_provider(tmp_path, monkeypatch):
    from hermes_cli.config import save_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    save_config({'model': {'provider': 'openai-codex',
                          'openai_runtime': 'codex_app_server'},
                 'providers': {'openai-codex': {'enabled': False}}})
    with pytest.raises(ValueError, match='disabled'):
        resolve_runtime_provider(requested='openai-codex')
