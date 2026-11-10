"""Real HTTP requests through a configured, revocable authorization provider."""
import asyncio
import sys
import threading
import types

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_provider_revocation_outage_and_nonblocking_identity_io(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('API_SERVER_KEY', raising=False)
    state = {'allowed': True, 'outage': False}
    release = threading.Event()
    main_thread = threading.get_ident()
    calls = []

    def authorize(**request):
        assert threading.get_ident() != main_thread
        assert release.wait(2), 'Identity I/O blocked the Gateway event loop'
        calls.append(request)
        if state['outage']:
            raise OSError('synthetic authority outage')
        if not state['allowed'] or request['authorization'] != 'Bearer fixture-key':
            return None
        return {'principal_id': 'account-a', 'caller': {'account_id': 'account-a', 'key_id': 'key-a'},
                'operations': {'status': {'method': 'GET', 'path': '/v1/health'}}}

    module = types.ModuleType('fixture_api_authorizer')
    module.authorize = authorize
    monkeypatch.setitem(sys.modules, module.__name__, module)
    adapter = APIServerAdapter(PlatformConfig(enabled=True, extra={
        'key': '', 'authorization_provider': 'fixture_api_authorizer:authorize'}))
    assert adapter._api_key_passes_startup_guard()
    app = web.Application(middlewares=[adapter._make_profile_prefix_middleware()])
    app.router.add_get('/v1/health', adapter._handle_health)
    app.router.add_get('/v1/capabilities', adapter._handle_capabilities)
    asyncio.get_running_loop().call_later(0.01, release.set)
    async with TestClient(TestServer(app)) as client:
        headers = {'Authorization': 'Bearer fixture-key'}
        response = await client.get('/v1/health', headers=headers)
        assert response.status == 200
        response = await client.get('/v1/capabilities', headers=headers)
        assert response.status == 200
        assert (await response.json())['auth']['caller']['account_id'] == 'account-a'
        state['allowed'] = False
        assert (await client.get('/v1/health', headers=headers)).status == 401
        state['outage'] = True
        assert (await client.get('/v1/health', headers=headers)).status == 503
    assert len(calls) == 4


def test_ambiguous_or_missing_provider_never_falls_back(tmp_path, monkeypatch):
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    with pytest.raises(ValueError, match='either'):
        APIServerAdapter(PlatformConfig(enabled=True, extra={
            'key': 'legacy-key', 'authorization_provider': 'absent_provider:authorize'}))
    with pytest.raises(ModuleNotFoundError):
        APIServerAdapter(PlatformConfig(enabled=True, extra={
            'key': '', 'authorization_provider': 'absent_provider:authorize'}))


def test_yaml_provider_is_enrolled_without_a_shared_key(tmp_path, monkeypatch):
    from gateway.config import Platform, load_gateway_config

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.delenv('API_SERVER_KEY', raising=False)
    (tmp_path / 'config.yaml').write_text(
        'platforms:\n  api_server:\n    enabled: true\n'
        '    authorization_provider: fixture_api_authorizer:authorize\n')
    config = load_gateway_config()
    assert Platform.API_SERVER in config.get_connected_platforms()
    assert config.platforms[Platform.API_SERVER].extra['authorization_provider'] == 'fixture_api_authorizer:authorize'
