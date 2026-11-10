from concurrent.futures import ThreadPoolExecutor

import pytest

from gateway.drain_control import read_drain_request, update_owned_drain


def test_concurrent_owners_cannot_replace_or_cancel_each_other(tmp_path):
    def claim(owner):
        try:
            update_owned_drain(principal=owner, action='begin', home=tmp_path)
            return owner
        except PermissionError:
            return None
    with ThreadPoolExecutor(max_workers=2) as pool:
        winners = [owner for owner in pool.map(claim, ['first', 'second']) if owner]
    assert len(winners) == 1
    owner = winners[0]
    other = 'second' if owner == 'first' else 'first'
    with pytest.raises(PermissionError):
        update_owned_drain(principal=other, action='cancel', home=tmp_path)
    update_owned_drain(principal=owner, action='refresh', home=tmp_path)
    assert read_drain_request(home=tmp_path)['principal'] == owner
    assert update_owned_drain(principal=owner, action='cancel', home=tmp_path)['present'] is False
    assert read_drain_request(home=tmp_path) is None
    with pytest.raises(ValueError):
        update_owned_drain(principal=owner, action='refresh', home=tmp_path)


def test_ambiguous_marker_and_failed_persistence_are_not_success(tmp_path, monkeypatch):
    import os

    marker = tmp_path / '.drain_request.json'
    marker.write_text('invalid')
    with pytest.raises(PermissionError):
        update_owned_drain(principal='owner', action='begin', home=tmp_path)
    assert marker.read_text() == 'invalid'
    marker.unlink()
    def fail(_fd):
        raise OSError('synthetic durability failure')
    monkeypatch.setattr(os, 'fsync', fail)
    with pytest.raises(OSError, match='durability'):
        update_owned_drain(principal='owner', action='begin', home=tmp_path)
