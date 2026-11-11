from argparse import Namespace
from pathlib import Path
import sqlite3
import zipfile

import pytest

from hermes_cli import backup


@pytest.mark.parametrize('mode', ['cli', 'automatic', 'quick', 'external'])
@pytest.mark.parametrize('filename', ['DIPS', 'History.sqlite', 'state.db'])
def test_live_wal_backup_is_self_contained(tmp_path, monkeypatch, mode, filename):
    home = tmp_path / '.hermes'
    directory = (tmp_path / 'provider') if mode == 'external' else home / 'browser'
    home.mkdir(exist_ok=True)
    directory.mkdir(parents=True)
    database = directory / filename
    writer = sqlite3.connect(database)
    try:
        writer.execute('pragma journal_mode=WAL')
        writer.execute('pragma wal_autocheckpoint=0')
        writer.execute('create table records(value text)')
        writer.commit()
        writer.execute('pragma wal_checkpoint(TRUNCATE)')
        writer.execute("insert into records values ('committed-in-wal')")
        writer.commit()
        assert Path(str(database) + '-wal').stat().st_size > 0
        # Leave the connection open throughout export; its committed row is
        # deliberately absent from the main file, so a raw copy cannot pass.
        monkeypatch.setenv('HERMES_HOME', str(home))
        monkeypatch.setattr(Path, 'home', lambda: tmp_path)
        monkeypatch.setattr(backup, '_collect_memory_provider_external_paths', lambda: [directory] if mode == 'external' else [])
        if mode == 'quick':
            monkeypatch.setattr(backup, '_QUICK_STATE_FILES', ['browser'])
            snapshot = backup.create_quick_snapshot(hermes_home=home)
            assert snapshot
            restored = home / 'state-snapshots' / snapshot / 'browser'
        else:
            archive = tmp_path / 'backup.zip'
            if mode in ('cli', 'external'):
                backup.run_backup(Namespace(output=str(archive)))
            else:
                assert backup._write_full_zip_backup(archive, home) == archive
            restored = tmp_path / 'restored' / 'browser'
            with zipfile.ZipFile(archive) as zf:
                zf.extractall(restored.parent)
            if mode == 'external':
                restored = restored.parent / backup._EXTERNAL_PREFIX / 'provider'
        for suffix in ('-wal', '-shm', '-journal'):
            assert not (restored / (filename + suffix)).exists()
        with sqlite3.connect('file:' + str(restored / filename) + '?mode=ro', uri=True) as recovered:
            assert recovered.execute('select value from records').fetchall() == [('committed-in-wal',)]
            assert recovered.execute('pragma integrity_check').fetchall() == [('ok',)]
    finally:
        writer.close()


def test_non_database_sidecar_name_remains_in_archive(tmp_path):
    home = tmp_path / '.hermes'
    home.mkdir()
    (home / 'notes').write_text('ordinary document')
    (home / 'notes-wal').write_text('ordinary companion')
    archive = tmp_path / 'backup.zip'
    assert backup._write_full_zip_backup(archive, home) == archive
    with zipfile.ZipFile(archive) as zf:
        assert zf.read('notes-wal') == b'ordinary companion'


@pytest.mark.parametrize('mode', ['cli', 'automatic'])
def test_extensionless_snapshot_failure_preserves_previous_archive(tmp_path, monkeypatch, mode):
    home = tmp_path / '.hermes'
    home.mkdir()
    with sqlite3.connect(home / 'DIPS') as db:
        db.execute('create table records(value text)')
    archive = tmp_path / 'backup.zip'
    with zipfile.ZipFile(archive, 'w') as zf:
        zf.writestr('prior.txt', 'recoverable')
    previous = archive.read_bytes()
    monkeypatch.setenv('HERMES_HOME', str(home))
    monkeypatch.setattr(Path, 'home', lambda: tmp_path)
    monkeypatch.setattr(backup, '_collect_memory_provider_external_paths', lambda: [])
    monkeypatch.setattr(backup, '_safe_copy_db', lambda *a, **kw: False)
    if mode == 'cli':
        with pytest.raises(SystemExit) as failed:
            backup.run_backup(Namespace(output=str(archive)))
        assert failed.value.code == 1
    else:
        assert backup._write_full_zip_backup(archive, home) is None
    assert archive.read_bytes() == previous
