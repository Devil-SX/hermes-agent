"""Regression: Codex app-server turns can read cached inbound attachments.

Inbound Telegram attachments are cached under Hermes-private media
directories and the turn text names those host paths, but the isolated
group-agent sandbox denies ``~/.hermes`` wholesale, so the model sees an
attachment it can never open (issue-20260831-152859-a6c78624).

The fix stages every referenced media-cache file into the Codex session's
workspace (``<cwd>/attachments``) and rewrites the turn text to the staged
path before ``turn/start``. Only the turn input changes; the messages list
and every persistent store stay untouched.
"""

from pathlib import Path

import pytest

from agent.codex_runtime import (
    _rewrite_turn_input_media_paths,
    _stage_cached_media_into_cwd,
)


@pytest.fixture()
def hermes_media_home(tmp_path, monkeypatch):
    """Point HERMES_HOME at a temp dir with a populated image cache."""
    home = tmp_path / "hermes-home"
    images = home / "cache" / "images"
    images.mkdir(parents=True)
    (images / "img_9ff602ad3384.jpg").write_bytes(b"\xff\xd8\xfffakejpg")
    monkeypatch.setenv("HERMES_HOME", str(home))
    return home


def _workspace(tmp_path) -> Path:
    workspace = tmp_path / "group-workspace"
    workspace.mkdir()
    return workspace


def test_stage_rewrites_image_path_to_workspace_copy(
    hermes_media_home, tmp_path
):
    """A cached image path in the text is staged and rewritten."""
    workspace = _workspace(tmp_path)
    host_path = (
        hermes_media_home / "cache" / "images" / "img_9ff602ad3384.jpg"
    )
    text = (
        "[The user sent an image. It is saved at: "
        f"{host_path} for inspection.]"
    )

    rewritten = _stage_cached_media_into_cwd(text, str(workspace))

    staged = workspace / "attachments" / "img_9ff602ad3384.jpg"
    assert staged.is_file()
    assert staged.read_bytes() == b"\xff\xd8\xfffakejpg"
    assert str(staged) in rewritten
    assert str(host_path) not in rewritten


def test_stage_document_note_style_text(hermes_media_home, tmp_path):
    """Document context notes with a saved-at path are rewritten too."""
    home = hermes_media_home
    docs = home / "cache" / "documents"
    docs.mkdir(parents=True, exist_ok=True)
    (docs / "report.pdf").write_bytes(b"%PDF-1.7 fake")
    workspace = _workspace(tmp_path)

    text = (
        "[The user sent a document: 'report.pdf'. It is saved at: "
        f"{docs / 'report.pdf'}. Its text is not inlined here.]"
    )
    rewritten = _stage_cached_media_into_cwd(text, str(workspace))

    staged = workspace / "attachments" / "report.pdf"
    assert staged.is_file()
    assert str(staged) in rewritten


def test_no_cache_path_in_text_returns_text_unchanged(
    hermes_media_home, tmp_path
):
    """Ordinary workspace paths pass through without side effects."""
    workspace = _workspace(tmp_path)
    text = f"please look at {workspace}/notes.md"

    assert (
        _stage_cached_media_into_cwd(text, str(workspace)) == text
    )
    assert not (workspace / "attachments").exists()


def test_missing_cache_file_leaves_text_unchanged(
    hermes_media_home, tmp_path
):
    """A cache path that does not exist on disk is not rewritten."""
    workspace = _workspace(tmp_path)
    missing = (
        hermes_media_home / "cache" / "images" / "img_missing.jpg"
    )
    text = f"saved at: {missing}"

    assert _stage_cached_media_into_cwd(text, str(workspace)) == text


def test_rewrite_turn_input_composed_text_items(
    hermes_media_home, tmp_path
):
    """Composed text items (context + message) are rewritten item-wise."""
    workspace = _workspace(tmp_path)
    host_path = (
        hermes_media_home / "cache" / "images" / "img_9ff602ad3384.jpg"
    )
    turn_input = [
        {"type": "text", "text": "[system] 当前模型: glm-5.3-flash"},
        {"type": "text", "text": f"saved at: {host_path}"},
    ]

    rewritten = _rewrite_turn_input_media_paths(turn_input, str(workspace))

    assert isinstance(rewritten, list)
    assert rewritten[0] == {
        "type": "text",
        "text": "[system] 当前模型: glm-5.3-flash",
    }
    staged = workspace / "attachments" / "img_9ff602ad3384.jpg"
    assert staged.is_file()
    assert str(staged) in rewritten[1]["text"]
    # Non-text items pass through untouched.
    multimodal = [
        {"type": "image", "url": str(host_path)},
        {"type": "text", "text": f"saved at: {host_path}"},
    ]
    rewritten_mm = _rewrite_turn_input_media_paths(
        multimodal, str(workspace)
    )
    assert rewritten_mm[0] == {"type": "image", "url": str(host_path)}
    assert str(staged) in rewritten_mm[1]["text"]


def test_rewrite_skipped_for_workspace_under_cache_root(
    tmp_path, monkeypatch
):
    """A workspace inside a media cache root must not be self-copied."""
    home = tmp_path / "hermes-home"
    images = home / "cache" / "images"
    images.mkdir(parents=True)
    (images / "img_x.jpg").write_bytes(b"x")
    monkeypatch.setenv("HERMES_HOME", str(home))
    workspace = images / "workspace"
    workspace.mkdir()
    text = f"saved at: {images / 'img_x.jpg'}"

    assert (
        _rewrite_turn_input_media_paths(text, str(workspace)) == text
    )


def test_rewrite_skipped_when_cwd_missing(hermes_media_home):
    """Without a cwd the input passes through untouched."""
    text = f"saved at: {hermes_media_home / 'cache' / 'images' / 'img_9ff602ad3384.jpg'}"

    assert _rewrite_turn_input_media_paths(text, None) == text


def test_staging_failure_degrades_to_original_text(
    hermes_media_home, tmp_path, monkeypatch
):
    """If the copy fails, the original path survives (no crash, no rewrite)."""
    from agent import codex_runtime

    workspace = _workspace(tmp_path)
    host_path = (
        hermes_media_home / "cache" / "images" / "img_9ff602ad3384.jpg"
    )
    text = f"saved at: {host_path}"
    monkeypatch.setattr(
        codex_runtime.shutil,
        "copy2",
        lambda *a, **k: (_ for _ in ()).throw(PermissionError("denied")),
    )

    assert _stage_cached_media_into_cwd(text, str(workspace)) == text


def test_symlink_escape_not_staged(tmp_path, monkeypatch):
    """A symlink under a cache root pointing outside is not followed."""
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_text("secret")
    home = tmp_path / "hermes-home"
    images = home / "cache" / "images"
    images.mkdir(parents=True)
    link = images / "img_link.jpg"
    link.symlink_to(outside / "secret.txt")
    monkeypatch.setenv("HERMES_HOME", str(home))
    workspace = _workspace(tmp_path)
    text = f"saved at: {link}"

    assert _stage_cached_media_into_cwd(text, str(workspace)) == text
    assert not (workspace / "attachments" / "img_link.jpg").exists()
