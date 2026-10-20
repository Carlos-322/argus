"""prepare/tools.py: the one place the repo finds ffmpeg and ffprobe. board/hands.py read $FFPROBE and exited without
it, board/serve.py and board/static.py took the ffprobe beside the server's ffmpeg and fell back to PATH, and
board/clips.py looked beside the ffmpeg it was handed, so one machine could probe a clip with one install and cut it
with another."""
from __future__ import annotations

from pathlib import Path

import pytest

from prepare import tools


def _program(folder: Path, name: str) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / name
    p.write_text("#!/bin/sh\nexit 0\n")
    p.chmod(0o755)
    return str(p)


@pytest.fixture
def bare(tmp_path, monkeypatch):
    """A machine with neither tool anywhere: nothing on PATH, no variables, no install places."""
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.delenv("FFMPEG", raising=False)
    monkeypatch.delenv("FFPROBE", raising=False)
    monkeypatch.setattr(tools, "INSTALL_DIRS", ())
    return tmp_path


def test_ffprobe_is_the_one_beside_the_ffmpeg_in_use_before_the_one_on_path(bare, monkeypatch):
    ffmpeg = _program(bare / "pinned", "ffmpeg")
    probe = _program(bare / "pinned", "ffprobe")
    _program(bare / "path", "ffprobe")
    monkeypatch.setenv("PATH", str(bare / "path"))
    monkeypatch.setenv("FFMPEG", ffmpeg)
    assert (tools.ffmpeg(), tools.ffprobe()) == (ffmpeg, probe)
    monkeypatch.setenv("FFPROBE", str(bare / "path" / "ffprobe"))
    assert tools.ffprobe() == str(bare / "path" / "ffprobe")      # $FFPROBE before anything else


def test_a_tool_on_path_or_in_an_install_place_is_found(bare, monkeypatch):
    _program(bare / "path", "ffmpeg")
    monkeypatch.setenv("PATH", str(bare / "path"))
    assert tools.ffmpeg() == str(bare / "path" / "ffmpeg")
    monkeypatch.setattr(tools, "INSTALL_DIRS", (str(bare / "brew"),))
    probe = _program(bare / "brew", "ffprobe")
    assert tools.ffprobe() == probe          # a server started outside a login shell still finds it


def test_a_missing_tool_is_a_clear_error_and_never_a_silent_fallback(bare, monkeypatch):
    with pytest.raises(tools.Missing, match="ffprobe not found: put it on PATH or set FFPROBE"):
        tools.ffprobe()
    with pytest.raises(tools.Missing, match="ffmpeg not found: put it on PATH or set FFMPEG"):
        tools.ffmpeg()
    assert tools.ffmpeg_or_none() is None and tools.ffprobe_or_none() is None
    _program(bare / "path", "ffprobe")
    monkeypatch.setenv("PATH", str(bare / "path"))
    monkeypatch.setenv("FFPROBE", str(bare / "no" / "ffprobe"))
    with pytest.raises(tools.Missing, match=r"\$FFPROBE is .*, which is not a program"):
        tools.ffprobe()
    assert isinstance(tools.Missing("ffprobe", "x"), FileNotFoundError)    # the machine's, board/clips.py machine_stopped


def test_every_caller_runs_the_same_ffprobe():
    """The callers that probe a clip all take it from prepare/tools.py, with no lookup of their own left."""
    root = Path(__file__).resolve().parents[1]
    own = []
    for folder in ("board", "checks", "label", "prepare"):
        for p in (root / folder).rglob("*.py"):
            if p.name == "tools.py" or "hand_pose" in p.parts or "vendor" in p.parts:
                continue
            text = p.read_text()
            for needle in ('which("ffprobe")', "which('ffprobe')", 'which("ffmpeg")', "which('ffmpeg')",
                           '"FFPROBE"', '"FFMPEG"', "['ffprobe'", "['ffmpeg'", '["ffprobe"', '["ffmpeg"',
                           "\n        'ffprobe',", "\n        'ffmpeg',"):
                if needle in text:
                    own.append(f"{p.relative_to(root)}: {needle.strip()}")
    assert not own, own
