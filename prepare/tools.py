"""Where the repo finds ffmpeg and ffprobe. Every caller (the reader, board clips, the board server, the static build,
the hand pose overlay) uses these two, so a probe, a frame and a clip all come from one install.

  ffmpeg    $FFMPEG, else ffmpeg on PATH, else the usual install places
  ffprobe   $FFPROBE, else the ffprobe beside that ffmpeg, else ffprobe on PATH, else the usual install places

The install places are there because a server started outside a login shell may not have Homebrew or /usr/local on
its PATH. A variable that is set but names no program is an error, never skipped for another copy. A tool that cannot
be found raises Missing, a FileNotFoundError, so a step that treats a missing tool as the machine's fault
(board/clips.py machine_stopped) still does, with a message that says how to fix it."""
from __future__ import annotations

import errno
import os
import shutil
from pathlib import Path

INSTALL_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "/usr/bin")


class Missing(FileNotFoundError):
    """ffmpeg or ffprobe cannot be found."""

    def __init__(self, tool: str, why: str):
        super().__init__(errno.ENOENT, f"{tool} not found: {why}", tool)


def _runnable(path) -> bool:
    return bool(path) and os.path.isfile(path) and os.access(path, os.X_OK)


def _from_env(var: str, tool: str) -> str | None:
    given = os.environ.get(var)
    if not given:
        return None
    if not _runnable(given):
        raise Missing(tool, f"${var} is {given}, which is not a program")
    return given


def _installed(tool: str) -> str | None:
    found = shutil.which(tool)
    if found:
        return found
    return next((str(Path(d) / tool) for d in INSTALL_DIRS if _runnable(str(Path(d) / tool))), None)


def ffmpeg() -> str:
    """The ffmpeg every caller runs (module docstring)."""
    found = _from_env("FFMPEG", "ffmpeg") or _installed("ffmpeg")
    if not found:
        raise Missing("ffmpeg", "put it on PATH or set FFMPEG")
    return found


def ffprobe() -> str:
    """The ffprobe every caller runs (module docstring)."""
    found = _from_env("FFPROBE", "ffprobe")
    if found:
        return found
    try:
        beside = str(Path(ffmpeg()).with_name("ffprobe"))
    except Missing:
        beside = None
    found = beside if _runnable(beside) else _installed("ffprobe")
    if not found:
        raise Missing("ffprobe", "put it on PATH or set FFPROBE")
    return found


def ffmpeg_or_none() -> str | None:
    """ffmpeg, or None when there is none, for a caller that works without it (the board server's frames)."""
    try:
        return ffmpeg()
    except Missing:
        return None


def ffprobe_or_none() -> str | None:
    """ffprobe, or None when there is none."""
    try:
        return ffprobe()
    except Missing:
        return None
