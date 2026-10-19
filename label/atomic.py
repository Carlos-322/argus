"""Writing a JSON file whole or not at all, and removing a shared file only for what it holds now, with nothing else
loaded: the readers, the checks, the board and the labelling harness all write through it.

Several processes (board servers, review and label runs) can share one folder. A process may keep its own copy of
what a shared file said, but that copy stands only for the version of the file it was read from (file_version), so a
decision is taken on a fresh read whenever the version differs (cached_while_unchanged keeps a result only for the
version it was read from), every writer publishes through a temporary file of its own (write_bytes_atomic), and a
file is removed only through remove_if, which reads it again and removes it only when the reason still holds for what
it has just read."""
from __future__ import annotations

import functools
import json
import os
import threading
import uuid
from collections import OrderedDict
from contextlib import nullcontext
from pathlib import Path


def write_atomic(out_path: Path, result, indent: int | None = 2, default=None, serializer=json.dumps, **options) -> None:
    """Write JSON through a temporary file in the same folder, then replace the file with it, so a kill mid-write never
    leaves a truncated file: a resume would take it for a result, and a context.json cut short would stop everything
    that reads the episode. A serializer can apply a consumer's JSON normalization, with the same write rule."""
    text = serializer(result, indent=indent, default=default, **options)
    write_bytes_atomic(out_path, text.encode('utf-8'))


# A file name may be up to NAME_MAX bytes, and a name derived from it (a temporary file, a receipt, a claim, a lock) adds
# a prefix and a suffix of at most DERIVED_RESERVE bytes. So every derived name is built on derived_base, which is the
# name itself when it leaves that room, and otherwise its start and a digest of the whole name: a valid name always
# yields valid derived names, two names never share one, and every derivation of one name shares one base, so a
# reader finds by glob what a writer derived.
NAME_MAX = 255
DERIVED_RESERVE = 64


def derived_base(name: str) -> str:
    """The part of every name derived from `name` that stands for it (see NAME_MAX)."""
    raw = name.encode('utf-8', 'surrogateescape')
    if len(raw) <= NAME_MAX - DERIVED_RESERVE:
        return name
    import hashlib
    head = raw[:NAME_MAX - DERIVED_RESERVE - 13].decode('utf-8', 'ignore')
    return f"{head}~{hashlib.sha256(raw).hexdigest()[:12]}"


def derived_name(name: str, prefix: str = '', suffix: str = '') -> str:
    """prefix + derived_base(name) + suffix, a valid file name whenever `name` is one."""
    if len((prefix + suffix).encode('utf-8', 'surrogateescape')) > DERIVED_RESERVE:
        raise ValueError(f'a derived name may add at most {DERIVED_RESERVE} bytes: {prefix!r} {suffix!r}')
    return prefix + derived_base(name) + suffix


def derived_path(path, prefix: str = '', suffix: str = '') -> Path:
    """The file beside `path` named derived_name(path.name, prefix, suffix)."""
    path = Path(path)
    return path.with_name(derived_name(path.name, prefix, suffix))


def temporary_path(out_path, suffix: str = '') -> Path:
    """A name beside out_path that no other writer, in this process or another, uses: .<name>.<random>.tmp<suffix>.
    `suffix` keeps an extension a tool chooses its format by (.mp4)."""
    return derived_path(out_path, '.', f'.{uuid.uuid4().hex}.tmp{suffix}')


def write_bytes_atomic(out_path, data: bytes) -> None:
    """Write `data` to a temporary file of this call's own in out_path's folder, flush it to disk, then rename it over
    out_path. Every writer has its own temporary file (created exclusively, like tempfile.mkstemp, but with the
    permissions the umask gives an ordinary file), so two processes writing one path each publish a whole file of their
    own, never the other's half-written one, and a sweep of a fixed temporary name cannot take a live write."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    while True:
        tmp = temporary_path(out_path)
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, 'O_BINARY', 0), 0o666)
            break
        except FileExistsError:
            continue
    try:
        with open(fd, 'wb') as out:
            out.write(data)
            out.flush()
            os.fsync(out.fileno())
        os.replace(tmp, out_path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def cached_while_unchanged(maxsize: int, unchanged):
    """A per-process memo, like functools.lru_cache, for a function whose arguments name a file and the version of it
    the caller saw before calling (a stat taken first). unchanged(*args) says whether the file still has that version;
    a result is kept only when it does once the function has read the file, so it stands for exactly the version it
    was read from, and a file replaced during the read is read again on the next call."""
    def decorate(fn):
        cache, lock = OrderedDict(), threading.Lock()

        @functools.wraps(fn)
        def call(*args):
            with lock:
                if args in cache:
                    cache.move_to_end(args)
                    return cache[args]
            value = fn(*args)
            try:
                keep = unchanged(*args)
            except OSError:
                keep = False
            if keep:
                with lock:
                    cache[args] = value
                    cache.move_to_end(args)
                    while len(cache) > maxsize:
                        cache.popitem(last=False)
            return value

        def cache_clear():
            with lock:
                cache.clear()
        call.cache_clear = cache_clear
        call.__wrapped__ = fn
        return call
    return decorate


REMOVING = '.removing'      # suffix of a file set aside by remove_if; sweep_removals clears ones a crash left behind


def file_version(path) -> tuple:
    """(size, modification time in ns) of the file as it is now (FileNotFoundError when it is missing). A rename keeps
    both, and an atomic replace or an edit changes them, on every filesystem the board runs on, so this is the version a
    per-process copy of a shared file is checked against before it is used again."""
    st = os.lstat(path)
    return st.st_size, st.st_mtime_ns


def small_file(limit: int = 65536):
    """A reader for remove_if that returns (size, modification time in ns, bytes) of a small file, so a condition can
    look at its content and two reads compare equal only when nothing changed. A symbolic link or a file over `limit`
    bytes reads as its version with None for the bytes."""
    def read(path) -> tuple:
        try:
            fd = os.open(path, os.O_RDONLY | getattr(os, 'O_NOFOLLOW', 0))
        except OSError:
            if not os.path.islink(path):
                raise
            return (*file_version(path), None)
        with open(fd, 'rb') as source:
            # the version of the open file is the version of the bytes read from it, whatever the name points to now
            st = os.fstat(source.fileno())
            data = source.read(limit + 1) if st.st_size <= limit else None
        return st.st_size, st.st_mtime_ns, data if data is not None and len(data) <= limit else None
    return read


def remove_if(path, holds, *, read=file_version, lock=None) -> bool:
    """Remove `path` only when holds(read(path)) is true for the file as it is now, and only that version of it.

    The file is read again here, never taken from an earlier read, then renamed to a name of this call's own, and what
    was renamed is read once more: when another process replaced the file between the read and the rename, the
    version it wrote is put back (unless a still newer one has taken the name since) and nothing is removed. `lock`
    (a function returning a context manager) is the lock the file's writers hold, when they hold one, so no writer
    can come between the read and the removal at all. Returns whether the file was removed."""
    path = Path(path)
    with lock() if lock is not None else nullcontext():
        try:
            seen = read(path)
        except FileNotFoundError:
            return False
        if not holds(seen):
            return False
        private = derived_path(path, '.', f'.{os.getpid()}.{uuid.uuid4().hex}{REMOVING}')
        try:
            os.rename(path, private)
        except FileNotFoundError:
            return False
        try:
            moved = read(private)
        except FileNotFoundError:
            return True
        if moved == seen or holds(moved):
            private.unlink(missing_ok=True)
            return True
        try:
            os.link(private, path)          # never over a version written after the rename
        except FileExistsError:
            pass                            # a newer version holds the name and supersedes the one set aside
        except OSError:
            # no hard links on this filesystem (exFAT, some shares): rename it back unless a newer version is there
            if not os.path.lexists(path):
                os.rename(private, path)
        private.unlink(missing_ok=True)
        return False


def sweep_removals(folder, older_than: float = 600.0) -> None:
    """Remove files remove_if set aside and a crash left behind. Its rename sets the change time, so a file set aside
    longer ago than `older_than` seconds belongs to no call still running."""
    import time
    try:
        entries = list(os.scandir(folder))
    except OSError:
        return
    for entry in entries:
        if entry.name.startswith('.') and entry.name.endswith(REMOVING):
            try:
                if entry.stat(follow_symlinks=False).st_ctime < time.time() - older_than:
                    os.unlink(entry.path)
            except OSError:
                continue
