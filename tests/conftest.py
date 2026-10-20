import os

import pytest

from sharding import assign, parse


@pytest.fixture(autouse=True)
def _uploads_read_in_this_process(monkeypatch):
    """Uploads are read in a child process (prepare.isolation), which never sees a function or limit a test replaced
    in this one, so tests read them here. tests/test_reader_isolation.py tests the child process itself, and the
    corrupted-input and timing harnesses run the command line, child process included."""
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '0')


_SHARD = {}


def _shard_of(config) -> tuple[int, dict] | None:
    """(this shard's index, {test file: shard index}) under ARGUS_TEST_SHARD=i/n, else None. Every test file under
    tests/ is placed, so each shard computes the same split from the same files (tests/sharding.py)."""
    value = os.environ.get('ARGUS_TEST_SHARD')
    if not value:
        return None
    if value not in _SHARD:
        index, count = parse(value)
        root = config.rootpath
        files = [f.relative_to(root).as_posix() for f in sorted((root / 'tests').rglob('test_*.py'))]
        _SHARD[value] = (index, assign(files, count))
    return _SHARD[value]


def pytest_ignore_collect(collection_path, config):
    """With ARGUS_TEST_SHARD=i/n, skip the test files of the other shards before they are imported. Without it, run
    all."""
    shard = _shard_of(config)
    if shard is None or not (collection_path.name.startswith('test_') and collection_path.suffix == '.py'):
        return None
    index, placed = shard
    try:
        rel = collection_path.relative_to(config.rootpath).as_posix()
    except ValueError:
        return None
    return True if placed.get(rel, index) != index else None
