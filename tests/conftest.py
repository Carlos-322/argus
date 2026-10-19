import pytest


@pytest.fixture(autouse=True)
def _uploads_read_in_this_process(monkeypatch):
    """Uploads are read in a child process (prepare.isolation), which never sees a function or limit a test replaced
    in this one, so tests read them here. tests/test_reader_isolation.py tests the child process itself, and the
    corrupted-input and timing harnesses run the command line, child process included."""
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '0')
