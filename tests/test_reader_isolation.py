"""A native read that never returns, or that kills its process, costs that read alone.

Each episode is read in a child process (prepare.isolation). A child that stops making progress is stopped, and one
that dies is reported, so neither can hang or crash the upload or the server that prepares it.
"""
import os
import signal
import time

import pytest

from prepare.isolation import ReaderStopped, cameras_only, run_isolated


def doubles(value):
    return value * 2, os.getpid()


def raises():
    raise ValueError('the recording names no camera')


def never_returns():
    time.sleep(3600)


def crashes():
    os.kill(os.getpid(), signal.SIGSEGV)


def reports_mode():
    return cameras_only()


def writes_slowly(folder, seconds):
    from pathlib import Path
    target = Path(folder) / 'episode' / 'video.bin'
    target.parent.mkdir(parents=True)
    with target.open('wb') as stream:
        for _ in range(int(seconds * 4)):
            stream.write(b'x' * 64)
            stream.flush()
            time.sleep(.25)
    return 'done'


def test_a_result_comes_back_from_another_process():
    value, pid = run_isolated(doubles, 21)
    assert value == 42 and pid != os.getpid()


def test_an_error_comes_back_with_its_type():
    with pytest.raises(ValueError, match='names no camera'):
        run_isolated(raises)


def test_a_read_that_never_returns_is_stopped():
    start = time.monotonic()
    with pytest.raises(ReaderStopped, match='stopped responding'):
        run_isolated(never_returns, stall_s=1.5)
    assert time.monotonic() - start < 30


def test_a_reader_that_crashes_its_process_is_reported():
    with pytest.raises(ReaderStopped, match='signal'):
        run_isolated(crashes)


def test_output_written_by_the_child_counts_as_progress(tmp_path):
    assert run_isolated(writes_slowly, str(tmp_path), 3, watch=tmp_path, stall_s=1.5) == 'done'


def records_pid_and_never_returns(path):
    from pathlib import Path
    Path(path).write_text(str(os.getpid()))
    time.sleep(3600)


def test_a_killed_parent_never_leaves_its_reader_running(tmp_path):
    import subprocess
    import sys
    from pathlib import Path
    pid_file = tmp_path / 'child.pid'
    script = ('import sys; sys.path[:0] = sys.argv[1:3]; from prepare.isolation import run_isolated; '
              'from test_reader_isolation import records_pid_and_never_returns as f; run_isolated(f, sys.argv[3])')
    parent = subprocess.Popen([sys.executable, '-c', script, str(Path(__file__).parents[1]),
                               str(Path(__file__).parent), str(pid_file)])
    deadline = time.monotonic() + 30
    while not pid_file.exists() and time.monotonic() < deadline:
        time.sleep(.1)
    child = int(pid_file.read_text())
    parent.kill()
    parent.wait()
    while time.monotonic() < deadline:
        try:
            os.kill(child, 0)
        except ProcessLookupError:
            return
        time.sleep(.2)
    os.kill(child, signal.SIGKILL)
    pytest.fail('the reader outlived its killed parent')


def test_the_second_attempt_reads_cameras_only():
    assert run_isolated(reports_mode) is False
    assert run_isolated(reports_mode, mode='cameras') is True


def test_an_hdf5_attribute_read_that_never_returns_costs_the_parts_not_the_camera(tmp_path, monkeypatch):
    # Corpus mutant n0145, seed 5145: metadata bytes flipped so that reading one attribute in HDF5's C library
    # never returns. The whole upload hung; now the episode is read again with its cameras alone.
    import gzip
    import json
    from pathlib import Path
    from prepare import formats, isolation
    monkeypatch.setattr(isolation, 'READER_STALL_S', 4)
    upload = tmp_path / 'upload'
    upload.mkdir()
    (upload / 'episode_1.hdf5').write_bytes(gzip.decompress(
        (Path(__file__).parent / 'fixtures' / 'hdf_attribute_read_hangs.h5.gz').read_bytes()))
    report = formats.convert(upload, 'teleop_arms', tmp_path / 'out', 'hang', 900)
    assert len(report['episodes']) == 1 and not report['failed'], report
    ctx = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert ctx['cameras']
    assert any(i['kind'] == 'part_unreadable' and 'cameras alone' in i['what'] for i in ctx['reader_issues'])
