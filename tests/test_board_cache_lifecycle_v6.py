"""Cache lifecycle boundaries through real HTTP and ffmpeg."""
import concurrent.futures
import errno
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import threading
import urllib.error
import urllib.request

import pytest

from board import serve, static
from test_prepared_footage import count_media_reads, media, serving, get


def test_reuse_refreshes_eviction_age_without_changing_movie_identity(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 2)
    with serving() as base:
        def prepare(a, b):
            return json.loads(get(base + f'api/footage?id=episode_x&t0={a}&t1={b}&prepare=1'))
        first = prepare(0, .1)
        path, _ = serve.ticket_footage(first['ticket'])
        identity = serve._footage_version(path)
        second = prepare(.05, .2)
        refreshed = prepare(0, .1)
        prepare(.1, .3)
        assert get(base + 'api/footage?ticket=' + refreshed['ticket'])
        assert serve._footage_version(path) == identity
        assert serve.ticket_footage(second['ticket']) is None


def test_cli_restart_preserves_ticket_with_disabled_file_count(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    board = tmp_path / 'board'
    (board / 'qa').mkdir(parents=True)
    with socket.socket() as reservation:
        reservation.bind(('127.0.0.1', 0))
        port = reservation.getsockname()[1]
    command = [sys.executable, '-m', 'board', 'serve', '--board', str(board), '--clips', str(serve.MP4_DIR), '--port', str(port)]
    environment = {**os.environ, 'BOARD_FOOTAGE_DISK_FILES': '0', 'BOARD_FOOTAGE_DIR': str(serve.FOOTAGE_DIR)}
    token = None
    for _ in range(2):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
        try:
            assert 'board at' in process.stdout.readline()
            base = f'http://127.0.0.1:{port}/api/footage?'
            if token is None:
                token = json.loads(get(base + 'id=episode_x&prepare=1'))['ticket']
            assert get(base + 'ticket=' + token)
        finally:
            process.terminate()
            process.communicate(timeout=5)


@pytest.mark.parametrize('stage', ['alias', 'canonical', 'trim'])
def test_receipt_bookkeeping_failure_keeps_verified_media_and_next_request(tmp_path, monkeypatch, stage):
    media(tmp_path, monkeypatch)
    initial = serve.footage('episode_x', 0, .1)
    atomic, trim, encode = serve.write_atomic, serve._trim_footage_receipts, serve._run_footage
    failed, calls = [True], []
    def write(path, *args, **kwargs):
        if failed[0] and path.parent.name == '.ready' and (stage == 'canonical' or stage == 'alias' and len(path.stem) == 40):
            raise PermissionError(errno.EACCES, 'receipt read only')
        return atomic(path, *args, **kwargs)
    def prune(*args, **kwargs):
        if failed[0] and stage == 'trim':
            raise PermissionError(errno.EACCES, 'unreadable unrelated receipt')
        return trim(*args, **kwargs)
    def run(command):
        calls.append(True)
        return encode(command)
    monkeypatch.setattr(serve, 'write_atomic', write)
    monkeypatch.setattr(serve, '_trim_footage_receipts', prune)
    monkeypatch.setattr(serve, '_run_footage', run)
    with serving() as base:
        assert get(base + 'api/footage?id=episode_x&t0=0&t1=0.100000001') == initial[0].read_bytes()
        prepared = json.loads(get(base + 'api/footage?id=episode_x&t0=.05&t1=.2&prepare=1'))
        assert get(base + 'api/footage?ticket=' + prepared['ticket'])
        failed[0] = False
        again = json.loads(get(base + 'api/footage?id=episode_x&t0=.05&t1=.2&prepare=1'))
        assert get(base + 'api/footage?ticket=' + again['ticket'])
        assert len(calls) == 1


def test_ticket_capacity_failure_is_not_reported_as_identity_change(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    atomic = serve.write_atomic
    def write(path, *args, **kwargs):
        if path.parent.name == '.tickets':
            raise OSError(errno.ENOSPC, 'no space for ticket')
        return atomic(path, *args, **kwargs)
    monkeypatch.setattr(serve, 'write_atomic', write)
    with serving() as base:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base + 'api/footage?id=episode_x&prepare=1')
        assert error.value.code == 507
        assert list(serve.FOOTAGE_DIR.glob('*.mp4'))
    monkeypatch.setattr(serve, 'write_atomic', atomic)
    with serving() as base:
        assert get(base + 'api/footage?id=episode_x&prepare=1')


def test_old_active_output_is_not_swept_during_source_validation(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    ready = serve.footage('episode_x', 0, .1)
    validate = serve._validate_active_sources
    entered, release = threading.Event(), threading.Event()
    def held():
        temporary = next(serve.FOOTAGE_DIR.glob('*.part.mp4'))
        os.utime(temporary, (1, 1))
        entered.set()
        assert release.wait(5)
        return validate()
    monkeypatch.setattr(serve, '_validate_active_sources', held)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        pending = pool.submit(serve.footage, 'episode_x', .05, .2)
        try:
            assert entered.wait(3)
            assert serve.footage('episode_x', 0, .1) == ready
        finally:
            release.set()
        assert pending.result()[0].is_file()


def test_receipt_orphan_sweep_skips_an_active_atomic_writer(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    result = serve.footage('episode_x', 0, .1)
    folder = serve.FOOTAGE_DIR / '.tickets'
    folder.mkdir(exist_ok=True)
    orphan = folder / ('.' + 'a' * 32 + '.json.tmp')
    orphan.write_text('partial')
    entered, release = threading.Event(), threading.Event()
    atomic = serve.write_atomic
    active = []
    from types import SimpleNamespace
    monkeypatch.setattr(serve.uuid, 'uuid4', lambda: SimpleNamespace(hex='b' * 32))
    def hold(path, *args, **kwargs):
        if path.parent == folder:
            temporary = path.with_name('.' + path.name + '.tmp')
            temporary.write_text('active')
            active.append(temporary)
            entered.set()
            assert release.wait(5)
        return atomic(path, *args, **kwargs)
    monkeypatch.setattr(serve, 'write_atomic', hold)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        pending = pool.submit(serve.prepare_footage_ticket, result)
        try:
            assert entered.wait(3)
            serve._trim_footage_receipts(folder, r'[a-f0-9]{32}\.json', expire=True)
            assert not orphan.exists()
            assert active[0].exists()
        finally:
            release.set()
        assert serve.ticket_footage(pending.result()) == result


def test_actual_download_client_retries_encoder_contention(tmp_path, monkeypatch):
    import shutil
    node = shutil.which('node')
    if not node:
        pytest.skip('node is required for the actual download client')
    source = media(tmp_path, monkeypatch)
    (source.parent / 'episode_y.mp4').write_bytes(source.read_bytes())
    entered, release = threading.Event(), threading.Event()
    encode = serve._run_footage
    def held(command):
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return encode(command)
    monkeypatch.setattr(serve, '_run_footage', held)
    page = Path(serve.__file__).with_name('index.html').read_text()
    client = page[page.index('async function vdDownload('):page.index('vdBtn.addEventListener', page.index('async function vdDownload('))]
    script = tmp_path / 'client.js'
    script.write_text('''const status={textContent:''}; let clicks=0;
const _vdEp={eid:'episode_x'}; const fmtBytes=String;
const document={getElementById:()=>status,body:{appendChild(){}},createElement:()=>({click(){clicks++},remove(){}})};
const original=fetch; global.fetch=async (url,options)=>{const response=await original(new URL(url,process.argv[2]),options); if(response.status===503) console.log('busy'); return response;};
''' + client + "vdDownload(0,.1).then(()=>console.log(JSON.stringify({clicks,status:status.textContent})));\n")
    with serving() as base, concurrent.futures.ThreadPoolExecutor(1) as pool:
        # episode_y holds the only encoder, so the client's new span is refused busy until it is free
        pending = pool.submit(serve.footage, 'episode_y', 0, .1)
        process = None
        try:
            assert entered.wait(3)
            process = subprocess.Popen([node, str(script), base], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            assert process.stdout.readline().strip() == 'busy'
        finally:
            release.set()
        stdout, stderr = process.communicate(timeout=8)
        assert process.returncode == 0, stderr
        report = json.loads(stdout.splitlines()[-1])
        assert report['clicks'] == 1, report
        assert pending.result()[0].is_file()


def test_untrusted_prepare_and_range_read_no_content_to_check_identity(tmp_path, monkeypatch):
    # Suggestion 9: receipts of size and modification time stand for the content, so a repeat prepare reads nothing
    # and a Range download reads exactly the bytes it sends.
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    with serving() as base:
        url = base + 'api/footage?id=episode_x&prepare=1'
        json.loads(get(url))
        reads = count_media_reads(monkeypatch)
        prepared = json.loads(get(url))
        assert reads == []
        request = urllib.request.Request(base + 'api/footage?ticket=' + prepared['ticket'], headers={'Range': 'bytes=0-31'})
        assert len(get(request)) == 32
        assert sum(size for name, size in reads if name != 'episode_x.mp4') == 32
        assert not [name for name, _ in reads if name == 'episode_x.mp4']


def _interrupted_ticket_writer(result, connection):
    def hold(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name('.' + path.name + '.tmp')
        temporary.write_text('{partial')
        connection.send(str(temporary))
        threading.Event().wait(30)
    serve.write_atomic = hold
    serve.prepare_footage_ticket(result)


@pytest.mark.skipif(os.name != 'posix', reason='forked crash and filesystem locks')
def test_killed_atomic_writer_releases_guard_and_next_http_request_cleans_orphan(tmp_path, monkeypatch):
    import multiprocessing
    media(tmp_path, monkeypatch)
    result = serve.footage('episode_x', 0, .1)
    context = multiprocessing.get_context('fork')
    parent, child = context.Pipe()
    process = context.Process(target=_interrupted_ticket_writer, args=(result, child))
    process.start()
    try:
        assert parent.poll(5)
        temporary = Path(parent.recv())
        serve._trim_footage_receipts(temporary.parent, r'[a-f0-9]{32}\.json', expire=True)
        assert temporary.exists()
        process.kill()
        process.join(5)
        with serving() as base:
            prepared = json.loads(get(base + 'api/footage?id=episode_x&t0=0&t1=.1&prepare=1'))
            assert get(base + 'api/footage?ticket=' + prepared['ticket'])
        assert not temporary.exists()
    finally:
        if process.is_alive():
            process.kill()
        process.join(5)


def test_sweep_removes_an_orphaned_work_guard_without_touching_operator_files(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    serve.FOOTAGE_DIR.mkdir()
    guard = serve.FOOTAGE_DIR / ('a' * 20 + '.123.456.' + 'b' * 32 + '.part.mp4.lock')
    unrelated = serve.FOOTAGE_DIR / 'operator.part.mp4.lock'
    for path in (guard, unrelated):
        path.write_text('lock')
        os.utime(path, (1, 1))
    serve._trim_footage_cache()
    assert not guard.exists()
    assert unrelated.read_text() == 'lock'


def test_new_untrusted_span_during_another_encode_is_refused_retryably_and_ready_span_is_not(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    serve.footage('episode_x', 0, .1)
    entered, release = threading.Event(), threading.Event()
    encode = serve._run_footage
    def hold(command):
        result = encode(command)
        if not entered.is_set():
            entered.set()
            assert release.wait(5)
        return result
    monkeypatch.setattr(serve, '_run_footage', hold)
    with serving() as base, concurrent.futures.ThreadPoolExecutor(1) as pool:
        first = pool.submit(get, base + 'api/footage?id=episode_x&t0=.05&t1=.2&prepare=1')
        try:
            assert entered.wait(3)
            with pytest.raises(urllib.error.HTTPError) as error:
                get(base + 'api/footage?id=episode_x&t0=.1&t1=.3&prepare=1')
            assert error.value.code == 503 and error.value.headers['Retry-After']
            # a span already prepared never waits for the encoder
            assert json.loads(get(base + 'api/footage?id=episode_x&t0=0&t1=.1&prepare=1'))['ticket']
        finally:
            release.set()
        assert get(base + 'api/footage?ticket=' + json.loads(first.result())['ticket'])
        second = json.loads(get(base + 'api/footage?id=episode_x&t0=.1&t1=.3&prepare=1'))
        assert get(base + 'api/footage?ticket=' + second['ticket'])


def test_prepare_ticket_eviction_is_retryable_media_loss(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    prepare = serve.prepare_footage_ticket
    def evict(result):
        result[0].unlink()
        return prepare(result)
    monkeypatch.setattr(serve, 'prepare_footage_ticket', evict)
    with serving() as base:
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base + 'api/footage?id=episode_x&prepare=1')
        assert error.value.code == 503
        message = json.loads(error.value.read())
        assert message.get('retryable') is True
        assert 'storage cannot' not in message['error'].lower()
    monkeypatch.setattr(serve, 'prepare_footage_ticket', prepare)
    with serving() as base:
        result = json.loads(get(base + 'api/footage?id=episode_x&prepare=1'))
        assert get(base + 'api/footage?ticket=' + result['ticket'])


def test_configured_untrusted_prepares_encode_different_episodes_in_parallel(tmp_path, monkeypatch):
    source = media(tmp_path, monkeypatch)
    (source.parent / 'episode_y.mp4').write_bytes(source.read_bytes())
    serve.footage('episode_x', 0, .1)
    monkeypatch.setattr(serve, '_FOOTAGE_SEM', threading.BoundedSemaphore(2))
    monkeypatch.setattr(serve, '_FOOTAGE_PROBE_SEM', threading.BoundedSemaphore(2))
    original = serve._run_footage
    completed, release = threading.Event(), threading.Event()
    lock, active = threading.Lock(), [0]
    def hold(command):
        result = original(command)
        with lock:
            active[0] += 1
            if active[0] == 2:
                completed.set()
        assert release.wait(5)
        return result
    monkeypatch.setattr(serve, '_run_footage', hold)
    with serving() as base, concurrent.futures.ThreadPoolExecutor(2) as pool:
        requests = [pool.submit(get, base + f'api/footage?id=episode_{name}&t0=.1&t1=.3&prepare=1') for name in ('x', 'y')]
        try:
            assert completed.wait(3)
        finally:
            release.set()
        for request in requests:
            prepared = json.loads(request.result())
            assert get(base + 'api/footage?ticket=' + prepared['ticket'])
