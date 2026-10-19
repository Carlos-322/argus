"""Prepared downloads keep one immutable output without repeating source preparation."""
import contextlib
import json
import os
from pathlib import Path
import socketserver
import subprocess
import threading
import urllib.error
import urllib.request

import pytest

from board import serve, static


@contextlib.contextmanager
def serving():
    with socketserver.ThreadingTCPServer(('127.0.0.1', 0), serve.Handler) as server:
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            yield f'http://127.0.0.1:{server.server_address[1]}/'
        finally:
            server.shutdown()
            worker.join(3)


def get(url):
    with urllib.request.urlopen(url, timeout=10) as response:
        return response.read()


def count_media_reads(monkeypatch):
    """[(file name, bytes)] of every read the server makes from an opened .mp4 (its sources and prepared videos)."""
    reads, original = [], Path.open
    class Counted:
        def __init__(self, file): self._file = file
        def read(self, *args):
            data = self._file.read(*args)
            reads.append((Path(self._file.name).name, len(data)))
            return data
        def readinto(self, buffer):
            size = self._file.readinto(buffer)
            reads.append((Path(self._file.name).name, size))
            return size
        def __getattr__(self, name): return getattr(self._file, name)
        def __enter__(self): return self
        def __exit__(self, *args): self._file.close()
    def opened(path, mode='r', *args, **kwargs):
        file = original(path, mode, *args, **kwargs)
        return Counted(file) if mode == 'rb' and Path(path).suffix == '.mp4' else file
    monkeypatch.setattr(Path, 'open', opened)
    return reads


def media(tmp_path, monkeypatch):
    serve._cached_probe_versioned.cache_clear()
    ffmpeg = serve.FFMPEG
    if not ffmpeg:
        pytest.skip('ffmpeg is required')
    folder = tmp_path / 'clips'
    folder.mkdir()
    for color in ('red', 'blue'):
        output = tmp_path / (color + '.mp4')
        subprocess.run([ffmpeg, '-v', 'error', '-f', 'lavfi', '-i',
                        f'color=c={color}:s=16x16:r=30:d=0.3', '-c:v', 'libx264',
                        '-pix_fmt', 'yuv420p', str(output)], check=True, timeout=5)
    source = folder / 'episode_x.mp4'
    source.write_bytes((tmp_path / 'red.mp4').read_bytes().ljust(4096, b'\0'))
    monkeypatch.setattr(serve, 'HERE', tmp_path / 'qa')
    monkeypatch.setattr(serve, 'MP4_DIR', folder)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path / 'footage')
    durable = static._durable_metadata
    monkeypatch.setattr(static, '_durable_metadata', lambda path, version:
                        False if Path(path).parent == folder else durable(path, version))
    return source


def test_http_prepare_ticket_keeps_output_and_skips_repeat_probe_and_encode(tmp_path, monkeypatch):
    source = media(tmp_path, monkeypatch)
    counts = {'copy': 0, 'probe': 0, 'encode': 0}
    for name, key in (('_copy_source_snapshot', 'copy'), ('_probe_uncached', 'probe'), ('_run_footage', 'encode')):
        original = getattr(serve, name, None)
        def track(*args, _original=original, _key=key, **kwargs):
            counts[_key] += 1
            return _original(*args, **kwargs)
        monkeypatch.setattr(serve, name, track, raising=False)
    with serving() as base:
        prepare = base + 'api/footage?id=episode_x&prepare=1'
        first = json.loads(get(prepare))
        assert 'ticket' in first and counts == {'copy': 0, 'probe': 1, 'encode': 1}
        original = get(base + 'api/footage?ticket=' + first['ticket'])
        assert counts == {'copy': 0, 'probe': 1, 'encode': 1}
        before = source.stat()
        source.write_bytes((tmp_path / 'blue.mp4').read_bytes().ljust(4096, b'\0'))
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        # A prepared ticket names that exact output even if the recording changes.
        assert get(base + 'api/footage?ticket=' + first['ticket']) == original
        assert counts == {'copy': 0, 'probe': 1, 'encode': 1}
        second = json.loads(get(prepare))
        assert get(base + 'api/footage?ticket=' + second['ticket']) != original
        assert counts == {'copy': 0, 'probe': 2, 'encode': 2}
        assert get(base + 'api/footage?id=episode_x') == get(base + 'api/footage?ticket=' + second['ticket'])
        assert counts == {'copy': 0, 'probe': 2, 'encode': 2}
    assert not list((serve.FOOTAGE_DIR / '.sources').glob('snapshot-*'))


def test_source_persistent_edit_during_probe_rejects_publication(tmp_path, monkeypatch):
    source = media(tmp_path, monkeypatch)
    serve._cached_probe_versioned.cache_clear()
    probe = serve._probe_uncached
    def edit(path):
        result = probe(path)
        source.write_bytes((tmp_path / 'blue.mp4').read_bytes().ljust(4096, b'\0'))
        return result
    monkeypatch.setattr(serve, '_probe_uncached', edit)
    with pytest.raises(ValueError, match='changed'):
        serve.footage('episode_x')
    assert not list(serve.FOOTAGE_DIR.glob('*.mp4'))


def test_ticket_survives_process_restart_and_receipts_have_a_per_client_count_bound(tmp_path, monkeypatch):
    import hashlib
    import sys
    media(tmp_path, monkeypatch)
    result = serve.footage('episode_x')
    monkeypatch.setattr(serve, '_TICKET_CLIENT_MAX', 2)
    tokens = [serve.prepare_footage_ticket(result) for _ in range(3)]
    assert len(list((serve.FOOTAGE_DIR / '.tickets').glob('*.json'))) == 2
    assert serve.ticket_footage(tokens[0]) is None
    script = '''
import hashlib, sys
from pathlib import Path
from board import serve
serve.FOOTAGE_DIR = Path(sys.argv[1])
serve.footage = lambda *args: (_ for _ in ()).throw(AssertionError('source preparation repeated'))
result = serve.ticket_footage(sys.argv[2])
print(hashlib.sha256(result[0].read_bytes()).hexdigest())
'''
    completed = subprocess.run([sys.executable, '-c', script, str(serve.FOOTAGE_DIR), tokens[-1]],
                               capture_output=True, text=True, check=True, timeout=5)
    assert completed.stdout.strip() == hashlib.sha256(result[0].read_bytes()).hexdigest()
    receipt = serve.FOOTAGE_DIR / '.tickets' / (tokens[-1] + '.json')
    record = json.loads(receipt.read_text())
    record['expires_at'] = 0
    receipt.write_text(json.dumps(record))
    assert serve.ticket_footage(tokens[-1]) is None and not receipt.exists()


@pytest.mark.parametrize('change', ['content', 'mtime', 'size'])
def test_untrusted_output_edit_invalidates_ticket_by_its_receipt_stat(tmp_path, monkeypatch, change):
    # Suggestion 9: on a weak filesystem a request compares size and modification time with the receipt written
    # when the video was made, never its content. An in-place edit changes the modification time.
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    with serving() as base:
        prepared = json.loads(get(base + 'api/footage?id=episode_x&prepare=1'))
        receipt = serve.FOOTAGE_DIR / '.tickets' / (prepared['ticket'] + '.json')
        output = serve.FOOTAGE_DIR / json.loads(receipt.read_text())['file']
        before, content = output.stat(), output.read_bytes()
        if change == 'size':
            output.write_bytes(content + b'\0')
            os.utime(output, ns=(before.st_atime_ns, before.st_mtime_ns))
        elif change == 'mtime':
            os.utime(output, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        else:
            output.write_bytes(bytes([content[0] ^ 1]) + content[1:])
        with pytest.raises(urllib.error.HTTPError) as error:
            get(base + 'api/footage?ticket=' + prepared['ticket'])
        assert error.value.code == 404
