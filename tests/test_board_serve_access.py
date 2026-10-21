"""Per-client download tickets, Host checks and media validators of the live board."""
import json
import os
import urllib.error
import urllib.request

import pytest

from board import serve, static
from test_prepared_footage import media, served, serving, get


def prepare(base, span, client=None):
    headers = {'X-Forwarded-For': client} if client else {}
    request = urllib.request.Request(base + f'api/footage?id=episode_x&t0={span[0]}&t1={span[1]}&prepare=1',
                                     headers=headers)
    return json.loads(get(request))['ticket']


def status(request):
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, dict(response.headers), response.read()
    except urllib.error.HTTPError as error:
        return error.code, dict(error.headers), error.read()


@pytest.mark.parametrize('weak', [False, True])
def test_other_clients_prepares_never_cancel_a_live_ticket(tmp_path, monkeypatch, weak):
    media(tmp_path, monkeypatch)
    if weak:
        monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 4)
    with serving() as base:
        victim = prepare(base, (0, .1), '10.0.0.1')
        spans = [(round(.01 * i, 2), .3) for i in range(1, 13)]
        codes = []
        for span in spans:
            try:
                prepare(base, span, '10.0.0.66')
                codes.append(200)
            except urllib.error.HTTPError as error:
                codes.append(error.code)
        assert get(base + 'api/footage?ticket=' + victim)
        # the other client keeps at most _TICKET_CLIENT_MAX tickets and withdraws its own oldest
        with serve._FOOTAGE_READY_LOCK:
            owners = [entry[0] for entry in serve._ticket_index().values()]
        assert owners.count(serve._ticket_owner('10.0.0.66')) <= serve._TICKET_CLIENT_MAX
        assert set(codes) <= {200, 503}
        assert len(list(serve.FOOTAGE_DIR.glob('*.mp4'))) <= 4


def test_repeat_prepare_renews_the_clients_own_ticket_for_a_full_lifetime(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    with serving() as base:
        first = prepare(base, (0, .1), '10.0.0.1')
        receipt = serve.FOOTAGE_DIR / '.tickets' / (first + '.json')
        record = json.loads(receipt.read_text())
        record['expires_at'] = record['issued_at'] = __import__('time').time() + 20
        receipt.write_text(json.dumps(record))
        # the server's ticket index sees a ticket file rewritten after it read it, with nothing cleared by hand
        again = prepare(base, (0, .1), '10.0.0.1')
        assert again == first
        assert json.loads(receipt.read_text())['expires_at'] > __import__('time').time() + 3000
        assert prepare(base, (0, .1), '10.0.0.2') != first


def test_own_load_withdraws_own_oldest_ticket_when_the_cache_has_no_room(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 2)
    with serving() as base:
        oldest = prepare(base, (0, .1))
        prepare(base, (.05, .2))
        newest = prepare(base, (.1, .3))
        assert served(oldest) is None and served(newest)


def test_tickets_of_other_clients_filling_the_cache_refuse_new_work_instead_of_evicting(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 2)
    with serving() as base:
        held = [prepare(base, (0, .1), '10.0.0.1'), prepare(base, (.05, .2), '10.0.0.2')]
        with pytest.raises(urllib.error.HTTPError) as error:
            prepare(base, (.1, .3), '10.0.0.3')
        assert error.value.code == 503 and json.loads(error.value.read())['retryable'] is True
        assert all(get(base + 'api/footage?ticket=' + token) for token in held)


@pytest.mark.parametrize('bound, host, allowed', [
    ('127.0.0.1', 'localhost', True), ('127.0.0.1', '127.0.0.1', True), ('127.0.0.1', '192.168.1.5', False),
    ('127.0.0.1', 'evil.example', False), ('127.0.0.1', 'box.tail1234.ts.net', False),
    ('0.0.0.0', 'evil.example', False), ('0.0.0.0', '192.168.1.5', True), ('0.0.0.0', 'localhost', True),
    ('192.168.1.5', 'rebind.attacker.example', False), ('192.168.1.5', '192.168.1.5', True),
    ('::', '::1', True), ('::', 'fe80::1', True), ('0.0.0.0', 'board.lan', False),
])
def test_host_check_on_loopback_and_network_boards(bound, host, allowed):
    assert serve.host_allowed(host, bound) is allowed
    assert serve.host_allowed(host.upper() + '.', bound, (host,))   # --allow-host admits any name, in any case


def test_network_board_answers_to_this_machines_name():
    assert all(serve.host_allowed(name, '0.0.0.0') for name in serve.machine_names())
    assert not serve.host_allowed(next(iter(serve.machine_names())), '127.0.0.1')


def test_refused_host_names_the_host_and_the_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    with serving() as base:
        code, _, body = status(urllib.request.Request(base + 'api/episodes', headers={'Host': 'box.tail1234.ts.net'}))
        assert code == 403
        message = json.loads(body)['error']
        assert 'box.tail1234.ts.net' in message and '--allow-host box.tail1234.ts.net' in message
        code, _, body = status(urllib.request.Request(base + 'api/episodes', headers={'Host': '<b>x'}))
        assert code == 403 and b'<' not in body


def test_clip_and_frame_revalidate_and_a_replaced_clip_is_never_served_stale(tmp_path, monkeypatch):
    source = media(tmp_path, monkeypatch)
    paths, tags = ('api/video?id=episode_x&cam=exo', 'api/frame?id=episode_x&cam=exo&t=0.1&w=64'), {}
    with serving() as base:
        for path in paths:
            code, headers, first = status(base + path)
            assert code == 200 and headers['Cache-Control'] == 'no-cache' and headers['ETag']
            tag = tags[path] = headers['ETag']
            code, headers, body = status(urllib.request.Request(base + path, headers={'If-None-Match': tag}))
            assert code == 304 and body == b'' and headers['ETag'] == tag
            if 'video' in path:
                ranged = urllib.request.Request(base + path, headers={'Range': 'bytes=0-9', 'If-Range': '"old"'})
                code, _, body = status(ranged)
                assert code == 200 and body == first
        before = source.stat()
        replacement = source.with_suffix('.new')
        replacement.write_bytes((tmp_path / 'blue.mp4').read_bytes().ljust(before.st_size, b'\0'))
        os.replace(replacement, source)
        for path in paths:
            code, headers, body = status(urllib.request.Request(base + path, headers={'If-None-Match': tags[path]}))
            assert code == 200 and body and headers['ETag'] != tags[path]
