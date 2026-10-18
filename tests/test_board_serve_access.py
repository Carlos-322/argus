"""Per-client download tickets of the live board."""
import json
import os
import urllib.error
import urllib.request

import pytest

from board import serve, static
from test_prepared_footage import media, serving, get


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
        serve._TICKETS.clear()
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
        assert serve.ticket_footage(oldest) is None and serve.ticket_footage(newest)


def test_tickets_of_other_clients_filling_the_cache_refuse_new_work_instead_of_evicting(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 2)
    with serving() as base:
        held = [prepare(base, (0, .1), '10.0.0.1'), prepare(base, (.05, .2), '10.0.0.2')]
        with pytest.raises(urllib.error.HTTPError) as error:
            prepare(base, (.1, .3), '10.0.0.3')
        assert error.value.code == 503 and json.loads(error.value.read())['retryable'] is True
        assert all(get(base + 'api/footage?ticket=' + token) for token in held)
