"""Release evidence keeps camera clocks, current bytes and source metadata."""
import json
import subprocess
import urllib.request

import pytest

from board import serve, static
from board.public_export import project
from test_prepared_footage import media, serving, get


def test_unpaired_camera_survives_export_and_trim_without_entering_synchronized_footage(tmp_path, monkeypatch):
    import av
    import io
    import numpy as np
    from board import build
    from prepare.camera_clock import unshown_span
    main = media(tmp_path, monkeypatch)
    other = main.parent / 'unshown1' / main.name
    other.parent.mkdir(); other.write_bytes((tmp_path / 'blue.mp4').read_bytes())
    np.savez(tmp_path / 'other_times.npz', presentation=100 + np.arange(9) / 30,
             capture=2000 + np.arange(9) / 30, pts=np.arange(9))
    entry = {'name': 'independent wrist', 'packed': str(other), 'why': 'Distinct publisher clock',
             'n_frames': 9, 'fps': 30, 'start_s': 0, 'camera_times': 'other_times.npz', 'clock_alignment': 'unverified',
             'presentation_clock': {'basis': 'native_unpaired'}}
    retained = unshown_span(tmp_path, [entry], .1, .2, shift=True)[0]
    assert retained['n_frames'] == 9 and retained.get('frame_span') is None
    assert retained['presentation_clock'] == entry['presentation_clock']
    from prepare.camera_clock import unshown_times
    np.testing.assert_array_equal(unshown_times(tmp_path, entry, 15)[0], 100 + np.arange(9) / 30)
    from prepare.formats import shift_context_times
    assert shift_context_times({'unshown_cameras': [entry]}, -10)['unshown_cameras'][0] == entry
    label = {'camera_views': ['exo'], 'camera_labels': ['head'], 'episode_prompt': 'fixture',
             '_meta': {'episode_id': 'episode_x'}}
    build.add_context(label, {'profile': 'ego_head', 'fps': 30, 'n_state_frames': 9,
                            'unshown_cameras': [entry]}, tmp_path)
    assert label['unshown_cameras'][0]['clock_alignment'] == 'unverified'
    label['camera_presentation'] = {'unshown1': {'role': 'scene'}}
    assert static.shown_cams(label) == ['exo', 'unshown1']
    qa = tmp_path / 'qa'; qa.mkdir()
    (qa / 'episode_x.json').write_text(json.dumps(label))
    with serving() as base:
        own = get(base + 'api/video?id=episode_x&cam=unshown1')
        combined = get(base + 'api/footage?id=episode_x')
    with av.open(io.BytesIO(own)) as container:
        blue = next(container.decode(video=0)).to_ndarray(format='rgb24').mean(axis=(0, 1))
    with av.open(io.BytesIO(combined)) as container:
        frame = next(container.decode(video=0))
        red = frame.to_ndarray(format='rgb24').mean(axis=(0, 1))
        assert frame.width == frame.height == 16
    assert blue[2] > 200 and blue[0] < 20
    assert red[0] > 200 and red[2] < 20


@pytest.mark.parametrize('prefix,count', [('sk-or-v1-',64), ('sk-proj-',100), ('sk-',48),
    ('hf_',34), ('glpat-',20), ('AIza',35), ('npm_',36), ('SG.'+'a'*22+'.',43)])
def test_provider_values_withheld_in_native_fields(prefix, count):
    assert project({'fields': {'note': prefix + 'a' * count}}) == {'fields': {}}


def test_native_semantics_survive_access_filter():
    native = {'pass': True, 'pwd': 'calib/run_03', 'tokens': ['pick', 'up'],
              'jwt': 'not-used', 'csrf': 'disabled', 'firmware': '10.3.0.17',
              'fw': '192.168.1.20', 'build': '10.0.0.5', 'bios': '10.2.1.0',
              'citation': 'https://example.org/paper#Token', 'eos_token': '</s>'}
    assert project({'fields': native}) == {'fields': native}
    assert project({'fields': {'hf': {'token': 'opaque'}, 'wandb_key': 'opaque',
                               'signing_key': 'opaque', 'npm_token': 'opaque'}}) == {'fields': {'hf': {}}}
    assert project({'endpoint': 'fd7a:115c:a1e0::5'}) == {}


def test_live_video_and_poster_require_browser_revalidation(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    with serving() as base:
        for route in ('api/video?id=episode_x&cam=exo', 'api/frame?id=episode_x&cam=exo&t=0'):
            with urllib.request.urlopen(base + route) as response:
                assert 'no-store' in response.headers['Cache-Control']


@pytest.mark.parametrize('camera_start,camera_duration,t0', [(5,3,6), (0,5,10)])
def test_span_holds_same_camera_pixels_as_whole(tmp_path, monkeypatch, camera_start, camera_duration, t0):
    import av
    import numpy as np
    clips = tmp_path / 'clips'
    (clips / 'wrist_left').mkdir(parents=True)
    for destination, color, duration, start in [(clips/'episode_x.mp4', 'red', 12, 0),
                                               (clips/'wrist_left'/'episode_x.mp4', 'blue', camera_duration, camera_start)]:
        subprocess.run([serve.FFMPEG, '-v', 'error', '-f', 'lavfi', '-i',
                        f'color=c={color}:s=32x32:r=10:d={duration}', '-vf', f'setpts=PTS+{start}/TB',
                        '-vsync', '0', '-c:v', 'libx264', '-pix_fmt', 'yuv420p', str(destination)],
                       check=True, timeout=5)
    qa = tmp_path / 'qa'; qa.mkdir()
    (qa/'episode_x.json').write_text(json.dumps({'episode_prompt':'test', 'duration_s':12,
                                               'camera_views':['exo','left']}))
    monkeypatch.setattr(serve, 'HERE', qa)
    monkeypatch.setattr(serve, 'MP4_DIR', clips)
    monkeypatch.setattr(serve, 'FOOTAGE_DIR', tmp_path/'footage')
    with serving() as base:
        whole = get(base+'api/footage?id=episode_x')
        span = get(base+f'api/footage?id=episode_x&t0={t0}&t1={t0+1}')
    def pixel(data, when):
        import io
        with av.open(io.BytesIO(data)) as container:
            frames = list(container.decode(video=0))
        frame = min(frames, key=lambda frame: abs(float(frame.time)-when))
        return frame.to_ndarray(format='rgb24')[16, 48].astype(float)
    assert pixel(whole,t0+.5)[2] > 200
    assert np.max(np.abs(pixel(whole,t0+.5)-pixel(span,.5))) < 8


def test_deep_post_returns_structured_error(tmp_path, monkeypatch):
    import urllib.error
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    with serving() as base:
        request = urllib.request.Request(base+'api/export', data=b'['*2000+b']'*2000,
                                         headers={'Content-Type':'application/json'})
        with pytest.raises(urllib.error.HTTPError) as failure:
            get(request)
        assert failure.value.code == 400


def test_weak_list_serves_last_complete_snapshot_while_background_refresh_blocks(tmp_path, monkeypatch):
    import threading
    import time
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    path = tmp_path/'episode_x.json'
    path.write_text(json.dumps({'episode_prompt':'original'}))
    first = serve.list_episodes(tmp_path)
    entered, release = threading.Event(), threading.Event()
    read = serve._refresh_weak_list if hasattr(serve, '_refresh_weak_list') else None
    if read is None:
        monkeypatch.setattr(serve, '_label_identity', lambda path: (_ for _ in ()).throw(ValueError('changed')))
    else:
        def blocked(*args):
            entered.set()
            assert release.wait(5)
            return read(*args)
        monkeypatch.setattr(serve, '_refresh_weak_list', blocked)
        monkeypatch.setattr(serve, '_LIST_REFRESH_INTERVAL', 0)
    try:
        started = time.monotonic()
        assert serve.list_episodes(tmp_path) == first
        assert time.monotonic()-started < .5
        if read:
            assert entered.wait(1)
    finally:
        release.set()


def test_weak_list_keeps_complete_row_during_partial_write_and_refreshes_same_stat(tmp_path, monkeypatch):
    import os
    import time
    monkeypatch.setattr(static, '_durable_metadata', lambda *args: False)
    monkeypatch.setattr(serve, '_LIST_REFRESH_INTERVAL', 0, raising=False)
    path = tmp_path/'episode_x.json'
    path.write_text(json.dumps({'episode_prompt':'original'}))
    original = path.stat()
    assert serve.list_episodes(tmp_path)[0]['episode_prompt'] == 'original'
    path.write_text('{')
    assert serve.list_episodes(tmp_path)[0]['episode_prompt'] == 'original'
    path.write_text(json.dumps({'episode_prompt':'replaced'}))
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))
    deadline = time.monotonic()+3
    while time.monotonic() < deadline:
        rows = serve.list_episodes(tmp_path)
        if rows and rows[0]['episode_prompt'] == 'replaced':
            break
        time.sleep(.01)
    assert rows[0]['episode_prompt'] == 'replaced'


@pytest.mark.parametrize('secret', [
    'Use header Bearer ' + 'a'*32 + ' for the API',
    'X-Api-Key: '+ 'a'*32, 'PASSWORD='+ 'a'*32,
    'DefaultEndpointsProtocol=https;AccountKey='+'a'*40+';AccountName=fixture',
    'sv=2022-11-02&sp=r&sig='+'a'*40,
    'postgres://fixture:'+'a'*20+'@db.example.com/data',
    'SK'+'a'*32, 'AC'+'a'*32+':'+'b'*32,
    'xapp-1-A'+'A'*10+'-'+'1'*13+'-'+'b'*64,
    '-----BEGIN PGP PRIVATE KEY BLOCK-----',
])
def test_recognizable_access_syntax_withheld_from_prose(secret):
    assert project({'fields': {'description':secret}}) == {'fields':{}}


@pytest.mark.parametrize('text', ['Authorization: pending review',
    'Authorization: not required for this dataset', 'Bearer B205walks across room',
    'Type api_key on the panel.', 'The password is reset after each demonstration.'])
def test_access_vocabulary_without_credential_syntax_survives(text):
    assert project({'fields': {'description':text}}) == {'fields':{'description':text}}


def test_browser_cross_site_and_loopback_host_rebinding_are_rejected(tmp_path, monkeypatch):
    import urllib.error
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    with serving() as base:
        for headers in ({'Sec-Fetch-Site':'cross-site'}, {'Host':'attacker.invalid'}):
            with pytest.raises(urllib.error.HTTPError) as failure:
                get(urllib.request.Request(base+'api/episodes', headers=headers))
            assert failure.value.code == 403


def test_deep_label_returns_structured_error(tmp_path, monkeypatch):
    import urllib.error
    monkeypatch.setattr(serve, 'HERE', tmp_path)
    (tmp_path/'episode_x.json').write_text('['*2000+']'*2000)
    with serving() as base:
        with pytest.raises(urllib.error.HTTPError) as failure:
            get(base+'api/episode?file=episode_x.json')
        assert failure.value.code == 500


def test_header_deadline_is_absolute_despite_slow_byte_trickle(monkeypatch):
    import socket
    import time
    monkeypatch.setattr(serve.Handler, 'request_read_timeout', .15, raising=False)
    with serving() as base:
        host, port = base.split('/')[2].split(':')
        with socket.create_connection((host, int(port)), timeout=1) as connection:
            connection.sendall(b'GET /api/episodes HTTP/1.1\r\nHost: localhost\r\nX-Test: ')
            for _ in range(5):
                time.sleep(.04)
                try:
                    connection.sendall(b'a')
                except OSError:
                    break
            connection.settimeout(.2)
            try:
                assert connection.recv(1024) == b''
            except ConnectionResetError:
                pass


def test_static_refuses_empty_board_without_creating_build(tmp_path, capsys):
    from types import SimpleNamespace
    qa = tmp_path/'qa'; qa.mkdir()
    args = SimpleNamespace(public_base=None, qa=qa, clips=tmp_path/'clips', compare=None, out=tmp_path/'static', hands=None, keypoints=None, sensors=None,
                           title='Test', build_id=None, force=False, board=tmp_path, header=None)
    assert static.cmd_site(args) == 1
    assert 'No episodes' in capsys.readouterr().out
    assert not args.out.exists()


def test_credential_values_in_dictionary_keys_are_withheld():
    source = {'fields': {'ghp_'+'a'*36: 7, 'https://example.org/?api_key=unusable':3,
                         'Bearer '+'a'*24:2, '/observations/gripper/force':12}}
    assert project(source) == {'fields': {'/observations/gripper/force':12}}


def test_native_ambiguous_attribute_names_retain_publisher_semantics():
    native = {'pass':True, 'pwd':'pulse-width', 'jwt':'not-used', 'csrf':'disabled',
              'secret':'hidden-object'}
    assert project({'fields': native}) == {'fields': native}


@pytest.mark.parametrize('key', ['contraseña','passwort','mot_de_passe','пароль','密码','clave_api'])
def test_translated_access_attribute_names_are_withheld(key):
    assert project({'fields':{key:'opaquecredential'}}) == {'fields':{}}


def test_repeat_http_prepare_reuses_live_ticket_instead_of_evicting_other_download(tmp_path, monkeypatch):
    media(tmp_path, monkeypatch)
    monkeypatch.setattr(serve, '_FOOTAGE_DISK_FILES', 2)
    with serving() as base:
        first = json.loads(get(base+'api/footage?id=episode_x&t0=0&t1=.1&prepare=1'))
        second = json.loads(get(base+'api/footage?id=episode_x&t0=.1&t1=.2&prepare=1'))
        for _ in range(4):
            repeated = json.loads(get(base+'api/footage?id=episode_x&t0=.1&t1=.2&prepare=1'))
            assert repeated['ticket'] == second['ticket']
        assert get(base+'api/footage?ticket='+first['ticket'])


def test_recognizable_encoded_registry_auth_and_private_prose_are_withheld():
    import base64
    auth = base64.b64encode(b'fixture:unusablecredential').decode()
    encoded = base64.b64encode(json.dumps({'auths': {'ghcr.io': {'auth': auth}}}).encode()).decode()
    values = [encoded, 'ssh fixture@100.101.7.9 to pull',
              'rsync from pluto.tailc7b2a4.ts.net:/data/episode nightly']
    assert project({'fields': {'notes': values}}) == {'fields': {'notes': [None,None,None]}}
    safe = ['The SSH connection failed during calibration.',
            'See https://example.org/paper for the public source.',
            base64.b64encode(json.dumps({'camera': {'auth': 'calibration-state'}}).encode()).decode()]
    assert project({'fields': {'notes': safe}}) == {'fields': {'notes': safe}}


@pytest.mark.parametrize('private_prefix', ['', '/private/upload/'])
def test_credential_shaped_source_names_are_aliased_without_losing_measurements(private_prefix):
    from board.public_export import reader_context_field
    credential = 'ghp_' + 'a' * 36
    value = {private_prefix + credential + '.json': {'fields': {'force': 12}},
             private_prefix + credential + '.csv': {'fields': {'force': 24}}}
    shown = reader_context_field('recorded_metadata', value)
    assert credential not in json.dumps(shown)
    assert len(shown) == 2
    assert sorted(item['fields']['force'] for item in shown.values()) == [12, 24]
