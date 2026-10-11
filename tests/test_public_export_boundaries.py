"""Exercise real public serializers using synthetic private access descriptors."""
import io
import json
from types import SimpleNamespace

import pytest

from board import serve, static


PRIVATE_PATH = '/srv/private-upload/source.h5'
PRIVATE_URL = 'https://example.org/object?X-Amz-Signature=synthetic-secret'
CITATION = 'https://example.org/public-dataset'


class Request(serve.Handler):
    def __init__(self, path, body=None, headers=None):
        self.path = path
        encoded = json.dumps(body or {}).encode()
        self.headers = {'Content-Length': str(len(encoded)), **(headers or {})}
        self.rfile, self.wfile = io.BytesIO(encoded), io.BytesIO()
        self.status = None

    def send_response(self, status):
        self.status = status

    def send_header(self, *args):
        pass

    def end_headers(self):
        pass


@pytest.fixture
def public_sources(tmp_path, monkeypatch):
    paths = {name: tmp_path / name for name in ('qa', 'compare', 'sensors', 'hands', 'keypoints', 'clips')}
    for path in paths.values():
        path.mkdir()
    descriptor = {'source_path': PRIVATE_PATH, 'url': PRIVATE_URL,
                  'source': {'key': '/observations/force'}, 'public_citation': CITATION}
    label = {'dataset': 'synthetic', 'episode_id': 'ep', 'episode_prompt': 'Place the cup.',
             'duration_s': 1, 'publisher_labels': {'label': 'public label'},
             'data_dictionary': {'fields': [descriptor]},
             'evidence_inspection': {'parts': [{'source_location': PRIVATE_PATH, 'record': descriptor}]},
             'sensor_evidence': {'sensors': [descriptor]},
             '_withheld_checks': {'sensor_check': {'reason': 'unavailable', 'result': {'error': 'unavailable',
                                                                                      'source_path': PRIVATE_PATH}}}}
    (paths['qa'] / 'ep.json').write_text(json.dumps(label))
    for kind in ('sensors', 'hands', 'keypoints'):
        (paths[kind] / 'ep.json').write_text(json.dumps({'series': [], **descriptor}))
        (paths[kind] / 'index.json').write_text(json.dumps({'files': {'ep.json': {'bytes': 17, **descriptor}}}))
    for name in ('index.json', 'metrics.json'):
        (paths['compare'] / name).write_text(json.dumps({'records': [descriptor]}))
    (paths['compare'] / 'model').mkdir()
    (paths['compare'] / 'model' / 'ep.json').write_text(json.dumps(label))
    for name, key in (('HERE', 'qa'), ('COMPARE_DIR', 'compare'), ('SENSORS_DIR', 'sensors'),
                      ('HANDS_DIR', 'hands'), ('KEYPOINTS_DIR', 'keypoints')):
        monkeypatch.setattr(serve, name, paths[key])
    monkeypatch.setattr(serve, 'DICTIONARY_JOB', None)
    return paths


@pytest.mark.parametrize('route,body', [
    ('/api/episode?file=ep.json', None), ('/api/episode?file=ep.json&download=1', None),
    ('/api/export', {'files': ['ep.json']}), ('/api/dictionary?file=ep.json', None),
    ('/api/sensors?file=ep.json', None), ('/api/sensors?file=index.json', None),
    ('/api/hands?file=ep.json', None), ('/api/keypoints?file=ep.json', None),
    ('/api/keypoints?file=ep.json&download=1', None),
    ('/api/compare/index', None), ('/api/compare/metrics', None),
    ('/api/compare/episode?key=model&file=ep.json', None),
])
def test_live_public_json_boundaries(public_sources, route, body):
    before = {p: p.read_bytes() for directory in public_sources.values() for p in directory.rglob('*.json')}
    request = Request(route, body)
    request.do_POST() if body else request.do_GET()
    text = request.wfile.getvalue().decode()
    assert request.status == 200
    assert PRIVATE_PATH not in text and PRIVATE_URL not in text
    assert CITATION in text
    assert '/observations/force' in text
    if route.startswith(('/api/episode', '/api/export', '/api/compare/episode')):
        assert 'public label' in text
    assert all(p.read_bytes() == content for p, content in before.items())


@pytest.mark.parametrize('route', ['/api/episode?file=../outside.json',
                                  '/api/dictionary?file=../outside.json',
                                  '/api/sensors?file=../outside.json'])
def test_projection_does_not_replace_containment(public_sources, route):
    (public_sources['qa'].parent / 'outside.json').write_text('{"private": true}')
    request = Request(route)
    request.do_GET()
    assert request.status == 404


def test_projection_does_not_enable_remote_dictionary_writes(public_sources, monkeypatch):
    monkeypatch.setattr(serve, 'DICTIONARY_JOB', public_sources['qa'].parent)
    request = Request('/api/dictionary', {'revision': 0},
                      {'Host': 'example.org', 'Origin': 'http://example.org', 'Content-Type': 'application/json'})
    request.do_POST()
    assert request.status == 403
    assert not (public_sources['qa'].parent / 'dictionary_overrides.json').exists()


def test_static_json_boundaries(public_sources, tmp_path, monkeypatch):
    monkeypatch.setattr(static, 'plan', lambda *a, **k: [
        {'rec': {'file': 'ep.json', 'episode_id': 'ep', 'duration_s': 1},
         'ds': 'synthetic', 'media': {}, 'frames': {}}])
    args = SimpleNamespace(board=tmp_path, **public_sources, out=tmp_path / 'site',
                           build_id='synthetic', force=False, public_base=None, title='Synthetic', header=None)
    before = {p: p.read_bytes() for directory in public_sources.values() for p in directory.rglob('*.json')}
    assert static.cmd_site(args) == 0
    exported = {p.relative_to(args.out).as_posix(): p.read_text() for p in args.out.rglob('*.json')}
    assert 'synthetic/data/ep/ep.json' in exported
    assert 'synthetic/data/sensors/ep.json' in exported
    assert 'synthetic/data/compare/index.json' in exported
    assert 'synthetic/data/keypoints/index.json' in exported
    assert any(name.startswith('media/k/') for name in exported)
    for name, text in exported.items():
        assert PRIVATE_PATH not in text and PRIVATE_URL not in text, name
        if name.endswith('BUILD.json'):
            assert str(tmp_path) not in text
    assert CITATION in exported['synthetic/data/ep/ep.json']
    assert 'public label' in exported['synthetic/data/ep/ep.json']
    assert all(p.read_bytes() == content for p, content in before.items())
