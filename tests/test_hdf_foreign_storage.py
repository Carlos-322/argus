"""An uploaded HDF5 file can never make prepare read a file outside the upload, and an object that names outside
storage costs that object, never the file's cameras.

HDF5 resolves external raw storage against the working directory and honours absolute names, so a dataset whose
raw storage names a file is read only when that name lands on an upload file however HDF5 resolves it. Links and
virtual sources resolve next to the file and are admitted only when that file is part of the upload.
"""
import json

import h5py
import numpy as np
import pytest

from prepare import formats
from prepare.hdf_metadata import authorized_sources, objects

SECRET = b'HOST-SECRET-0123456789'


def camera(h, rows=6):
    h['observations/images/cam_high'] = np.zeros((rows, 24, 32, 3), dtype=np.uint8)
    h['observations/qpos'] = np.zeros((rows, 7), dtype=np.float32)


@pytest.fixture
def host(tmp_path, monkeypatch):
    """A working directory holding host files, and an upload folder holding same-named decoys."""
    cwd, upload = tmp_path / 'cwd', tmp_path / 'upload'
    cwd.mkdir()
    upload.mkdir()
    for name in ('secret.bin', 'pyproject.toml', 'other.h5'):
        (cwd / name).write_bytes(SECRET)
        (upload / name).write_bytes(b'decoy-in-the-upload!!')
    (tmp_path / 'outside.bin').write_bytes(SECRET)
    monkeypatch.chdir(cwd)
    return cwd, upload


def walked(path, upload):
    with authorized_sources(list(upload.iterdir())), h5py.File(path, 'r') as f:
        issues = []
        names = [name for name, _ in objects(f, issues=issues)]
    return names, issues


def assert_no_secret(out, report):
    for written in (out.rglob('*') if out.exists() else []):
        if written.is_file():
            assert SECRET not in written.read_bytes(), written
    assert SECRET.decode() not in json.dumps(report, default=str)


@pytest.mark.parametrize('name', ['secret.bin', 'pyproject.toml', '../outside.bin', '../../outside.bin', 'ABSOLUTE'])
def test_external_raw_storage_outside_the_upload_is_excluded_whatever_its_name(host, tmp_path, name):
    _, upload = host
    target = str(tmp_path / 'outside.bin') if name == 'ABSOLUTE' else name
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        h.create_dataset('note', shape=(len(SECRET),), dtype='u1', external=[(target, 0, len(SECRET))])
    names, issues = walked(path, upload)
    assert 'note' not in names and 'observations/images/cam_high' in names
    assert any(i['kind'] == 'external_source_unread' and '/note' in i['what'] for i in issues)


def test_external_link_missing_next_to_the_file_is_excluded(host):
    cwd, upload = host
    with h5py.File(cwd / 'host.h5', 'w') as h:
        h['v'] = np.frombuffer(SECRET, dtype=np.uint8)
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        h['observations/qvel'] = h5py.ExternalLink('host.h5', '/v')   # HDF5 falls back to the working directory
    names, issues = walked(path, upload)
    assert 'observations/qvel' not in names and 'observations/qpos' in names
    assert any(i['kind'] == 'external_source_unread' for i in issues)


def test_virtual_source_outside_the_upload_is_excluded(host, tmp_path):
    _, upload = host
    with h5py.File(tmp_path / 'source.h5', 'w') as h:
        h['v'] = np.frombuffer(SECRET, dtype=np.uint8)
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        layout = h5py.VirtualLayout(shape=(len(SECRET),), dtype='u1')
        layout[:] = h5py.VirtualSource(str(tmp_path / 'source.h5'), 'v', shape=(len(SECRET),))
        h.create_virtual_dataset('vds', layout)
    names, _ = walked(path, upload)
    assert 'vds' not in names and 'observations/images/cam_high' in names


def test_virtual_source_inside_the_upload_that_itself_names_outside_storage_is_excluded(host):
    _, upload = host
    with h5py.File(upload / 'inner.h5', 'w') as h:
        h.create_dataset('v', shape=(len(SECRET),), dtype='u1', external=[('secret.bin', 0, len(SECRET))])
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        layout = h5py.VirtualLayout(shape=(len(SECRET),), dtype='u1')
        layout[:] = h5py.VirtualSource('inner.h5', 'v', shape=(len(SECRET),))
        h.create_virtual_dataset('vds', layout)
    names, _ = walked(path, upload)
    assert 'vds' not in names


def test_links_and_virtual_sources_inside_the_upload_are_read(host):
    _, upload = host
    with h5py.File(upload / 'inner.h5', 'w') as h:
        h['v'] = np.arange(6, dtype=np.float32)
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        h['linked'] = h5py.ExternalLink('inner.h5', '/v')
        layout = h5py.VirtualLayout(shape=(6,), dtype='f4')
        layout[:] = h5py.VirtualSource('inner.h5', 'v', shape=(6,))
        h.create_virtual_dataset('vds', layout)
        h['soft'] = h5py.SoftLink('/observations/qpos')
    names, issues = walked(path, upload)
    assert {'linked', 'vds', 'soft'} <= set(names) and not issues


@pytest.mark.parametrize('name', ['pyproject.toml', 'ABSOLUTE'])
def test_prepare_keeps_the_camera_and_never_publishes_host_bytes(host, tmp_path, name):
    cwd, upload = host
    for p in list(upload.iterdir()):
        p.unlink()
    (upload / 'pyproject.toml').write_bytes(b'decoy-in-the-upload!!')
    target = str(tmp_path / 'outside.bin') if name == 'ABSOLUTE' else name
    with h5py.File(upload / 'ep0.h5', 'w') as h:
        camera(h)
        h.attrs['fps'] = 10
        h.create_dataset('leaked_note', shape=(len(SECRET),), dtype='u1', external=[(target, 0, len(SECRET))])
    out = tmp_path / 'out'
    report = formats.convert(upload, 'teleop_arms', out, 'test', 3600)
    assert len(report['episodes']) == 1 and not report['failed']
    context = json.loads((out / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    assert context['cameras']
    assert any(i['kind'] == 'external_source_unread' for i in context['reader_issues'])
    assert_no_secret(out, report)
