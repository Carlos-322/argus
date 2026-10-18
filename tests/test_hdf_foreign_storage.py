"""An uploaded HDF5 file can never make prepare read a file outside the upload.

HDF5 resolves external raw storage against the working directory, falls back to the working directory for links
and virtual sources missing next to the file, and honours absolute names. A file that names any outside storage is
refused as a whole at open, so no reader, whatever dataset name it asks for, can be steered to a host file.
"""
import json
import os

import h5py
import numpy as np
import pytest

from prepare import formats
from prepare.hdf_metadata import ForeignStorage, authorized_sources, foreign_references, open_hdf5

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


@pytest.mark.parametrize('name', ['secret.bin', 'pyproject.toml', '../outside.bin', '../../outside.bin', 'ABSOLUTE'])
def test_external_raw_storage_is_refused_whatever_its_name(host, tmp_path, name):
    _, upload = host
    target = str(tmp_path / 'outside.bin') if name == 'ABSOLUTE' else name
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        h.create_dataset('note', shape=(len(SECRET),), dtype='u1', external=[(target, 0, len(SECRET))])
    with authorized_sources(list(upload.iterdir())):
        assert foreign_references(path)
        with pytest.raises(ForeignStorage):
            open_hdf5(path)


def test_external_link_missing_next_to_the_file_is_refused(host):
    cwd, upload = host
    with h5py.File(cwd / 'host.h5', 'w') as h:
        h['v'] = np.frombuffer(SECRET, dtype=np.uint8)
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        h['observations/qvel'] = h5py.ExternalLink('host.h5', '/v')   # HDF5 falls back to the working directory
    with authorized_sources(list(upload.iterdir())):
        with pytest.raises(ForeignStorage):
            open_hdf5(path)


def test_virtual_source_outside_the_upload_is_refused(host, tmp_path):
    _, upload = host
    with h5py.File(tmp_path / 'source.h5', 'w') as h:
        h['v'] = np.frombuffer(SECRET, dtype=np.uint8)
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        layout = h5py.VirtualLayout(shape=(len(SECRET),), dtype='u1')
        layout[:] = h5py.VirtualSource(str(tmp_path / 'source.h5'), 'v', shape=(len(SECRET),))
        h.create_virtual_dataset('vds', layout)
    with authorized_sources(list(upload.iterdir())):
        with pytest.raises(ForeignStorage):
            open_hdf5(path)


def test_virtual_source_inside_the_upload_that_itself_names_outside_storage_is_refused(host):
    _, upload = host
    with h5py.File(upload / 'inner.h5', 'w') as h:
        h.create_dataset('v', shape=(len(SECRET),), dtype='u1', external=[('secret.bin', 0, len(SECRET))])
    path = upload / 'ep0.h5'
    with h5py.File(path, 'w') as h:
        camera(h)
        layout = h5py.VirtualLayout(shape=(len(SECRET),), dtype='u1')
        layout[:] = h5py.VirtualSource('inner.h5', 'v', shape=(len(SECRET),))
        h.create_virtual_dataset('vds', layout)
    with authorized_sources(list(upload.iterdir())):
        problems = foreign_references(path)
    assert problems and 'inner.h5' in problems[0]


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
    with authorized_sources(list(upload.iterdir())):
        assert foreign_references(path) == []
        with open_hdf5(path) as f:
            np.testing.assert_array_equal(f['vds'][:], np.arange(6))


def test_prepare_never_publishes_host_bytes(host, tmp_path):
    cwd, upload = host
    for p in list(upload.iterdir()):
        p.unlink()
    (upload / 'pyproject.toml').write_bytes(b'decoy-in-the-upload!!')
    with h5py.File(upload / 'ep0.h5', 'w') as h:
        camera(h)
        h.create_dataset('leaked_note', shape=(), dtype=h5py.string_dtype(length=len(SECRET)))
    # rewrite the dataset as external fixed-length storage, as a crafted upload would
    with h5py.File(upload / 'ep0.h5', 'a') as h:
        del h['leaked_note']
        h.create_dataset('leaked_note', shape=(len(SECRET),), dtype='u1',
                         external=[('pyproject.toml', 0, len(SECRET))])
    out = tmp_path / 'out'
    try:
        report = formats.convert(upload, 'teleop_arms', out, 'test', 3600)
    except formats.UploadInputError as error:
        report = {'error': str(error)}
    text = json.dumps(report)
    for written in (out.rglob('*') if out.exists() else []):
        if written.is_file():
            assert SECRET not in written.read_bytes(), written
    assert SECRET.decode() not in text
    assert 'outside the upload' in text
