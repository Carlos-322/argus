"""An episode is shown and labelled beside its cameras, so nothing that goes wrong reading another part of it (its
state, signals, text, metadata, a depth stream, a sensor file, its table) may cost the cameras. Every such failure is
recorded on the episode with its type, and clean inputs record none, which is what keeps a reader regression visible."""
import io
import json

import h5py
import numpy as np
import pytest
from PIL import Image

from prepare import formats
from test_ingestion_v5 import lerobot_source

DAMAGE = ('part_unreadable', 'attribute_unreadable', 'adapter_failed')


def issues(ep_dir):
    ctx = json.loads((ep_dir / 'context.json').read_text())
    return ctx, [i for i in ctx.get('reader_issues') or [] if i['kind'] in DAMAGE]


def jpeg(value):
    out = io.BytesIO()
    Image.new('RGB', (64, 48), (value, 40, 90)).save(out, 'JPEG')
    return np.frombuffer(out.getvalue(), dtype=np.uint8)


def hdf5_episode(path, *, junk_front=False, rows=8):
    with h5py.File(path, 'w') as h:
        h.attrs['fps'] = 30
        images = h.create_group('observations/images')
        vlen = h5py.vlen_dtype(np.uint8)
        front = images.create_dataset('front', (rows,), dtype=vlen)
        wrist = images.create_dataset('wrist', (rows,), dtype=vlen)
        for i in range(rows):
            front[i] = np.frombuffer(b'not a picture at all', dtype=np.uint8) if junk_front else jpeg(10 * i)
            wrist[i] = jpeg(200 - 10 * i)
        h['observations/qpos'] = np.cumsum(np.full((rows, 7), .01, np.float32), 0)
        h['action'] = np.cumsum(np.full((rows, 7), .01, np.float32), 0)
        h['timestamp'] = np.arange(rows) / 30


def only_episode(out):
    (ep,) = [p for p in out.iterdir() if p.name.startswith('episode_')]
    return ep


def test_clean_inputs_record_no_damage(tmp_path):
    lerobot_source(tmp_path / 'lr', length=6, offset=0)
    report = formats.convert(tmp_path / 'lr', 'teleop_arms', tmp_path / 'out_lr', 'clean', 900)
    assert report['episodes'] and not report['failed']
    assert issues(only_episode(tmp_path / 'out_lr'))[1] == []
    (tmp_path / 'h5').mkdir()
    hdf5_episode(tmp_path / 'h5' / 'ep.h5')
    report = formats.convert(tmp_path / 'h5', 'teleop_arms', tmp_path / 'out_h5', 'clean', 900)
    assert report['episodes'] and not report['failed']
    assert issues(only_episode(tmp_path / 'out_h5'))[1] == []


def test_table_reader_fault_costs_the_table_alone_and_is_named(tmp_path, monkeypatch):
    root = tmp_path / 'source'
    lerobot_source(root, length=6, offset=0)

    def broken_reader(*args, **kwargs):
        raise RuntimeError('table reader implementation regression')
    monkeypatch.setattr(formats, '_read_episode_table', broken_reader)
    report = formats.convert(root, 'teleop_arms', tmp_path / 'out', 'reader fault', 900)
    assert report['episodes'] and not report['failed']
    ctx = json.loads((only_episode(tmp_path / 'out') / 'context.json').read_text())
    assert ctx['cameras'] and ctx['state_kind'] == 'none'
    assert 'RuntimeError' in json.dumps(ctx) and 'table reader implementation regression' in json.dumps(ctx)


def test_hdf5_signals_fault_keeps_the_cameras(tmp_path, monkeypatch):
    (tmp_path / 'up').mkdir()
    hdf5_episode(tmp_path / 'up' / 'ep.h5')

    def broken(*args, **kwargs):
        raise OSError('Unable to synchronously read data (wrong B-tree signature)')
    monkeypatch.setattr(formats, 'h5_signals', broken)
    report = formats.convert(tmp_path / 'up', 'teleop_arms', tmp_path / 'out', 'signals fault', 900)
    assert report['episodes'] and not report['failed']
    ctx, damage = issues(only_episode(tmp_path / 'out'))
    assert ctx['cameras'] and any('recorded signals' in i['what'] and 'OSError' in i['what'] for i in damage)


def test_unreadable_chosen_hdf5_camera_is_replaced_by_a_readable_one(tmp_path):
    (tmp_path / 'up').mkdir()
    hdf5_episode(tmp_path / 'up' / 'ep.h5', junk_front=True)
    report = formats.convert(tmp_path / 'up', 'ego_head', tmp_path / 'out', 'spare camera', 900)
    assert report['episodes'] and not report['failed'], report
    ctx = json.loads((only_episode(tmp_path / 'out') / 'context.json').read_text())
    assert [c.get('key') or c.get('source') or c for c in ctx['cameras'].values()]
    assert 'observations/images/front' not in json.dumps(ctx['cameras'])


def test_failed_state_write_keeps_the_episode_with_its_cameras(tmp_path, monkeypatch):
    (tmp_path / 'up').mkdir()
    hdf5_episode(tmp_path / 'up' / 'ep.h5')
    original = formats.video_views_episode

    def fails_with_state(ep, files, rig, dataset, extra, **kw):
        if kw.get('state') is not None:
            raise ValueError('state rows could not be written')
        return original(ep, files, rig, dataset, extra, **kw)
    monkeypatch.setattr(formats, 'video_views_episode', fails_with_state)
    report = formats.convert(tmp_path / 'up', 'teleop_arms', tmp_path / 'out', 'write fault', 900)
    assert report['episodes'] and not report['failed']
    ctx, damage = issues(only_episode(tmp_path / 'out'))
    assert ctx['cameras'] and any('could not be written' in i['what'] for i in damage)
