"""A member whose owner an archive does not prove is an upload of its own in every way (prepare/formats.py
_unpack_archives, _convert_upload): its storage belongs to the conversion that read it, an archive it holds is opened
on the same budget, and nothing of another upload (a table, a task, folder metadata) is joined to its recordings.
Every name derived from another (label/atomic.py derived_name) is valid whenever its source is. Every count and list of
the pictures a video shows reads one definition (label/presented.py)."""
from __future__ import annotations

import io
import json
import shutil
import struct
import subprocess
import zipfile

import av
import numpy as np
import pytest

from label import atomic, episode
from label.presented import presented, presented_pts
from prepare import formats


def movie(level, frames=8):
    out = io.BytesIO()
    with av.open(out, 'w', format='mp4') as container:
        stream = container.add_stream('mpeg4', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        for _ in range(frames):
            picture = av.VideoFrame.from_ndarray(np.full((48, 64, 3), level, np.uint8), format='rgb24')
            for packet in stream.encode(picture):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return out.getvalue()


def zipped(members: dict) -> bytes:
    sink = io.BytesIO()
    with zipfile.ZipFile(sink, 'w', zipfile.ZIP_STORED) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return sink.getvalue()


def rename_first_local(raw: bytes, old: str, new: str) -> bytes:
    """The first member's own header names it `new`; its directory entry still says `old`."""
    raw = bytearray(raw)
    length = struct.unpack_from('<H', raw, 26)[0]
    assert raw[30:30 + length] == old.encode() and len(new) == len(old)
    raw[30:30 + length] = new.encode()
    return bytes(raw)


def contexts(out, report):
    return {e['episode_id']: json.loads((out / e['episode_id'] / 'context.json').read_text())
            for e in report['episodes']}


def mean_level(out, episode_id):
    loaded = episode.load(out / episode_id)
    view = next(iter(loaded['sources']))
    return round(float(np.asarray(episode._decode_view(loaded, view, [0])[0]).mean()))


@pytest.mark.parametrize('upload', ['folder', 'archive'])
def test_a_later_conversion_never_changes_the_footage_of_an_earlier_one(tmp_path, upload):
    """Two uploads converted beside each other: the second never writes where the first one's episodes read."""
    levels = {}
    for name, level in (('a', 40), ('b', 180)):
        raw = rename_first_local(zipped({'takeA/first.mp4': movie(level), 'healthy/front.mp4': movie(level)}),
                                 'takeA/first.mp4', 'takeB/first.mp4')
        source = tmp_path / f'source_{name}'
        source.mkdir()
        (source / 'recordings.zip').write_bytes(raw)
        root = source if upload == 'folder' else source / 'recordings.zip'
        out = tmp_path / f'out_{name}'
        report = formats.convert(root, 'ego_head', out, name, 900)
        assert len(report['episodes']) == 2 and not report['failed'], report
        levels[name] = (out, report)
    out_a, report_a = levels['a']
    for e in report_a['episodes']:
        assert abs(mean_level(out_a, e['episode_id']) - 40) < 8, e
    sources = [json.loads((out_a / e['episode_id'] / 'sources.json').read_text()) for e in report_a['episodes']]
    files_a = {v['packed'] for s in sources for v in s.values() if isinstance(v, dict) and v.get('packed')}
    out_b, report_b = levels['b']
    sources = [json.loads((out_b / e['episode_id'] / 'sources.json').read_text()) for e in report_b['episodes']]
    files_b = {v['packed'] for s in sources for v in s.values() if isinstance(v, dict) and v.get('packed')}
    assert files_a and files_b and not files_a & files_b


def test_a_member_kept_apart_that_is_an_archive_is_opened(tmp_path):
    inner = zipped({'take/front.mp4': movie(60)})
    raw = rename_first_local(zipped({'takeA/first.zip': inner, 'healthy/front.mp4': movie(200)}),
                             'takeA/first.zip', 'takeB/first.zip')
    path = tmp_path / 'up.zip'
    path.write_bytes(raw)
    out = tmp_path / 'out'
    report = formats.convert(path, 'ego_head', out, 'nested apart', 900)
    assert len(report['episodes']) == 2, report
    levels = sorted(mean_level(out, e['episode_id']) for e in report['episodes'])
    assert abs(levels[0] - 60) < 8 and abs(levels[1] - 200) < 8, levels
    apart = [e for e in report['episodes'] if 'name_unverified' in e['episode_id']]
    assert len(apart) == 1


def test_no_table_of_the_upload_describes_a_member_kept_apart(tmp_path):
    """The upload's own table names member.h5; the member kept apart is read as member.h5 too, but the table is not its
    upload's, so it gets no row, no task and no table in its source."""
    h5py = pytest.importorskip('h5py')
    files = {}
    for name, level in (('source.h5', 30), ('member.h5', 200)):
        with h5py.File(tmp_path / name, 'w') as h:
            h.attrs['fps'] = 30
            for group in ('demo_00', 'demo_01'):
                h[group + '/observations/images/front'] = np.full((8, 48, 64, 3), level, dtype=np.uint8)
        files[name] = (tmp_path / name).read_bytes()
    raw = bytearray(zipped({**files, 'annotations.csv': b'file,group,task\nmember.h5,demo_00,TASK OF THE INTACT FILE\n'}))
    central = raw.find(b'PK\x01\x02')
    raw[central + 46:central + 46 + 9] = b'member.h5'            # source.h5's directory entry now says member.h5
    path = tmp_path / 'up.zip'
    path.write_bytes(bytes(raw))
    out = tmp_path / 'out'
    report = formats.convert(path, 'ego_head', out, 'table ownership', 900)
    ctx = contexts(out, report)
    apart = {eid: c for eid, c in ctx.items() if 'name_unverified' in eid}
    intact = {eid: c for eid, c in ctx.items() if 'name_unverified' not in eid}
    assert apart and intact, report
    tasked = [eid for eid, c in ctx.items() if 'TASK OF THE INTACT FILE' in json.dumps(c)]
    assert tasked and all(eid in intact for eid in tasked), tasked
    for eid, c in apart.items():
        assert not c.get('uploader_notes') and not (c.get('source') or {}).get('tables'), c
        assert not (out / eid / 'instruction.txt').exists() or 'INTACT' not in (out / eid / 'instruction.txt').read_text()


def test_a_cut_with_preroll_has_no_damaged_frames_and_every_count_agrees(tmp_path):
    """A stream copy cut away from a key frame keeps the frames before its start as discarded preroll, which the decoder
    needs and never shows: no picture of it is damaged, and the probe, the frame count and the decoder agree."""
    ffmpeg = shutil.which('ffmpeg')
    if not ffmpeg:
        pytest.skip('ffmpeg is needed to cut a stream copy')
    clean = tmp_path / 'clean.mp4'
    with av.open(str(clean), 'w') as out:
        s = out.add_stream('libx264', rate=30)
        s.width, s.height, s.pix_fmt = 64, 48, 'yuv420p'
        s.options = {'g': '30', 'keyint_min': '30', 'sc_threshold': '0'}
        for i in range(90):
            for packet in s.encode(av.VideoFrame.from_ndarray(np.full((48, 64, 3), (i * 3) % 256, np.uint8),
                                                             format='rgb24')):
                out.mux(packet)
        for packet in s.encode():
            out.mux(packet)
    cut = tmp_path / 'cut.mp4'
    subprocess.run([ffmpeg, '-y', '-v', 'error', '-ss', '0.5', '-i', str(clean), '-t', '1', '-c:v', 'copy', str(cut)],
                   check=True)
    with av.open(str(cut)) as c:
        discarded = sum(1 for p in c.demux(video=0) if p.size and p.is_discard)
    with av.open(str(cut)) as c:
        decoded = sorted(int(f.pts) for f in c.decode(video=0))
    with av.open(str(cut)) as c:
        shown = presented_pts(c, c.streams.video[0])
    with av.open(str(cut)) as c:
        counted = sum(1 for _ in presented(c, c.streams.video[0]))
    assert discarded, 'the cut should carry preroll for this test to mean anything'
    assert shown == decoded and counted == len(decoded)
    assert formats.decoded_damage(cut) == set()
    out = tmp_path / 'out'
    report = formats.convert(cut, 'ego_head', out, 'preroll', 900)
    ctx = contexts(out, report)
    assert not any(i['kind'] == 'frames_damaged' for c in ctx.values() for i in c.get('reader_issues') or [])


@pytest.mark.parametrize('length', [1, 150, 191, 192, 220, 255])
def test_every_name_derived_from_a_valid_name_is_valid_and_its_own(tmp_path, length):
    name = 'x' * (length - 5) + '.json' if length > 5 else 'x' * length
    other = 'y' + name[1:]
    for prefix, suffix in (('.', f'.{"0" * 32}.tmp.mp4'), ('noreply_', ''), ('failed_', ''), ('.', '.route.lock'),
                           ('', '.round12.retry3.claim'), ('.', '.unpacked'), ('.noreply_', '.route.lock')):
        derived = atomic.derived_name(name, prefix, suffix)
        assert len(derived.encode()) <= atomic.NAME_MAX, (prefix, suffix, len(derived))
        assert derived != atomic.derived_name(other, prefix, suffix)
        assert derived.startswith(prefix) and derived.endswith(suffix)
        if len(name.encode()) <= atomic.NAME_MAX - atomic.DERIVED_RESERVE:
            assert derived == prefix + name + suffix          # a name that leaves room is kept as it is
    target = tmp_path / name
    atomic.write_atomic(target, {'ok': True})                  # its temporary file fits too
    assert json.loads(target.read_text()) == {'ok': True}
    assert sorted(p.name for p in tmp_path.iterdir()) == [name]
