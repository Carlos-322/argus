"""A video used as it was uploaded is decoded once while it is prepared, so damage the decoder conceals by showing
another picture is named on the episode with its span instead of playing as footage."""
import json

import av
import numpy as np

from prepare import formats


def clip(path, frames=60):
    rng = np.random.default_rng(7)
    with av.open(str(path), 'w') as out:
        stream = out.add_stream('h264', rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 48, 'yuv420p'
        stream.options = {'g': '15'}
        for i in range(frames):
            picture = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_ndarray(picture, format='rgb24')):
                out.mux(packet)
        for packet in stream.encode():
            out.mux(packet)


def damage(path):
    raw = bytearray(path.read_bytes())
    data = raw.find(b'mdat') + 4
    middle = data + (len(raw) - data) // 2
    for i in range(middle, middle + 400, 7):
        raw[i] ^= 0x5A
    path.write_bytes(bytes(raw))


def test_a_clean_video_has_no_damaged_frames(tmp_path):
    clip(tmp_path / 'clean.mp4')
    assert formats.decoded_damage(tmp_path / 'clean.mp4') == set()


def test_concealed_damage_is_named_with_its_span(tmp_path):
    upload = tmp_path / 'upload'
    upload.mkdir()
    clip(upload / 'front.mp4')
    damage(upload / 'front.mp4')
    assert formats.decoded_damage(upload / 'front.mp4')
    report = formats.convert(upload, 'ego_head', tmp_path / 'out', 'damaged video', 900)
    assert len(report['episodes']) == 1 and not report['failed'], report
    ctx = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    damaged = [i for i in ctx['reader_issues'] if i['kind'] == 'frames_damaged']
    assert damaged and all(0 <= i['t0_s'] <= i['t1_s'] <= 2.0 for i in damaged)
