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
    """Cut the picture data of the middle frame short: its slice is declared 8 bytes long, so the decoder runs out of
    data before the picture's last macroblock whatever encoder wrote the file. Bytes flipped at a fixed offset are
    not enough, because flipped bytes that still parse as valid data are invisible to any decoder, and where a fixed
    offset lands depends on the encoder's build (it was caught on macOS and missed on Linux)."""
    with av.open(str(path)) as container:
        packets = [(p.pos, p.size, p.pts) for p in container.demux(container.streams.video[0]) if p.size]
    pos, size, pts = packets[len(packets) // 2]
    raw = bytearray(path.read_bytes())
    at = pos
    while at + 5 <= pos + size:                     # length-prefixed NAL units, as MP4 stores H.264
        length, kind = int.from_bytes(raw[at:at + 4], 'big'), raw[at + 4] & 0x1F
        if kind in (1, 5):                          # a coded slice
            raw[at:at + 4] = (8).to_bytes(4, 'big')
            break
        at += 4 + length
    else:
        raise AssertionError('no coded slice in the middle packet')
    path.write_bytes(bytes(raw))
    return pts


def test_a_clean_video_has_no_damaged_frames(tmp_path):
    clip(tmp_path / 'clean.mp4')
    assert formats.decoded_damage(tmp_path / 'clean.mp4') == set()


def test_concealed_damage_is_named_with_its_span(tmp_path):
    upload = tmp_path / 'upload'
    upload.mkdir()
    clip(upload / 'front.mp4')
    cut = damage(upload / 'front.mp4')
    # the damaged picture itself is named, and nothing decoded before it: on several threads the decoder reports an
    # error packets later, and naming the packet in hand named a clean picture and missed the damaged one
    named = formats.decoded_damage(upload / 'front.mp4')
    assert cut in named and min(named) == cut, (cut, sorted(named))
    report = formats.convert(upload, 'ego_head', tmp_path / 'out', 'damaged video', 900)
    assert len(report['episodes']) == 1 and not report['failed'], report
    ctx = json.loads((tmp_path / 'out' / report['episodes'][0]['episode_id'] / 'context.json').read_text())
    damaged = [i for i in ctx['reader_issues'] if i['kind'] == 'frames_damaged']
    assert damaged and all(0 <= i['t0_s'] <= i['t1_s'] <= 2.0 for i in damaged)
