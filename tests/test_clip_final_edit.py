"""A fresh encode can discard its final packet even when every packet has a duration."""
import shutil
import struct
import subprocess

import av
import numpy as np
import pytest

from board import clips

pytestmark = pytest.mark.skipif(not shutil.which('ffmpeg'), reason='no ffmpeg')


def _shorten_final_edit(path, duration_ms):
    data = bytearray(path.read_bytes())
    def visit(start, end):
        offset = start
        while offset + 8 <= end:
            size, kind = struct.unpack_from('>I4s', data, offset)
            assert size >= 8
            if kind in (b'moov', b'trak', b'edts'):
                visit(offset + 8, offset + size)
            elif kind == b'elst':
                assert data[offset + 8] == 0
                count = struct.unpack_from('>I', data, offset + 12)[0]
                for index in range(count):
                    entry = offset + 16 + index * 12
                    media_time = struct.unpack_from('>i', data, entry + 4)[0]
                    if media_time >= 0:
                        struct.pack_into('>I', data, entry, duration_ms)
            offset += size
    visit(0, len(data))
    path.write_bytes(data)


@pytest.mark.parametrize('start', [0, 0.5])
def test_nonzero_packet_lengths_do_not_hide_the_last_encoded_frame(tmp_path, start):
    source, output = tmp_path/'source.mp4', tmp_path/'clip.mp4'
    subprocess.run([clips.find_ffmpeg(), '-y', '-v', 'error', '-f', 'lavfi', '-i',
                    'testsrc2=size=64x48:rate=30', '-frames:v', '90', '-c:v', 'libx264',
                    '-bf', '0', '-video_track_timescale', '30', '-threads', '1', str(source)], check=True)
    command = [clips.find_ffmpeg(), '-y', '-v', 'error', '-i', str(source), '-frames:v', '31',
               '-an', '-fps_mode', 'passthrough', *clips.video_args(64, 48, True, 1)]
    if start:
        command += ['-output_ts_offset', str(start)]
    subprocess.run([*command, str(output)], check=True)
    # Some muxers end the edit at the final presentation instant, excluding that frame.
    _shorten_final_edit(output, 1000)
    with av.open(str(output)) as container:
        packets = [packet for packet in container.demux(video=0) if packet.size]
    assert len(packets) == 31 and all(packet.duration > 0 for packet in packets)
    assert max(packets, key=lambda packet: packet.pts).is_discard
    assert clips.clip_frames(output) == 30
    with av.open(str(output), options={'ignore_editlist': '1'}) as container:
        original = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    assert len(original) == 31
    assert clips.frame_lengths(output) is True
    with av.open(str(output)) as container:
        stream = container.streams.video[0]
        frames = list(container.decode(stream))
        assert len(frames) == 31
        assert [float(frame.pts * frame.time_base) for frame in frames] == pytest.approx(
            [start + index/30 for index in range(31)], abs=1e-4)
        assert float(stream.duration * stream.time_base) == pytest.approx(31/30, abs=1e-4)
        assert all(np.array_equal(before, after.to_ndarray(format='rgb24'))
                   for before, after in zip(original, frames))
    assert clips.frame_lengths(output) is False
