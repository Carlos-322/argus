"""Declared marker geometry, native-clock events and private-source cache ownership."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from prepare import jaws


DECLARATION = {'profile': 'orange_pair_rows_640x480_v1', 'basis': 'declared',
               'evidence': 'Camera calibration supplied with the recording.'}


def marker_frames():
    # The donor's two rows and paired orange jaw markers, with a closing motion.
    yy, xx = np.ogrid[:480, :640]
    for spacing in [200] * 6 + [190, 180, 170, 160, 150, 140] + [140] * 6:
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        for row, width in ((254, spacing), (374, spacing + 40)):
            for x in (320 - width / 2, 320 + width / 2):
                image[(xx - x) ** 2 + (yy - row) ** 2 <= 16] = [10, 90, 240]
        yield image


def test_declared_geometry_is_measured_but_unknown_hardware_abstains(tmp_path):
    times = np.arange(18) * .04 + 37
    result = jaws.analyse_frames(marker_frames, times, DECLARATION)
    assert result['status'] == 'measured'
    assert result['quality']['observed_frames'] == 18
    assert result['quality']['interpolated_frames'] == 0
    assert result['times_s'] == times.tolist()
    assert result['spacing_px'][0] > result['spacing_px'][-1]
    assert any(e['kind'] == 'close' for e in result['events'])
    assert all(37 <= e['start_s'] <= e['end_s'] <= times[-1] for e in result['events'])
    assert not any(k in json.dumps(result) for k in ('shut', 'retry', 'pressure', 'failed_grasp'))
    unknown = jaws.analyse_frames(lambda: iter(()), times, None)
    assert unknown['status'] == 'abstained' and unknown['reason'] == 'unsupported_geometry'
    blank = jaws.analyse_frames(lambda: (np.zeros((480, 640, 3), np.uint8) for _ in times), times, DECLARATION)
    assert blank['status'] == 'abstained' and blank['reason'] == 'markers_not_recognized'
    partial = list(marker_frames())[:12] + [np.zeros((480, 640, 3), np.uint8)] * 6
    low = jaws.analyse_frames(lambda: iter(partial), times, DECLARATION)
    assert low['status'] == 'abstained' and low['reason'] == 'insufficient_marker_coverage'
    assert low['quality']['observed_frames'] == 12
    assert low['quality']['observed_fraction'] == pytest.approx(2 / 3)


def test_events_use_native_times_and_never_bridge_unmeasured_gaps():
    times = np.array([12, 12.03, 12.11, 12.21, 12.37, 12.51, 13.8, 13.9, 14, 14.1, 14.2, 14.3])
    opening = np.array([200, 190, 175, 160, 140, 140, np.nan, 100, 100, 100, 100, 100])
    events = jaws.events(opening, times)
    assert events and events[0]['start_s'] == 12 and events[0]['end_s'] == 12.37
    assert events[0]['from_px'] == 200 and events[0]['to_px'] == 140
    assert all(not (e['start_s'] < 13.8 < e['end_s']) for e in events)
    # No synthetic close across a gap in the source PTS even with finite samples.
    assert jaws.events(np.array([200] * 6 + [100] * 6), np.array([0, .1, .2, .3, .4, .5, 5, 5.1, 5.2, 5.3, 5.4, 5.5])) == []


def test_native_gap_interpolation_reports_observed_and_interpolated_coverage():
    spacing = np.array([[150 + i, 190 + i] for i in range(20)], dtype=float)
    spacing[5:7] = np.nan
    spacing[12:14] = np.nan
    times = np.arange(20) * .05
    times[13:] += 2
    opening, observed, filled = jaws.opening(spacing, times)
    assert observed.sum() == 16
    assert filled.sum() == 2 and filled[5:7].all()
    assert np.isnan(opening[12:14]).all()


def test_cache_identity_and_source_ownership(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    video = source / 'camera.mp4'
    video.write_bytes(b'synthetic decoder input')
    cache = tmp_path / 'output'
    calls = []

    def decode(path, *, frame_start, frame_count):
        calls.append((frame_start, frame_count))
        yield from marker_frames()

    monkeypatch.setattr(jaws, 'read_frames', decode)
    args = dict(times_s=np.arange(18) * .04 + 20, camera='wrist', declaration=DECLARATION,
                output_dir=cache, source_proof={'sha256': 'a' * 64})
    first = jaws.analyse(video, **args)
    assert jaws.analyse(video, **args) == first and len(calls) == 2
    assert first['status'] == 'measured'
    second = jaws.analyse(video, **{**args, 'frame_start': 18, 'times_s': args['times_s'] + 4})
    assert second['cache_key'] != first['cache_key'] and len(calls) == 4
    third = jaws.analyse(video, **{**args, 'camera': 'other_wrist'})
    assert third['cache_key'] != first['cache_key'] and len(calls) == 6
    calibrated = jaws.analyse(video, **{**args, 'declaration': {**DECLARATION, 'evidence': 'Updated camera calibration.'}})
    assert calibrated['cache_key'] != first['cache_key'] and len(calls) == 8
    reclocked = jaws.analyse(video, **{**args, 'times_s': args['times_s'] + 1})
    assert reclocked['cache_key'] != first['cache_key'] and len(calls) == 10
    video.write_bytes(b'same pathname, replaced video')
    changed = jaws.analyse(video, **args)
    assert changed['cache_key'] != first['cache_key'] and len(calls) == 12
    assert sorted(p.name for p in source.iterdir()) == ['camera.mp4']
    with pytest.raises(ValueError, match='outside'):
        jaws.analyse(video, **{**args, 'output_dir': source / 'cache'})


def test_wrong_frame_count_or_clock_abstains():
    result = jaws.analyse_frames(marker_frames, np.arange(19) * .04, DECLARATION)
    assert result['status'] == 'abstained' and result['reason'] == 'frame_count_mismatch'
    result = jaws.analyse_frames(marker_frames, [1, 1, 2], DECLARATION)
    assert result['status'] == 'abstained' and result['reason'] == 'invalid_native_clock'
    result = jaws.analyse_frames(marker_frames, ['unknown'], DECLARATION)
    assert result['status'] == 'abstained' and result['reason'] == 'invalid_native_clock'


def test_actual_decoder_refuses_unqualified_geometry(tmp_path):
    video = Path(__file__).parent / 'fixtures/clip_without_frame_lengths.mp4'
    with pytest.raises(ValueError, match='unsupported_image_geometry'):
        list(jaws.read_frames(video, frame_start=0, frame_count=20))


def test_decoder_retains_exact_native_frame_window(tmp_path):
    frames = list(marker_frames())
    source = tmp_path / 'source'
    source.mkdir()
    video = source / 'controlled_markers.mkv'
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24',
                    '-s:v', '640x480', '-r', '25', '-i', 'pipe:0', '-c:v', 'ffv1', str(video)],
                   input=b''.join(frame.tobytes() for frame in frames), check=True, timeout=20)
    selected = list(jaws.read_frames(video, frame_start=6, frame_count=6))
    assert len(selected) == 6
    assert all(np.array_equal(a, b) for a, b in zip(selected, frames[6:12]))
    # The request consumer binds decoded native PTS to the source proof. A saved
    # measurement stops being current after that source context changes.
    from board.dictionary_projection import project
    prepared = tmp_path / 'episode_a'
    prepared.mkdir()
    context = {'state_kind': 'none', 'cameras': {'wrist': {'jaw_geometry': DECLARATION}}}
    sources = {'wrist': {'packed': str(video), 'n_frames': len(frames)}}
    (prepared / 'context.json').write_text(json.dumps(context))
    (prepared / 'sources.json').write_text(json.dumps(sources))
    native = {'wrist': np.arange(len(frames)) * .04, 'wrist_pts': np.arange(len(frames)) * 40}
    result = jaws.for_episode({'dir': prepared, 'context': context, 'sources': sources, 'times': native},
                              tmp_path / 'derived')
    assert result['wrist']['status'] == 'measured' and result['wrist']['source_proof']
    saved = {'jaw_evidence': result}
    assert project(saved, prepared)['jaw_evidence']['wrist']['source_current'] is True
    (prepared / 'context.json').write_text(json.dumps({**context, 'source_replaced': True}))
    changed = project(saved, prepared)
    assert changed['jaw_evidence'] == {}
    assert changed['withheld_jaw_evidence']['wrist']['spacing_px'] == result['wrist']['spacing_px']
    assert 'withheld_jaw_evidence' not in saved


def test_existing_small_video_without_declaration_is_never_decoded(tmp_path, monkeypatch):
    video = Path(__file__).parent / 'fixtures/clip_without_frame_lengths.mp4'
    monkeypatch.setattr(jaws, 'read_frames', lambda *a, **k: pytest.fail('unknown camera was decoded'))
    result = jaws.analyse(video, times_s=[0, .1], camera='unknown', declaration=None,
                          output_dir=tmp_path / 'analysis', source_proof={})
    assert result['status'] == 'abstained'
    assert result['reason'] == 'unsupported_geometry'
    assert not (tmp_path / 'analysis').exists()
