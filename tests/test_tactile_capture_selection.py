"""Native tactile comparisons can reveal brief changes between scene samples."""
import json

import av
import numpy as np
import pytest
from PIL import Image

from label import episode as me, optical_motion as om
from prepare.formats import probe
from tests.test_label import _packed_mp4
from tests.test_optical_motion import pad


def recording(tmp_path, *, fps=30, frames=91, pulses=((12, 15),), human_excluded=False):
    movie = tmp_path / 'tactile.mp4'
    with av.open(str(movie), 'w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = 240, 160, 'yuv420p'
        for k in range(frames):
            image = pad(2 if any(start <= k < end for start, end in pulses) else 0)
            for packet in stream.encode(av.VideoFrame.from_image(image)):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    ctx = {'episode_id': 'episode_pad', 'profile': 'teleop_arms', 'state_kind': 'none', 'fps': fps,
           'cameras': {'exo': {'name': 'gelsight', 'width': 240, 'height': 160}}}
    if human_excluded:
        ctx['data_dictionary'] = {'episode_id': 'episode_pad',
            'fields': [{'id': 'camera', 'kind': 'camera', 'name': 'gelsight',
                        'bindings': [{'episode': 'episode_pad', 'context_path': 'cameras/exo'}]}],
            'entries': {'camera': {'role': 'scene', 'provenance': 'human'}}}
    (tmp_path / 'context.json').write_text(json.dumps(ctx))
    (tmp_path / 'sources.json').write_text(json.dumps({'exo': {'packed': str(movie), 'base_s': 0., 'n_frames': frames}}))
    return tmp_path


def test_default_request_supplies_brief_native_tactile_change_without_global_fps_override(tmp_path):
    folder = recording(tmp_path)
    request = me.build_request(folder, cell_w=224)
    assert request['timesteps'] == [0., 1.5, 3.]
    sensor = request['sensor_evidence']['sensors'][0]
    assert any(.4 <= t < .5 for t in sensor['times'])
    assert sensor['native_captures']
    assert len(sensor['native_captures']) <= 24
    assert request['n_images'] == sum(part['type'] == 'image_url' for part in request['content'])
    assert request['sensor_evidence']['source_proof']
    assert any('native tactile captures' in part.get('text', '') for part in request['content'])


def test_repeated_brief_changes_are_sampled_in_multiple_recording_phases(tmp_path):
    folder = recording(tmp_path, frames=181, pulses=((12, 15), (72, 75), (132, 135)))
    request = me.build_request(folder, cell_w=224)
    times = request['sensor_evidence']['sensors'][0]['times']
    for start in (.4, 2.4, 4.4):
        assert any(start <= t < start + .1 for t in times)


def test_human_camera_exclusion_prevents_native_capture_scanning(tmp_path):
    request = me.build_request(recording(tmp_path, human_excluded=True), cell_w=224)
    assert not request['sensor_evidence']['sensors']
    assert not any('native tactile captures' in part.get('text', '') for part in request['content'])


def test_native_side_captures_between_every_anchor_mapping_remain_available(tmp_path):
    folder = recording(tmp_path)
    tactile = folder / 'tactile.mp4'
    scene = folder / 'scene.mp4'
    _packed_mp4(scene, 4)
    ctx = {'episode_id': 'episode_pad', 'profile': 'teleop_arms', 'state_kind': 'none', 'fps': 1,
           'real_times': 'times.npz', 'cameras': {'exo': {'name': 'scene'}, 'extra1': {'name': 'gelsight'}}}
    (folder / 'context.json').write_text(json.dumps(ctx))
    (folder / 'sources.json').write_text(json.dumps({
        'exo': {'packed': str(scene), 'base_s': 0., 'n_frames': 4},
        'extra1': {'packed': str(tactile), 'base_s': 0., 'n_frames': 91, 'kmap': 'map.npy'}}))
    np.save(folder / 'map.npy', np.array([0, 30, 60, 90]))
    np.savez(folder / 'times.npz', exo=np.arange(4.), extra1=np.arange(91) / 30,
             exo_pts=probe(scene)['pts'], extra1_pts=probe(tactile)['pts'])
    request = me.build_request(folder, cell_w=224)
    sensor = request['sensor_evidence']['sensors'][0]
    assert any(.4 <= t < .5 for t in sensor['times'])
    assert any(12 <= capture['source_frame'] < 15 for capture in sensor['native_captures'])


def test_capture_image_budget_withholds_unsupplied_native_citations(tmp_path, monkeypatch):
    from label import tactile_capture
    monkeypatch.setattr(tactile_capture, 'MAX_IMAGE_BYTES', 1)
    request = me.build_request(recording(tmp_path), cell_w=224)
    sensor = request['sensor_evidence']['sensors'][0]
    assert sensor['times'] == [0., 1.5, 3.]
    assert not sensor.get('native_captures')
    assert sensor['capture_selection']['withheld_by_image_budget'] > 0
    assert sensor['capture_selection']['selected_source_frames'] == []


def test_periodic_marker_alias_is_not_reported_as_reliable_zero_displacement():
    motion = om.measure({0: pad(), 1: pad(20)}, {0: 0., 1: 1.5})
    assert motion is None or motion['pairs'][0]['p95_px'] is None


def test_small_coherent_marker_motion_remains_available():
    motion = om.measure({0: pad(), 1: pad(2)}, {0: 0., 1: 1.5})
    assert motion['pairs'][0]['p95_px'] == pytest.approx(2, abs=.25)


def test_ambiguous_pairs_keep_their_capture_times_beside_valid_comparisons():
    motion = om.measure({0: pad(), 1: pad(2), 2: pad(22)}, {0: 0., 1: .1, 2: .2})
    assert motion['pairs'][0]['p95_px'] == pytest.approx(2, abs=.25)
    assert motion['pairs'][1]['from_s'] == .1
    assert motion['pairs'][1]['to_s'] == .2
    assert motion['pairs'][1]['p95_px'] is None
