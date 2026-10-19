"""Depth units and source holes survive ingestion and viewer conversion."""
import json

import av
import numpy as np
import pytest

from board import clips
from label import depth
from prepare import formats
from test_label import _packed_mp4


@pytest.mark.parametrize('descriptor,expected', [
    ({'info': {'units': 'mm'}}, .001),
    ({'units': 'm'}, 1.),
    ({'scale_m': .001}, .001),
    ({'info': {'depth_scale': .001, 'units': 'mm'}}, .001),
    ({'info': {'depth_scale': .001, 'units': 'm'}}, None),
    ({'info': {'depth_scale': .001, 'depth_units': .01}}, None),
])
def test_video_depth_uses_complete_consistent_unit_declarations(tmp_path, descriptor, expected):
    path = tmp_path / 'depth.mkv'
    writer = formats.DepthWriter(path)
    for k in range(3):
        writer.add(k / 30, np.full((36, 64), 1000, np.uint16), None)
    writer.close()
    key = 'observation.depth.cam_high'
    entries, _ = formats.lerobot_depth(tmp_path / 'episode',
        {'depth_cams': [key], 'features': {key: descriptor}}, {'videos': {key: str(path)}},
        {'exo': 'observation.images.cam_high'}, 30, 3, [])
    entry = entries['exo']
    assert entry['scale_m'] == expected
    picture = depth.decode(entry, formats.probe_depth(path)['pts'], [0])[0]
    assert depth.spatial_grid(picture, entry)['unit'] == ('m' if expected else None)


@pytest.mark.parametrize('camera,offset,trim,clocked', [
    ('exo', 0., False, True),
    ('left', 1 / 120, False, True),
    ('exo', 0., True, True),
    ('exo', 0., False, False),
])
def test_viewer_depth_keeps_authoritative_missing_rows(tmp_path, monkeypatch, camera, offset, trim, clocked):
    anchor_times = np.arange(6) / 30
    camera_times = anchor_times + offset
    source_rows = [0, 1, 3, 4, 5]
    stored_map = np.array([0, 1, -1, 2, 3, 4])
    source = tmp_path / 'source.mp4'
    _packed_mp4(source, 6)
    colour = tmp_path / 'colour.mp4'
    _packed_mp4(colour, 4 if trim else 6)
    path = tmp_path / 'depth.mkv'
    writer = formats.DepthWriter(path)
    for row in source_rows:
        writer.add(float(camera_times[row]), np.full((36, 64), 1000 + row * 100, np.uint16), .001)
    writer.close()
    entry, times = formats.depth_entry(tmp_path, camera, path, camera_times[source_rows], anchor_times,
                                      formats.probe_depth(path)['pts'], .001, 'depth')
    np.save(tmp_path / entry['kmap'], stored_map)
    (tmp_path / 'depth.json').write_text(json.dumps({camera: entry}))
    np.savez(tmp_path / 'depth_times.npz', **times)
    sources = {'exo': {'packed': str(source), 'n_frames': 6}}
    if camera != 'exo':
        sources[camera] = {'packed': str(source), 'n_frames': 6}
    (tmp_path / 'sources.json').write_text(json.dumps(sources))
    context = {'fps': 30, 'cameras': {view: {'name': view} for view in sources}}
    if trim:
        context['clock_zero_s'] = 2 / 30
    (tmp_path / 'context.json').write_text(json.dumps(context))
    if clocked:
        np.savez(tmp_path / 'times.npz', **{'exo': anchor_times, camera: camera_times})
    loaded = depth.load(tmp_path)
    assert sorted(depth.at_anchor({}, loaded, camera, list(range(6)))) == source_rows
    selected = []
    picture = depth.picture

    def recorded_picture(array, *args, **kwargs):
        selected.append(int(array[0, 0]))
        return picture(array, *args, **kwargs)

    monkeypatch.setattr(depth, 'picture', recorded_picture)
    output = tmp_path / 'preview.mp4'
    assert clips.extract_depth(tmp_path, camera, colour, output) == []
    with av.open(str(output)) as container:
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    expected_rows = [2, 3, 4, 5] if trim else list(range(6))
    assert len(frames) == len(expected_rows)
    assert [int(frame.max()) > 40 for frame in frames] == [row != 2 for row in expected_rows]
    assert selected == [1000 + row * 100 for row in expected_rows if row != 2]


def test_depth_preview_cache_rebuilds_old_same_length_clips(tmp_path):
    episode = tmp_path / 'episode_cache'
    episode.mkdir()
    path = episode / 'depth.mkv'
    writer = formats.DepthWriter(path)
    for row in range(3):
        writer.add(row / 30, np.full((36, 64), 1000, np.uint16), .001)
    writer.close()
    entry, times = formats.depth_entry(episode, 'exo', path, np.arange(3) / 30, np.arange(3) / 30,
                                      formats.probe_depth(path)['pts'], .001, 'depth')
    (episode / 'depth.json').write_text(json.dumps({'exo': entry}))
    np.savez(episode / 'depth_times.npz', **times)
    (episode / 'sources.json').write_text(json.dumps({'exo': {'packed': str(path), 'n_frames': 3}}))
    (episode / 'context.json').write_text(json.dumps({'fps': 30}))
    media = tmp_path / 'clips'
    media.mkdir()
    colour = clips.clip_path(media, episode.name, 'exo')
    output = clips.depth_clip(media, episode.name, 'exo')
    output.parent.mkdir()
    _packed_mp4(colour, 3)
    _packed_mp4(output, 3)
    (job,) = clips.depth_jobs(episode, media, False)
    assert job[3] == output
    assert clips.extract_depth(*job[:4], fps=job[4]) == []
    assert clips.depth_jobs(episode, media, False) == []
