"""Supplemental scene captures preserve actual clocks without changing scene cadence."""
import copy
import hashlib
import json

import av
import numpy as np
from PIL import Image

from label import episode as me
from prepare.formats import nearest, probe


def movie(path, n, fps=10, size=(96, 64)):
    with av.open(str(path), 'w') as container:
        stream = container.add_stream('libx264', rate=fps)
        stream.width, stream.height, stream.pix_fmt = *size, 'yuv420p'
        for k in range(n):
            pixels = np.random.default_rng(k).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
            for packet in stream.encode(av.VideoFrame.from_image(Image.fromarray(pixels))):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)


def episode(tmp_path, *, aliases=False):
    path = tmp_path / 'scene.mp4'
    movie(path, 7)
    clock = np.arange(7) / 10
    ep = {'dir': tmp_path, 'state': np.zeros((7, 0)), 'signals': {}, 'action': None,
          'context': {'episode_id': 'sample', 'profile': 'teleop_arms', 'state_kind': 'none', 'fps': 10,
                      'cameras': {'exo': {'name': 'overview', 'content_role': 'scene', 'width': 96, 'height': 64}}},
          'sources': {'exo': {'packed': str(path), 'base_s': 0., 'n_frames': 7}},
          'times': {'exo': clock, 'exo_pts': probe(path)['pts']}, 'kmap': {}}
    if aliases:
        ep['context']['cameras']['extra1'] = dict(ep['context']['cameras']['exo'], name='overview alias')
        ep['sources']['extra1'] = dict(ep['sources']['exo'])
        ep['times'].update(extra1=clock.copy(), extra1_pts=probe(path)['pts'])
    return ep


def arrays(*times):
    return {'sensors': [{'id': 'signal:pad', 'kind': 'numeric', 'episode_alignment_verified': True,
                         'array_evidence': {'frames': [{'time_s': t, 'values': [1, None, 3, 4]}
                                                       for t in times]}}]}


def pairing(ep, doc, **kwargs):
    from label import sensor_scene
    shown = kwargs.pop('shown', {'exo': [0, 6]})
    return sensor_scene.collect(ep, doc, shown=shown, width=96, **kwargs)


def test_missing_feature_does_not_supply_scene_context_between_baseline_samples(tmp_path):
    """Without pairing the actual between-sample scene frame is absent."""
    ep = episode(tmp_path)
    parts, receipt, size = pairing(ep, arrays(.2, .4))
    assert [row['source_frame'] for row in receipt['captures'] if row['status'] == 'supplemental'] == [2, 4]
    assert len([p for p in parts if p['type'] == 'image_url']) == 1
    assert size > 0


def test_async_capture_time_is_separate_from_sensor_and_presentation_time(tmp_path):
    ep = episode(tmp_path)
    side = tmp_path / 'side.mp4'
    movie(side, 4)
    own = np.array([.025, .225, .425, .625])
    ep['context']['cameras'] = {'exo': {'name': 'sensor view', 'content_role': 'tactile'},
                               'extra1': {'name': 'overview', 'content_role': 'scene'}}
    ep['sources']['extra1'] = {'packed': str(side), 'base_s': 0., 'n_frames': 4}
    ep['times'].update(extra1=own, extra1_pts=probe(side)['pts'])
    ep['kmap']['extra1'] = nearest(own, ep['times']['exo'])
    parts, receipt, _ = pairing(ep, arrays(.2))
    capture = next(row for row in receipt['captures'] if row['status'] == 'supplemental')
    assert capture['source_frame'] == 1
    assert capture['source_time_s'] == .225
    assert capture['display_time_s'] == .2
    assert abs(capture['capture_offset_s'] - .025) < 1e-12
    assert capture['paired_sensor_refs'] == [{'sensor_id': 'signal:pad', 'time_s': .2}]
    assert not any('equals' in part.get('text', '') for part in parts)


def test_duplicate_scene_slots_reuse_one_capture_and_keep_both_aliases(tmp_path):
    ep = episode(tmp_path, aliases=True)
    doc = arrays(.2, .4)
    doc['sensors'].append(copy.deepcopy(doc['sensors'][0]))
    doc['sensors'][1]['id'] = 'signal:other-pad'
    parts, receipt, _ = pairing(ep, doc)
    new = [row for row in receipt['captures'] if row['status'] == 'supplemental']
    assert len(new) == 2
    assert all(set(row['view_aliases']) == {'exo', 'extra1'} for row in new)
    assert all({ref['sensor_id'] for ref in row['paired_sensor_refs']} == {'signal:pad', 'signal:other-pad'} for row in new)
    # An existing frame in either alias needs no new image.
    _, reused, _ = pairing(ep, arrays(.2), shown={'extra1': [2]})
    assert reused['supplemental_captures'] == 0
    assert reused['reused_captures'] == 1


def test_assumed_or_unaligned_clocks_and_sensor_roles_do_not_gain_pairing(tmp_path):
    ep = episode(tmp_path)
    for change in ['camera clock', 'sensor clock', 'sensor role', 'outside']:
        e, doc = copy.deepcopy(ep), arrays(.2)
        if change == 'camera clock':
            e['context']['camera_clock'] = {'exo': {'method': 'assumed FPS', 'what': 'Clock alignment unknown.'}}
        elif change == 'sensor clock':
            doc['sensors'][0]['episode_alignment_verified'] = False
        elif change == 'sensor role':
            e['context']['cameras']['exo']['content_role'] = 'tactile'
        else:
            doc = arrays(-.1, .8)
        parts, receipt, size = pairing(e, doc)
        assert not parts and size == 0
        assert receipt['supplemental_captures'] == 0
        assert sum(receipt['omissions'].values()) > 0


def test_capture_and_byte_limits_omit_images_honestly_without_source_mutation(tmp_path):
    from label import sensor_scene
    ep = episode(tmp_path)
    doc = arrays(.1, .2, .3, .4, .5)
    before = copy.deepcopy(doc)
    source_hash = hashlib.sha256((tmp_path / 'scene.mp4').read_bytes()).hexdigest()
    _, complete, _ = pairing(ep, doc)
    parts, bounded, size = pairing(ep, doc, max_captures=2, max_bytes=1)
    assert not parts and size == 0
    assert bounded['omissions']['capture_budget'] == 3
    assert bounded['omissions']['image_byte_budget'] == 2
    assert bounded['supplemental_captures'] == 0
    assert complete['supplemental_captures'] == 5
    assert doc == before
    assert hashlib.sha256((tmp_path / 'scene.mp4').read_bytes()).hexdigest() == source_hash


def test_unclassified_baseline_camera_uses_neutral_context_without_role_invention(tmp_path):
    ep = episode(tmp_path)
    ep['context']['cameras']['exo'].pop('content_role')
    parts, receipt, _ = pairing(ep, arrays(.2))
    capture = next(row for row in receipt['captures'] if row['status'] == 'supplemental')
    assert capture['role']['role'] is None
    assert capture['context_kind'] == 'camera context'
    assert any('camera context' in part.get('text', '') for part in parts)
    assert not any('scene context' in part.get('text', '') for part in parts)


def test_reused_async_capture_retains_each_display_placement(tmp_path):
    ep = episode(tmp_path)
    side = tmp_path / 'side.mp4'
    movie(side, 4)
    own = np.array([.025, .225, .425, .625])
    ep['context']['cameras'] = {'exo': {'name': 'sensor', 'content_role': 'tactile'},
                               'extra1': {'name': 'overview', 'content_role': 'scene'}}
    ep['sources']['extra1'] = {'packed': str(side), 'base_s': 0., 'n_frames': 4}
    ep['times'].update(extra1=own, extra1_pts=probe(side)['pts'])
    ep['kmap']['extra1'] = nearest(own, ep['times']['exo'])
    _, receipt, _ = pairing(ep, arrays(.2, .3))
    row = receipt['captures'][0]
    assert row['source_frame'] == 1
    assert [p['display_time_s'] for p in row['placements']] == [.2, .3]
    assert np.allclose([p['capture_offset_s'] for p in row['placements']], [.025, -.075])
    assert [p['paired_sensor_refs'][0]['time_s'] for p in row['placements']] == [.2, .3]


def test_normal_request_pairs_array_snapshots_without_changing_scene_sampling(tmp_path):
    ep = episode(tmp_path)
    ctx = copy.deepcopy(ep['context'])
    ctx['real_times'] = 'times.npz'
    ctx['signals'] = [{'name': 'right_pressure', 'key': 'pressure', 'dims': 16, 'shape': [4, 4]}]
    (tmp_path / 'context.json').write_text(json.dumps(ctx))
    (tmp_path / 'sources.json').write_text(json.dumps(ep['sources']))
    np.savez(tmp_path / 'times.npz', **ep['times'])
    np.savez(tmp_path / 'signals.npz', pressure=np.arange(112).reshape(7, 16))
    request = me.build_request(tmp_path, cell_w=224)
    assert request['timesteps'] == [0., .6]
    assert request['plan']['ks'] == [0, 6]
    assert request['sensor_scene_pairing']['supplemental_captures'] == 5
    assert request['n_images'] == sum(p['type'] == 'image_url' for p in request['content'])
    import base64
    assert request['image_bytes'] == sum(len(base64.b64decode(p['image_url']['url'].split(',', 1)[1]))
                                         for p in request['content'] if p['type'] == 'image_url')
    from label import harness
    output = tmp_path / 'run' / 'reply.json'
    output.parent.mkdir()
    saved = harness.label_episode(tmp_path, output, model='openai/gpt-6-astra', reasoning='medium',
        api_key='unused', max_tokens=16000, timeout=1, cell_w=224, dry_run=True)
    assert saved['config']['sensor_scene_pairing'] == request['sensor_scene_pairing']
    actual_hashes = [hashlib.sha256(base64.b64decode(part['image_url']['url'].split(',', 1)[1])).hexdigest()
                     for part in request['content'] if part['type'] == 'image_url']
    final_receipt = saved['config']['request_images']
    assert final_receipt['dispatch_status'] == 'dry_run'
    assert [image['sha256'] for image in final_receipt['images']] == actual_hashes
    assert saved['request']['n_images'] == len(actual_hashes)
    assert saved['config']['n_frames_sent'] == 7
    assert json.loads(output.read_text())['config']['sensor_scene_pairing'] == request['sensor_scene_pairing']
    # Explicitly unaligned array clocks retain omission metadata even with no paired snapshot times.
    ctx['signals'][0]['aligned_by'] = 'assumed camera placement'
    (tmp_path / 'context.json').write_text(json.dumps(ctx))
    unaligned = harness.label_episode(tmp_path, output.with_name('unaligned.json'), model='openai/gpt-6-astra',
        reasoning='medium', api_key='unused', max_tokens=16000, timeout=1, cell_w=224, dry_run=True)
    omission = unaligned['config']['sensor_scene_pairing']
    assert omission['snapshot_times'] == 0 and omission['supplemental_captures'] == 0
    assert omission['omissions']['unaligned_sensor_snapshots'] == 7
    assert not omission['captures']


def test_nonmonotone_camera_clock_is_not_paired_by_its_bounds(tmp_path):
    ep = episode(tmp_path)
    ep['recorded_times'] = {'exo': np.array([0., .1, .4, .3, .2, .5, .6])}
    parts, receipt, size = pairing(ep, arrays(.2))
    assert not parts and size == 0
    assert receipt['omissions']['unaligned_camera_pairs'] == 1


def test_global_capture_cap_covers_time_without_multiplying_unknown_views(tmp_path):
    path = tmp_path / 'long.mp4'
    movie(path, 50)
    clock = np.arange(50) / 10
    ep = {'dir': tmp_path, 'state': np.zeros((50, 0)), 'signals': {}, 'action': None,
          'context': {'episode_id': 'sample', 'profile': 'teleop_arms', 'state_kind': 'none', 'fps': 10,
                      'cameras': {'exo': {'name': 'ordinary'}, 'extra1': {'name': 'another ordinary'}}},
          'sources': {v: {'packed': str(path), 'base_s': 0., 'n_frames': 50} for v in ['exo', 'extra1']},
          'times': {'exo': clock, 'extra1': clock, 'exo_pts': probe(path)['pts'], 'extra1_pts': probe(path)['pts']}, 'kmap': {}}
    from label import sensor_scene
    _, receipt, size = sensor_scene.collect(ep, arrays(*clock.tolist()), shown={'exo': [0, 49], 'extra1': [0, 49]}, width=96, max_captures=100)
    new = [row for row in receipt['captures'] if row['status'] == 'supplemental']
    assert len(new) == 32
    assert {row['view'] for row in new} == {'exo'}
    assert new[0]['source_frame'] == 1 and new[-1]['source_frame'] == 48
    assert receipt['omissions']['capture_budget'] == 16
    assert receipt['omissions']['unclassified_extra_camera_pairs'] == 50
    assert size <= 2_000_000


def test_source_caption_does_not_repeat_structured_receipt_proof(tmp_path):
    parts, receipt, _ = pairing(episode(tmp_path), arrays(.2, .4))
    text = ' '.join(part.get('text', '') for part in parts)
    assert 'overview' in text and '0.2' in text and '0.4' in text
    assert 'placements' not in text and 'sensor_id' not in text and 'sha256' not in text
    assert receipt['captures'][0]['paired_sensor_refs'][0]['sensor_id'] == 'signal:pad'


def test_nearest_native_capture_is_not_nearest_anchor_kmap_capture(tmp_path):
    from label import frames as mf
    ep = episode(tmp_path)
    side = tmp_path / 'native.mp4'
    movie(side, 2)
    own = np.array([.09, .11])
    ep['context']['cameras'] = {'exo': {'name': 'sensor', 'content_role': 'tactile'},
                               'extra1': {'name': 'overview', 'content_role': 'scene'}}
    ep['sources']['extra1'] = {'packed': str(side), 'base_s': 0., 'n_frames': 2}
    ep['times'].update(extra1=own, extra1_pts=probe(side)['pts'])
    # Stored baseline pairing is deliberately earlier; supplemental selection is native-clock nearest.
    ep['kmap']['extra1'] = np.array([0, 0, 1, 1, 1, 1, 1])
    before = ep['kmap']['extra1'].copy()
    _, receipt, _ = pairing(ep, arrays(.105))
    row = receipt['captures'][0]
    assert row['source_frame'] == 1 and row['source_time_s'] == .11
    assert row['display_time_s'] == .1
    expected = mf.extract_frames(side, 0., 2, [1], pts=probe(side)['pts'], fps=10)[1]
    assert row['tile_jpeg_sha256'] == hashlib.sha256(mf.to_jpeg(expected, 96, 90)).hexdigest()
    assert np.array_equal(ep['kmap']['extra1'], before)


def test_duplicate_capture_clock_needs_explicit_source_identity_mapping(tmp_path):
    ep = episode(tmp_path)
    side = tmp_path / 'ties.mp4'
    movie(side, 4)
    own = np.array([.025, .225, .225, .625])
    ep['context']['cameras'] = {'exo': {'name': 'sensor', 'content_role': 'tactile'},
                               'extra1': {'name': 'overview', 'content_role': 'scene'}}
    ep['sources']['extra1'] = {'packed': str(side), 'base_s': 0., 'n_frames': 4}
    ep['times'].update(extra1=own, extra1_pts=probe(side)['pts'])
    ep['kmap']['extra1'] = nearest(own, ep['times']['exo'])
    _, receipt, _ = pairing(ep, arrays(.225))
    assert receipt['supplemental_captures'] == 1
    assert receipt['captures'][0]['source_frame'] == 1
    ep['times'].pop('extra1_pts')
    parts, receipt, size = pairing(ep, arrays(.225))
    assert not parts and size == 0
    assert receipt['omissions']['unaligned_camera_pairs'] == 1


def test_limited_sensor_request_does_not_pair_a_capture_after_its_own_window(tmp_path):
    ep = episode(tmp_path)
    side = tmp_path / 'late.mp4'
    movie(side, 3)
    own = np.array([0., .35, .55])
    ep['context']['cameras'] = {'exo': {'name': 'sensor', 'content_role': 'tactile'},
                               'extra1': {'name': 'overview', 'content_role': 'scene'}}
    ep['sources']['extra1'] = {'packed': str(side), 'base_s': 0., 'n_frames': 3}
    ep['times'].update(extra1=own, extra1_pts=probe(side)['pts'])
    ep['kmap']['extra1'] = nearest(own, ep['times']['exo'])
    doc = arrays(.2)
    doc['episode_window_s'] = [0., .3]
    parts, receipt, size = pairing(ep, doc)
    assert not parts and size == 0
    assert receipt['omissions']['outside_capture_window'] == 1
