"""Normal requests carry bounded source-bound response and shape comparisons."""
import copy
import json

import numpy as np
import pytest

from label import episode as me, sensor_evidence as se
from tests.test_label import _packed_mp4
from tests.test_tactile_contract import episode
from tests.test_tactile_constituent_projection import expand_prompt


def descriptor(**extra):
    return {'version': 1, 'regions': [{'name': 'branch A', 'indices': [0, 2]},
                                      {'name': 'branch B', 'indices': [1, 3]}],
            'response_direction': 'down', 'zero_exclusion_sensitivity': True,
            'provenance': {'authority': 'publisher_candidate', 'source': 'https://example.org/layout'}, **extra}


def region_episode(n=5, semantics=None, entry=None):
    values = np.full((n, 4), 100.)
    values[3] = [0, 90, 80, 70]
    return episode({'pad': values}, [{'name': 'pad', 'key': 'pad', 'shape': [4], 'unit': 'ADC',
                    'sensor_semantics': descriptor() if semantics is None else semantics}],
                   {'pad': {'role': 'touch', **(entry or {})}})


def derived(doc):
    return [row for row in doc['series'] if row.get('derivation_id')]


def test_regional_response_and_zero_sensitivity_are_distinct_citable_series():
    ep = region_episode()
    doc = se.build(ep, {'n': 5, 'ks': [0, 4]})
    rows = derived(doc)
    assert rows
    main = next(row for row in rows if row['label'] == 'branch A relative response')
    sensitivity = next(row for row in rows if row['label'] == 'branch A zero-exclusion sensitivity')
    assert main['values'] == [0, 0, 0, 60, 0]
    assert sensitivity['values'] == [0, 0, 0, 20, 0]
    assert main['source_columns'] == [0, 2]
    assert sensitivity['source_columns'] == [2]
    assert main['operation'] != 'recorded value'
    assert main['coverage']['valid_count_min'] == 2
    audit = next(item for item in doc['derivations'] if item['id'] == main['derivation_id'])
    assert audit['reference']['values'] == [100, 100, 100, 100]
    assert audit['provenance']['authority'] == 'publisher_candidate'
    finding = {'start_s': .3, 'end_s': .3, 'headline': 'The branch responses differ',
               'claim_type': 'response_change', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'signal:pad', 'series_id': main['id'], 'time_s': [.3]}]}
    assert se.bind(doc, [finding])['findings']


@pytest.mark.parametrize('shape', [[4], [2, 2]])
def test_human_layout_replaces_candidate_regions_and_role_exclusion_wins(shape):
    ep = region_episode(entry={'provenance': 'human', 'layout': [
        {'name': 'reviewed left', 'start': 0, 'count': 2}, {'name': 'reviewed right', 'start': 2, 'count': 2}]})
    ep['context']['signals'][0]['shape'] = shape
    doc = se.build(ep, {'n': 5})
    labels = [row['label'] for row in derived(doc)]
    assert 'reviewed left relative response' in labels
    assert not any('branch A' in label for label in labels)
    ep['context']['data_dictionary']['entries']['field_0']['role'] = 'annotation'
    assert not derived(se.build(ep, {'n': 5}))


def test_landmark_shape_is_nonanatomical_without_explicit_chains():
    base = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    points = np.stack([base, base, base + [8, 9], base * 2, base * 2])
    ep = episode({'points': points}, [{'name': 'points', 'key': 'p', 'shape': [4, 2]}],
                 {'points': {'role': 'landmarks'}})
    doc = se.build(ep, {'n': 5})
    sensor = next(sensor for sensor in doc['sensors'] if sensor['id'] == 'signal:points')
    assert sensor['quantity'] == 'landmark_shape'
    rows = derived(doc)
    assert {row['label'] for row in rows} == {'Point-set shape from first frame', 'Point-set shape from previous frame',
                                           'Point-set shape between selected poses'}
    assert rows[0]['values'][2] == pytest.approx(0, abs=1e-12)
    assert any(row['values'][3] > 0 for row in rows)
    assert sensor['derivations'][0]['coverage']['repeated_pairs'] == 2
    assert sensor['derivations'][0]['stale_intervals'][0]['start_s'] == 0
    assert sensor['derivations'][0]['stale_intervals'][0]['end_s'] == .1


def test_selected_pose_comparisons_measure_later_endpoints_directly_with_native_rows():
    base = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    points = np.repeat(base[None], 100, axis=0)
    points[30:70] *= 1.25
    points[70:] *= .75
    ep = episode({'points': points}, [{'name': 'points', 'key': 'p', 'shape': [4, 2]}],
                 {'points': {'role': 'landmarks'}})
    doc = se.build(ep, {'n': 100})
    rows = derived(doc)
    comparison = next(row for row in rows if row['label'] == 'Point-set shape between selected poses')
    assert len(comparison['times']) <= 48
    assert comparison['source_rows'][0] == 0
    assert comparison['source_rows'][-1] == 99
    assert comparison['comparison_from_s'][0] is None
    intervals = [(start, end, value) for start, end, value in zip(
        comparison['comparison_from_s'], comparison['times'], comparison['values'])
        if start is not None and 3 <= start < 7 <= end]
    assert intervals
    assert intervals[0][2] == pytest.approx(.5)
    assert comparison['valid_counts'][comparison['times'].index(intervals[0][1])] == 4
    audit = next(trace for item in doc['derivations'] for trace in item['traces']
                 if trace['series_id'] == comparison['id'])
    assert audit['rows'] == comparison['source_rows']
    assert audit['comparison_from_s'] == comparison['comparison_from_s']


def test_declared_chains_supply_curl_and_human_semantic_change_withdraws_candidates():
    base = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    bend = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    semantics = {'version': 1, 'normalization_pair': [0, 3],
                 'chains': [{'name': 'declared branch', 'indices': [0, 1, 2, 3]}],
                 'provenance': {'authority': 'publisher_candidate'}}
    ep = episode({'points': np.stack([base, base, base, bend, bend])},
                 [{'name': 'points', 'key': 'p', 'shape': [4, 2], 'sensor_semantics': semantics}],
                 {'points': {'role': 'landmarks'}})
    doc = se.build(ep, {'n': 5})
    assert next(row for row in derived(doc) if row['label'] == 'declared branch curl')['values'][3] == pytest.approx(2/3)
    ep['context']['data_dictionary']['entries']['field_0'].update(provenance='human', meaning='Different point semantics')
    assert not any('curl' in row['label'] for row in derived(se.build(ep, {'n': 5})))


def test_unknown_layout_keeps_raw_path_and_derived_payload_is_bounded():
    ep = region_episode(5000)
    doc = se.build(ep, {'n': 5000})
    assert derived(doc)
    assert all(len(row['times']) <= 48 for row in derived(doc))
    assert len(json.dumps(doc['derivations'])) < 50000
    payload = expand_prompt(json.loads(se.prompt(doc).splitlines()[-1]))
    assert any(row.get('derivation_id') for row in payload['series'])
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES
    unknown = region_episode(semantics={'version': 42})
    assert not derived(se.build(unknown, {'n': 5}))
    assert se.build(unknown, {'n': 5})['series']


def test_normal_build_request_includes_derived_series_without_an_extra_model_call(tmp_path):
    movie = tmp_path / 'scene.mp4'
    _packed_mp4(movie, 5)
    ep = region_episode()
    ctx = ep['context']
    ctx.update(profile='teleop_arms', state_kind='none', fps=30)
    ctx.pop('data_dictionary')
    ctx['signals'][0]['name'] = 'tactile_pad'
    np.savez(tmp_path / 'signals.npz', pad=ep['signals']['pad'])
    (tmp_path / 'context.json').write_text(json.dumps(ctx))
    (tmp_path / 'sources.json').write_text(json.dumps({'exo': {'packed': str(movie), 'base_s': 0., 'n_frames': 5}}))
    request = me.build_request(tmp_path, cell_w=64)
    assert derived(request['sensor_evidence'])
    assert 'branch A relative response' in request['prompt']


def test_semantic_changes_invalidate_descriptor_and_interpretation_proofs():
    ep = region_episode()
    before = se.signature(ep['context'])
    meta = ep['context']['signals'][0]
    digest = se._descriptor_digest(meta)
    meta['sensor_semantics']['regions'][0]['indices'] = [0]
    assert se.signature(ep['context']) != before
    assert se._descriptor_digest(meta) != digest


def test_pressure_and_pointset_groups_survive_default_prompt_budget():
    from pathlib import Path
    contract = json.loads((Path(__file__).parents[1] / 'configs/sensor_schemas/opentouch_hdf5_v1.json').read_text())
    pressure = np.full((500, 256), 100.)
    pressure[250:300, :140] = 80
    base = np.arange(63.).reshape(21, 3)
    points = np.repeat(base[None], 500, axis=0)
    points[250:, 4] += [0, 4, 0]
    ep = episode({'right_pressure': pressure, 'right_hand_landmarks': points}, [
        {'name': 'right_pressure', 'key': 'p', 'shape': [16, 16],
         'sensor_semantics': contract['signals']['right_pressure']},
        {'name': 'right_hand_landmarks', 'key': 'h', 'shape': [21, 3],
         'sensor_semantics': contract['signals']['right_hand_landmarks']}])
    doc = se.build(ep, {'n': 500, 'ks': [0, 250, 499]})
    rows = derived(doc)
    assert sum(row['label'].endswith('relative response') for row in rows) == 6
    assert any(row['label'] == 'Point-set shape from previous frame' for row in rows)
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES
    assert doc['sensors'][0]['derivations'][0]['semantic_limits']['hardware_revision_verified'] is False


def test_piece_stitch_preserves_derived_reference_and_exact_citation_clock():
    ep = region_episode()
    doc = se.build(ep, {'n': 5})
    trace = next(row for row in derived(doc) if row['label'] == 'branch A relative response')
    finding = {'start_s': .3, 'end_s': .3, 'headline': 'The branches diverge',
               'claim_type': 'response_change', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': trace['sensor_id'], 'series_id': trace['id'], 'time_s': [.3]}]}
    bound = se.bind(doc, [finding])
    stitched = se.merge([({'piece': {'index': 2, 't0_s': 10}}, {'sensor_evidence': bound})], ep['context'])
    audit = stitched['derivations'][0]
    saved = next(row for row in stitched['series'] if row['id'] == 'part2:' + trace['id'])
    assert saved['derivation_id'] == audit['id'] == 'part2:' + trace['derivation_id']
    assert audit['sensor_id'] == saved['sensor_id'] == 'part2:signal:pad'
    assert audit['reference']['times_s'] == pytest.approx([10, 10.1, 10.2])
    assert audit['traces'][0]['times_s'] == pytest.approx(saved['times'])
    assert audit['traces'][0]['series_id'] == saved['id']
    assert audit['traces'][0]['raw_means'] == doc['derivations'][0]['traces'][0]['raw_means']
    summary = stitched['sensors'][0]['derivations'][0]
    assert summary['id'] == audit['id']
    assert saved['id'] in summary['trace_ids']
    assert summary['reference']['times_s'] == audit['reference']['times_s']
    assert stitched['findings'][0]['evidence'][0]['time_s'] == [10.3]
    assert doc['derivations'][0]['reference']['times_s'][0] == 0


def test_piece_stitch_shifts_pointset_normalization_and_stale_intervals():
    base = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]])
    ep = episode({'points': np.repeat(base[None], 5, axis=0)},
                 [{'name': 'points', 'key': 'p', 'shape': [4, 2]}], {'points': {'role': 'landmarks'}})
    doc = se.build(ep, {'n': 5})
    stitched = se.merge([({'piece': {'index': 0, 't0_s': 20}}, {'sensor_evidence': doc})], ep['context'])
    audit = stitched['derivations'][0]
    summary = stitched['sensors'][0]['derivations'][0]
    assert audit['normalization']['reference_time_s'] == summary['reference']['reference_time_s'] == 20
    assert audit['stale_intervals'][0]['start_s'] == summary['stale_intervals'][0]['start_s'] == 20
    assert audit['stale_intervals'][0]['end_s'] == summary['stale_intervals'][0]['end_s'] == pytest.approx(20.4)
    pairs = next(row for row in stitched['series'] if row.get('comparison_from_s'))
    original = next(row for row in doc['series'] if row.get('comparison_from_s'))
    assert pairs['comparison_from_s'] == [t + 20 if t is not None else None for t in original['comparison_from_s']]
    assert pairs['source_rows'] == original['source_rows']
    audit_pairs = next(row for row in audit['traces'] if row.get('comparison_from_s'))
    assert audit_pairs['comparison_from_s'] == pairs['comparison_from_s']


def test_mixed_coordinate_units_cannot_disappear_through_generic_pose_fallback():
    base = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    points = np.stack([base, base, base * 2, base * 2, base])
    ep = episode({'points': points}, [{'name': 'points', 'shape': [4, 3], 'units': ['m', 'm', 'pixels']}],
                 {'points': {'role': 'landmarks'}})
    assert not derived(se.build(ep, {'n': 5}))
    ep['context']['signals'][0]['units'] = ['m', 'm', 'm']
    assert derived(se.build(ep, {'n': 5}))


def test_human_replacement_layout_does_not_inherit_candidate_cell_exclusions():
    ep = region_episode(semantics=descriptor(valid_indices=[0, 2]), entry={'provenance': 'human', 'layout': [
        {'name': 'reviewed whole pad', 'start': 0, 'count': 4}]})
    ep['signals']['pad'][3] = [100, 80, 100, 60]
    doc = se.build(ep, {'n': 5})
    row = next(row for row in derived(doc) if row['label'] == 'reviewed whole pad relative response')
    assert row['source_columns'] == [0, 1, 2, 3]
    assert row['values'][3] == 15


def test_mixed_modality_finding_can_cite_context_without_granting_context_force_capability():
    base = np.array([[0., 0.], [1., 0.], [0., 1.]])
    ep = episode({'force': np.arange(5.).reshape(5, 1), 'points': np.repeat(base[None], 5, axis=0)},
        [{'name': 'force', 'shape': [1], 'unit': 'N'}, {'name': 'points', 'shape': [3, 2]}],
        {'force': {'role': 'normal_force'}, 'points': {'role': 'landmarks'}})
    doc = se.build(ep, {'n': 5})
    force = next(row for row in doc['series'] if row['sensor_id'] == 'signal:force')
    pose = next(row for row in derived(doc) if 'first frame' in row['label'])
    finding = {'start_s': .2, 'end_s': .3, 'headline': 'Force rises with similar estimated hand shape',
        'claim_type': 'force_change', 'adds_beyond_video': True,
        'evidence': [{'sensor_id': force['sensor_id'], 'series_id': force['id'], 'time_s': [.2, .3]},
                     {'sensor_id': pose['sensor_id'], 'series_id': pose['id'], 'time_s': [.2, .3]}]}
    assert se.bind(doc, [finding])['findings']
    finding['evidence'] = finding['evidence'][1:]
    assert not se.bind(doc, [finding])['findings']


def test_actual_prompt_preserves_sample_specific_coverage_when_means_have_missing_cells(monkeypatch):
    semantics = {'version': 1, 'regions': [{'name': 'pad', 'indices': [0, 1]}], 'response_direction': 'up'}
    values = np.array([[100., 100.], [100., 100.], [100., 100.], [200., np.nan], [200., 100.]])
    ep = episode({'tactile_pad': values}, [{'name': 'tactile_pad', 'shape': [2], 'sensor_semantics': semantics}])
    doc = se.build(ep, {'n': 5})
    row = next(row for row in derived(doc) if row['label'] == 'pad relative response')
    assert row['valid_counts'] == [2, 2, 2, 1, 2]
    assert row['values'][-2:] == [100, 50]
    monkeypatch.setattr(se, 'MAX_PROMPT_BYTES', 15000)
    larger = region_episode(500)
    larger['signals']['pad'][30:50, 0] = np.nan
    prompt = expand_prompt(json.loads(se.prompt(se.build(larger, {'n': 500})).splitlines()[-1]))
    for trace in prompt['series']:
        if trace.get('derivation_id'):
            assert len(trace['valid_counts']) == len(trace['times']) == len(trace['values'])
            audit = next(row for row in derived(se.build(larger, {'n': 500})) if row['id'] == trace['id'])
            by_time = dict(zip(audit['times'], audit['valid_counts']))
            assert trace['valid_counts'] == [by_time[t] for t in trace['times']]


def test_explicit_point_schema_survives_a_coarser_machine_role():
    base = np.array([[0., 0.], [1., 0.], [0., 1.]])
    ep = episode({'points': np.repeat(base[None], 5, axis=0)}, [
        {'name': 'points', 'shape': [3, 2], 'sensor_semantics': {'version': 1,
         'normalization_pair': [0, 1], 'provenance': {'authority': 'publisher_candidate'}}}],
        {'points': {'role': 'end_effector_pose', 'provenance': 'machine'}})
    assert any(row['sensor_id'] == 'signal:points' for row in derived(se.build(ep, {'n': 5})))
    ep['context']['data_dictionary']['entries']['field_0'].update(role='annotation', provenance='human')
    assert not any(row['sensor_id'] == 'signal:points' for row in derived(se.build(ep, {'n': 5})))


def test_changed_human_meaning_withdraws_candidate_regional_anatomy_and_polarity():
    ep = region_episode(entry={'provenance': 'human', 'role': 'touch',
        'meaning': 'A flat pressure sheet without anatomical locations or a known polarity.'})
    doc = se.build(ep, {'n': 5})
    assert not derived(doc)
    assert doc['series']
    assert any('withdrawn' in note for note in doc['limitations'])


def test_quiet_pose_comparisons_retain_temporal_context_for_other_modalities():
    base = np.array([[0., 0.], [1., 0.], [0., 1.]])
    ep = episode({'points': np.repeat(base[None], 126, axis=0)},
        [{'name': 'points', 'shape': [3, 2]}], {'points': {'role': 'landmarks'}})
    doc = se.build(ep, {'n': 126})
    trace = next(row for row in derived(doc) if row.get('comparison_from_s'))
    assert len(trace['times']) == 48
    assert max(b-a for a,b in zip(trace['source_rows'], trace['source_rows'][1:])) <= 4


def test_declared_regions_work_for_a_64_by_64_pad_over_a_short_recording():
    raw = np.full((1000, 4096), 100., dtype=np.float32)
    raw[450:550, :2048] = 80
    ep = episode({'tactile_pad': raw}, [{'name': 'tactile_pad', 'shape': [64, 64], 'sensor_semantics': {
        'version': 1, 'response_direction': 'down', 'regions': [
            {'name': 'left half', 'indices': list(range(2048))},
            {'name': 'right half', 'indices': list(range(2048, 4096))}]}}])
    doc = se.build(ep, {'n': 1000})
    trace = next(row for row in derived(doc) if row['label'] == 'left half relative response')
    assert max(trace['values']) == 20
    assert all(value == 2048 for value in trace['valid_counts'])


def test_short_source_trajectories_keep_native_regional_pose_and_curl_times():
    from pathlib import Path
    from label import derived_measurements as dm
    schemas = json.loads((Path(__file__).parents[1] / 'configs/sensor_schemas/opentouch_hdf5_v1.json').read_text())['signals']
    n = 186
    times = np.arange(n) / 30
    pressure = np.full((n, 256), 100.)
    pressure[17:27, schemas['right_pressure']['regions'][0]['indices']] = np.arange(10)[:, None] + 50
    base = np.column_stack((np.arange(21.), np.arange(21.) % 3, np.arange(21.) % 5))
    points = np.repeat(base[None], n, axis=0)
    points[18:27, 4, 1] += np.arange(9) / 100
    ep = episode({'right_pressure': pressure, 'right_hand_landmarks': points}, [
        {'name': 'right_pressure', 'shape': [16, 16], 'sensor_semantics': schemas['right_pressure']},
        {'name': 'right_hand_landmarks', 'shape': [21, 3], 'sensor_semantics': schemas['right_hand_landmarks']}])
    ep['times']['exo'] = times
    doc = se.build(ep, {'n': n, 'ks': [0, n - 1]})
    sent = expand_prompt(json.loads(se.prompt(doc).splitlines()[-1]))
    ordinary = [row for row in sent['series'] if row.get('derivation_id') and 'comparison_from_s' not in row]
    assert ordinary
    assert all(row['times'] == times.tolist() for row in ordinary)
    assert all(not row['sampled'] and row['coverage']['omitted_samples'] == 0 for row in ordinary)
    thumb = next(row for row in ordinary if row['label'] == 'thumb relative response')
    pose = next(row for row in ordinary if row['label'] == 'Point-set shape from previous frame')
    curl = next(row for row in ordinary if row['label'] == 'thumb curl')
    expected_region = dm.regional_response(pressure, times, schemas['right_pressure'])['traces'][0]['values']
    expected_pose = dm.pointset_shape(points, times, schemas['right_hand_landmarks'])
    assert thumb['values'] == expected_region
    assert pose['values'] == expected_pose['adjacent_rms']
    assert curl['values'] == expected_pose['curls'][0]['values']
    # Selected endpoint comparisons retain their separate source-row receipts.
    selected = next(row for row in sent['series'] if row.get('comparison_from_s'))
    assert len(selected['times']) <= 48
    assert len(selected['times']) < len(pose['times'])
    for feature in [thumb, pose, curl]:
        finding = {'start_s': float(times[19]), 'end_s': float(times[20]), 'headline': 'Regional response and estimated geometry differ',
                   'claim_type': 'response_change', 'adds_beyond_video': True,
                   'evidence': [{'sensor_id': thumb['sensor_id'], 'series_id': thumb['id'], 'time_s': [float(times[19]), float(times[20])]},
                                {'sensor_id': feature['sensor_id'], 'series_id': feature['id'], 'time_s': [float(times[19]), float(times[20])]}]}
        assert se.bind(doc, [finding])['findings']


def test_many_short_derived_trajectories_share_the_native_value_cap(monkeypatch):
    from label import semantic_evidence as sem
    monkeypatch.setattr(sem, 'MAX_NATIVE_VALUES', 1024)
    signals = {f'pad_{i}': np.tile(np.arange(4.), (186, 1)) for i in range(6)}
    ep = episode(signals, [{'name': name, 'shape': [4], 'sensor_semantics': descriptor()} for name in signals],
                 {name: {'role': 'touch'} for name in signals})
    doc = se.build(ep, {'n': 186})
    rows = derived(doc)
    assert {row['sensor_id'] for row in rows} == {f'signal:{name}' for name in signals}
    assert sum(len(row['values']) for row in rows) <= 1024
    assert all(row['sampled'] and row['coverage']['omitted_samples'] > 0 for row in rows)
    assert all(len(row['times']) <= 48 for row in rows)
