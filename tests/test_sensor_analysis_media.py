"""The regional pass may cite only the image bytes and clocks it actually receives."""
import copy
import hashlib
import json

import numpy as np

from label import sensor_analysis as sa, sensor_evidence as se, sensor_scene
from tests.test_semantic_evidence import descriptor
from tests.test_sensor_scene import episode as scene_episode
from tests.test_tactile_contract import episode as signal_episode


def source(tmp_path):
    ep = scene_episode(tmp_path)
    ep['context']['profile'] = 'ego_head'
    ep['context']['cameras']['exo']['name'] = 'head'
    values = np.full((7, 16), 100.)
    values[3:, :8] = 80
    meta = {'name': 'pad', 'shape': [4, 4], 'unit': 'ADC', 'sensor_semantics': descriptor(
        regions=[{'name': 'region A', 'indices': list(range(8))},
                 {'name': 'region B', 'indices': list(range(8, 16))}])}
    signal = signal_episode({'pad': values}, [meta], {'pad': {'role': 'pressure'}})
    ep['signals'] = signal['signals']
    ep['signal_meta'] = signal['signal_meta']
    ep['context'].update({key: signal['context'][key] for key in ('episode_id', 'signals', 'data_dictionary')})
    doc = se.build(ep, {'n': 7, 'ks': [0, 6]})
    parts, pairing, _ = sensor_scene.collect(ep, doc, shown={}, width=96)
    content = [*se.image_parts(doc), *parts]
    config = {'model': 'fixture', 'reasoning': 'medium', 'max_tokens': 16000,
              'sensor_copy_contract_version': 2, 'sensor_scene_pairing': pairing}
    return ep, doc, content, config


def labels(doc, refs):
    row = next(row for row in doc['series'] if row.get('derivation_id'))
    return {'start_s': .3, 'end_s': .3, 'headline': 'A regional response differs',
            'detail': 'The declared regions respond differently in the supplied camera context.',
            'claim_type': 'response_change', 'adds_beyond_video': True,
            'evidence': [{'sensor_id': row['sensor_id'], 'series_id': row['id'], 'time_s': [.3]}, *refs]}


def bind(stage, findings):
    return sa.bind(stage, {'digest': stage['digest'], 'state': 'received', 'finish_reason': 'stop',
        'response_text': json.dumps({'sensor_findings': findings})})


def test_image_backed_array_snapshot_is_citable_without_admitting_cell_readings(tmp_path):
    ep, doc, content, config = source(tmp_path)
    stage = sa.build(content, doc, config=config, ep=ep)
    result = bind(stage, [labels(doc, [{'sensor_id': 'signal:pad', 'time_s': [.3]}])])
    assert len(result['findings']) == 1 and not result['unbound_findings']
    sensor = next(s for s in stage['sensor_evidence']['sensors'] if s['id'] == 'signal:pad')
    array = sensor['array_evidence']
    assert 'native_constituents' not in array and 'constituents' not in array
    assert 'values' not in array['reference']
    assert all('values' not in frame for frame in array['frames'])
    rejected = bind(stage, [labels(doc, [{'sensor_id': 'signal:pad', 'series_id': 'signal:pad:array:0', 'time_s': [.3]}])])
    assert not rejected['findings'] and rejected['unbound_findings']


def test_source_declared_camera_label_binds_as_context_without_mechanics_capability(tmp_path):
    ep, doc, content, config = source(tmp_path)
    stage = sa.build(content, doc, config=config, ep=ep)
    # The actual tongs response uses its supplied camera label, head, as sensor_id.
    result = bind(stage, [labels(doc, [{'sensor_id': 'head', 'time_s': [.3]}])])
    assert len(result['findings']) == 1 and not result['unbound_findings']
    context = next(s for s in stage['sensor_evidence']['sensors'] if s['id'] == 'head')
    assert context['kind'] == 'image' and context['view'] == 'exo'
    assert context['descriptor_digest'] == se._descriptor_digest(ep['context']['cameras']['exo'])
    camera_only = labels(doc, [])
    camera_only['evidence'] = [{'sensor_id': 'head', 'time_s': [.3]}]
    assert not bind(stage, [camera_only])['findings']
    combined = {'sensor_evidence': se.bind(doc, [])}
    sa.attach(combined, result)
    alias = next(s for s in combined['sensor_evidence']['sensors'] if s['id'] == 'mechanics:head')
    assert alias['citation_alias_of'] == 'mechanics:camera:exo'
    joined = se.merge([({**ep['context'], 'piece': {'index': 3, 't0_s': 2}}, combined)], ep['context'])
    alias = next(s for s in joined['sensors'] if s['id'] == 'part3:mechanics:head')
    assert alias['citation_alias_of'] == 'part3:mechanics:camera:exo'
    assert alias['citation_alias_of'] in {s['id'] for s in joined['sensors']}


def test_missing_grid_hash_or_unshown_clock_never_authorizes_camera_context(tmp_path):
    ep, doc, content, config = source(tmp_path)
    for change in ('missing image', 'changed hash', 'changed clock'):
        parts, settings = copy.deepcopy(content), copy.deepcopy(config)
        if change == 'missing image':
            parts = parts[:len(se.image_parts(doc))]
        elif change == 'changed hash':
            settings['sensor_scene_pairing']['grids'][0]['jpeg_sha256'] = '0' * 64
        else:
            settings['sensor_scene_pairing']['captures'][0]['placements'][0]['source_time_s'] = .321
        stage = sa.build(parts, doc, config=settings, ep=ep)
        at = .321 if change == 'changed clock' else .3
        assert not bind(stage, [labels(doc, [{'sensor_id': 'head', 'time_s': [at]}])])['findings']
    stage = sa.build(content, doc, config=config, ep=ep)
    assert not bind(stage, [labels(doc, [{'sensor_id': 'invented-camera', 'time_s': [.3]}])])['findings']
    assert not bind(stage, [labels(doc, [{'sensor_id': 'head', 'time_s': [.35]}])])['findings']


def test_partial_snapshot_sheet_does_not_admit_unshown_array_clocks(tmp_path):
    values = np.full((17, 16), 100.)
    values[3:, :8] = 80
    ep = signal_episode({'pad': values}, [{'name': 'pad', 'shape': [4, 4], 'unit': 'ADC',
        'sensor_semantics': descriptor(regions=[{'name': 'A', 'indices': list(range(8))},
                                                {'name': 'B', 'indices': list(range(8, 16))}])}],
        {'pad': {'role': 'pressure'}})
    doc = se.build(ep, {'n': 17})
    content = se.image_parts(doc)[:2]
    stage = sa.build(content, doc, config={'sensor_copy_contract_version': 2}, ep=None)
    sensor = next(s for s in stage['sensor_evidence']['sensors'] if s['id'] == 'signal:pad')
    assert sensor['times'] == [0., .1, .2, .3, .4, .5, .6, .7]
    rejected = labels(doc, [{'sensor_id': 'signal:pad', 'time_s': [.9]}])
    rejected.update(start_s=.9, end_s=.9)
    rejected['evidence'][0]['time_s'] = [.9]
    assert not bind(stage, [rejected])['findings']


def test_declared_camera_label_collision_does_not_replace_another_source_view(tmp_path):
    ep, doc, content, config = source(tmp_path)
    ep['context']['cameras']['extra1'] = dict(ep['context']['cameras']['exo'], name='other source')
    ep['context']['cameras']['exo']['name'] = 'camera:extra1'
    ep['sources']['extra1'] = dict(ep['sources']['exo'])
    ep['times']['extra1'] = ep['times']['exo'].copy()
    ep['times']['extra1_pts'] = ep['times']['exo_pts'].copy()
    parts, config['sensor_scene_pairing'], _ = sensor_scene.collect(ep, doc, shown={}, width=96)
    stage = sa.build([*se.image_parts(doc), *parts], doc, config=config, ep=ep)
    camera = next(s for s in stage['sensor_evidence']['sensors'] if s['id'] == 'camera:extra1')
    assert camera['view'] == 'extra1'
    assert not camera.get('citation_alias_of')


def test_ambiguous_source_labels_and_current_alignment_changes_stay_uncitable(tmp_path):
    ep, doc, content, config = source(tmp_path)
    ep['context']['cameras']['extra1'] = dict(ep['context']['cameras']['exo'])
    ep['sources']['extra1'] = dict(ep['sources']['exo'])
    ep['times']['extra1'] = ep['times']['exo'].copy()
    ep['times']['extra1_pts'] = ep['times']['exo_pts'].copy()
    parts, config['sensor_scene_pairing'], _ = sensor_scene.collect(ep, doc, shown={}, width=96)
    stage = sa.build([*se.image_parts(doc), *parts], doc, config=config, ep=ep)
    assert 'head' not in {s['id'] for s in stage['sensor_evidence']['sensors']}
    assert bind(stage, [labels(doc, [{'sensor_id': 'camera:extra1', 'time_s': [.3]}])])['findings']
    ep['context']['camera_clock'] = {'exo': {'clock_problem': 'alignment withdrawn'}}
    stage = sa.build(content, doc, config=config, ep=ep)
    assert not bind(stage, [labels(doc, [{'sensor_id': 'head', 'time_s': [.3]}])])['findings']


def test_capture_top_level_clock_cannot_disagree_with_supplied_placement(tmp_path):
    ep, doc, content, config = source(tmp_path)
    config['sensor_scene_pairing']['captures'][3]['source_time_s'] = .321
    stage = sa.build(content, doc, config=config, ep=ep)
    assert not bind(stage, [labels(doc, [{'sensor_id': 'head', 'time_s': [.3]}])])['findings']
