"""Independent regional analysis retains its own exact source admission."""
import copy
import json

import numpy as np
import pytest

from label import sensor_evidence as se
from tests.test_semantic_evidence import descriptor
from tests.test_tactile_contract import episode


def source(*, with_context=False):
    values = np.full((5, 16), 100.)
    values[3, :8] = 80
    semantics = descriptor(regions=[{'name': 'region A', 'indices': list(range(8))},
                                    {'name': 'region B', 'indices': list(range(8, 16))}])
    ep = episode({'pad': values}, [{'name': 'pad', 'shape': [4, 4], 'unit': 'ADC',
                 'sensor_semantics': semantics}], {'pad': {'role': 'pressure'}})
    doc = se.build(ep, {'n': 5, 'ks': [0, 4]})
    return (doc, ep['context']) if with_context else doc


def request(doc):
    from label import sensor_analysis as sa
    content = [{'type': 'text', 'text': 'VISUAL OUTPUT SCHEMA MUST NOT ENTER SENSOR PASS'},
               {'type': 'text', 'text': '=== grid t=0-0.4s | original camera ==='},
               {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,cGl4ZWxz', 'detail': 'high'}}]
    return sa.build(content, doc, config={'model': 'fixture', 'reasoning': 'medium', 'max_tokens': 16000,
                    'input_identity': {'source': 'one'}, 'sensor_copy_contract_version': 2})


def finding(doc, *, local=False):
    row = next(row for row in doc['series'] if bool(row.get('derivation_id')) is not local)
    return {'start_s': .3, 'end_s': .3, 'headline': 'A regional response increases',
            'detail': 'One region responds more strongly than the other.', 'claim_type': 'response_change',
            'adds_beyond_video': True, 'evidence': [{'sensor_id': row['sensor_id'],
                'series_id': row['id'], 'time_s': [.3]}]}


def receipt(labels, stage):
    return {'state': 'received', 'response_text': json.dumps(labels), 'finish_reason': 'stop',
            'model': 'fixture', 'generation_id': 'offline-fixture', 'digest': stage['digest']}


def test_regional_pass_preserves_derived_values_and_media_but_excludes_local_admission():
    doc = source()
    original = copy.deepcopy(doc)
    stage = request(doc)
    assert stage is not None
    assert [r for r in stage['sensor_evidence']['series'] if not r.get('regional_output')] == [
        r for r in doc['series'] if r.get('derivation_id')]
    assert {r.get('regional_output') for r in stage['sensor_evidence']['series']} >= {
        'raw_mean', 'raw_never_zero_mean', 'zero_cells'}
    assert all('array_evidence' not in s for s in stage['sensor_evidence']['sensors'])
    text = '\n'.join(p.get('text', '') for p in stage['content'])
    assert 'VISUAL OUTPUT SCHEMA MUST NOT ENTER SENSOR PASS' not in text
    assert 'overlay_detail' not in text
    assert [p for p in stage['content'] if p['type'] == 'image_url'] == [
        {'type': 'image_url', 'image_url': {'url': 'data:image/jpeg;base64,cGl4ZWxz', 'detail': 'high'}}]
    assert doc == original
    from label import sensor_analysis as sa
    result = sa.bind(stage, receipt({'sensor_findings': [finding(doc), finding(doc, local=True)]}, stage))
    assert len(result['findings']) == 1 and len(result['unbound_findings']) == 1
    assert result['findings'][0]['copy_contract_version'] == 2


def test_stage_is_capability_gated_and_its_digest_includes_exact_source_and_settings():
    from label import sensor_analysis as sa
    doc = source()
    first = request(doc)
    changed = copy.deepcopy(doc)
    next(r for r in changed['series'] if r.get('derivation_id'))['values'][3] += 1
    assert request(changed)['digest'] != first['digest']
    raw = receipt({'sensor_findings': []}, first)
    with pytest.raises(ValueError, match='request proof changed'):
        sa.bind(request(changed), raw)
    altered = copy.deepcopy(first)
    altered['sensor_evidence']['series'][0]['values'][3] += 1
    with pytest.raises(ValueError, match='request proof changed'):
        sa.bind(altered, raw)
    without_local = copy.deepcopy(doc)
    for sensor in without_local['sensors']:
        sensor.pop('array_evidence', None)
    assert request(without_local) is None
    without_regions = copy.deepcopy(doc)
    without_regions['series'] = [r for r in doc['series'] if not r.get('derivation_id')]
    assert request(without_regions) is None


def test_empty_analysis_is_valid_but_missing_or_truncated_findings_are_not_retried_as_empty():
    from label import sensor_analysis as sa
    stage = request(source())
    assert sa.bind(stage, receipt({'sensor_findings': []}, stage))['findings'] == []
    for bad in [receipt({}, stage), receipt({'sensor_findings': 'not a list'}, stage),
                {**receipt({'sensor_findings': []}, stage), 'finish_reason': 'length'}]:
        with pytest.raises(ValueError):
            sa.bind(stage, bad)


def test_attached_findings_keep_their_own_namespace_provenance_and_raw_outputs():
    from label import sensor_analysis as sa
    doc, context = source(with_context=True)
    stage = request(doc)
    raw = receipt({'sensor_findings': [finding(doc)]}, stage)
    bound = sa.bind(stage, raw)
    main = se.bind(doc, [finding(doc)], copy_contract_version=2)
    result = {'labels': {'sensor_findings': [finding(doc)]}, 'sensor_evidence': main}
    original_labels, original_raw = copy.deepcopy(result['labels']), copy.deepcopy(raw)
    sa.attach(result, bound)
    added = result['sensor_evidence']['findings'][-1]
    assert added['evidence'][0]['sensor_id'].startswith('mechanics:')
    assert added['annotation_provenance']['generation_id'] == raw['generation_id']
    sensors = {s['id'] for s in result['sensor_evidence']['sensors']}
    series = {r['id'] for r in result['sensor_evidence']['series']}
    assert added['evidence'][0]['sensor_id'] in sensors and added['evidence'][0]['series_id'] in series
    assert result['labels'] == original_labels and raw == original_raw
    cleaned = sa.without_attached(result['sensor_evidence'])
    assert cleaned['sensors'] == main['sensors'][:len(doc['sensors'])]
    assert all(not row['sensor_id'].startswith('mechanics:') for row in cleaned['series'])
    joined = se.merge([({**context, 'piece': {'index': 1, 't0_s': 2}}, result)], context)
    attached = next(f for f in joined['findings'] if f.get('annotation_provenance'))
    assert attached['start_s'] == 2.3
    assert attached['annotation_provenance']['generation_id'] == raw['generation_id']
    assert any(d.get('source') == 'part1:mechanics:signal:pad' for d in joined['derivations'])


def test_reparse_rebinds_saved_regional_response_once_and_preserves_human_caption(tmp_path, monkeypatch):
    from label import evidence_access as ea, harness, reparse, sensor_analysis as sa
    from tests.test_shared_sensor_copy import source as stored_source
    ep = stored_source(tmp_path)
    ep['signals']['pad'] = np.full((5, 16), 100.)
    ep['signals']['pad'][3, :8] = 80
    ep['context']['signals'][0].update(shape=[4, 4], sensor_semantics=descriptor(
        regions=[{'name': 'region A', 'indices': list(range(8))},
                 {'name': 'region B', 'indices': list(range(8, 16))}]))
    (tmp_path / 'context.json').write_text(json.dumps(ep['context']))
    np.savez(tmp_path / 'signals.npz', pad=ep['signals']['pad'])
    doc = se.build(ep, {'n': 5, 'ks': [0, 4]})
    stage = request(doc)
    raw = receipt({'sensor_findings': [finding(doc)]}, stage)
    raw['request'] = {key: copy.deepcopy(value) for key, value in stage.items() if key != 'content'}
    access = ea.Access(ep, doc, plan={'n': 5}, copy_contract_version=2)
    record = {'episode_dir': str(tmp_path), 'config': {'sensor_copy_contract_version': 2},
              'labels': {'sensor_findings': [finding(doc)]}, 'sensor_evidence': se.bind(doc, [finding(doc)]),
              'evidence_inspection': access.record(), 'sensor_analysis': raw}
    sa.attach(record, sa.bind(stage, raw))
    record['sensor_evidence']['findings'][-1].update(review_status='accepted', overlay_detail='')
    before = copy.deepcopy(record['sensor_analysis'])
    monkeypatch.setattr(harness.me, 'load', lambda folder: ep)
    for _ in range(2):
        reparse.rebind(record)
        added = [f for f in record['sensor_evidence']['findings'] if f.get('annotation_provenance')]
        assert len(added) == 1
        assert added[0]['overlay_detail'] == '' and added[0]['review_status'] == 'accepted'
        assert record['sensor_analysis'] == before
    original_context = copy.deepcopy(ep['context'])
    ep['context']['data_dictionary']['entries']['field_0']['role'] = 'annotation'
    (tmp_path / 'context.json').write_text(json.dumps(ep['context']))
    for _ in range(2):
        reparse.rebind(record)
        assert not any(f.get('annotation_provenance') for f in record['sensor_evidence']['findings'])
        assert len([f for f in record['sensor_evidence']['withheld_findings'] if f.get('annotation_provenance')]) == 1
    ep['context'] = original_context
    (tmp_path / 'context.json').write_text(json.dumps(ep['context']))
    reparse.rebind(record)
    recovered = next(f for f in record['sensor_evidence']['findings'] if f.get('annotation_provenance'))
    assert recovered['overlay_detail'] == '' and recovered['review_status'] == 'accepted'
    assert record['sensor_analysis'] == before
