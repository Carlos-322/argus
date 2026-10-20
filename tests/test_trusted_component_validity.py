"""Trusted component masks restrict physical sensor claims without deleting source values."""
import copy

import numpy as np
import pytest

from label import sensor_evidence as se, reparse
from tests.test_tactile_contract import episode


def source(authority='recording_verified', width=4, valid=None, layout=None):
    values = np.full((5, width), 100.)
    values[3, width - 1] = 900
    semantics = {'version': 1, 'valid_indices': [0] if valid is None else valid,
                 'provenance': {'authority': authority}}
    entry = {'role': 'pressure', 'provenance': 'human'}
    if layout:
        entry['layout'] = layout
    ep = episode({'pressure': values}, [{'name': 'pressure', 'shape': [width],
        'sensor_semantics': semantics}], {'pressure': entry})
    return ep


def claim(ident, time=.3):
    return {'start_s': time, 'end_s': time, 'headline': 'A pressure response increases',
            'claim_type': 'pressure_response_change', 'adds_beyond_video': True,
            'evidence': [{'sensor_id': 'signal:pressure', 'series_id': ident, 'time_s': [time]}]}


def legacy(ep):
    doc = se.build(ep, {'n': 5})
    doc['sensors'][0].pop('component_validity', None)
    doc['series'] = [{'id': 'signal:pressure:columns:3', 'sensor_id': 'signal:pressure',
        'label': 'pressure [3]', 'columns': [3], 'unit': None, 'times': [0., .1, .2, .3, .4],
        'values': ep['signals']['pressure'][:, 3].tolist(), 'operation': 'recorded value',
        'native_samples': 5, 'sampled': False}]
    return doc


@pytest.mark.parametrize('authority', ['recorded', 'recording_verified'])
def test_trusted_mask_selection_does_not_offer_excluded_cells_as_pressure(authority):
    ep = source(authority)
    before = copy.deepcopy(ep['signals']['pressure'])
    doc = se.build(ep, {'n': 5})
    assert [row['columns'] for row in doc['series']] == [[0]]
    assert doc['sensors'][0]['component_validity'] == {'valid_indices': [0], 'authority': authority}
    assert np.array_equal(ep['signals']['pressure'], before)


def test_trusted_mask_filters_group_members_before_averaging():
    ep = source(width=16, valid=list(range(13)), layout=[{'name': 'recorded pad', 'start': 0, 'count': 16}])
    doc = se.build(ep, {'n': 5})
    row = next(row for row in doc['series'] if not row.get('derivation_id'))
    assert row['columns'] == list(range(13))
    assert row['values'] == [100.] * 5
    assert se.bind(doc, [claim(row['id'])])['findings']


def test_new_saved_descriptor_rejects_stale_excluded_component_row():
    ep = source()
    doc = se.build(ep, {'n': 5})
    old_row = legacy(ep)['series'][0]
    doc['series'].append(old_row)
    result = se.bind(doc, [claim(old_row['id'])])
    assert not result['findings'] and result['unbound_findings']
    assert next(row for row in result['series'] if row['id'] == old_row['id'])['values'] == old_row['values']


def test_legacy_binding_uses_descriptor_matched_owned_source_context():
    ep = source()
    doc = legacy(ep)
    result = se.bind(doc, [claim(doc['series'][0]['id'])], source_context=ep['context'])
    assert not result['findings'] and result['unbound_findings']
    assert result['series'][0]['values'] == doc['series'][0]['values']


def test_legacy_reparse_applies_owned_source_validity(tmp_path, monkeypatch):
    import json
    import label
    from label import harness
    ep = source()
    doc = legacy(ep)
    (tmp_path / 'context.json').write_text(json.dumps(ep['context']))
    record = {'episode_dir': str(tmp_path), 'labels': {'sensor_findings': [claim(doc['series'][0]['id'])]},
              'sensor_evidence': doc}
    monkeypatch.setattr(label, 'sensor_evidence', se)
    monkeypatch.setattr(harness, 'saved_input_problem', lambda *args: None)
    reparse.rebind(record)
    assert not record['sensor_evidence']['findings'] and record['sensor_evidence']['unbound_findings']


def test_candidate_mask_does_not_delete_unverified_recorded_components():
    ep = source('publisher_candidate')
    doc = se.build(ep, {'n': 5})
    assert [row['columns'] for row in doc['series']] == [[0], [1], [2], [3]]
    assert 'component_validity' not in doc['sensors'][0]


def test_human_replacement_of_candidate_layout_removes_candidate_exclusions():
    ep = source('publisher_candidate', layout=[{'name': 'reviewed pad', 'start': 0, 'count': 4}])
    doc = se.build(ep, {'n': 5})
    assert [row['columns'] for row in doc['series'] if not row.get('derivation_id')] == [[0], [1], [2], [3]]
    derived = next(row for row in doc['series'] if row.get('derivation_id'))
    assert derived['source_columns'] == [0, 1, 2, 3]


def test_empty_trusted_mask_does_not_offer_numeric_pressure_claims():
    doc = se.build(source(valid=[]), {'n': 5})
    assert not doc['series']


@pytest.mark.parametrize('context', [None, [], 'legacy context', {'signals': [{'name': 'pressure'}]}, {'signals': [None, {}, {'name': 'pressure'}]}])
def test_legacy_sensor_without_name_or_shape_does_not_invent_validity(context):
    doc = {'sensors': [{'id': 'camera:legacy', 'kind': 'image'}], 'series': [], 'limitations': []}
    assert se.bind(doc, [], source_context=context)['findings'] == []


def test_metadata_mismatch_does_not_invent_component_validity():
    ep = source('publisher_candidate')
    doc = legacy(ep)
    changed = copy.deepcopy(ep['context'])
    changed['signals'][0]['sensor_semantics']['provenance']['authority'] = 'recording_verified'
    result = se.bind(doc, [claim(doc['series'][0]['id'])], source_context=changed)
    assert len(result['findings']) == 1


def test_owned_human_override_withdrawal_is_authoritative_over_saved_mask():
    ep = source('recorded')
    doc = legacy(ep)
    doc['sensors'][0]['component_validity'] = {'valid_indices': [0], 'authority': 'recorded'}
    ep['context']['data_dictionary']['entries']['field_0']['layout'] = [
        {'name': 'withdrawn layout', 'start': 0, 'count': 10}]
    result = se.bind(doc, [claim(doc['series'][0]['id'])], source_context=ep['context'])
    assert len(result['findings']) == 1


def test_legacy_named_descriptor_without_shape_does_not_invent_validity():
    meta = {'name': 'pressure'}
    doc = {'sensors': [{'id': 'signal:pressure', 'name': 'pressure', 'kind': 'numeric',
                       'descriptor_digest': se._descriptor_digest(meta)}], 'series': [], 'limitations': []}
    assert se.bind(doc, [], source_context={'signals': [meta]})['findings'] == []


@pytest.mark.parametrize('review_status', ['accepted', 'rejected'])
def test_reparse_preserves_review_when_trusted_mask_newly_unbinds_finding(tmp_path, monkeypatch, review_status):
    import json
    from label import harness
    ep = source()
    doc = legacy(ep)
    raw = claim(doc['series'][0]['id'])
    previously_bound = se.bind(doc, [raw])
    assert len(previously_bound['findings']) == 1
    previously_bound['findings'][0]['review_status'] = review_status
    (tmp_path / 'context.json').write_text(json.dumps(ep['context']))
    record = {'episode_dir': str(tmp_path), 'labels': {'sensor_findings': [raw]},
              'sensor_evidence': previously_bound}
    monkeypatch.setattr(harness, 'saved_input_problem', lambda *args: None)
    original_labels = copy.deepcopy(record['labels'])
    for _ in range(2):
        reparse.rebind(record)
        result = record['sensor_evidence']
        assert not result['findings']
        assert len(result['unbound_findings']) == 1
        unbound = result['unbound_findings'][0]
        assert 'recorded validity mask' in unbound['reason']
        assert unbound['finding']['review_status'] == review_status
        assert record['labels'] == original_labels
