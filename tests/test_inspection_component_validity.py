"""Trusted channel validity stays attributed without deleting raw inspection evidence."""
import copy
import json

import numpy as np
import pytest

from label import evidence_access as ea, sensor_evidence as se
from tests.test_tactile_contract import episode


def source(tmp_path, authority='recording_verified', width=4, valid=None, description=''):
    values = np.full((5, width), 100.)
    values[3, -1] = 900.
    meta = {'name': 'pad', 'shape': [width], 'description': description,
            'sensor_semantics': {'version': 1, 'valid_indices': [0] if valid is None else valid,
                                 'provenance': {'authority': authority}}}
    ep = episode({'pad': values}, [meta], {'pad': {'role': 'pressure', 'provenance': 'human'}})
    ep.update(dir=tmp_path, sources={}, action=None, depth={})
    return ep


def inspect_component(access, column):
    field = next(row for row in access.inventory() if row['name'] == 'pad')
    return field, access.inspect({'field_id': field['id'], 'mode': 'samples', 'columns': [column]})


def bind_component(access, receipt):
    return access.bind([{'start_s': .3, 'end_s': .3, 'headline': 'Raw component has a pulse',
        'adds_beyond_video': True, 'evidence': [{'evidence_id': receipt['id'], 'time_s': [.3],
                                                'columns': receipt['columns']}]}])


@pytest.mark.parametrize('authority', ['recorded', 'recording_verified'])
def test_trusted_validity_survives_selection_receipt_and_save_without_hiding_raw(tmp_path, authority):
    ep = source(tmp_path, authority, description='Original recording descriptor ' + 'x' * 2400)
    native = se.build(ep, {'n': 5})
    access = ea.Access(ep, native, plan={'n': 5})
    field, receipt = inspect_component(access, 3)
    expected = {'valid_indices': [0], 'authority': authority}
    assert field['descriptor']['truncated'] is True
    assert field['component_validity'] == expected
    context = json.loads(access.selection_content([])[0]['text'].split('\n', 1)[1])
    summary = next(s for s in context['already_supplied_sensor_evidence']['sensors'] if s['id'] == 'signal:pad')
    assert summary['component_validity'] == expected
    selected = {'columns': [3], 'valid': [False], 'authority': authority}
    assert receipt['selected_component_validity'] == selected
    assert receipt['values'] == [[100.], [100.], [100.], [900.], [100.]]
    assert any('raw recorded values' in note for note in receipt['limitations'])
    saved = bind_component(access, receipt)
    assert len(saved['findings']) == 1
    assert saved['findings'][0]['claim_type'] == 'recorded_evidence'
    assert saved['sensors'][0]['selected_component_validity'] == selected
    assert saved['sensors'][0]['capabilities'] == ['recorded_evidence']


def test_candidate_layout_does_not_become_trusted_component_validity(tmp_path):
    ep = source(tmp_path, 'publisher_candidate')
    access = ea.Access(ep, se.build(ep, {'n': 5}), plan={'n': 5})
    field, receipt = inspect_component(access, 3)
    assert field['descriptor']['sensor_semantics']['provenance']['authority'] == 'publisher_candidate'
    assert 'component_validity' not in field
    assert 'selected_component_validity' not in receipt
    saved = bind_component(access, receipt)
    assert saved['sensors'][0]['capabilities'] == ['recorded_evidence']
    assert 'selected_component_validity' not in saved['sensors'][0]
    assert receipt['values'][3] == [900.]


def test_current_owned_interpretation_can_withdraw_saved_inspection_validity(tmp_path):
    ep = source(tmp_path)
    access = ea.Access(ep, plan={'n': 5})
    _, receipt = inspect_component(access, 3)
    assert receipt['selected_component_validity']['valid'] == [False]
    current = copy.deepcopy(ep)
    current['context']['data_dictionary']['entries']['field_0']['layout'] = [
        {'name': 'withdrawn layout', 'start': 0, 'count': 10}]
    refreshed = ea.Access(current, plan={'n': 5})
    refreshed.receipts = [copy.deepcopy(receipt)]
    saved = bind_component(refreshed, receipt)
    assert len(saved['findings']) == 1
    assert 'selected_component_validity' not in json.dumps(saved['sensors'][0])
    assert saved['series'][0]['values'][3] == 900.
    assert saved['sensors'][0]['capabilities'] == ['recorded_evidence']


def test_large_trusted_mask_is_bounded_but_selected_validity_remains_exact(tmp_path):
    ep = source(tmp_path, width=8192, valid=list(range(8191)))
    native = se.build(ep, {'n': 5})
    access = ea.Access(ep, native, plan={'n': 5})
    field, receipt = inspect_component(access, 8191)
    assert 'component_validity' not in field
    assert field['component_validity_summary']['authority'] == 'recording_verified'
    assert field['component_validity_summary']['valid_components'] == 8191
    assert len(ea.packed(field).encode()) < 4000
    context = json.loads(access.selection_content([])[0]['text'].split('\n', 1)[1])
    summary = next(s for s in context['already_supplied_sensor_evidence']['sensors'] if s['id'] == 'signal:pad')
    assert 'component_validity' not in summary
    assert summary['component_validity_summary'] == field['component_validity_summary']
    assert len(ea.packed(receipt).encode()) < ea.MAX_RESPONSE_BYTES
    assert receipt['selected_component_validity'] == {
        'columns': [8191], 'valid': [False], 'authority': 'recording_verified'}
    assert receipt['values'][3] == [900.]
    saved = bind_component(access, receipt)
    assert saved['sensors'][0]['selected_component_validity'] == receipt['selected_component_validity']
    assert saved['sensors'][0]['capabilities'] == ['recorded_evidence']


@pytest.mark.parametrize('shape', [[None], [3]])
def test_unknown_or_mismatched_descriptor_shape_keeps_raw_access_without_validity(tmp_path, shape):
    ep = source(tmp_path)
    ep['context']['signals'][0]['shape'] = shape
    access = ea.Access(ep, plan={'n': 5})
    field, receipt = inspect_component(access, 3)
    assert 'component_validity' not in field
    assert 'selected_component_validity' not in receipt
    assert receipt['values'][3] == [900.]
    assert bind_component(access, receipt)['sensors'][0]['capabilities'] == ['recorded_evidence']


def test_unaligned_prepared_array_keeps_raw_rows_with_selected_validity(tmp_path):
    ep = source(tmp_path)
    values = ep['signals'].pop('pad')
    ep['context']['signals'][0]['key'] = 'p'
    ep['context']['state_unaligned'] = True
    np.savez(tmp_path / 'signals.npz', p=values)
    access = ea.Access(ep, plan={'n': 5})
    field = next(row for row in access.inventory() if row['name'] == 'pad')
    assert field['kind'] == 'array'
    receipt = access.inspect({'field_id': field['id'], 'mode': 'samples',
                              'start_row': 0, 'end_row': 5, 'columns': [3]})
    assert 'times_s' not in receipt
    assert receipt['selected_component_validity'] == {
        'columns': [3], 'valid': [False], 'authority': 'recording_verified'}
    assert receipt['values'] == [[100.], [100.], [100.], [900.], [100.]]
    saved = access.bind([{'start_s': None, 'end_s': None, 'headline': 'Raw component has a pulse',
        'adds_beyond_video': True, 'evidence': [{'evidence_id': receipt['id'], 'rows': [3], 'columns': [3]}]}])
    assert len(saved['untimed_findings']) == 1
    assert access.record()['inspections'][0]['selected_component_validity'] == receipt['selected_component_validity']
