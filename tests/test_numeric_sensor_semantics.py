"""Literal sensor semantics take precedence over names and response heuristics."""
import numpy as np
import pytest

from label import contacts, sensor_evidence as se, signals as sg
from tests.test_tactile_contract import episode


def test_vector_layout_preserves_opposing_axes_when_the_components_fit():
    a = np.array([[0., 0, 0], [5, -5, 0], [10, -10, 0], [0, 0, 0]])
    ep = episode({'load': a}, [{'name': 'load', 'names': ['Fx', 'Fy', 'Fz'], 'unit': 'N'}], {
        'load': {'role': 'force_torque', 'layout': [{'name': 'force XYZ', 'start': 0, 'count': 3}]}})
    rows = se.build(ep, {'n': 4})['series']
    assert [r['label'] for r in rows] == ['Fx', 'Fy', 'Fz']
    assert rows[0]['values'] == [0, 5, 10, 0]
    assert rows[1]['values'] == [0, -5, -10, 0]
    assert all(r['operation'] == 'recorded value' for r in rows)


def test_contact_token_does_not_turn_a_continuous_pressure_map_into_a_state():
    raw = np.arange(80.).reshape(5, 4, 4)
    ep = episode({'contact_pressure': raw}, [{'name': 'contact_pressure', 'shape': [4, 4], 'unit': 'kPa'}])
    sensor = se.build(ep, {'n': 5})['sensors'][0]
    assert sensor['quantity'] == 'sensor_response'
    assert sensor['array_evidence']
    assert 'contact_change' not in sensor['capabilities']
    assert se._quantity('contact', {}) == 'sensor_response'
    assert se._quantity('switch', {'role': 'contact_state'}) == 'contact_state'


@pytest.mark.parametrize('direction,zero,loaded,reduced', [('up', 0, 100, 80), ('down', 1000, 900, 920)])
def test_explicit_zero_and_direction_keep_a_loaded_sensor_active(direction, zero, loaded, reduced):
    a = np.full((100, 1), float(loaded)); a[30:70] = reduced
    meta = {'name': 'right_force', 'unit': 'N', 'calibration': {'zero': zero},
            'response_direction': direction, 'dictionary': {'role': 'normal_force'}}
    rows = contacts.find({'right_force': a}, {'right_force': meta}, np.arange(100) / 10)
    assert len(rows) == 1
    assert (rows[0]['start_s'], rows[0]['end_s']) == (0, 9.9)
    assert rows[0]['peak_s'] == 0
    assert rows[0]['reference_kind'] == 'declared_zero'
    assert rows[0]['evidence_kind'] == 'response_activity'


def test_declared_zero_does_not_treat_negative_response_or_missing_rows_as_load():
    a = np.array([[100.], [80.], [np.nan], [-5.], [0.], [10.], [10.]])
    meta = {'calibration': {'zero': 0}, 'response_direction': 'up'}
    activity = sg.declared_zero_activity(a, meta)
    assert activity[:2].tolist() == [100, 80]
    assert np.isnan(activity[2])
    assert activity[3:].tolist() == [0, 0, 10, 10]
    rows = contacts.find({'right_force': a}, {'right_force': meta}, np.arange(7.))
    assert [(c['start_s'], c['end_s']) for c in rows] == [(0, 1), (5, 6)]
    assert sg.declared_zero_activity(a, {'calibration': {'zero': 0}}) is None


def test_inferred_response_reference_is_qualified_even_in_cached_contacts():
    a = np.full((100, 1), 100.); a[30:70] = 80
    ep = episode({'right_force': a}, [{'name': 'right_force', 'unit': 'ADC'}])
    row = contacts.of_episode(ep)[0]
    assert row['reference_kind'] == 'inferred'
    assert row['evidence_kind'] == 'response_activity'
    assert 'not a verified unloaded' in row['reference_note']
    ep['context']['contacts'] = [{'id': 'c1', 'signals': ['right_force'], 'start_s': 3., 'end_s': 6.9}]
    assert contacts.of_episode(ep)[0]['reference_kind'] == 'inferred'


def test_cached_inferred_contact_boundaries_are_recomputed_for_declared_zero():
    a = np.full((100, 1), 100.); a[30:70] = 80
    ep = episode({'right_force': a}, [{'name': 'right_force', 'unit': 'N',
        'calibration': {'zero': 0}, 'response_direction': 'up'}])
    ep['context']['contacts'] = [{'id': 'c1', 'signals': ['right_force'], 'start_s': 3., 'end_s': 6.9}]
    assert contacts.of_episode(ep)[0]['start_s'] == 0
    assert contacts.of_episode(ep)[0]['end_s'] == 9.9


def test_quiet_calibrated_sensor_does_not_fall_back_to_an_all_episode_contact():
    meta = {'calibration': {'zero': 0}, 'response_direction': 'up'}
    assert contacts.find({'right_force': np.zeros((10, 1))}, {'right_force': meta}, np.arange(10.)) == []


def test_per_cell_calibration_reference_is_reported_for_sparse_loaded_readings():
    meta = {'calibration': {'zero': [0, 10]}, 'response_direction': 'up'}
    values = np.array([[2., 13.], [1., 12.], [1., 11.]])
    row = contacts.find({'right_force': values}, {'right_force': meta}, np.arange(3.))[0]
    assert row['reference_kind'] == 'declared_zero'
    assert sg.declared_zero_activity(values, meta).tolist() == [5, 3, 2]


def test_force_axis_groups_do_not_become_means_when_only_one_trace_fits():
    a = np.array([[0., 0., 0.], [5., -5., 0.]])
    groups = se._groups({'name': 'load', 'names': ['Fx', 'Fy', 'Fz'], 'unit': 'N'}, {
        'role': 'force_torque', 'layout': [{'name': 'force XYZ', 'start': 0, 'count': 3}]}, a, 1)
    assert len(groups) == 1
    assert len(groups[0][1]) == 1


def test_constant_calibrated_load_is_eligible_and_its_map_uses_the_declared_zero():
    from label import episode as me
    values = np.full((10, 4), 100.)
    ep = episode({'right_force': values}, [{'name': 'right_force', 'shape': [2, 2],
        'unit': 'N', 'calibration': {'zero': 0}, 'response_direction': 'up', 'touch': False}])
    assert 'right_force' in me.touch_verdicts(ep, 10)
    assert np.asarray(me._map_tile(ep, 'right_force', 0)).max() == 255


def test_activity_prompt_does_not_present_an_inferred_reference_as_recorded_touch(monkeypatch):
    from label import episode as me
    a = np.full((100, 1), 100.); a[30:70] = 80
    ep = episode({'right_force': a}, [{'name': 'right_force'}])
    row = contacts.of_episode(ep)[0]
    ep['contacts'] = ep['contacts_shown'] = [row]
    monkeypatch.setattr(me, '_contact_views', lambda *args: ([], [], []))
    text = me.contacts_block(ep, {'n': 100, 'touch': {'right_force'}})
    assert 'response activity' in text.lower()
    assert 'not a verified unloaded' in text
    assert 'The recording\'s touch signals say a hand is touching' not in text
    assert 'signal activity begins' in text
    assert 'signal activity ends' in text
    line = me._contact_line({**row, 'from_start': True, 'to_end': True})
    assert 'already active' in line and 'still active' in line
    assert 'already touching' not in line


def test_explicit_contact_state_retains_recorded_state_identity():
    rows = contacts.mark_aligned([{'signals': ['switch']}], {'switch': {'dictionary': {'role': 'contact_state'}}})
    assert rows[0]['evidence_kind'] == 'recorded_contact_state'
    assert rows[0]['reference_kind'] == 'declared_contact_state'


def test_activity_timing_disagreement_is_not_reported_as_sensor_clock_error():
    from checks import contacts as cc
    rows = [{'id': 'c' + str(i), 'start_s': float(i), 'end_s': i + .5,
             'hand': 'right', 'evidence_kind': 'response_activity'} for i in range(3)]
    labels = {'contacts': [{'id': r['id'], 'first_touch_frame': 2, 'last_touch_frame': 2,
                            'touch_seen': 'no', 'hand': 'left'} for r in rows]}
    strips = {r['id']: {'begin': [r['start_s']+.1, r['start_s']+.2],
                        'end': [r['end_s']+.1, r['end_s']+.2, r['end_s']+.3]} for r in rows}
    result = cc.check(labels, rows, strips, 30)
    assert 'offset_ms' not in result
    assert 'timing_basis' not in result
    assert not {'clock_offset', 'hand_mismatch'} & {n['check'] for n in result['notes']}
    assert 'response activity' in result['notes'][0]['evidence']
    assert 'not establish' in result['notes'][0]['evidence']
