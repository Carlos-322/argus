"""Inspection retains verified presentation coordinates without asserting a physical event clock."""
import copy

import numpy as np
import pytest

from label import episode, evidence_access
from prepare.signal_alignment import RECORDER_ARRIVAL, RECORDER_ARRIVAL_LIMITATION


@pytest.fixture
def arrival_episode(tmp_path):
    from test_abc_generic import capture_mcap, converted
    source = tmp_path / 'source'; source.mkdir()
    capture_mcap(source / 'recording.mcap', n=60, bad_header='/left-arm-state')
    path, context = converted(source, tmp_path / 'prepared')
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    loaded = episode.load(path)
    name = next(name for name, meta in loaded['signal_meta'].items()
                if 'velocity' in name and meta.get('aligned_by') == RECORDER_ARRIVAL
                and np.any(np.diff(loaded['signals'][name][:, 0])))
    return loaded, name


def numeric_access(loaded, name):
    access = evidence_access.Access(loaded, plan=episode.plan(loaded))
    field = next(field for field in access.inventory() if field['kind'] == 'numeric' and field['name'] == name)
    return access, field


def finding(receipt, *, timed):
    refs = {'evidence_id': receipt['id'], 'columns': [0]}
    interval = {}
    if timed:
        interval = {'start_s': receipt['times_s'][0], 'end_s': receipt['times_s'][-1]}
        refs['time_s'] = [receipt['times_s'][0], receipt['times_s'][-1]]
    else:
        refs['rows'] = [receipt['rows'][0], receipt['rows'][-1]]
    return {**interval, 'headline': 'Recorded velocity changes',
            'observation': f"The inspected value changes from {receipt['values'][0][0]} to {receipt['values'][-1][0]}.",
            'claim': 'The recorded channel changes across the inspected samples.',
            'confidence': 'medium', 'adds_beyond_video': True, 'evidence': [refs]}


def test_materialized_arrival_numeric_preserves_exact_log_times_and_binding_qualification(arrival_episode):
    loaded, name = arrival_episode
    access, field = numeric_access(loaded, name)
    receipt = access.inspect({'field_id': field['id'], 'mode': 'samples', 'columns': [0]})
    with np.load(loaded['dir'] / loaded['context']['recorded_camera_clocks']) as clocks:
        log = clocks[episode.anchor(loaded) + '_log_ns']
    expected_times = (log - log[0]) / 1e9
    np.testing.assert_array_equal(receipt['times_s'], expected_times[receipt['rows']])
    meta = next(row for row in loaded['context']['signals'] if row['name'] == name)
    with np.load(loaded['dir'] / 'signals.npz') as arrays:
        np.testing.assert_array_equal(receipt['values'], arrays[meta['key']][receipt['rows'], :1])
    clock = loaded['context']['source']['presentation_clock']
    assert receipt['presentation_clock'] == clock
    assert RECORDER_ARRIVAL_LIMITATION in receipt['limitations']
    bound = access.bind([finding(receipt, timed=True)])
    assert len(bound['findings']) == 1 and not bound['unbound_findings'] and not bound['untimed_findings']
    sensor = next(row for row in bound['sensors'] if row['id'] == receipt['id'])
    assert sensor['presentation_clock'] == clock
    assert RECORDER_ARRIVAL_LIMITATION in sensor['limitations']
    assert RECORDER_ARRIVAL_LIMITATION in bound['limitations']
    series = next(row for row in bound['series'] if row['sensor_id'] == receipt['id'])
    assert series['times'] == receipt['times_s']
    assert series['values'] == [row[0] for row in receipt['values']]


@pytest.mark.parametrize('assumption', ['assumed start', 'row per frame', 'coarse clock',
    'assumed camera clock', 'clock_problem', 'camera_aligned_by', 'anchor_clock'])
def test_arrival_numeric_does_not_promote_assumed_placement_to_timed_evidence(arrival_episode, assumption):
    original, name = arrival_episode
    loaded = copy.deepcopy(original)
    meta = loaded['signal_meta'][name]
    if assumption == 'anchor_clock':
        loaded['context']['camera_clock'] = {episode.anchor(loaded): {'what': 'assumed camera presentation timing'}}
    elif assumption in ('clock_problem', 'camera_aligned_by'):
        meta[assumption] = 'unverified placement'
    else:
        meta['aligned_by'] = assumption
    access, field = numeric_access(loaded, name)
    receipt = access.inspect({'field_id': field['id'], 'mode': 'samples', 'columns': [0]})
    assert 'times_s' not in receipt
    assert receipt['rows'] and receipt['values'][0] != receipt['values'][-1]
    bound = access.bind([finding(receipt, timed=False)])
    assert len(bound['untimed_findings']) == 1 and not bound['findings'] and not bound['sensors']
    assert bound['untimed_findings'][0]['evidence'][0]['rows'] == [receipt['rows'][0], receipt['rows'][-1]]
