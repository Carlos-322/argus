"""Model input must preserve local constituent contrasts without regional averaging."""
import copy
import json

import numpy as np

from label import sensor_evidence as se
from tests.test_tactile_contract import episode
from tests.test_tactile_spatial_evidence import spatial_case


def test_shared_clock_numeric_matrix_preserves_opposing_neighboring_cells():
    raw, times, semantics = spatial_case()
    raw[60, 10] = 599.25
    raw[66, 90] = np.nan
    ep = episode({'right_pressure': raw}, [{'name': 'right_pressure', 'shape': [16, 16],
                                          'sensor_semantics': semantics}])
    ep['times']['exo'] = times
    before = raw.copy()
    doc = se.build(ep, {'n': len(raw), 'ks': [0, 103, 185]})
    sent = json.loads(se.prompt(doc).splitlines()[-1])
    array = sent['sensors'][0]['array_evidence']
    table = array['constituents']
    assert sent['sensors'][0]['array_supplied_columns'] == len(semantics['valid_indices'])
    assert table['source_columns'] == semantics['valid_indices']
    assert table['series_ids'] == ['signal:right_pressure:array:' + str(i) for i in semantics['valid_indices']]
    assert table['times_s'] == times.tolist()
    assert all(frame['time_s'] in table['times_s'] for frame in array['frames'])
    assert table['reference_values'] == [1000] * 4
    assert all(type(value) is int for value in table['reference_values'])
    for time, values in zip(table['times_s'], table['values']):
        row = int(np.argmin(abs(times - time)))
        expected = [None if not np.isfinite(raw[row, cell]) else raw[row, cell]
                    for cell in table['source_columns']]
        assert values == expected
    event = [(time, values) for time, values in zip(table['times_s'], table['values']) if 1.8 <= time <= 2.3]
    assert len(event) >= 2
    assert any(values[:2] == [600, 600] and values[2:] == [1400, 1400] for _, values in event)
    assert array['physical_layout']['positions'][0]['index'] == table['source_columns'][0]
    assert np.array_equal(raw, before, equal_nan=True)
    assert 'constituents' not in doc['sensors'][0]['array_evidence']
    assert 'native_constituents' in doc['sensors'][0]['array_evidence']
    assert se._array_projection(se._array_projection(doc)) == se._array_projection(doc)
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES


def test_constituent_projection_keeps_unknown_layout_and_fractional_values():
    raw = np.full((5, 4, 4), 10.25)
    raw[3, 0, 0] = 11.125
    raw[4, 3, 3] = np.nan
    ep = episode({'pressure': raw}, [{'name': 'pressure', 'shape': [4, 4]}])
    doc = se.build(ep, {'n': 5})
    array = se._array_projection(doc)['sensors'][0]['array_evidence']
    table = array['constituents']
    assert table['source_columns'] == list(range(16))
    assert table['values'][3][0] == 11.125
    assert table['values'][4][15] is None
    assert table['reference_values'] == [10.25] * 16
    assert 'physical_layout' not in array
    assert 'Source array row and column' in array['coordinates']


def test_prompt_pressure_reduces_array_rows_before_removing_local_evidence(monkeypatch):
    raw, times, semantics = spatial_case()
    signals = {'right_pressure': raw, 'left_pressure': raw + 1}
    ep = episode(signals, [{'name': name, 'shape': [16, 16], 'sensor_semantics': semantics} for name in signals])
    ep['times']['exo'] = times
    doc = se.build(ep, {'n': len(raw), 'ks': [0, 103, 185]})
    initial = copy.deepcopy(doc)
    # Force the reducer to share its byte budget between several distinct
    # modalities while keeping complete columns for every retained snapshot.
    monkeypatch.setattr(se, 'MAX_PROMPT_BYTES', 24000)
    sent = json.loads(se.prompt(doc).splitlines()[-1])
    for sensor in sent['sensors']:
        array = sensor['array_evidence']
        table = array['constituents']
        original = next(s for s in initial['sensors'] if s['id'] == sensor['id'])['array_evidence']
        assert 4 <= len(table['times_s']) < len(original['frames'])
        assert all(frame['time_s'] in table['times_s'] for frame in original['frames'] if frame['selection_priority'] < 2)
        assert max(np.diff(table['times_s'])) <= (times[-1] - times[0]) / 2
        assert table['source_columns'] == semantics['valid_indices']
        assert all(len(row) == 4 for row in table['values'])
        assert table['times_s'][0] == 0 and table['times_s'][-1] == times[-1]
    assert doc == initial
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES


def test_saved_materialized_constituents_do_not_duplicate_the_prompt_matrix():
    raw, times, semantics = spatial_case()
    ep = episode({'right_pressure': raw}, [{'name': 'right_pressure', 'shape': [16, 16],
                                          'sensor_semantics': semantics}])
    ep['times']['exo'] = times
    doc = se.build(ep, {'n': len(raw)})
    initial_prompt = se.prompt(doc)
    doc['series'].append({'id': 'signal:right_pressure:array:10', 'sensor_id': 'signal:right_pressure',
                          'array_constituent': True, 'times': times.tolist(), 'values': raw[:, 10].tolist(),
                          'columns': [10]})
    before = copy.deepcopy(doc)
    assert se.prompt(doc) == initial_prompt
    assert doc == before


def test_inspection_context_keeps_structured_measurement_coverage_instead_of_a_json_prefix():
    from label.evidence_access import Access
    raw = np.arange(100 * 256, dtype=float).reshape(100, 16, 16)
    semantics = {'version': 1, 'positions': [[i % 16, i // 16] for i in range(256)],
                 'provenance': {'authority': 'recorded'}}
    ep = episode({'pressure': raw, 'left_pressure': raw}, [
        {'name': 'pressure', 'shape': [16, 16], 'sensor_semantics': semantics},
        {'name': 'left_pressure', 'shape': [16, 16]}],
        entries={'left_pressure': {'role': 'annotation', 'provenance': 'human'}})
    doc = se.build(ep, {'n': 100})
    access = object.__new__(Access)
    access.ep, access.supplied = ep, doc
    context = json.loads(access.selection_content([])[0]['text'].split('\n', 1)[1])
    assert context['sensor_annotations_generated'] is False
    summary = context['already_supplied_sensor_evidence']
    assert 'preview' not in summary and 'truncated' not in summary
    assert summary['coverage'] == doc['coverage']
    assert next(row for row in summary['coverage'] if row['id'] == 'signal:left_pressure')['status'] == 'excluded by interpretation'
    assert {r['id'] for r in summary['series']} == {r['id'] for r in doc['series']}
    array = summary['sensors'][0]['array_evidence']
    assert array['native_samples'] == 100
    assert array['constituent_coverage']['source_columns'] == list(range(256))
    assert array['layout']['positions'][-1] == [255, 15, 15]
    assert array['snapshot_times_s'][-1] == doc['sensors'][0]['array_evidence']['frames'][-1]['time_s']


def test_native_constituents_preserve_an_event_between_map_snapshots():
    raw, times, semantics = spatial_case()
    ep = episode({'right_pressure': raw}, [{'name': 'right_pressure', 'shape': [16, 16],
                                          'sensor_semantics': semantics}])
    ep['times']['exo'] = times
    doc = se.build(ep, {'n': len(raw)})
    array = doc['sensors'][0]['array_evidence']
    map_times = {frame['time_s'] for frame in array['frames']}
    event = next(i for i, time in enumerate(times) if 1.8 < time < 2.3 and time not in map_times)
    raw[event, [10, 26]] = 500
    raw[event, [90, 106]] = 1500
    doc = se.build(ep, {'n': len(raw)})
    array = json.loads(se.prompt(doc).splitlines()[-1])['sensors'][0]['array_evidence']
    table = array['constituents']
    assert times[event] not in {frame['time_s'] for frame in array['frames']}
    assert table['times_s'] == times.tolist()
    assert table['values'][event] == [500, 500, 1500, 1500]
    assert table['omitted_rows'] == 0
    assert table['sampled'] is False
    assert len(array['frames']) <= 32


def test_native_constituent_value_budget_is_shared_without_losing_columns(monkeypatch):
    from label import tactile_arrays as ta
    monkeypatch.setattr(ta, 'MAX_NATIVE_VALUES', 128)
    raw, times, semantics = spatial_case()
    ep = episode({'right_pressure': raw, 'left_pressure': raw + 1},
                 [{'name': name, 'shape': [16, 16], 'sensor_semantics': semantics}
                  for name in ['right_pressure', 'left_pressure']])
    ep['times']['exo'] = times
    doc = se.build(ep, {'n': len(raw)})
    arrays = [sensor['array_evidence'] for sensor in json.loads(se.prompt(doc).splitlines()[-1])['sensors']]
    tables = [array['constituents'] for array in arrays]
    assert sum(len(table['values']) * len(table['source_columns']) for table in tables) <= 128
    for table in tables:
        assert table['source_columns'] == semantics['valid_indices']
        assert table['times_s'][0] == times[0] and table['times_s'][-1] == times[-1]
        assert table['sampled'] and table['omitted_rows'] == len(times) - len(table['times_s'])
        assert max(np.diff(table['times_s'])) <= 2 * (times[-1] - times[0]) / (len(table['times_s']) - 1)


def test_byte_thinned_native_rows_cannot_be_cited_as_supplied(monkeypatch):
    raw, times, semantics = spatial_case()
    ep = episode({'right_pressure': raw}, [{'name': 'right_pressure', 'shape': [16, 16],
                                          'sensor_semantics': semantics}])
    ep['times']['exo'] = times
    doc = se.build(ep, {'n': len(raw)})
    full = json.loads(se.prompt(doc).splitlines()[-1])['sensors'][0]['array_evidence']['constituents']
    assert full['times_s'] == times.tolist()
    monkeypatch.setattr(se, 'MAX_PROMPT_BYTES', int(len(se.prompt(doc).encode()) * .8))
    projected = json.loads(se.prompt(doc).splitlines()[-1])['sensors'][0]['array_evidence']['constituents']
    omitted = next(time for time in full['times_s'] if time not in projected['times_s'])
    assert projected['omitted_rows'] == len(times) - len(projected['times_s'])
    assert projected['sampled']
    finding = {'start_s': omitted, 'end_s': omitted, 'headline': 'Opposing local responses',
               'claim_type': 'response_change', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'signal:right_pressure',
                             'series_id': 'signal:right_pressure:array:10', 'time_s': [omitted]}]}
    result = se.bind(doc, [finding])
    assert not result['findings']
    assert result['unbound_findings'][0]['reason'] == 'citation is not an available supplied sample'


def test_two_small_pads_keep_every_native_valid_cell_including_constant_cells():
    from label import tactile_arrays as ta
    raw, times, semantics = spatial_case()
    semantics['valid_indices'] = list(range(169))
    ep = episode({'right_pressure': raw, 'left_pressure': raw + 1},
                 [{'name': name, 'shape': [16, 16], 'sensor_semantics': semantics}
                  for name in ['right_pressure', 'left_pressure']])
    ep['times']['exo'] = times
    tables = [ta.comparison(values, [16, 16], times, [], 4096, semantics=semantics,
                            native_value_budget=ta.MAX_NATIVE_VALUES // 2)['native_constituents']
              for values in [raw, raw + 1]]
    assert sum(len(table['values']) * len(table['source_columns']) for table in tables) == 62868
    assert 62868 <= ta.MAX_NATIVE_VALUES
    for table in tables:
        assert table['source_columns'] == list(range(169))
        assert table['times_s'] == times.tolist()
        assert not table['sampled'] and table['omitted_rows'] == 0
    # Two full native pads fit the value budget, but their readable JSON must
    # still coexist within the independent model byte budget.
    doc = se.build(ep, {'n': len(raw)})
    projected = json.loads(se.prompt(doc).splitlines()[-1])
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES
    assert len(projected['sensors']) == 2
    for sensor in projected['sensors']:
        table = sensor['array_evidence']['constituents']
        assert table['source_columns'] == list(range(169))
        assert table['omitted_rows'] == len(times) - len(table['times_s'])
        assert table['times_s'][0] == times[0] and table['times_s'][-1] == times[-1]


def test_long_sources_keep_bounded_native_rows_with_explicit_omissions():
    from label import tactile_arrays as ta
    times = np.arange(5000) / 30
    raw = np.arange(5000 * 16, dtype=np.int64).reshape(5000, 16)
    semantics = {'version': 1, 'valid_indices': [0, 1, 2, 3]}
    array = ta.comparison(raw, [4, 4], times, [], 8192, semantics=semantics)
    table = array['native_constituents']
    assert len(table['times_s']) <= ta.MAX_NATIVE_ROWS
    assert len(table['values']) * len(table['source_columns']) <= ta.MAX_NATIVE_VALUES
    assert table['times_s'][0] == times[0] and table['times_s'][-1] == times[-1]
    assert table['omitted_rows'] == len(times) - len(table['times_s'])
    assert table['sampled']
    for row, values in zip(table['source_rows'], table['values']):
        assert values == raw[row, :4].tolist()


def test_native_tables_preserve_large_integer_codes_and_float_representation():
    integers = (np.arange(5 * 16, dtype=np.int64) + 2**53).reshape(5, 4, 4)
    ep = episode({'pressure': integers}, [{'name': 'pressure', 'shape': [4, 4]}])
    array = json.loads(se.prompt(se.build(ep, {'n': 5})).splitlines()[-1])['sensors'][0]['array_evidence']
    assert array['constituents']['values'][1] == integers[1].reshape(-1).tolist()
    assert all(type(value) is int for value in array['constituents']['values'][1])
    floating = np.full((5, 4, 4), 1e100)
    ep = episode({'pressure': floating}, [{'name': 'pressure', 'shape': [4, 4]}])
    array = json.loads(se.prompt(se.build(ep, {'n': 5})).splitlines()[-1])['sensors'][0]['array_evidence']
    assert type(array['constituents']['values'][0][0]) is float
    assert array['constituents']['values'][0][0] == 1e100
