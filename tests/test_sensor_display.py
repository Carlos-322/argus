"""Card rows come only from a finding's citations, are named for what they measure and point the declared way."""
import copy

from label import sensor_display as sd

TIMES = [i * 0.1 for i in range(20)]
REF = 3000.0


def _doc(cells, *, direction='down', zero_censored=True, regions=None):
    """An array sensor of len(cells) components with full-rate values cells[c] over TIMES."""
    columns = list(range(len(cells)))
    regions = regions or [{'name': 'thumb', 'indices': [0, 1, 2, 3]}, {'name': 'index finger', 'indices': [4, 5]}]
    sensor = {'id': 'signal:p', 'name': 'p', 'descriptor_digest': 'd1', 'sample_period_s': 0.1,
              'array_evidence': {'reference': {'values': [REF] * len(cells)},
                                 'response_convention': {'direction': direction},
                                 'native_constituents': {'source_columns': columns, 'times_s': TIMES,
                                                         'values': [[cells[c][i] for c in columns] for i in range(len(TIMES))]}}}
    derivation = {'id': 'signal:p:derivation:regional_response', 'kind': 'regional_response', 'sensor_id': 'signal:p',
                  'response_direction': direction,
                  'semantics': {'regions': regions, 'zero_exclusion_sensitivity': zero_censored}}
    series = []
    for n, region in enumerate(regions):
        ids = region['indices']
        sign = -1 if direction == 'down' else 1
        positive = [sum(max(0.0, sign * (cells[c][i] - REF)) for c in ids) / len(ids) for i in range(len(TIMES))]
        signed = [sum(cells[c][i] - REF for c in ids) / len(ids) for i in range(len(TIMES))]
        series.append({'id': f'signal:p:derivation:regional_response:{n}', 'sensor_id': 'signal:p',
                       'derivation_id': derivation['id'], 'source_columns': ids, 'label': f'{region["name"]} relative response',
                       'operation': 'mean positive response in declared direction from initial reference',
                       'times': TIMES, 'values': positive})
        series.append({'id': f'signal:p:derivation:regional_response:{n + 10}', 'sensor_id': 'signal:p',
                       'derivation_id': derivation['id'], 'source_columns': ids,
                       'label': f'{region["name"]} signed raw minus initial reference',
                       'operation': 'mean signed raw minus initial reference', 'times': TIMES, 'values': signed})
    for c in columns:
        series.append({'id': f'signal:p:array:{c}', 'sensor_id': 'signal:p', 'columns': [c], 'array_constituent': True,
                       'label': f'p [{c}]', 'times': TIMES, 'values': cells[c]})
    return {'sensors': [sensor], 'derivations': [derivation], 'series': series, 'findings': []}


def _cite(*ids):
    return [{'sensor_id': 'signal:p', 'series_id': i, 'time_s': [0.5]} for i in ids]


def _finding(evidence, start=0.5, end=1.5):
    return {'start_s': start, 'end_s': end, 'headline': 'h', 'evidence': evidence}


def _flat(value=REF):
    return [value] * len(TIMES)


def _ramp(lo, hi):
    return [lo + (hi - lo) * i / (len(TIMES) - 1) for i in range(len(TIMES))]


def test_a_cited_region_shows_its_oriented_response_under_its_own_name():
    # Lower raw codes mean a stronger response, so the thumb's falling readings plot as a rising row.
    cells = [_ramp(REF, 2000), _ramp(REF, 2000), _flat(), _flat(), _flat(), _flat()]
    doc = _doc(cells)
    rows = sd.rows_for(doc, _finding(_cite('signal:p:derivation:regional_response:10')))
    assert [r['label'] for r in rows] == ['Thumb']
    assert rows[0]['values'][-1] > rows[0]['values'][0] >= 0
    assert rows[0]['source_ids'] == ['signal:p:derivation:regional_response:0']


def test_opposite_cited_patches_in_one_region_stay_two_areas():
    # Cells 0 and 1 press harder than their reference while 2 and 3 press less: the regional mean would hide both.
    cells = [_ramp(REF, 2000), _ramp(REF, 2100), _ramp(REF, 3900), _ramp(REF, 3800), _flat(), _flat()]
    doc = _doc(cells, zero_censored=False)
    rows = sd.rows_for(doc, _finding(_cite(*(f'signal:p:array:{c}' for c in (0, 2, 1, 3)))))
    assert [r['label'] for r in rows] == ['Thumb area 1', 'Thumb area 2']
    assert sorted(rows[0]['source_ids']) == ['signal:p:array:0', 'signal:p:array:1']
    assert sorted(rows[1]['source_ids']) == ['signal:p:array:2', 'signal:p:array:3']
    assert rows[0]['values'][-1] > 0 > rows[1]['values'][-1]


def test_a_response_that_moves_between_patches_stays_two_areas():
    early = [2000.0 if 5 <= i <= 9 else REF for i in range(len(TIMES))]
    late = [2000.0 if 10 <= i <= 14 else REF for i in range(len(TIMES))]
    doc = _doc([_flat(), _flat(), _flat(), _flat(), early, late])
    rows = sd.rows_for(doc, _finding(_cite('signal:p:array:4', 'signal:p:array:5'), start=0.5, end=1.4))
    assert [r['label'] for r in rows] == ['Index finger area 1', 'Index finger area 2']


def test_components_that_move_together_are_one_area():
    cells = [_ramp(REF, 2000), _ramp(REF, 2050), _flat(), _flat(), _flat(), _flat()]
    rows = sd.rows_for(_doc(cells), _finding(_cite('signal:p:array:0', 'signal:p:array:1')))
    assert [r['label'] for r in rows] == ['Thumb area']


def test_a_censored_zero_reading_is_missing_not_a_maximal_response():
    cells = [_ramp(REF, 2000), _flat(), _flat(), _flat(), _flat(), _flat()]
    cells[0][7] = 0.0
    rows = sd.rows_for(_doc(cells, zero_censored=True), _finding(_cite('signal:p:array:0')))
    assert rows[0]['values'][7] is None
    rows = sd.rows_for(_doc(cells, zero_censored=False), _finding(_cite('signal:p:array:0')))
    assert rows[0]['values'][7] == REF


def test_only_cited_components_are_shown():
    cells = [_ramp(REF, 2000), _ramp(REF, 3900), _flat(), _flat(), _flat(), _flat()]
    rows = sd.rows_for(_doc(cells), _finding(_cite('signal:p:array:0')))
    assert [r['source_ids'] for r in rows] == [['signal:p:array:0']]


def test_unknown_direction_plots_the_signed_change():
    cells = [_ramp(REF, 2000), _flat(), _flat(), _flat(), _flat(), _flat()]
    rows = sd.rows_for(_doc(cells, direction=None), _finding(_cite('signal:p:array:0')))
    assert rows[0]['values'][-1] < 0


def test_at_most_two_rows_and_other_series_are_left_to_the_board():
    cells = [_ramp(REF, 2000), _flat(), _flat(), _flat(), _ramp(REF, 2500), _flat()]
    doc = _doc(cells)
    doc['series'].append({'id': 'depth:median', 'sensor_id': 'depth', 'label': 'Median depth in inspected region',
                          'times': TIMES, 'values': _ramp(10, 20)})
    depth_only = _finding([{'sensor_id': 'depth', 'series_id': 'depth:median', 'time_s': [0.5]}])
    assert sd.rows_for(doc, depth_only) == [{'series_id': 'depth:median'}]
    rows = sd.rows_for(doc, _finding(_cite('signal:p:derivation:regional_response:10')
                                     + [{'sensor_id': 'depth', 'series_id': 'depth:median', 'time_s': [0.5]}]))
    assert [r.get('label') or r['series_id'] for r in rows] == ['Thumb', 'depth:median']
    rows = sd.rows_for(doc, _finding(_cite('signal:p:derivation:regional_response:10', 'signal:p:derivation:regional_response:11')
                                     + [{'sensor_id': 'depth', 'series_id': 'depth:median', 'time_s': [0.5]}]))
    assert [r['label'] for r in rows] == ['Thumb', 'Index finger']
    doc['findings'] = [depth_only]
    sd.attach(doc)
    assert 'display' not in doc['findings'][0]


def test_an_inspected_read_joins_its_array_only_by_descriptor_digest():
    cells = [_ramp(REF, 2000), _flat(), _flat(), _flat(), _flat(), _flat()]
    doc = _doc(cells)
    doc['sensors'].append({'id': 'inspection:x', 'name': 'p', 'access_descriptor_digest': 'd1'})
    doc['series'].append({'id': 'inspection:x:0', 'sensor_id': 'inspection:x', 'columns': [0], 'label': 'p [0]',
                          'times': TIMES[:5], 'values': cells[0][:5]})
    rows = sd.rows_for(doc, _finding([{'sensor_id': 'inspection:x', 'series_id': 'inspection:x:0', 'time_s': [0.1]}]))
    assert rows[0]['label'] == 'Thumb area' and rows[0]['source_ids'] == ['signal:p:array:0']
    other = copy.deepcopy(doc)
    other['sensors'][-1]['access_descriptor_digest'] = 'another'
    rows = sd.rows_for(other, _finding([{'sensor_id': 'inspection:x', 'series_id': 'inspection:x:0', 'time_s': [0.1]}]))
    assert rows == [{'series_id': 'inspection:x:0'}]


def test_attach_adds_display_and_leaves_citations_untouched():
    cells = [_ramp(REF, 2000), _flat(), _flat(), _flat(), _flat(), _flat()]
    doc = _doc(cells)
    doc['findings'] = [_finding(_cite('signal:p:array:0')), _finding([])]
    before = copy.deepcopy(doc['findings'][0]['evidence'])
    sd.attach(doc)
    assert doc['findings'][0]['display']['version'] == sd.VERSION
    assert doc['findings'][0]['evidence'] == before
    assert 'display' not in doc['findings'][1]
