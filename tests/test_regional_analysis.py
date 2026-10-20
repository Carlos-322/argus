"""The independent pass receives original regional outputs with exact citation rows."""
import copy

import numpy as np
import pytest

from label import sensor_evidence as se
from tests.test_semantic_evidence import descriptor
from tests.test_tactile_contract import episode


def source(n=5):
    raw = np.full((n, 18), 100.)
    raw[3, 0], raw[3, 1:17], raw[3, 17] = 88, 163.125, 0
    semantics = descriptor(regions=[{'name': 'declared region', 'indices': list(range(18))}])
    ep = episode({'pad': raw}, [{'name': 'pad', 'shape': [18], 'unit': 'ADC',
                  'sensor_semantics': semantics}], {'pad': {'role': 'touch'}})
    doc = se.build(ep, {'n': n})
    doc['series'] = [row for row in doc['series'] if row.get('derivation_id')]
    return ep, doc


def restored(doc, kind):
    return next(row for row in doc['series'] if row.get('regional_output') == kind)


def test_original_signed_sensitivity_and_raw_outputs_survive_prompt_and_binding():
    from label import regional_analysis as ra
    ep, doc = source()
    before, raw = copy.deepcopy(doc), ep['signals']['pad'].copy()
    assert next(row for row in doc['series'] if row['label'].endswith('zero-exclusion sensitivity'))['values'][3] == pytest.approx(12 / 17)
    result = ra.augment(doc, ep=ep)
    signed = restored(result, 'signed_never_zero')
    assert signed['values'][3] == pytest.approx(-998 / 17)
    assert signed['source_columns'] == list(range(17))
    assert restored(result, 'raw_mean')['values'][3] == pytest.approx(raw[3].mean())
    assert restored(result, 'raw_never_zero_mean')['values'][3] == pytest.approx(raw[3, :17].mean())
    assert restored(result, 'zero_cells')['values'] == [0, 0, 0, 1, 0]
    admitted = se._prompt(result)[1]
    assert next(row for row in admitted['series'] if row['id'] == signed['id'])['values'][3] == pytest.approx(-998 / 17)
    finding = {'start_s': .3, 'end_s': .3, 'headline': 'The signed regional response falls',
               'claim_type': 'response_change', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'signal:pad', 'series_id': signed['id'], 'time_s': [.3]}]}
    assert se.bind(result, [finding])['findings']
    assert result['sensors'][0]['capabilities'] == doc['sensors'][0]['capabilities']
    assert doc == before and np.array_equal(ep['signals']['pad'], raw)
    assert ra.augment(result, ep=ep) == result


def test_shared_restored_value_budget_withholds_omitted_native_rows_from_citations():
    from label import regional_analysis as ra
    ep, doc = source(20)
    result = ra.augment(doc, ep=ep, max_rows=3, max_values=12)
    rows = [r for r in result['series'] if r.get('regional_output')]
    assert len(rows) == 4 and sum(len(r['values']) for r in rows) <= 12
    signed = restored(result, 'signed_never_zero')
    assert signed['coverage']['native_samples'] == 20
    assert signed['coverage']['omitted_samples'] == 17
    omitted = next(float(t) for t in np.arange(20) / 10 if float(t) not in signed['times'])
    finding = {'start_s': omitted, 'end_s': omitted, 'headline': 'An omitted response',
               'claim_type': 'response_change', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'signal:pad', 'series_id': signed['id'], 'time_s': [omitted]}]}
    bound = se.bind(result, [finding])
    assert not bound['findings'] and 'available supplied sample' in bound['unbound_findings'][0]['reason']
    assert any('regional output' in note.lower() for note in result['limitations'])


def test_saved_audits_remain_usable_without_source_but_do_not_invent_missing_signed_sensitivity():
    from label import regional_analysis as ra
    _, doc = source()
    result = ra.augment(doc)
    assert restored(result, 'raw_mean')['values'][3] == pytest.approx((88 + 16 * 163.125) / 18)
    assert restored(result, 'raw_never_zero_mean')['values'][3] == pytest.approx((88 + 16 * 163.125) / 17)
    assert not any(r.get('regional_output') == 'signed_never_zero' for r in result['series'])
    assert any('signed' in note.lower() and 'unavailable' in note.lower() for note in result['limitations'])


def test_current_source_mismatch_and_human_withdrawal_do_not_restore_obsolete_outputs():
    from label import regional_analysis as ra
    ep, doc = source()
    changed = copy.deepcopy(ep)
    changed['signals']['pad'][3, 0] += 1
    assert not any(r.get('regional_output') for r in ra.augment(doc, ep=changed)['series'])
    ep['context']['data_dictionary']['entries']['field_0'].update(provenance='human', role='annotation')
    result = ra.augment(doc, ep=ep)
    assert not any(r.get('regional_output') for r in result['series'])
    assert any('withdraws' in note for note in result['limitations'])


def test_raw_missing_values_and_unknown_response_direction_do_not_gain_physical_capabilities():
    from label import regional_analysis as ra
    ep, _ = source()
    ep['signals']['pad'][4] = np.nan
    ep['context']['signals'][0]['sensor_semantics'].pop('response_direction')
    doc = se.build(ep, {'n': 5})
    result = ra.augment(doc, ep=ep)
    assert restored(result, 'raw_mean')['values'][-1] is None
    assert restored(result, 'signed_never_zero')['values'][-1] is None
    assert restored(result, 'zero_cells')['values'][-1] == 0
    assert restored(result, 'zero_cells')['valid_counts'][-1] == 0
    assert result['sensors'][0]['capabilities'] == doc['sensors'][0]['capabilities']


def test_existing_compact_regional_source_columns_are_restored_without_expanding_the_prompt():
    from label import regional_analysis as ra
    values = np.full((5, 300), 100.)
    values[3, 0] = 80
    ep = episode({'pad': values}, [{'name': 'pad', 'shape': [300], 'unit': 'ADC',
        'sensor_semantics': descriptor(regions=[{'name': 'wide region', 'indices': list(range(300))}])}],
        {'pad': {'role': 'touch'}})
    doc = se.build(ep, {'n': 5})
    row = restored(ra.augment(doc, ep=ep), 'raw_mean')
    assert row['source_columns'] == {'start': 0, 'count': 300}
    assert row['values'][3] == pytest.approx(values[3].mean())
