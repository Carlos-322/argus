"""Inspected numeric curves retain source context without widening citations."""
import copy

import numpy as np
import pytest

from label import evidence_access as ea, sensor_evidence as se
from tests.test_tactile_contract import episode


def numeric_access(tmp_path, n=126, raw_rows=None, frame_count=None):
    values = np.column_stack([np.arange(n), np.sin(np.arange(n) / 7)])
    values[40, 1] = np.nan
    raw = values[:raw_rows].copy() if raw_rows is not None else values.copy()
    np.savez(tmp_path / 'signals.npz', pad=raw)
    ep = episode({'pressure': values.copy()}, [{'name': 'pressure', 'key': 'pad', 'shape': [2], 'unit': 'ADC'}])
    ep['dir'] = tmp_path
    ep['times']['exo'][70:] += .2
    access = ea.Access(ep, plan={'n': frame_count or n})
    field = next(row for row in access.inventory() if row['name'] == 'pressure')
    return access, field['id'], raw


def inspect_finding(access, field_id, mode='samples', end_s=None):
    request = {'field_id': field_id, 'mode': mode, 'columns': [1]}
    if end_s is not None:
        request['end_s'] = end_s
    receipt = access.inspect(request)
    times = [time for time, row in zip(receipt['times_s'], receipt['values']) if row[0] is not None]
    finding = {'headline': 'The recorded response changes', 'start_s': times[0], 'end_s': times[-1],
               'adds_beyond_video': True,
               'evidence': [{'evidence_id': receipt['id'], 'time_s': [times[0], times[-1]], 'columns': [1]}]}
    return receipt, finding


@pytest.mark.parametrize('mode', ['samples', 'extrema'])
def test_accepted_inspections_get_native_curves_without_new_citations(tmp_path, mode):
    access, field_id, raw = numeric_access(tmp_path)
    receipt, finding = inspect_finding(access, field_id, mode)
    before = copy.deepcopy(receipt)
    # The raw prepared dtype used by inspection is authoritative over a changed
    # in-memory view of the same named field.
    access.fields[field_id]['value'] += 20
    bound = access.bind([finding])
    row = next(row for row in bound['series'] if row['id'] == receipt['id'] + ':1')
    assert bound['findings']
    assert row['times'] == receipt['times_s']
    assert row['values'] == [values[0] for values in receipt['values']]
    assert row['plot']['times'] == access.times().tolist()
    assert row['plot']['values'] == ea.clean(raw[:, 1])
    assert row['native_samples'] == len(raw)
    assert row['sample_period_s'] == pytest.approx(.1)
    assert access.receipts[0] == before
    omitted = next(time for time in row['plot']['times'] if time not in row['times'])
    invalid = copy.deepcopy(finding)
    invalid['evidence'][0]['time_s'] = [omitted]
    assert not access.bind([invalid])['findings']
    assert all('plot' not in row for row in access.bind([invalid])['series'])
    assert all('plot' not in row for row in access.bind([])['series'])
    rebound = access.bind([finding])
    assert rebound['series'] == bound['series']
    reloaded = ea.Access(access.ep, plan={'n': 126})
    reloaded.receipts = [copy.deepcopy(receipt)]
    assert reloaded.bind([finding])['series'] == bound['series']
    merged = se.merge([({'piece': {'index': 1, 't0_s': 20}}, {'sensor_evidence': bound})], access.ep['context'])
    assert merged['series'][0]['plot']['times'] == pytest.approx(access.times() + 20)


def test_native_inspection_plot_preserves_absent_raw_tail(tmp_path):
    access, field_id, raw = numeric_access(tmp_path, raw_rows=90)
    receipt, finding = inspect_finding(access, field_id)
    row = access.bind([finding])['series'][0]
    assert row['times'] == receipt['times_s']
    assert row['plot']['values'] == ea.clean(raw[:, 1]) + [None] * 36
    assert row['native_samples'] == 126


@pytest.mark.parametrize('change', ['values', 'clock', 'alignment'])
def test_changed_source_or_clock_does_not_receive_native_plot(tmp_path, change):
    access, field_id, raw = numeric_access(tmp_path)
    _, finding = inspect_finding(access, field_id)
    if change == 'values':
        np.savez(tmp_path / 'signals.npz', pad=raw + 1)
    elif change == 'clock':
        access.ep['times']['exo'] += .01
    else:
        access.state_usable = False
    assert all('plot' not in row for row in access.bind([finding])['series'])


def test_native_inspection_plot_uses_only_owned_request_rows(tmp_path):
    access, field_id, raw = numeric_access(tmp_path, frame_count=80)
    _, finding = inspect_finding(access, field_id, end_s=float(access.times()[79]))
    row = access.bind([finding])['series'][0]
    assert row['native_samples'] == 80
    assert row['plot']['times'] == access.times()[:80].tolist()
    assert row['plot']['values'] == ea.clean(raw[:80, 1])


def test_long_inspection_sources_do_not_read_native_curves(tmp_path, monkeypatch):
    access, field_id, _ = numeric_access(tmp_path, n=4097)
    _, finding = inspect_finding(access, field_id)
    monkeypatch.setattr(ea, 'array_rows', lambda *a, **k: pytest.fail('unbounded native source read'))
    assert all('plot' not in row for row in access.bind([finding])['series'])


def test_native_inspection_plot_checks_source_byte_budget_before_read(tmp_path, monkeypatch):
    access, field_id, _ = numeric_access(tmp_path)
    _, finding = inspect_finding(access, field_id)
    monkeypatch.setattr(ea, 'MAX_SCAN_BYTES', 64)
    monkeypatch.setattr(ea, 'array_rows', lambda *a, **k: pytest.fail('source byte budget exceeded'))
    assert all('plot' not in row for row in access.bind([finding])['series'])


def test_native_inspection_plot_reads_fortran_source_columns(tmp_path):
    access, field_id, raw = numeric_access(tmp_path)
    np.savez(tmp_path / 'signals.npz', pad=np.asfortranarray(raw))
    _, finding = inspect_finding(access, field_id)
    assert access.bind([finding])['series'][0]['plot']['values'] == ea.clean(raw[:, 1])


@pytest.mark.parametrize('excluded', ['depth', 'native', 'array', 'spectrum'])
def test_noncomponent_inspections_do_not_receive_numeric_source_curves(tmp_path, monkeypatch, excluded):
    access, field_id, _ = numeric_access(tmp_path)
    receipt, finding = inspect_finding(access, field_id)
    if excluded == 'spectrum':
        access.receipts[0]['mode'] = excluded
    else:
        access.fields[field_id]['public']['kind'] = excluded
        if excluded == 'depth':
            access.fields[field_id]['view'] = 'exo'
    monkeypatch.setattr(ea, 'array_rows', lambda *a, **k: pytest.fail('ineligible source was read'))
    assert all('plot' not in row for row in access.bind([finding])['series'])


def test_in_memory_components_and_reloaded_receipts_receive_the_same_curve(tmp_path):
    access, field_id, raw = numeric_access(tmp_path)
    (tmp_path / 'signals.npz').unlink()
    field = access.fields[field_id]
    for key in ('raw_path', 'raw_key', 'raw_rows'):
        field.pop(key)
    receipt, finding = inspect_finding(access, field_id)
    bound = access.bind([finding])
    assert bound['series'][0]['plot']['values'] == ea.clean(raw[:, 1])
    reloaded = ea.Access(access.ep, plan={'n': 126})
    reloaded.receipts = [copy.deepcopy(receipt)]
    assert reloaded.bind([finding])['series'] == bound['series']
