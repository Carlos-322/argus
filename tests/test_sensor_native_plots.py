"""Viewer context retains native readings without widening the model's citation set."""
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from label import sensor_evidence as se
from tests.test_semantic_evidence import region_episode


def test_native_plot_survives_prompt_sampling_without_becoming_citable():
    ep = region_episode(126)
    doc = se.build(ep, {'n': 126})
    row = next(r for r in doc['series'] if r['label'] == 'branch A relative response')
    assert len(row['times']) < 126
    assert row['plot']['times'] == pytest.approx(np.arange(126) / 10)
    assert row['plot']['values'][3] == 60
    assert len(row['plot']['values']) == 126
    payload = json.loads(se.prompt(doc).split('\n')[-1])
    assert all('plot' not in r for r in payload['series'])
    missing = next(t for t in row['plot']['times'] if t not in row['times'])
    finding = {'start_s': missing, 'end_s': missing, 'headline': 'An unsupported quoted instant',
               'adds_beyond_video': True, 'claim_type': 'response_change',
               'evidence': [{'sensor_id': row['sensor_id'], 'series_id': row['id'], 'time_s': [missing]}]}
    assert not se.bind(doc, [finding])['findings']
    merged = se.merge([({'piece': {'index': 1, 't0_s': 20}}, {'sensor_evidence': doc})], ep['context'])
    shifted = next(r for r in merged['series'] if r['id'] == 'part1:' + row['id'])
    assert shifted['plot']['times'] == pytest.approx(np.arange(126) / 10 + 20)
    assert shifted['plot']['values'] == row['plot']['values']
    long = se.build(region_episode(5000), {'n': 5000})
    assert all('plot' not in r for r in long['series'])


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_sensor_plot_uses_native_context_in_evidence_order_and_breaks_missing_gaps():
    script = r"""
const fs=require('fs'),assert=require('assert');
const source=fs.readFileSync(process.argv[1],'utf8');
const begin=source.indexOf('// ================= sensor evidence:');
const end=source.indexOf('// ================= touch:',begin);
const api=new Function('esc','fmtT',source.slice(begin,end)+'return {genericSensorSeries,gripOverlayPressureHtml};')(String,t=>t+'s');
const row=(id,values)=>({id,sensor_id:'pad',label:id,times:[0,.2],values:[0,4],native_samples:5,
 sample_period_s:.05,plot:{kind:'native_samples',times:[0,.05,.1,.15,.2],values}});
const G={generic:true,sensors:[{id:'pad',name:'Pad'}],series:[row('A',[0,1,null,3,4]),row('B',[0,3,2,1,4])]};
const f={start:.05,end:.15,evidence:[{sensor_id:'pad',series_id:'B'},{sensor_id:'pad',series_id:'A'}]};
const original=JSON.stringify(G),traces=api.genericSensorSeries(G,f,2);
assert.deepEqual(traces.map(r=>r.id),['B','A'],'model evidence priority controls the two plotted traces');
assert.deepEqual(traces[1].times,[0,.05,.1,.15,.2]);
const html=api.gripOverlayPressureHtml(G,f,2);
const paths=[...html.matchAll(/<path d="([^"]*)"/g)].map(m=>m[1]);
assert(paths[0].includes('L'),'native samples form the recorded trajectory');
assert.equal((paths[1].match(/M/g)||[]).length,2,'missing native rows split the path');
assert(!html.includes('data-sensor-recorded-point'),'native traces do not masquerade as sparse points');
assert.equal(JSON.stringify(G),original);
G.series[0].plot.values=[0,null,2,null,4];
assert(api.gripOverlayPressureHtml(G,f,2).includes('data-sensor-recorded-point'),'isolated native readings stay visible across missing rows');
delete G.series[0].plot;delete G.series[1].plot;
const sparse=api.gripOverlayPressureHtml(G,f,2);
assert(sparse.includes('data-sensor-recorded-point'),'legacy sparse citations remain isolated points');
assert(![...sparse.matchAll(/<path d="([^"]*)"/g)].some(m=>m[1].includes('L')));
"""
    root = Path(__file__).resolve().parents[1]
    run = subprocess.run([shutil.which('node'), '-e', script, str(root / 'board/index.html')], capture_output=True, text=True)
    assert run.returncode == 0, run.stdout + run.stderr


def test_canonical_candidate_layout_exposes_physical_taxel_positions():
    from prepare.sensor_semantics import validate
    root = Path(__file__).resolve().parents[1]
    schema = json.loads((root / 'configs/sensor_schemas/opentouch_hdf5_v1.json').read_text())
    pad = schema['signals']['right_pressure']
    assert validate(pad, [16, 16]) == pad
    assert pad['positions'][221] == pytest.approx([502.5, 261])
    assert len({tuple(pad['positions'][i]) for i in pad['valid_indices']}) == 169


def test_spatial_snapshot_budget_covers_both_pads():
    from tests.test_tactile_contract import episode
    a = np.broadcast_to(np.arange(100)[:, None, None], (100, 16, 16)).copy()
    ep = episode({'left_pressure': a, 'right_pressure': 100 - a}, [
        {'name': name, 'shape': [16, 16]} for name in ['left_pressure', 'right_pressure']])
    doc = se.build(ep, {'n': 100, 'ks': [0, 99]})
    arrays = [s['array_evidence'] for s in doc['sensors'] if s.get('array_evidence')]
    assert len(arrays) == 2
    assert all(len(array['frames']) >= 12 for array in arrays)
    assert sum((len(array['frames']) + 1) * 256 for array in arrays) <= 8192
    for sensor, expected in zip(doc['sensors'], [a, 100 - a]):
        for frame in sensor['array_evidence']['frames']:
            row = int(round(frame['time_s'] * 10))
            assert frame['values'] == expected[row].reshape(-1).tolist()


def test_constituent_citations_resolve_only_supplied_cells_and_samples():
    from tests.test_tactile_contract import episode
    a = np.full((126, 16, 16), 100., dtype=float)
    a[:, 0, 1] += np.arange(126)
    a[:, 1, 1] -= np.arange(126)
    ep = episode({'pressure': a}, [{'name': 'pressure', 'shape': [16, 16], 'units': 'ADC'}])
    doc = se.build(ep, {'n': 126})
    table = json.loads(se.prompt(doc).split('\n')[-1])['sensors'][0]['array_evidence']['constituents']
    times = table['times_s'][:2]
    finding = {'start_s': times[0], 'end_s': times[-1], 'headline': 'Neighboring cells diverge',
               'adds_beyond_video': True, 'claim_type': 'response_change',
               'evidence': [{'sensor_id': 'signal:pressure', 'series_id': 'signal:pressure:array:' + str(col),
                             'time_s': times} for col in [1, 17]]}
    result = se.bind(doc, [finding], ep=ep)
    assert result['findings'] == [finding]
    rows = [r for r in result['series'] if ':array:' in r['id']]
    assert len(rows) == 2
    assert {r['columns'][0] for r in rows} == {1, 17}
    assert all(r['unit'] == 'ADC' for r in rows)
    for r in rows:
        col = r['columns'][0]
        assert r['times'] == table['times_s']
        assert r['values'] == [row[table['source_columns'].index(col)] for row in table['values']]
        assert r['plot']['values'] == a.reshape(126, -1)[:, col].tolist()
        assert len(r['plot']['times']) == 126
        assert r['sampled'] is (len(table['times_s']) < 126)
    rebound = se.bind(result, [finding])
    assert rebound['findings'] == result['findings']
    assert [r for r in rebound['series'] if r.get('array_constituent')] == rows
    merged = se.merge([({'piece': {'index': 1, 't0_s': 20}}, {'sensor_evidence': result})], ep['context'])
    merged_table = se._array_projection(merged)['sensors'][0]['array_evidence']['constituents']
    assert merged_table['times_s'] == pytest.approx(np.array(table['times_s']) + 20)
    assert merged_table['values'] == table['values']
    import copy
    for changed in [
            {**finding['evidence'][0], 'series_id': 'signal:pressure:array:256'},
            {**finding['evidence'][0], 'series_id': 'signal:another:array:1'},
            {**finding['evidence'][0], 'time_s': [.012345]}]:
        invalid = copy.deepcopy(finding)
        invalid['evidence'] = [changed]
        rejected = se.bind(doc, [invalid], ep=ep)
        assert not rejected['findings']
        assert not any(':array:' in r['id'] for r in rejected['series'])


def test_excluded_cells_cannot_make_an_empty_displayed_array_sample_citable():
    from label import tactile_arrays
    from tests.test_tactile_contract import episode
    from tests.test_tactile_spatial_evidence import spatial_case
    raw, times, semantics = spatial_case()
    raw, times = raw[:12], times[:12]
    raw[10, semantics['valid_indices']] = np.nan
    ep = episode({'pressure': raw.reshape(12, 16, 16)}, [{'name': 'pressure', 'shape': [16, 16]}])
    doc = se.build(ep, {'n': 12})
    sensor = doc['sensors'][0]
    sensor['array_evidence'] = tactile_arrays.comparison(raw, [16, 16], times, [], 8192, semantics)
    missing = sensor['array_evidence']['frames'][10]
    assert missing['valid_cells'] == 0
    assert missing['raw_quantiles'] == [None] * 3
    sensor['times'] = times.tolist()
    finding = {'start_s': float(times[10]), 'end_s': float(times[10]), 'headline': 'No displayed reading supports this',
               'adds_beyond_video': True, 'claim_type': 'response_change',
               'evidence': [{'sensor_id': sensor['id'], 'time_s': [float(times[10])]}]}
    before = {**finding, 'start_s': float(times[9]), 'end_s': float(times[9]),
              'evidence': [{'sensor_id': sensor['id'], 'time_s': [float(times[9])]}]}
    assert se.bind(doc, [before])['findings'] == [before]
    assert not se.bind(doc, [finding])['findings']
