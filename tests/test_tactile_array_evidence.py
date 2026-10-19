"""The normal request must expose distributed responses outside its few selected traces."""
import copy
import json

import numpy as np

from label import sensor_evidence as se
from tests.test_tactile_contract import episode


def test_array_comparison_retains_opposing_cells_and_missing_readings():
    raw = np.full((7, 4, 4), 100.0)
    raw[3, 0, :] = 80
    raw[3, 3, :] = 120
    raw[4, 2, 2] = np.nan
    raw[5, 1, 1] = 0
    ep = episode({'right_pressure': raw}, [{'name': 'right_pressure', 'shape': [4, 4]}])
    before = copy.deepcopy(ep['context'])
    doc = se.build(ep, {'n': 7, 'ks': [0, 3, 6]})
    sensor = doc['sensors'][0]
    array = sensor['array_evidence']
    assert array['shape'] == [4, 4]
    assert array['reference']['values'] == [100.0] * 16
    frames = {frame['time_s']: frame for frame in array['frames']}
    assert frames[.3]['values'][:4] == [80.0] * 4
    assert frames[.3]['values'][12:] == [120.0] * 4
    assert frames[.4]['values'][10] is None
    assert frames[.5]['values'][5] == 0
    assert ep['context'] == before
    assert sensor['response_direction'] is None
    # The actual prompt projection and request images must agree about available snapshots.
    sent = json.loads(se.prompt(doc).split('\n')[-1])
    assert [f['time_s'] for f in sent['sensors'][0]['array_evidence']['frames']] == sorted(frames)
    assert all('values' not in f for f in sent['sensors'][0]['array_evidence']['frames'])
    parts = se.image_parts(doc)
    assert parts and any(p['type'] == 'image_url' for p in parts)
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES


def test_whole_array_citations_are_bound_and_shift_with_their_part():
    ep = episode({'pressure': np.arange(80).reshape(5, 4, 4)},
                 [{'name': 'pressure', 'shape': [4, 4]}])
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    finding = {'start_s': .1, 'end_s': .3, 'headline': 'Different cells respond at different times',
               'claim_type': 'response_change', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'signal:pressure', 'time_s': [.1, .3]}]}
    bound = se.bind(doc, [finding])
    assert bound['findings'] == [finding]
    bad = {**finding, 'evidence': [{'sensor_id': 'signal:pressure', 'time_s': [.15]}]}
    assert not se.bind(doc, [bad])['findings']
    merged = se.merge([({'piece': {'index': 1, 't0_s': 10}}, {'sensor_evidence': bound})], ep['context'])
    sensor = merged['sensors'][0]
    assert sensor['times'] == [10, 10.1, 10.2, 10.3, 10.4]
    assert [f['time_s'] for f in sensor['array_evidence']['frames']] == sensor['times']
    assert sensor['array_evidence']['reference']['times_s'] == [10, 10.1, 10.2]
    assert merged['findings'][0]['evidence'][0]['time_s'] == [10.1, 10.3]
    # A numeric signal without actual array snapshots still requires a supplied series.
    scalar = episode({'pressure': np.arange(5)[:, None]}, [{'name': 'pressure', 'shape': [1]}])
    assert not se.bind(se.build(scalar, {'n': 5}), [finding])['findings']


def test_normal_request_includes_array_images_in_image_accounting(tmp_path):
    import base64
    from tests.test_label import _packed_episode
    from label import episode as me
    ep_dir, _ = _packed_episode(tmp_path)
    context = json.loads((ep_dir / 'context.json').read_text())
    # Match the real clip's frame count, rather than an unrelated synthetic signal clock.
    baseline = me.load(ep_dir)
    n = len(baseline['state'])
    context['signals'] = [{'name': 'pressure', 'key': 'pad', 'shape': [4, 4]}]
    (ep_dir / 'context.json').write_text(json.dumps(context))
    np.savez(ep_dir / 'signals.npz', pad=np.arange(n * 16).reshape(n, 4, 4))
    request = me.build_request(ep_dir, cell_w=128)
    images = [base64.b64decode(p['image_url']['url'].split(',', 1)[1])
              for p in request['content'] if p['type'] == 'image_url']
    assert len(images) == request['n_images']
    assert sum(map(len, images)) == request['image_bytes']
    assert any(p['type'] == 'text' and 'TACTILE ARRAY SNAPSHOTS' in p['text'] for p in request['content'])
    assert request['sensor_evidence']['sensors'][0]['array_evidence']['frames']


def test_large_arrays_do_not_crowd_out_other_sensor_families():
    rng = np.random.default_rng(3)
    signals = {'pressure_grid': rng.random((100, 32, 32))}
    signals.update({f'pressure_{i}': rng.random((100, 16, 16)) for i in range(33)})
    ep = episode(signals, [{'name': k, 'shape': list(v.shape[1:])} for k, v in signals.items()])
    doc = se.build(ep, {'n': 100, 'ks': [0, 50, 99]})
    sent = json.loads(se.prompt(doc).split('\n')[-1])
    assert {r['sensor_id'] for r in sent['series']} == {s['id'] for s in sent['sensors']}
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES
    assert len(se.image_parts(doc)) // 2 <= 4


def test_missing_initial_cells_do_not_appear_in_signed_comparison_coverage():
    raw = np.full((5, 4, 4), 100.0)
    raw[:3] = np.nan
    raw[:3, 0, 0] = 100
    raw[3:] += 1
    ep = episode({'pressure': raw}, [{'name': 'pressure', 'shape': [4, 4]}])
    doc = se.build(ep, {'n': 5, 'ks': [0, 3, 4]})
    frames = doc['sensors'][0]['array_evidence']['frames']
    assert frames[3]['valid_cells'] == 16
    assert frames[3]['signed_change_valid_cells'] == 1
    assert frames[3]['signed_change_quantiles'] == [1, 1, 1]


def test_too_narrow_grids_keep_traces_instead_of_claiming_invisible_map_cells():
    ep = episode({'pressure': np.arange(2560).reshape(5, 1, 512)},
                 [{'name': 'pressure', 'shape': [1, 512]}])
    doc = se.build(ep, {'n': 5})
    assert doc['series']
    assert 'array_evidence' not in doc['sensors'][0]
    assert not se.image_parts(doc)


def test_prompt_reduction_keeps_event_extrema_before_ordinary_rgb_instants():
    rng = np.random.default_rng(7)
    signals = {f'pressure_{i}': rng.random((100, 16, 16)) for i in range(40)}
    signals['pressure_0'][66, 0, 0] = 10000
    signals['pressure_0'][75] = 500
    ep = episode(signals, [{'name': k, 'shape': [16, 16]} for k in signals])
    doc = se.build(ep, {'n': 100, 'ks': list(range(0, 100, 10))})
    times = doc['sensors'][0]['times']
    assert {0, 6.6, 7.5, 9.9}.issubset(times)
    assert len(se.prompt(doc).encode()) <= se.MAX_PROMPT_BYTES


def test_inspection_sees_array_images_and_a_complete_supplied_field_summary():
    from label.evidence_access import Access
    ep = episode({'pressure': np.arange(1600).reshape(100, 4, 4)},
                 [{'name': 'pressure', 'shape': [4, 4]}])
    doc = se.build(ep, {'n': 100, 'ks': list(range(0, 100, 10))})
    access = object.__new__(Access)
    access.ep, access.supplied = ep, doc
    image = next(p for p in se.image_parts(doc) if p['type'] == 'image_url')
    content = []
    for i in range(6):
        content += [{'type': 'text', 'text': 'TACTILE ARRAY SNAPSHOTS | signal:pressure' if i == 2 else f'=== grid {i} ==='},
                    image]
    parts = access.selection_content(content)
    assert any(p.get('text', '').startswith('TACTILE ARRAY SNAPSHOTS') for p in parts)
    summary = json.loads(parts[0]['text'].split('\n', 1)[1])['already_supplied_sensor_evidence']
    assert not summary.get('truncated')
    assert {r['id'] for r in summary['series']} == {r['id'] for r in doc['series']}
    assert all('values' not in r for r in summary['series'])


def test_sensor_clock_accepts_timestamp_rounding_but_rejects_real_overrun():
    ep = episode({'pressure': np.arange(80).reshape(5, 4, 4)},
                 [{'name': 'pressure', 'shape': [4, 4]}])
    ep['times']['exo'] = np.linspace(0, 4.4992719999999995, 5)
    doc = se.build(ep, {'n': 5, 'ks': [0, 2, 4]})
    # Isolate sensor coverage inside a wider scene window.
    doc['episode_window_s'] = [-1., 6.]
    finding = {'start_s': 0, 'end_s': 4.499272, 'headline': 'Cell readings fluctuate',
               'claim_type': 'response_change', 'adds_beyond_video': True,
               'evidence': [{'sensor_id': 'signal:pressure', 'time_s': [0, 4.499272]}]}
    assert se.bind(doc, [finding])['findings'] == [finding]
    for interval in ({'start_s': -0.001}, {'end_s': 4.500272}):
        rejected = se.bind(doc, [{**finding, **interval}])
        assert not rejected['findings']
        assert rejected['unbound_findings'][0]['reason'] == 'finding exceeds the evidence clock'
