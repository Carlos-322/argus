"""Recorder log order is usable evidence, without establishing physical synchronization."""
import json
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from board import build
from checks import timebase
from label import episode as me


def arrival_source():
    return {'presentation_clock': {'basis': 'mcap_log_time', 'physical_synchronization_verified': False}}


def rendered_checks(record):
    node = shutil.which('node')
    if not node:
        pytest.skip('node is required for actual viewer checks')
    script = """const {src,piece}=require(process.argv[1])();
const a=src.indexOf('const asSentence = '), b=src.indexOf('const sentences = ');
const setup='let CHECKS_OPEN=false; const famName=s=>s; const OUR_CHECKS=[["timebase","sped_up_recording","sped-up"]];'
  +'const sentences=t=>String(t);';
const render=new Function(setup+src.slice(a,b)+piece('esc')+piece('placementText')+piece('checksSection')
  +'return checksSection;')();
process.stdout.write(render(JSON.parse(require('fs').readFileSync(0,'utf8'))));"""
    result = subprocess.run([node, '-e', script, str(Path(__file__).parent / 'page_functions.js')],
                            input=json.dumps(record), text=True, capture_output=True, check=True)
    return result.stdout


def test_prepared_log_axis_keeps_motion_but_cannot_flag_physical_follower_delay(tmp_path):
    from test_checks import _recording, write_episode
    state, action = _recording(0, sped_up=True, skips=True)
    ep = write_episode(tmp_path / 'episode_000000', state, {'exo': np.full(len(state), 100.)},
                       profile='teleop_arms', kind='joints', source=arrival_source(),
                       action_semantics={'version': 1, 'role': 'command', 'producer': 'recorded leader'},
                       timebase_neighbour_lag_frames=2.7)
    np.savez(ep / 'state.npz', state=state, action=action)
    pl = me.plan(me.load(ep))
    measured = pl['checks']['timebase']
    assert measured['sped_up_recording'] is False
    assert measured['follower_lag_frames'] is None
    assert measured['recorded_order_lag_frames'] is not None and measured['skipped_frac'] > 0
    assert measured['status'] == 'not evaluated' and 'physical' in measured['reason'].lower()
    html = rendered_checks({'dataset_checks': {'timebase': measured}})
    assert 'ck-row na' in html and 'physical event synchronization' in html
    assert 'ck-row clear' not in html and 'ck-row issue' not in html
    assert pl['state_usable'] is True
    path = ep / 'context.json'; ctx = json.loads(path.read_text()); ctx.pop('source')
    path.write_text(json.dumps(ctx))
    control = me.plan(me.load(ep))
    assert control['checks']['timebase']['sped_up_recording'] is True
    assert pl['spans'] == control['spans'] and pl['contact'] == control['contact']


def test_neighbour_scan_does_not_calibrate_physical_delay_from_arrival_order(tmp_path):
    from test_checks import _tb_episode
    for index in range(5):
        _tb_episode(tmp_path, f'episode_{index:06d}', index, 'recorded-log', lag=2)
        path = tmp_path / f'episode_{index:06d}' / 'context.json'
        ctx = json.loads(path.read_text()); ctx['source'].update(arrival_source())
        ctx.update(timebase_neighbour_lag_frames=2., timebase_neighbours_in_upload=5)
        path.write_text(json.dumps(ctx))
    assert timebase.measure_folder(tmp_path) == 0
    for path in tmp_path.glob('episode_*/context.json'):
        ctx = json.loads(path.read_text())
        assert 'timebase_neighbour_lag_frames' not in ctx and ctx['timebase_neighbours_in_upload'] == 0


def test_board_preserves_arrival_contact_values_without_physical_timing_claims(tmp_path):
    from test_board_touch import _episode
    ep = _episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text()); ctx['source'] = arrival_source()
    ctx['signals'][0]['aligned_by'] = 'recorder arrival'
    recorded = ctx['contacts']
    labels = {'contacts_model': [{'id': row['id'], 'touch_seen': 'no', 'hand': 'left',
                                  'first_touch_frame': 4, 'last_touch_frame': 3} for row in recorded],
              'contact_views': {'shown': [row['id'] for row in recorded], 'strips': {
                  row['id']: {'begin': [row['start_s'] + offset for offset in (-.3, -.15, 0, .15, .3)],
                              'end': [row['end_s'] + offset for offset in (-.3, -.15, 0, .15, .3)]}
                  for row in recorded}},
              'contacts_missing': [{'hand': 'right', 't_s': 5., 'object': 'cup'}]}
    build.add_contacts(labels, ctx)
    result = labels['dataset_checks']['contact_checks']
    assert result['notes'] == [] and 'offset_ms' not in result
    assert result['placed_on_recorder_arrival_clock'] == [row['id'] for row in recorded]
    assert labels['contact_sensor_coverage'] == {'right': ['right_pressure']}
    assert labels['contacts_missing'][0]['withheld_from_final'] is True
    assert 'synchronization' in labels['contacts_missing'][0]['withheld_reason'].lower()
    for before, after in zip(recorded, labels['contacts']):
        assert all(after[key] == before[key] for key in ('start_s', 'end_s', 'peak_strength', 'regions'))
    html = rendered_checks(labels)
    assert 'ck-row na' in html and 'recorder arrival' in html
    assert 'ck-row clear' not in html and 'physical event synchronization' in html


def test_quiet_arrival_sensor_does_not_establish_missing_physical_contact(tmp_path):
    from test_board_touch import _episode
    ep = _episode(tmp_path)
    ctx = json.loads((ep / 'context.json').read_text()); ctx.update(source=arrival_source(), contacts=[])
    labels = {'contacts_missing': [{'hand': 'right', 't_s': 5., 'object': 'cup'}]}
    build.add_contacts(labels, ctx)
    assert labels['contact_sensor_coverage'] == {'right': ['right_pressure']}
    assert labels['contacts_missing'][0]['withheld_from_final'] is True
    assert labels['dataset_checks']['contact_checks']['notes'] == []


def test_apply_cannot_restore_physical_delay_flag_to_arrival_episode(tmp_path):
    from test_checks import _tb_episode
    ep = tmp_path / 'episode_000000'; _tb_episode(tmp_path, ep.name, 0, 'recorded-log', lag=2)
    path = ep / 'context.json'; ctx = json.loads(path.read_text()); ctx['source'].update(arrival_source())
    path.write_text(json.dumps(ctx))
    run = tmp_path / 'labels'; run.mkdir()
    raw = {'episode_dir': str(ep), 'dataset_checks': {'timebase': {
        'follower_lag_frames': 2., 'skipped_frac': .1, 'repeated_frac': 0., 'sped_up_recording': True}}}
    receipt = run / (ep.name + '.json'); receipt.write_text(json.dumps(raw))
    csv = tmp_path / 'scan.csv'
    pd.DataFrame([{'episode_index': 0, 'neighbour_lag_frames': 2., 'sped_up_recording': True}]).to_csv(csv, index=False)
    assert timebase.cmd_apply(Namespace(timebase=csv, roots=[tmp_path], labels=run)) == 0
    measured = json.loads(receipt.read_text())['dataset_checks']['timebase']
    assert measured['sped_up_recording'] is False and measured['follower_lag_frames'] is None
    assert measured['recorded_order_lag_frames'] == 2. and measured['status'] == 'not evaluated'
    assert 'timebase_neighbour_lag_frames' not in json.loads(path.read_text())


def test_generic_mcap_request_keeps_state_on_log_axis_without_physical_delay_claim(tmp_path):
    from test_abc_generic import capture_mcap, converted
    source = tmp_path / 'source'; source.mkdir()
    capture_mcap(source / 'episode.mcap', n=60, bad_header='/left-arm-state')
    ep, context = converted(source, tmp_path / 'prepared')
    assert context['state_kind'] == 'joints'
    assert context['source']['presentation_clock']['basis'] == 'mcap_log_time'
    # A recorded leader declaration makes this fixture eligible for the follower-delay consumer.
    context['action_semantics'] = {'version': 1, 'role': 'command', 'producer': 'controlled recorded leader declaration'}
    (ep / 'context.json').write_text(json.dumps(context))
    with np.load(ep / 'state.npz') as arrays:
        assert arrays['state'].shape == arrays['action'].shape == (60, 14)
        assert np.isfinite(arrays['state']).all() and np.isfinite(arrays['action']).all()
    request = me.build_request(ep)
    checks = request['plan']['checks']['timebase']
    assert checks['follower_lag_frames'] is None and checks['sped_up_recording'] is False
    assert checks['status'] == 'not evaluated'
    assert request['plan']['state_usable'] is True
    assert 'RECORDED MOTION' in request['prompt'] and 'recorder arrival' in request['prompt'].lower()
    assert all(signal.get('aligned_by') == 'recorder arrival' for signal in context['signals'])
    assert 'fact: check it against the video' not in request['prompt']
    assert 'A disagreement between the recording and the video is' not in request['prompt']
    assert 'only the timing of a change can be compared with the video' not in request['prompt']


def test_arrival_still_spans_do_not_instruct_model_to_diagnose_physical_video_disagreement(tmp_path):
    from test_checks import write_episode
    state = np.zeros((240, 14)); state[120:, 6] = .5
    ep = write_episode(tmp_path / 'episode_000000', state, {'exo': np.full(240, 100.)},
                       profile='teleop_arms', kind='joints', source=arrival_source())
    request = me.build_request(ep)
    assert request['plan']['spans'] and request['plan']['state_usable'] is True
    assert 'RECORDED STILL SPANS' in request['prompt']
    assert 'If the views show motion during a span, the recording is wrong' not in request['prompt']
    assert 'Do not infer a physical state/video mismatch' in request['prompt']


def test_arrival_touch_prompt_and_rate_name_actual_clock_without_assumed_start(tmp_path):
    from test_board_touch import _episode
    from label import contacts, signals
    ep = _episode(tmp_path)
    path = ep / 'context.json'; context = json.loads(path.read_text()); context['source'] = arrival_source()
    context['state_kind'] = 'none'
    path.write_text(json.dumps(context))
    # Older saved preparations predate the signal enum; the episode's explicit recorder clock still applies.
    loaded = me.load(ep)
    found = contacts.of_episode(loaded)
    assert found and all(row['aligned_by'] == 'recorder arrival' for row in found)
    assert 'recorder arrival' in me._contact_timing(found[0])
    assert 'physical synchronization' in me._contact_timing(found[0])
    text = signals.describe('right_pressure', np.arange(60).reshape(-1, 1), rate_hz=10, fps=30,
                            aligned_by='recorder arrival')
    assert 'recorder arrivals at 10 Hz' in text and 'physical synchronization' in text
    assert 'recorded at 10 Hz' not in text and 'both starts' not in text


@pytest.mark.parametrize('hand', [{}, [], None, 3])
def test_malformed_model_hand_is_retained_without_contact_timing_claim(hand):
    from checks import contacts
    row = {'hand': hand, 't_s': 1., 'object': 'cup'}
    labels = {'contacts_missing': [row]}
    result = contacts.check(labels, [], {}, 30, instrumented_hands={'right'})
    assert result['notes'] == [] and labels['contacts_missing'] == [row]
    result = contacts.check(labels, [], {}, 30, instrumented_hands={'right'},
                            timing_context={'source': arrival_source()})
    assert result['notes'] == [] and labels['contacts_missing'] == [row]


@pytest.mark.parametrize('placement,field', [('assumed start', 'placed_from_both_starts'),
    ('row per frame', 'placed_row_per_frame'), ('coarse clock', 'placed_within_stamp_intervals'),
    ('assumed camera clock', 'placed_on_assumed_camera_clock')])
def test_arrival_context_preserves_more_specific_contact_placement(tmp_path, placement, field):
    from test_board_touch import _episode
    from label import contacts
    ep = _episode(tmp_path)
    path = ep / 'context.json'; context = json.loads(path.read_text())
    context.update(source=arrival_source(), state_kind='none')
    if placement == 'assumed camera clock':
        context['signals'][0]['camera_aligned_by'] = placement
    else:
        context['signals'][0]['aligned_by'] = placement
    path.write_text(json.dumps(context))
    found = contacts.of_episode(me.load(ep))
    assert found and all(row['aligned_by'] == placement for row in found)
    labels = {'contacts_model': [{'id': row['id'], 'touch_seen': 'no', 'hand': 'left'} for row in found]}
    build.add_contacts(labels, context)
    checked = labels['dataset_checks']['contact_checks']
    assert checked[field] == [row['id'] for row in found]
    assert checked['notes'] == [] and 'placed_on_recorder_arrival_clock' not in checked
    assert all(row['aligned_by'] == placement for row in labels['contacts'])
