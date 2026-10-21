"""Publisher layouts require their complete recording schema and remain candidate evidence."""
import io
import json

import h5py
import numpy as np
import pytest
from PIL import Image

from prepare import formats


def recording(path, group='data/demo_00', omit=(), pressure_shape=(16, 16), image='jpeg', metadata=None):
    with h5py.File(path, 'w') as source:
        source.attrs['fps'] = 10
        episode = source.require_group(group)
        episode['timestamps'] = np.arange(6) / 10
        shapes = {'right_pressure': pressure_shape, 'right_hand_landmarks': (21, 3),
                  'right_palm_pos': (3,), 'right_wrist_pos': (3,), 'camera_poses': (4, 4)}
        for name, shape in shapes.items():
            if name not in omit:
                dataset = episode.create_dataset(name, data=np.ones((6, *shape), np.float32))
                for key, value in (metadata or {}).get(name, {}).items():
                    dataset.attrs[key] = json.dumps(value) if isinstance(value, dict) else value
        if 'rgb' not in omit:
            if image == 'raw':
                episode['rgb'] = np.zeros((6, 36, 64, 3), np.uint8)
            else:
                buffer = io.BytesIO()
                Image.new('RGB', (64, 36), (20, 80, 130)).save(buffer, format=image.upper())
                dataset = episode.create_dataset('rgb', (6,), dtype=h5py.vlen_dtype(np.uint8))
                for row in range(6):
                    dataset[row] = np.frombuffer(buffer.getvalue(), np.uint8)


def convert(root, output, dataset='unfamiliar upload'):
    report = formats.convert(root, 'ego_head', output, dataset, 900)
    assert not report['failed'], report
    assert len(report['episodes']) == 1
    directory = output / report['episodes'][0]['episode_id']
    context = json.loads((directory / 'context.json').read_text())
    return {row['name'].rsplit('/', 1)[-1]: row for row in context.get('signals', [])}


def test_complete_publisher_schema_persists_candidate_contract_through_normal_conversion(tmp_path):
    upload = tmp_path / 'upload'
    upload.mkdir()
    recording(upload / 'anything.h5')
    rows = convert(upload, tmp_path / 'prepared')
    pressure = rows['right_pressure']['sensor_semantics']
    assert pressure['version'] == 1
    assert pressure['provenance']['authority'] == 'publisher_candidate'
    assert pressure['provenance']['revision'] == '6d1fa38f19d034da431f7a81ef376ad09381ca89'
    assert pressure['provenance']['hash'] == 'e3ae0c6028b1714fccbcb77cdfb1e78449891a64083aae0034641ba2ceef7182'
    assert len(pressure['valid_indices']) == 169
    regions = {region['name']: region['indices'] for region in pressure['regions']}
    assert regions['thumb'][:3] == [203, 204, 205]
    assert regions['little finger'][:2] == [0, 1]
    assert pressure['response_direction'] == 'down'
    assert pressure['zero_exclusion_sensitivity'] is True
    assert pressure['hardware_revision_verified'] is False
    assert pressure['force_calibration_verified'] is False
    assert pressure['anatomical_claims_conditional'] is True
    landmarks = rows['right_hand_landmarks']['sensor_semantics']
    assert landmarks['normalization_pair'] == [5, 17]
    assert landmarks['chains'][0] == {'name': 'thumb', 'indices': [1, 2, 3, 4]}
    assert landmarks['chains'][-1]['indices'] == [17, 18, 19, 20]
    assert 'sensor_semantics' not in rows['right_palm_pos']


def test_filename_episode_group_and_dataset_label_do_not_control_recognition(tmp_path):
    values = []
    for i, (filename, group, label) in enumerate([
        ('opentouch.h5', 'data/demo_00', 'opentouch'),
        ('renamed.h5', 'arbitrary/renamed_group', 'some other label'),
    ]):
        upload = tmp_path / f'upload{i}'
        upload.mkdir()
        recording(upload / filename, group=group)
        rows = convert(upload, tmp_path / f'prepared{i}', dataset=label)
        values.append(rows['right_pressure'].get('sensor_semantics'))
    assert values[0] is not None
    assert values[0] == values[1]


@pytest.mark.parametrize('options', [
    {'omit': ('right_hand_landmarks',)}, {'omit': ('right_palm_pos',)},
    {'omit': ('right_wrist_pos',)}, {'omit': ('camera_poses',)},
    {'omit': ('right_hand_landmarks', 'right_palm_pos', 'right_wrist_pos', 'camera_poses')},
    {'pressure_shape': (8, 32)}, {'image': 'png'}, {'image': 'raw'},
])
def test_partial_or_shape_only_matches_stay_generic(tmp_path, options):
    upload = tmp_path / 'upload'
    upload.mkdir()
    recording(upload / 'opentouch.h5', **options)
    rows = convert(upload, tmp_path / 'prepared', dataset='opentouch')
    assert all('sensor_semantics' not in row for row in rows.values())


def test_explicit_recorded_semantics_calibration_and_polarity_win(tmp_path):
    upload = tmp_path / 'upload'
    upload.mkdir()
    explicit = {'version': 7, 'custom_layout': 'recorded hardware'}
    recording(upload / 'recording.h5', metadata={
        'right_hand_landmarks': {'sensor_semantics': explicit},
        'right_pressure': {'response_direction': 'up', 'unit': 'N', 'calibration': {'zero': 12}},
    })
    rows = convert(upload, tmp_path / 'prepared')
    assert rows['right_hand_landmarks']['sensor_semantics'] == explicit
    pressure = rows['right_pressure']
    assert pressure['response_direction'] == 'up'
    assert pressure['unit'] == 'N'
    assert pressure['calibration'] == '{"zero": 12}'
    assert pressure['sensor_semantics']['response_direction'] == 'up'
    assert pressure['sensor_semantics']['unit'] == 'N'


def test_pooled_publisher_arrays_do_not_keep_candidate_geometry(tmp_path, monkeypatch):
    upload = tmp_path / 'upload'
    upload.mkdir()
    recording(upload / 'recording.h5')
    monkeypatch.setattr(formats, 'SIGNAL_EPISODE_BYTES', 0)
    rows = convert(upload, tmp_path / 'prepared')
    assert rows['right_pressure']['summary_of'] == 256
    assert 'sensor_semantics' not in rows['right_pressure']
    assert 'sensor_semantics' not in rows['right_hand_landmarks']


@pytest.mark.parametrize('changed', ['row_count', 'different_parent', 'missing_rgb'])
def test_schema_companions_must_belong_to_the_same_recording_rows(tmp_path, changed):
    from prepare.publisher_schemas import hdf5_candidates
    path = tmp_path / 'source.h5'
    recording(path)
    with h5py.File(path, 'a') as source:
        group = source['data/demo_00']
        if changed == 'row_count':
            del group['right_palm_pos']
            group['right_palm_pos'] = np.ones((5, 3))
        elif changed == 'different_parent':
            source.move('data/demo_00/right_palm_pos', 'another_recording/right_palm_pos')
        else:
            del group['rgb']
        assert hdf5_candidates(source, formats.h5_streams(source, '')) == {}


def test_unmatched_left_arrays_stay_generic_and_calibration_suppresses_candidate_polarity(tmp_path):
    upload = tmp_path / 'upload'
    upload.mkdir()
    path = upload / 'source.h5'
    recording(path, metadata={'right_pressure': {'calibration': {'recorded_transfer': 'unresolved'}}})
    with h5py.File(path, 'a') as source:
        source['data/demo_00/left_pressure'] = np.ones((6, 16, 16), np.float32)
    rows = convert(upload, tmp_path / 'prepared')
    assert 'sensor_semantics' not in rows['left_pressure']
    assert 'response_direction' not in rows['right_pressure']['sensor_semantics']


@pytest.mark.parametrize('recorded_units', [{'unit': 'kPa'}, {'unit': 'N'}, {'units': ['kPa'] * 256}])
def test_incompatible_recorded_units_withdraw_the_publishers_raw_adc_polarity(recorded_units):
    from prepare.publisher_schemas import SCHEMA_DIR, with_candidate
    candidate = json.loads((SCHEMA_DIR / 'opentouch_hdf5_v1.json').read_text())['signals']['right_pressure']
    result = with_candidate(recorded_units, candidate, [16, 16])
    semantics = result['sensor_semantics']
    assert 'response_direction' not in semantics
    assert semantics['regions'] == candidate['regions']
    for key, value in recorded_units.items():
        assert semantics[key] == value
    # The inherited convention must not turn a positive recorded pressure change into zero response.
    from label.derived_measurements import regional_response
    values = np.full((5, 256), 100.)
    values[3:] = 120
    trace = regional_response(values, np.arange(5.), semantics)['traces'][0]
    assert trace['values'] == [0, 0, 0, 20, 20]


def test_matching_adc_units_keep_candidate_direction_and_recorded_direction_still_wins():
    from prepare.publisher_schemas import SCHEMA_DIR, with_candidate
    candidate = json.loads((SCHEMA_DIR / 'opentouch_hdf5_v1.json').read_text())['signals']['right_pressure']
    for units in ({'unit': 'ADC'}, {'units': ['ADC'] * 256}):
        assert with_candidate(units, candidate, [16, 16])['sensor_semantics']['response_direction'] == 'down'
    explicit = with_candidate({'unit': 'kPa', 'response_direction': 'up'}, candidate, [16, 16])
    assert explicit['sensor_semantics']['response_direction'] == 'up'


def test_json_encoded_hdf_calibration_keeps_a_continuously_loaded_pad_active(tmp_path):
    from label import contacts
    upload = tmp_path / 'upload'
    upload.mkdir()
    recording(upload / 'recording.h5', metadata={'right_pressure': {
        'unit': 'N', 'response_direction': 'up', 'calibration': {'zero': 0}}})
    rows = convert(upload, tmp_path / 'prepared')
    meta = rows['right_pressure']
    events = contacts.find({'right_pressure': np.ones((6, 256))}, {'right_pressure': meta}, np.arange(6) / 10)
    assert len(events) == 1
    assert events[0]['reference_kind'] == 'declared_zero'
    assert (events[0]['start_s'], events[0]['end_s']) == (0, .5)
