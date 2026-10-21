"""Explicit layouts stay bounded, source attributed and separate from pooled values."""
import copy
import importlib
import json

import h5py
import numpy as np
import pytest

from prepare import formats


def descriptor(**extra):
    return {'version': 1, 'regions': [{'name': 'recorded patch', 'indices': [0, 3]}],
            'valid_indices': [0, 1, 3], 'response_direction': 'down', 'unit': 'ADC',
            'provenance': {'authority': 'recorded', 'source': 'https://example.org/layout', 'revision': 'r2'},
            **extra}


def validate(value, shape):
    return importlib.import_module('prepare.sensor_semantics').validate(value, shape)


def test_sparse_semantics_preserve_unknown_metadata_without_inventing_anatomy():
    value = descriptor(extension={'recorded_codes': [17, 19]}, zero_exclusion_sensitivity=True,
                       positions=[[0, 0], [1, 0], [0, 1], [1, 1]])
    before = copy.deepcopy(value)
    actual = validate(json.dumps(value), [2, 2])
    assert actual == before
    actual['regions'][0]['indices'].append(2)
    assert value == before
    assert validate({'version': 1}, [21, 3]) == {'version': 1}


@pytest.mark.parametrize('change', [
    {'version': 2}, {'version': True}, {'regions': [{'name': 'bad', 'indices': [4]}]},
    {'valid_indices': [0, 0]}, {'response_direction': 'positive'},
    {'positions': [[0, 0]]}, {'positions': [[0, 0], [1, 0], [0, 1], [np.nan, 1]]},
    {'units': ['N', 'C']}, {'units': 'N'}, {'provenance': {'authority': 'guessed'}},
    {'provenance': {'authority': 'recorded', 'source': '/local/path'}},
    {'zero_exclusion_sensitivity': 'yes'}, {'extension': {'bad': np.inf}},
    {'extension': 'x' * 65536},
    {'regions': [{'name': 'same', 'indices': [0]}, {'name': 'same', 'indices': [1]}]},
])
def test_inconsistent_or_unbounded_semantics_are_not_applied(change):
    assert validate(descriptor(**change), [2, 2]) is None


def test_landmark_indices_address_points_rather_than_flattened_coordinates():
    value = {'version': 1, 'normalization_pair': [0, 3],
             'chains': [{'name': 'recorded chain', 'indices': [0, 1, 2, 3]}], 'units': ['m', 'm', 'm']}
    assert validate(value, [4, 3]) == value
    assert validate({**value, 'normalization_pair': [0, 4]}, [4, 3]) is None
    assert validate({**value, 'normalization_pair': [0, 0]}, [4, 3]) is None
    assert validate(value, [12]) is None
    assert validate({**value, 'chains': [{'name': 'bad', 'indices': [0, 4, 5]}]}, [4, 3]) is None


def test_hdf_and_feature_metadata_keep_explicit_semantics_but_pooled_values_drop_them(tmp_path, monkeypatch):
    value = descriptor()
    with h5py.File(tmp_path / 'recording.h5', 'w') as source:
        ds = source.create_dataset('measurement', data=np.ones((3, 2, 2)))
        ds.attrs['sensor_semantics'] = json.dumps(value).encode()
        ds.attrs['unit'] = 'ADC'
        metadata = formats.h5_signal_metadata(ds)
        assert metadata['sensor_semantics'] == value
        assert formats.h5_signal_metadata(ds, summarised=True) == {'unit': 'ADC'}
    signals = formats.Signals()
    signals.add('measurement', np.ones((3, 4)), shape=[2, 2],
                metadata={'info': {'sensor_semantics': json.dumps(value)}, 'unit': 'ADC'})
    assert signals.meta['measurement']['sensor_semantics'] == value
    monkeypatch.setattr(formats, 'SIGNAL_EPISODE_BYTES', 0)
    signals.add('pooled', np.ones((3, 4)), shape=[2, 2], metadata={'sensor_semantics': value, 'unit': 'ADC'})
    assert signals['pooled'].shape == (3, 3)
    assert 'sensor_semantics' not in signals.meta['pooled']
    assert signals.meta['pooled']['unit'] == 'ADC'


def test_unknown_or_invalid_recorded_semantics_remain_metadata_without_becoming_valid():
    future = {'version': 7, 'publisher_extension': {'layout': [1, 9]}}
    signals = formats.Signals()
    signals.add('measurement', np.ones((3, 4)), metadata={'sensor_semantics': future})
    assert signals.meta['measurement']['sensor_semantics'] == future
    assert validate(future, [4]) is None


def test_scalar_hdf_string_arrays_decode_the_recorded_descriptor(tmp_path):
    value = descriptor()
    with h5py.File(tmp_path / 'recording.h5', 'w') as source:
        ds = source.create_dataset('measurement', data=np.ones((3, 4)))
        ds.attrs['sensor_semantics'] = np.array([json.dumps(value).encode()])
        assert formats.h5_signal_metadata(ds)['sensor_semantics'] == value


def test_filtering_bookkeeping_columns_does_not_reassign_original_sensor_regions(tmp_path):
    value = {'version': 1, 'regions': [{'name': 'original component zero', 'indices': [0]}]}
    signals = formats.Signals()
    signals.add('measurement', np.ones((3, 2)), names=['rss_kb', 'measurement'],
                metadata={'sensor_semantics': value})
    context = {'n_state_frames': 3, 'fps': 30}
    formats.write_signals(tmp_path, context, signals)
    (reading,) = context['signals']
    assert reading['dims'] == 1
    assert 'sensor_semantics' not in reading
