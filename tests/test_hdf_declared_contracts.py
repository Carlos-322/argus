"""Publisher clock, measurement and recording declarations survive generic HDF preparation."""
import json

import h5py
import numpy as np
import pytest

from prepare import formats


def rgb(group, rows=6):
    group['rgb'] = np.zeros((rows, 32, 48, 3), dtype=np.uint8)


def convert(source, tmp_path):
    item = {'file': source, 'group': '', 'name': source.stem}
    ctx = formats.convert_hdf5(item, 'ego_head', tmp_path / 'episodes', 'test')
    return next((tmp_path / 'episodes').iterdir()), ctx


@pytest.mark.parametrize('unit_key', ['unit', 'units'])
def test_declared_seconds_survive_slow_camera_conversion(tmp_path, unit_key):
    source = tmp_path / 'slow.h5'
    raw = np.array([0., 2., 4.], dtype=np.float64)
    with h5py.File(source, 'w') as h:
        rgb(h, 3)
        h.create_dataset('timestamp', data=raw).attrs[unit_key] = 'seconds'
    ep, ctx = convert(source, tmp_path)
    with np.load(ep / 'times.npz') as times:
        np.testing.assert_allclose(times['exo'], [0., 2., 4.], atol=1e-6)
    with np.load(ep / ctx['recorded_container_times']['file']) as retained:
        np.testing.assert_array_equal(retained['timestamp'], raw)
        assert retained['timestamp'].dtype == raw.dtype
    assert ctx['recorded_container_times']['clocks']['timestamp']['units'] == 'seconds'


def test_declared_sensor_unit_cannot_be_changed_to_match_camera_span(tmp_path):
    with h5py.File(tmp_path / 'units.h5', 'w') as h:
        rgb(h)
        h.create_dataset('timestamp', data=[0., .1, .2, .3, .4, .5]).attrs['unit'] = 's'
        pad = h.create_group('pad')
        pad.create_dataset('timestamp', data=np.arange(600)).attrs['units'] = 'seconds'
        pad['pressure'] = np.arange(600, dtype=np.float32)
        streams = formats.h5_streams(h, '')
        np.testing.assert_array_equal(streams['clock']['pad/timestamp'], np.arange(600))
        np.testing.assert_allclose(streams['clock']['timestamp'], [0., .1, .2, .3, .4, .5])


@pytest.mark.parametrize('unit,raw', [('ms', [0, 100, 200]), ('microseconds', [0, 100000, 200000]),
                                   ('ns', [0, 100000000, 200000000])])
def test_declared_clock_units_set_camera_seconds(tmp_path, unit, raw):
    with h5py.File(tmp_path / 'clock.h5', 'w') as h:
        rgb(h, 3)
        h.create_dataset('timestamp', data=np.asarray(raw, dtype=np.int64)).attrs['units'] = unit
        np.testing.assert_allclose(formats.h5_streams(h, '')['clock']['timestamp'], [0., .1, .2])


def test_conflicting_clock_units_keep_source_but_only_assume_frame_placement(tmp_path):
    source = tmp_path / 'conflict.h5'
    raw = np.array([0, 100, 200, 300, 400, 500], dtype=np.int64)
    with h5py.File(source, 'w') as h:
        h.attrs['fps'] = 10
        rgb(h)
        clock = h.create_dataset('timestamp', data=raw)
        clock.attrs.update(unit='seconds', units='milliseconds')
        h['pressure'] = np.array([0., 0., 10., 10., 0., 0.])
    ep, ctx = convert(source, tmp_path)
    assert ctx.get('presentation_times')
    assert any(i['kind'] == 'clock_units_unresolved' for i in ctx['reader_issues'])
    assert 'declarations conflict' in ctx['camera_clock']['exo']['what']
    assert 'nonfinite' not in ctx['camera_clock']['exo']['what']
    pressure = next(s for s in ctx['signals'] if s['name'] == 'pressure')
    assert pressure.get('camera_aligned_by') == 'assumed camera clock'
    with np.load(ep / ctx['recorded_container_times']['file']) as retained:
        np.testing.assert_array_equal(retained['timestamp'], raw)
    with h5py.File(source) as h:
        assert h['timestamp'].attrs['unit'] == 'seconds'
        assert h['timestamp'].attrs['units'] == 'milliseconds'


@pytest.mark.parametrize('name,shape,dtype', [('tactile_pressure', (128, 128), 'uint16'),
                                           ('taxels', (32, 48, 3), 'float32'),
                                           ('depth', (128, 128), 'uint16')])
def test_declared_tactile_maps_are_signals_with_original_units(tmp_path, name, shape, dtype):
    source = tmp_path / 'skin.h5'
    with h5py.File(source, 'w') as h:
        rgb(h)
        h['timestamp'] = np.arange(6) / 10
        ds = h.create_dataset(name, data=np.ones((6, *shape), dtype=dtype))
        ds.attrs.update(units='kPa', sensor_type='tactile pressure')
    ep, ctx = convert(source, tmp_path)
    assert not ctx.get('depth')
    measurement = next(s for s in ctx['signals'] if s['name'] == name)
    assert measurement['shape'] == list(shape) and measurement['units'] == 'kPa'
    with np.load(ep / 'signals.npz') as signals:
        np.testing.assert_array_equal(signals[measurement['key']], np.ones((6, np.prod(shape))))


def test_unnamed_numeric_map_does_not_acquire_depth_semantics(tmp_path):
    with h5py.File(tmp_path / 'array.h5', 'w') as h:
        field = h.create_dataset('readings', data=np.ones((6, 128, 128), dtype=np.float32))
        assert formats.h5_kind('readings', field) == 'signal'
        depth = h.create_dataset('depth', data=np.ones((6, 128, 128), dtype=np.float32))
        assert formats.h5_kind('depth', depth) == 'depth'


@pytest.mark.parametrize('name,representation,units,scale,expected_scale', [
    ('readings', 'depth', 'm', None, 1.),
    ('force_sensor/depth', 'depth', 'm', None, 1.),
    ('readings', 'disparity', 'pixels', None, None),
    ('readings', 'depth', 'mm', np.float32(.001), float(np.float32(.001))),
    ('readings', 'depth', 'mm', 1., None)])
def test_declared_depth_representation_survives_classification_and_conversion(
        tmp_path, name, representation, units, scale, expected_scale):
    from label import depth
    source = tmp_path / 'depth.h5'
    values = np.full((6, 64, 80), 25, dtype=np.uint16)
    with h5py.File(source, 'w') as h:
        rgb(h)
        h['timestamp'] = np.arange(6) / 10
        field = h.create_dataset(name, data=values)
        field.attrs['sensor_type'] = representation
        field.attrs['units'] = units
        if scale is not None:
            field.attrs['depth_scale'] = scale
        streams = formats.h5_streams(h, '')
        assert [s['path'] for s in streams['depth']] == [name]
        assert not streams['signal']
    ep, ctx = convert(source, tmp_path)
    entry = depth.load(ep)['exo']
    assert entry['kind'] == representation
    assert depth.metric_scale(entry) == expected_scale
    decoded = depth.decode(entry, entry['pts'], [0])[0]
    np.testing.assert_array_equal(decoded, values[0])
    assert decoded.dtype == np.uint16
    if representation == 'disparity':
        pixels = np.array([[1, 2]], dtype=np.uint16)
        disparity = np.asarray(depth.picture(pixels, entry, (1., 2.)))
        ordinary = np.asarray(depth.picture(pixels, {'kind': 'depth'}, (1., 2.)))
        np.testing.assert_array_equal(disparity, ordinary[:, ::-1])
        assert ctx['depth']['exo']['units'] == 'relative'


@pytest.mark.parametrize('sensor_type', ['depth pressure', 'depth disparity'])
def test_incompatible_sensor_declarations_do_not_establish_depth(tmp_path, sensor_type):
    with h5py.File(tmp_path / 'ambiguous.h5', 'w') as h:
        rgb(h)
        field = h.create_dataset('depth', data=np.ones((6, 64, 80), dtype=np.uint16))
        field.attrs['sensor_type'] = sensor_type
        assert not formats.h5_streams(h, '')['depth']


@pytest.mark.parametrize('name,sensor_type', [('tactile_image', 'optical tactile sensor'),
                                             ('rgb', 'hardware model 7'),
                                             ('rgb', ['vendor_model_7', 'revision_B']),
                                             ('rgb', ['pressure', 'camera'])])
def test_optical_touch_images_and_unknown_hardware_types_remain_cameras(tmp_path, name, sensor_type):
    with h5py.File(tmp_path / 'optical.h5', 'w') as h:
        image = h.create_dataset(name, data=np.zeros((6, 32, 48, 3), dtype=np.uint8))
        image.attrs['sensor_type'] = sensor_type
        assert formats.h5_kind(name, image) == 'camera'
        np.testing.assert_array_equal(image.attrs['sensor_type'], sensor_type)


@pytest.mark.parametrize('group_names', [('left', 'right'), ('camera_0', 'camera_1'),
                                           ('left_recording', 'right_recording')])
@pytest.mark.parametrize('cameras_in_groups', [False, True])
def test_sensor_siblings_remain_one_recording(tmp_path, cameras_in_groups, group_names):
    with h5py.File(tmp_path / 'actors.h5', 'w') as h:
        if not cameras_in_groups:
            rgb(h)
        for side in group_names:
            group = h.create_group(side)
            group['qpos'] = np.zeros((6, 7), dtype=np.float32)
            group['timestamp'] = np.arange(6) / 10
            if cameras_in_groups:
                rgb(group)
        assert formats.h5_episodes(h) == ['']


@pytest.mark.parametrize('group_names', [('demo_0', 'demo_1'),
                                       ('first_recording', 'second_recording'),
                                       ('morning_take', 'afternoon_take')])
@pytest.mark.parametrize('outside', ['camera', 'sensor', None])
def test_named_recording_siblings_split_only_when_all_recorded_streams_are_owned(tmp_path, outside, group_names):
    with h5py.File(tmp_path / 'demos.h5', 'w') as h:
        h['calibration'] = np.eye(4)
        for name in group_names:
            group = h.create_group('data/' + name)
            rgb(group)
            group['timestamp'] = np.arange(6) / 10
        if outside == 'camera':
            rgb(h)
        elif outside == 'sensor':
            h['force'] = np.arange(6, dtype=np.float32)
            h['timestamp'] = np.arange(6) / 10
        assert formats.h5_episodes(h) == (['data/' + name for name in sorted(group_names)] if outside is None else [''])


def test_numbered_demo_missing_camera_is_reported_without_collapsing_recordings(tmp_path):
    upload = tmp_path / 'upload'
    upload.mkdir()
    with h5py.File(upload / 'demos.h5', 'w') as h:
        for index in range(2):
            group = h.create_group('data/demo_' + str(index))
            group['qpos'] = np.full((6, 7), index, dtype=np.float32)
            group.create_dataset('timestamps', data=np.arange(6) / 10 + index * 100).attrs['units'] = 'seconds'
            if index == 0:
                rgb(group)
    out = tmp_path / 'episodes'
    report = formats.convert(upload, 'teleop_arms', out, 'test', 900)
    assert [episode['name'] for episode in report['episodes']] == ['demos/demo_0']
    assert len(report['failed']) == 1 and report['failed'][0]['name'] == 'demos/demo_1'
    assert 'no camera' in report['failed'][0]['why']
    ep = out / report['episodes'][0]['episode_id']
    ctx = json.loads((ep / 'context.json').read_text())
    assert ctx['state_kind'] == 'none'
    signal = next(row for row in ctx['signals'] if row['name'].endswith('qpos'))
    with np.load(ep / 'signals.npz') as arrays:
        np.testing.assert_array_equal(arrays[signal['key']], np.zeros((6, 7)))


def test_hdf_group_cycle_does_not_discard_owned_camera(tmp_path):
    import subprocess
    import sys
    upload = tmp_path / 'upload'; upload.mkdir()
    with h5py.File(upload / 'recording.h5', 'w') as h:
        h.attrs['fps'] = 10
        rgb(h)
        h['timestamp'] = np.arange(6) / 10
        h['cycle'] = h['/']
    result = subprocess.run([sys.executable, '-c',
        'import json,sys; from pathlib import Path; from prepare.formats import convert; '
        'print(json.dumps(convert(Path(sys.argv[1]), "ego_head", Path(sys.argv[2]), "fixture", 20)))',
        str(upload), str(tmp_path / 'out')], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout.splitlines()[-1])
    assert len(report['episodes']) == 1 and not report['failed']


@pytest.mark.parametrize('inside', [False, True])
def test_hdf_external_instruction_requires_an_authorized_source_file(tmp_path, inside):
    from label import episode
    upload = tmp_path / 'upload'; upload.mkdir()
    external = (upload if inside else tmp_path) / 'instructions.h5'
    with h5py.File(external, 'w') as source:
        source['instruction'] = 'Unshared external instruction 9284517'
    with h5py.File(upload / 'recording.h5', 'w') as h:
        h.attrs['fps'] = 10
        rgb(h)
        h['timestamp'] = np.arange(6) / 10
        h['task'] = h5py.ExternalLink(str(external), '/instruction')
    out = tmp_path / 'out'
    report = formats.convert(upload, 'ego_head', out, 'fixture', 20)
    assert len(report['episodes']) == 1
    ep = out / report['episodes'][0]['episode_id']
    context = json.loads((ep / 'context.json').read_text())
    prompt = episode.build_request(ep)['prompt']
    assert ('Unshared external instruction 9284517' in prompt) is inside
    if not inside:
        assert any(i['kind'] == 'external_source_unread' for i in context['reader_issues'])


@pytest.mark.parametrize('damaged', ['depth', 'force', 'timestamp'])
@pytest.mark.parametrize('row', [0, 2])
def test_hdf_damaged_non_rgb_chunk_preserves_other_owned_rows(tmp_path, damaged, row):
    upload = tmp_path / 'upload'; upload.mkdir(); source = upload / 'recording.h5'
    with h5py.File(source, 'w') as h:
        h.attrs['fps'] = 10
        rgb(h)
        h.create_dataset('timestamp', data=np.arange(6) / 10, chunks=(1,), fletcher32=True)
        h.create_dataset('force', data=np.arange(6, dtype=float), chunks=(1,), fletcher32=True)
        h.create_dataset('depth', data=np.stack([np.full((8, 8), 1000+i, dtype=np.uint16) for i in range(6)]),
                         chunks=(1, 8, 8), fletcher32=True).attrs['sensor_type'] = 'depth'
        index = (row,) + (0,) * (h[damaged].ndim - 1)
        chunk = h[damaged].id.get_chunk_info_by_coord(index)
    with source.open('r+b') as stream:
        stream.seek(chunk.byte_offset + chunk.size - 1)
        value = stream.read(1); stream.seek(-1, 1); stream.write(bytes([value[0] ^ 1]))
    out = tmp_path / 'out'; report = formats.convert(upload, 'ego_head', out, 'fixture', 20)
    assert len(report['episodes']) == 1 and not report['failed']
    ep = out / report['episodes'][0]['episode_id']
    context = json.loads((ep / 'context.json').read_text())
    assert context['n_state_frames'] == 6
    assert context.get('reader_issues')
    if damaged == 'depth':
        entries = json.loads((ep / 'depth.json').read_text())
        mapping = np.load(ep / next(iter(entries.values()))['kmap'])
        assert mapping[row] == -1 and all(mapping[i] >= 0 for i in range(6) if i != row)
        from label.depth import at_anchor, load
        restored = load(ep)
        for index in range(6):
            picture = at_anchor({}, restored, next(iter(restored)), [index])
            assert bool(picture) is (index != row)
            if picture:
                assert np.all(next(iter(picture.values())) == 1000 + index)
    if damaged == 'force':
        force = next(signal for signal in context['signals'] if signal['name'] == 'force')
        with np.load(ep / 'signals.npz') as values:
            expected = np.arange(6, dtype=float); expected[row] = np.nan
            np.testing.assert_allclose(values[force['key']].ravel(), expected, equal_nan=True)


@pytest.mark.parametrize('linked_storage', ['external_link', 'virtual', 'virtual_external_link'])
def test_single_selected_hdf_file_does_not_authorize_neighbor_files(tmp_path, linked_storage):
    source = tmp_path / 'recording.h5'
    private = tmp_path / 'private.h5'
    with h5py.File(private, 'w') as h:
        h['instruction'] = 'Private task 4829971'
        h['force'] = np.arange(6, dtype=float) + 4829971
    with h5py.File(source, 'w') as h:
        rgb(h); h.attrs['fps'] = 10
        if linked_storage == 'external_link':
            h['task'] = h5py.ExternalLink(str(private), '/instruction')
        else:
            layout = h5py.VirtualLayout(shape=(6,), dtype=float)
            if linked_storage == 'virtual_external_link':
                h['linked_force'] = h5py.ExternalLink(str(private), '/force')
                layout[:] = h5py.VirtualSource('.', '/linked_force', shape=(6,))
            else:
                layout[:] = h5py.VirtualSource(str(private), 'force', shape=(6,))
            h.create_virtual_dataset('force', layout)
    out = tmp_path / 'out'
    report = formats.convert(source, 'ego_head', out, 'fixture', 20)
    ep = out / report['episodes'][0]['episode_id']
    from label.episode import build_request
    prompt = build_request(ep)['prompt']
    assert '4829971' not in prompt
    context = json.loads((ep / 'context.json').read_text())
    assert any(issue['kind'] == 'external_source_unread' for issue in context['reader_issues'])


def test_unrelated_damaged_hdf_object_metadata_does_not_hide_a_readable_camera(tmp_path):
    # Synthetic corpus mutant m0285, seed 1285: wrist object metadata fails h5o.get_info,
    # while all 60 front-camera rows are readable. Keep the original damaged bytes.
    import gzip
    from pathlib import Path
    from label import episode
    source = tmp_path / 'recording.h5'
    source.write_bytes(gzip.decompress((Path(__file__).parent / 'fixtures' /
                                       'corrupt_hdf_camera_metadata.h5.gz').read_bytes()))
    with h5py.File(source, 'r') as h:
        expected = h['observations/images/cam_high'][()]
    out = tmp_path / 'out'
    report = formats.convert(source, 'teleop_arms', out, 'corrupt metadata', 20)
    assert len(report['episodes']) == 1 and not report['failed']
    loaded = episode.load(out / report['episodes'][0]['episode_id'])
    assert loaded['context']['n_state_frames'] == len(expected) == 60
    decoded = episode._decode_view(loaded, 'exo', list(range(60)))
    assert len(decoded) == 60
    for row, picture in decoded.items():
        assert abs(np.asarray(picture).mean() - expected[row].mean()) < 4
    assert loaded['context'].get('reader_issues')
