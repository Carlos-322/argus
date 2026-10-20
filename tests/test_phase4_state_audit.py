"""Controlled native observations from MASTER item 21, not borrowed original uploads."""
import hashlib
import json

import h5py
import numpy as np
import pandas as pd
import pytest

from prepare import formats
from test_prepare import _lerobot_v21


JOINTS = [f'joint{i}' for i in range(1, 7)]


def native_state(root, names, *, joint_names=None, split=False):
    _lerobot_v21(root, n=8)
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    values = np.arange(56, dtype=np.float32).reshape(8, 7) / 100
    df['action'] = list(values + np.float32(.01))
    info['features']['action'].update(shape=[7])
    if split:
        df = df.drop(columns=['observation.state'])
        info['features'].pop('observation.state')
        for column, array, labels in [('observation.state.left_arm', values[:, :6], names[:6]),
                                      ('observation.state.left_gripper', values[:, 6:], names[6:])]:
            df[column] = list(array)
            info['features'][column] = {'dtype': 'float32', 'shape': [array.shape[1]], 'names': labels}
    else:
        df['observation.state'] = list(values)
        info['features']['observation.state'].update(shape=[7], names=names)
        if joint_names is not None:
            info['features']['observation.state']['joint_names'] = joint_names
    df.to_parquet(path)
    info_path.write_text(json.dumps(info, ensure_ascii=False))
    return values


def converted(root, out):
    before = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in root.rglob('*') if p.is_file()}
    report = formats.convert(root, 'teleop_arms', out, 'controlled named state', 900)
    assert not report['failed'] and len(report['episodes']) == 1, report
    path = out / report['episodes'][0]['episode_id']
    assert before == {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in root.rglob('*') if p.is_file()}
    return path, json.loads((path / 'context.json').read_text())


@pytest.mark.parametrize('names', [
    ['末端位置x', '末端位置y', '末端位置z', '末端姿态roll', '末端姿态pitch', '末端姿态yaw', '夹爪'],
    [f'未解释值{i}' for i in range(7)]])
def test_chinese_pose_and_unknown_names_never_become_joints_by_width(tmp_path, names):
    root = tmp_path / 'native'
    original = native_state(root, names)
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == 'none'
    assert ctx.get('state_note') and ctx['state_why'] == 'layout'
    field = next(s for s in ctx['signals'] if s['name'] == 'observation.state')
    assert field['names'] == names
    with np.load(path / 'signals.npz') as signals, np.load(path / 'state.npz') as archive:
        np.testing.assert_array_equal(signals[field['key']], original)
        assert archive['state'].dtype == original.dtype and archive['state'].shape == original.shape
        assert archive['state'].tobytes() == original.tobytes()


@pytest.mark.parametrize('split', [False, True])
def test_drive_joint_requires_recorded_scalar_gripper_column(tmp_path, split):
    root = tmp_path / 'native'
    names = JOINTS + ['drive_joint']
    original = native_state(root, names, split=split)
    if split:
        info_path = root / 'meta/info.json'
        info = json.loads(info_path.read_text())
        info['features']['observation.state.left_arm']['units'] = 'rad'
        info_path.write_text(json.dumps(info))
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == ('joints' if split else 'none')
    if split:
        assert ctx['source']['state_columns'] == ['observation.state.left_arm', 'observation.state.left_gripper']
        assert ctx['state_value_roles'] == [None] * 6 + ['gripper']
        assert ctx['state_value_role_sources'][-1] == 'observation.state.left_gripper'
        assert ctx['state_identity']['names'] == names
        with np.load(path / 'state.npz') as z:
            assert z['state'].tobytes() == original.tobytes()
    else:
        assert next(s for s in ctx['signals'] if s['name'] == 'observation.state')['names'] == names


@pytest.mark.parametrize('declared_joints', [False, True])
def test_sliding_base_axes_do_not_establish_an_end_effector_pose(tmp_path, declared_joints):
    root = tmp_path / 'native'
    names = ['base_' + axis for axis in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']] + ['gripper']
    original = native_state(root, names, joint_names=names[:6] if declared_joints else None)
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == 'none' and ctx.get('state_note')
    field = next(s for s in ctx['signals'] if s['name'] == 'observation.state')
    assert field['names'] == names
    with np.load(path / 'state.npz') as z:
        assert z['state'].tobytes() == original.tobytes()
    assert formats.state_layout(7, 'teleop_arms', ['ee_' + a for a in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']]
                                + ['gripper'])[0] == 'ee_pose'


@pytest.mark.parametrize('key,names,expected', [
    ('joint_action', JOINTS + ['gripper'], True),
    ('joint_action/vector', JOINTS + ['gripper'], True),
    ('joint_action', None, False),
    ('commands/joint_action', JOINTS + ['gripper'], False),
    ('actions/joint_action', JOINTS + ['gripper'], False)])
def test_joint_action_native_state_requires_named_joint_and_gripper_authority(tmp_path, key, names, expected):
    path = tmp_path / 'controlled.h5'
    times = 1_790_000_000 + np.arange(8) / 30
    original = np.arange(56, dtype=np.float64).reshape(8, 7) / 100
    with h5py.File(path, 'w') as f:
        f['timestamps'] = times
        ds = f.create_dataset(key, data=original)
        if names:
            ds.attrs['names'] = names
            ds.attrs['units'] = ['rad'] * 6 + ['fraction']
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with h5py.File(path, 'r') as f:
        streams = formats.h5_streams(f, '')
        signals = formats.h5_signals(f, streams, times, 30, 8)
    native_name = next(s['name'] for s in streams['signal'] if s['path'] == key)
    assert signals.meta[native_name]['source'] == 'HDF5 dataset ' + key
    identity = {}
    state, action, got_names, source, note = formats.h5_state(signals, 'teleop_arms', times, identity_ctx=identity)
    if expected:
        assert state is not None and state.tobytes() == original.tobytes()
        assert source == key and got_names == names
        assert identity['state_identity']['source'] == key and identity['state_identity']['names'] == names
    else:
        assert state is None and native_name in signals
        if key == 'joint_action':
            assert note and note.why == 'layout'
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_pressure_columns_remain_original_signals(tmp_path):
    root = tmp_path / 'native'
    names = [f'pressure_{i}' for i in range(7)]
    original = native_state(root, names)
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == 'none' and ctx['state_why'] == 'layout'
    field = next(s for s in ctx['signals'] if s['name'] == 'observation.state')
    assert field['names'] == names
    with np.load(path / 'signals.npz') as values:
        np.testing.assert_array_equal(values[field['key']], original)


@pytest.mark.parametrize('names', [None, [''] * 7, ['x', 'y', 'z', 'yaw', 'pitch', 'roll', 'gripper'],
                                  ['x', 'y', 'z', 'rx', 'ry', 'rz', 'gripper']])
def test_unknown_or_noncanonical_axis_order_cannot_be_physical_state(names):
    assert formats.state_layout(7, 'teleop_arms', names)[0] == 'none'


@pytest.mark.parametrize('unit,expected', [(None, 'none'), ('rad', 'joints')])
def test_generic_joint_units_are_declared_without_rescaling_values(tmp_path, unit, expected):
    root = tmp_path / 'native'
    original = native_state(root, JOINTS + ['gripper'])
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    if unit:
        info['features']['observation.state']['units'] = [unit] * 6 + ['fraction']
    info_path.write_text(json.dumps(info))
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == expected
    with np.load(path / 'state.npz') as values:
        np.testing.assert_array_equal(values['state'], original)
    if expected == 'none':
        assert 'unit' in ctx['state_note']
        assert any(s['name'] == 'observation.state' for s in ctx['signals'])


def degree_state(root, readings, *, units=None, action_units=None):
    """A LeRobot episode whose state is the given readings and whose action is half a degree past it, both named six
    joints and a gripper."""
    native_state(root, JOINTS + ['gripper'])
    path = root / 'data/chunk-000/episode_000000.parquet'
    df = pd.read_parquet(path)
    df['observation.state'] = list(readings)
    df['action'] = list(readings + np.float32(.5))
    df.to_parquet(path)
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features']['action']['names'] = JOINTS + ['gripper']
    if units:
        info['features']['observation.state']['units'] = units
    if action_units:
        info['features']['action']['units'] = action_units
    info_path.write_text(json.dumps(info))


# per-joint ranges of a recorded Realman arm, which declares no unit: all six of its joints read past 2 pi
REALMAN = np.linspace([-25.2, 5.6, 20.1, -46.6, 5.5, -211.4, 0], [98.0, 79.6, 130.1, 81.3, 99.6, 120.0, 1],
                      8).astype(np.float32)


@pytest.mark.parametrize('units,basis', [(None, 'readings'), (['degrees'] * 6 + ['fraction'], 'declared')])
def test_joint_degrees_are_read_as_radians_when_declared_or_proven_by_the_readings(tmp_path, units, basis):
    root = tmp_path / 'native'
    degree_state(root, REALMAN, units=units, action_units=units)
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == 'joints' and 'state_note' not in ctx
    with np.load(path / 'state.npz') as values:
        for key, recorded in (('state', REALMAN), ('action', REALMAN + np.float32(.5))):
            expected = recorded.astype(np.float64)
            expected[:, :6] = np.deg2rad(expected[:, :6])
            np.testing.assert_allclose(values[key], expected, rtol=1e-6)
    proof = ctx['state_contract']
    assert proof['units'][:6] == ['rad'] * 6 and proof['unit_conversion']['basis'] == basis
    assert proof['unit_conversion']['columns'] == list(range(6))
    assert proof['action']['unit_conversion']['basis'] == basis
    from prepare.state_contract import prepared_state_problem
    assert prepared_state_problem(ctx, 7, has_action=True) is None


@pytest.mark.parametrize('change', ['two joints past one turn', 'a joint past two turns of degrees',
                                    'a joint with no reading', 'a declared radian'])
def test_joint_readings_that_do_not_prove_degrees_keep_the_state_as_recorded(tmp_path, change):
    readings = REALMAN.copy()
    units = None
    if change == 'two joints past one turn':
        readings[:, [0, 1, 3, 4]] = np.clip(readings[:, [0, 1, 3, 4]], -6, 6)
    elif change == 'a joint past two turns of degrees':
        readings[3, 5] = 900
    elif change == 'a joint with no reading':
        readings[:, 2] = np.nan
    else:
        units = ['rad'] * 6 + ['fraction']
    root = tmp_path / 'native'
    degree_state(root, readings, units=units)
    path, ctx = converted(root, tmp_path / 'prepared')
    if change == 'a declared radian':
        # the declared unit is kept, readings unchanged, but readings only degrees explain contradict it, so the state
        # is left out of kinematic checks with a reader issue saying so, never trusted as 33 turns of a joint
        assert ctx['state_kind'] == 'none' and 'state_contract' not in ctx
        assert 'radians' in ctx['state_note'] and any(s['name'] == 'observation.state' for s in ctx['signals'])
        assert [i['kind'] for i in ctx['reader_issues']] == ['state_unit_contradicted']
        with np.load(path / 'state.npz') as values:
            np.testing.assert_array_equal(values['state'], readings)
    else:
        assert ctx['state_kind'] == 'none' and 'state_contract' not in ctx
        assert any(s['name'] == 'observation.state' for s in ctx['signals'])


def test_two_arms_convert_only_when_each_arm_proves_degrees():
    names = [f'{side}_{name}' for side in ('left', 'right') for name in JOINTS + ['gripper']]
    both = np.concatenate([REALMAN, REALMAN], axis=1)
    found = formats.joint_degrees(both, names, None, [None] * 14, 'teleop_arms')
    assert found['columns'] == list(range(6)) + list(range(7, 13)) and found['joints_beyond_one_turn'] == [6, 6]
    small = both.copy()
    small[:, 7:13] = np.deg2rad(small[:, 7:13])
    assert formats.joint_degrees(small, names, None, [None] * 14, 'teleop_arms') is None
    # a name that carries its own unit declares it: names ending in _deg are degrees, names ending in _rad radians
    suffixed = [f'{name}_deg' if i % 7 < 6 else name for i, name in enumerate(names)]
    found = formats.joint_degrees(both, suffixed, None, formats.state_units({}, suffixed, 14), 'teleop_arms')
    assert found['basis'] == 'declared' and found['columns'] == list(range(6)) + list(range(7, 13))
    radians = [f'{name}_rad' if i % 7 < 6 else name for i, name in enumerate(names)]
    assert formats.joint_degrees(both, radians, None, formats.state_units({}, radians, 14), 'teleop_arms') is None


def h5_joints(path, readings, names, units):
    times = np.arange(len(readings)) / 30
    with h5py.File(path, 'w') as f:
        f['timestamps'] = times
        ds = f.create_dataset('state', data=readings)
        ds.attrs['names'] = names
        if units:
            ds.attrs['units'] = units
    with h5py.File(path, 'r') as f:
        signals = formats.h5_signals(f, formats.h5_streams(f, ''), times, 30, len(readings))
    identity = {}
    state, _, _, _, note = formats.h5_state(signals, 'teleop_arms', times, identity_ctx=identity)
    return state, note, identity, signals


def mcap_joints(path, readings, names, units):
    from test_formats import _json_mcap
    messages = []
    for index, row in enumerate(readings):
        message = {'name': list(names), 'position': [float(v) for v in row]}
        if units:
            message['units'] = list(units)
        messages.append((index / 30, message))
    _json_mcap(path, {'/left/joint_states': messages}, 1000)
    q = 1000 + np.arange(len(readings)) / 30
    streams = formats.mcap_joint_streams([path], q)
    issues = []
    state, _, note = formats.joint_state(streams, q, issues=issues)
    identity = {'state_kind': 'joints'}
    if state is not None:
        formats.record_joint_state_identity(identity, streams)
    return state, note, identity, issues


DEGREE_DECLARATIONS = {'declared by the units': (JOINTS + ['gripper'], ['degrees'] * 6 + ['fraction']),
                       'declared by the names': ([f'{j}_deg' for j in JOINTS] + ['gripper'], None)}


@pytest.mark.parametrize('reader', ['hdf5', 'mcap', 'lerobot'])
@pytest.mark.parametrize('declared', list(DEGREE_DECLARATIONS))
def test_every_reader_reads_declared_joint_degrees_as_radians(tmp_path, reader, declared):
    # degree conversion had run only for LeRobot: HDF5 and MCAP refused the same declared degrees as outside the
    # radian contract, and names ending in _deg declared nothing to any reader
    names, units = DEGREE_DECLARATIONS[declared]
    from prepare.state_contract import prepared_state_problem
    expected = REALMAN.astype(np.float64)
    expected[:, :6] = np.deg2rad(expected[:, :6])
    if reader == 'lerobot':
        root = tmp_path / 'native'
        degree_state(root, REALMAN, units=units)
        info_path = root / 'meta/info.json'
        info = json.loads(info_path.read_text())
        info['features']['observation.state']['names'] = names
        info_path.write_text(json.dumps(info))
        path, ctx = converted(root, tmp_path / 'prepared')
        assert ctx['state_kind'] == 'joints', ctx.get('state_note')
        with np.load(path / 'state.npz') as values:
            state = values['state']
        identity = ctx
    elif reader == 'hdf5':
        state, note, identity, _ = h5_joints(tmp_path / 'joints.h5', REALMAN, names, units)
        assert state is not None, note
        identity = {**identity, 'state_kind': 'joints'}
    else:
        state, note, identity, _ = mcap_joints(tmp_path / 'joint.mcap', REALMAN, names, units)
        assert state is not None, note
    np.testing.assert_allclose(state, expected, rtol=1e-6)
    proof = identity['state_contract']
    assert proof['units'][:6] == ['rad'] * 6 and proof['unit_conversion']['basis'] == 'declared'
    assert proof['unit_conversion']['columns'] == list(range(6))
    assert prepared_state_problem(identity, 7) is None


@pytest.mark.parametrize('reader', ['hdf5', 'mcap'])
def test_every_reader_leaves_out_declared_radians_its_readings_contradict(tmp_path, reader):
    # a declared radian was trusted whatever the readings, so a Realman arm's degrees reached kinematic checks as
    # 33 turns of a joint; the declared unit is kept, the state is left out of kinematic checks and an issue says why
    names, units = JOINTS + ['gripper'], ['rad'] * 6 + ['fraction']
    if reader == 'hdf5':
        state, note, _, signals = h5_joints(tmp_path / 'joints.h5', REALMAN, names, units)
        issues = signals.issues
        np.testing.assert_array_equal(signals['state'], REALMAN)
    else:
        state, note, _, issues = mcap_joints(tmp_path / 'joint.mcap', REALMAN, names, units)
    assert state is None and 'radians' in str(note)
    assert [i['kind'] for i in issues] == ['state_unit_contradicted']
    # radians whose readings stay within a turn are read as declared, with no issue
    small = REALMAN.copy()
    small[:, :6] = np.deg2rad(small[:, :6])
    if reader == 'hdf5':
        state, note, _, signals = h5_joints(tmp_path / 'small.h5', small, names, units)
        issues = signals.issues
    else:
        state, note, _, issues = mcap_joints(tmp_path / 'small.mcap', small, names, units)
    assert state is not None and not issues, note
    np.testing.assert_allclose(state, small, rtol=1e-6)


@pytest.mark.parametrize('units,expected', [(None, 'none'), (['mm'] * 3 + ['rad'] * 3 + ['m'], 'none'),
                                          (['m'] * 3 + ['rad'] * 3 + ['m'], 'ee_pose')])
def test_hdf_pose_requires_canonical_units(tmp_path, units, expected):
    names = ['ee_' + axis for axis in ['x', 'y', 'z', 'roll', 'pitch', 'yaw']] + ['gripper']
    path = tmp_path / 'pose.h5'
    times = np.arange(8) / 30
    original = np.arange(56, dtype=np.float32).reshape(8, 7) / 100
    with h5py.File(path, 'w') as f:
        f['timestamps'] = times
        ds = f.create_dataset('state', data=original)
        ds.attrs['names'] = names
        if units:
            ds.attrs['units'] = units
    with h5py.File(path, 'r') as f:
        streams = formats.h5_streams(f, '')
        signals = formats.h5_signals(f, streams, times, 30, 8)
    identity = {}
    state, _, _, _, note = formats.h5_state(signals, 'teleop_arms', times, identity_ctx=identity)
    assert (state is not None) == (expected != 'none')
    if state is None:
        assert note and 'unit' in note
        np.testing.assert_array_equal(signals['state'], original)
    else:
        np.testing.assert_array_equal(state, original)
        from prepare.state_contract import prepared_state_problem
        assert prepared_state_problem({**identity, 'state_kind': expected}, 7) is None
        assert identity['state_contract']['producer'] == 'prepare.formats:hdf5'


def test_unqualified_commands_remain_signals_beside_valid_state(tmp_path):
    root = tmp_path / 'native'
    native_state(root, JOINTS + ['gripper'])
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features']['observation.state']['units'] = ['rad'] * 6 + ['fraction']
    info['features']['action'].update(names=JOINTS + ['gripper'], units=['mm'] * 6 + ['fraction'])
    info_path.write_text(json.dumps(info))
    path, ctx = converted(root, tmp_path / 'prepared')
    assert ctx['state_kind'] == 'joints'
    with np.load(path / 'state.npz') as values:
        assert 'action' not in values
    assert any(s['name'] == 'action' for s in ctx['signals'])


@pytest.mark.parametrize('unit,expected', [(None, False), ('degrees', True), ('rad', True)])
def test_mcap_joint_units_survive_decode_and_native_name_order(tmp_path, unit, expected):
    from test_formats import _json_mcap
    path = tmp_path / 'joint.mcap'
    names = JOINTS + ['gripper']
    readings = []
    for index in range(8):
        order = list(range(7)) if index % 2 == 0 else list(reversed(range(7)))
        message = {'name': [names[i] for i in order], 'position': [index / 100 + i / 10 for i in order]}
        if unit:
            units = [unit] * 6 + ['fraction']
            message['units'] = [units[i] for i in order]
        readings.append((index / 30, message))
    _json_mcap(path, {'/left/joint_states': readings}, 1000)
    q = 1000 + np.arange(8) / 30
    streams = formats.mcap_joint_streams([path], q)
    state, action, note = formats.joint_state(streams, q)
    assert (state is not None) == expected
    if expected:
        recorded = np.arange(8)[:, None] / 100 + np.arange(6)[None, :] / 10
        # declared degrees are read in radians, as every reader reads them (qualified_state)
        np.testing.assert_allclose(state[:, :6], np.deg2rad(recorded) if unit == 'degrees' else recorded)
        from prepare.state_contract import prepared_state_problem
        ctx = {'state_kind': 'joints'}
        formats.record_joint_state_identity(ctx, streams)
        assert prepared_state_problem(ctx, 7) is None
        assert ctx['state_contract']['producer'] == 'prepare.formats:mcap-joints'
    else:
        assert note and 'unit' in note
        assert formats.state_fields(streams, state, action) == {}


@pytest.mark.parametrize('metadata,expected', [({}, 'joints'), ({'units': 'radians'}, 'joints'),
                                              ({'units': 'degrees'}, 'none'),
                                              ({'unit': 'rad', 'units': 'degrees'}, 'none')])
def test_unit_suffix_proof_rejects_conflicting_declarations(metadata, expected):
    names = [f'joint{i}_rad' for i in range(1, 7)] + ['gripper']
    assert formats.physical_state_layout(7, 'teleop_arms', names, metadata=metadata)[0] == expected


@pytest.mark.parametrize('rig', ['teleop_arms', 'handheld_gripper'])
def test_numbered_pose_axes_remain_unknown_on_every_rig(rig):
    names = [f'wrist_rot_{i}' for i in range(6)] + ['gripper']
    assert formats.physical_state_layout(7, rig, names, metadata={'units': 'rad'})[0] == 'none'


def test_fresh_generic_state_stamps_qualified_producer_contract(tmp_path):
    root = tmp_path / 'native'
    native_state(root, JOINTS + ['gripper'])
    info_path = root / 'meta/info.json'
    info = json.loads(info_path.read_text())
    info['features']['observation.state']['units'] = ['rad'] * 6 + ['fraction']
    info_path.write_text(json.dumps(info))
    _, ctx = converted(root, tmp_path / 'prepared')
    proof = ctx['state_contract']
    assert proof['version'] == 1 and proof['producer'] == 'prepare.formats:lerobot'
    assert proof['names'] == JOINTS + ['gripper']
    assert proof['units'] == ['rad'] * 6 + ['fraction']


def test_legacy_generic_state_needs_proof_but_named_producer_keeps_contract():
    from prepare.state_contract import prepared_state_problem
    generic = {'source': {'format': 'lerobot v2.1'}, 'state_kind': 'joints', 'profile': 'teleop_arms'}
    assert 'rebuild' in prepared_state_problem(generic, 14).lower()
    assert prepared_state_problem({**generic, 'state_kind': 'none'}, 14) is None
    assert prepared_state_problem({**generic, 'source': {'adapter': 'galaxea', 'format': 'lerobot v2.1'}}, 14) is None
    assert prepared_state_problem({'state_kind': 'joints', 'profile': 'teleop_arms'}, 14) is None
    assert prepared_state_problem({**generic, 'dataset': 'galaxea', 'robot_type': 'r1lite'}, 14)


def test_legacy_explicit_unit_suffix_contract_is_qualified_without_rewriting_context():
    import copy
    from prepare.state_contract import prepared_state_problem
    ctx = {'source': {'format': 'hdf5'}, 'state_kind': 'joints', 'profile': 'teleop_arms',
           'state_identity': {'source': 'qpos', 'names': [f'joint{i}_rad' for i in range(6)] + ['gripper']}}
    before = copy.deepcopy(ctx)
    assert prepared_state_problem(ctx, 7) is None
    assert ctx == before
    assert prepared_state_problem(ctx, 7, has_action=True)


@pytest.mark.parametrize('change', ['version', 'units', 'names', 'kind', 'action', 'width'])
def test_prepared_contract_rejects_stale_or_inconsistent_physical_claims(change):
    import copy
    from prepare.state_contract import declared_contract, prepared_state_problem
    proof = declared_contract('prepare.formats:lerobot', 'joints', JOINTS + ['gripper'], ['rad'] * 6 + ['fraction'])
    ctx = {'source': {'format': 'lerobot v2.1'}, 'state_kind': 'joints', 'state_contract': copy.deepcopy(proof)}
    dims, has_action = 7, False
    assert prepared_state_problem(ctx, dims) is None
    if change == 'version':
        ctx['state_contract']['version'] = 0
    elif change == 'units':
        ctx['state_contract']['units'][0] = 'degrees'
    elif change == 'names':
        ctx['state_contract']['names'] = [f'pressure{i}' for i in range(7)]
    elif change == 'kind':
        ctx['state_kind'] = 'ee_pose'
    elif change == 'action':
        has_action = True
    else:
        dims = 14
    before = copy.deepcopy(ctx)
    assert prepared_state_problem(ctx, dims, has_action=has_action)
    assert ctx == before


def test_physical_command_requires_its_own_prepared_contract():
    from prepare.state_contract import declared_contract, prepared_state_problem
    command = declared_contract('prepare.formats:lerobot', 'joints', JOINTS + ['gripper'], ['rad'] * 6 + ['fraction'])
    proof = declared_contract('prepare.formats:lerobot', 'joints', JOINTS + ['gripper'], ['rad'] * 6 + ['fraction'],
                              action=command)
    ctx = {'state_kind': 'joints', 'state_contract': proof}
    assert prepared_state_problem(ctx, 7, has_action=True) is None
    proof['action']['units'][0] = 'degrees'
    assert prepared_state_problem(ctx, 7, has_action=True)
