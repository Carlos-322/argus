"""Request policy must reach actual frames and receipt identity, not just a recommendation table."""
import json

import pytest

from label import episode, harness, policy
from test_label import _packed_episode


@pytest.mark.parametrize('rig', list(policy.PERIODS))
def test_model_policy_controls_real_request_cadence_and_resume_identity(tmp_path, rig):
    ep, _ = _packed_episode(tmp_path)
    path = ep / 'context.json'
    ctx = json.loads(path.read_text())
    ctx['profile'] = rig
    ctx['annotation_policy'] = policy.resolve(ctx, model=policy.SOL61)
    path.write_text(json.dumps(ctx))
    request = episode.build_request(ep, cell_w=128)
    assert request['sampling'] == f'{rig}-every-1s'
    assert 'one every 1 s' in request['prompt']
    default = harness.input_identity(ep, model=policy.SOL61, reasoning='high', max_tokens=100,
                                     cell_w=128, example_dir=None)
    ctx['annotation_policy'] = policy.resolve(ctx, sample_fps=2)
    path.write_text(json.dumps(ctx))
    dense = episode.build_request(ep, cell_w=128)
    assert dense['sampling'] == f'{rig}-every-0.5s'
    assert len(dense['timesteps']) > len(request['timesteps'])
    assert ctx['annotation_policy']['part_max_s'] == 330
    changed = harness.input_identity(ep, model=policy.SOL61, reasoning='high', max_tokens=100,
                                     cell_w=128, example_dir=None)
    assert not harness.same_input_identity(default, changed)


def test_policy_preserves_declared_session_context_and_explicit_model_precedence():
    ctx = {'profile': 'handheld_gripper', 'instruction': 'Several assembly tasks at this station.', 'sessions': True}
    ctx['annotation_policy'] = policy.resolve(ctx, dataset_note='An open-ended assembly session.')
    resolved = policy.resolve(ctx, model=policy.SOL61)
    assert resolved['task_mode'] == 'session' and resolved['reasoning'] == 'high'
    assert resolved['dataset_note'] == 'An open-ended assembly session.'
    assert policy.resolve(ctx, task_mode='episode')['task_mode'] == 'episode'
    assert policy.resolve({'profile': 'handheld_gripper', 'instruction': ''})['task_mode'] == 'episode'


def test_inherited_provider_policy_is_checked_before_routing(tmp_path, monkeypatch):
    ep, _ = _packed_episode(tmp_path)
    path = ep / 'context.json'
    ctx = json.loads(path.read_text())
    ctx['annotation_policy'] = policy.resolve(ctx, model='other/vendor-model')
    path.write_text(json.dumps(ctx))
    monkeypatch.setattr(harness, 'route_width', lambda *a, **k: pytest.fail('incompatible provider was routed'))
    with pytest.raises(ValueError, match='requires an OpenRouter key'):
        harness.label_episode(ep, tmp_path / 'label.json', model=None, reasoning=None,
                              api_key='sk-proj-test', max_tokens=100, timeout=10)
    assert not (tmp_path / 'label.json').exists()


def test_the_paid_calls_beside_the_annotation_are_on_by_default_and_written_only_when_turned_off(tmp_path):
    ctx = {'profile': 'teleop_arms'}
    default = policy.resolve(ctx)
    assert 'inspection_rounds' not in default and 'field_dictionary' not in default
    assert policy.resolve(ctx, inspection_rounds=policy.INSPECTION_ROUNDS, field_dictionary=True) == default
    off = policy.resolve(ctx, inspection_rounds=0, field_dictionary=False)
    assert off == {**default, 'inspection_rounds': 0, 'field_dictionary': False}
    assert policy.resolve({**ctx, 'annotation_policy': off}) == off             # a saved choice is kept on resume
    assert policy.resolve({**ctx, 'annotation_policy': off}, inspection_rounds=2, field_dictionary=True) == default
    for bad in (3, -1, 1.0, True, '0'):
        with pytest.raises(ValueError):
            policy.resolve(ctx, inspection_rounds=bad)
    with pytest.raises(ValueError):
        policy.resolve(ctx, field_dictionary='off')
    ep, _ = _packed_episode(tmp_path)
    kw = dict(model=policy.SOL61, reasoning='high', max_tokens=100, cell_w=128, example_dir=None)
    same = harness.input_identity(ep, **kw, inspection_rounds=policy.INSPECTION_ROUNDS)
    assert harness.same_input_identity(harness.input_identity(ep, **kw), same)
    assert not harness.same_input_identity(same, harness.input_identity(ep, **kw, inspection_rounds=0))
