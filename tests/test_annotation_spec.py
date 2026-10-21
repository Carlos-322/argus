"""The supplied station contract must control prompts, parsing and saved result reuse."""
import copy
import io
import json

import pytest

from label import annotation_spec as spec, harness, policy


def answer():
    return {'spec_id': 'heating-v2', 'task_summary': '左右臂完成加热操作。',
            'timeline': [{'cycle': 1, 'subtask_id': f'H{i}', 'start_s': (i - 1) * .5, 'end_s': i * .5,
                          'start_evidence': '观察到符合起点规则的动作', 'end_evidence': '观察到符合终点规则的状态',
                          'boundary_status': 'observed', 'notes': ''} for i in range(1, 7)],
            'unobserved_subtasks': [], 'rule_flags': []}


def parse(value):
    return harness.parse_response(json.dumps(value), annotation_spec=spec.resolve('heating-v2'), duration_s=4.)


def test_six_templates_replace_model_names_and_preserve_boundary_review():
    value = answer()
    value['timeline'][2]['action'] = 'invented label'
    labels, ok = parse(value)
    assert ok
    assert [r['action'] for r in labels['timeline']] == [s['name'] for s in spec.resolve('heating-v2')['definition']['subtasks']]
    assert labels['timeline'][3]['boundary_status'] == labels['timeline'][4]['boundary_status'] == 'uncertain'
    assert labels['station_review']['status'] == 'needs_review'
    assert parse(labels) == (labels, True)


@pytest.mark.parametrize('problem', ['unknown', 'overlap', 'range', 'nan', 'boolean', 'missing', 'duplicate_missing',
                                    'wrong_spec', 'no_evidence', 'free_form'])
def test_invalid_contract_is_not_silently_repaired(problem):
    value = answer()
    if problem == 'unknown': value['timeline'][0]['subtask_id'] = 'wait'
    if problem == 'overlap': value['timeline'][1]['start_s'] = .1
    if problem == 'range': value['timeline'][-1]['end_s'] = 9
    if problem == 'nan': value['timeline'][0]['start_s'] = float('nan')
    if problem == 'boolean': value['timeline'][0]['start_s'] = False
    if problem == 'missing': value['timeline'].pop()
    if problem == 'duplicate_missing': value['unobserved_subtasks'] = [{'cycle': 1, 'subtask_id': 'H1', 'reason': 'missing'}]
    if problem == 'wrong_spec': value['spec_id'] = 'old-five-step'
    if problem == 'no_evidence': value['timeline'][0]['start_evidence'] = ''
    if problem == 'free_form': value['tasks'] = [{'task': 'new label'}]
    labels, ok = parse(value)
    assert not ok and labels['_raw'] and labels['_parse_error']


def test_missing_and_repeated_subtasks_are_preserved_without_forcing_six_segments():
    value = answer()
    value['timeline'] = value['timeline'][:3]
    repeated = copy.deepcopy(value['timeline'][-1])
    repeated.update(start_s=1.5, end_s=2.0)
    value['timeline'].append(repeated)
    value['unobserved_subtasks'] = [{'cycle': 1, 'subtask_id': f'H{i}', 'reason': '未录到'} for i in (4, 5, 6)]
    value['rule_flags'] = [{'code': 'multiple_presses', 'reason': '1.0至2.0秒有两次按键'}]
    labels, ok = parse(value)
    assert ok and len(labels['timeline']) == 4
    assert labels['station_review']['status'] == 'excluded_by_spec'


def test_spec_snapshot_and_changed_rules_are_not_interchangeable(tmp_path, monkeypatch):
    snapshot = spec.resolve('heating-v2')
    changed = copy.deepcopy(snapshot['definition'])
    changed['subtasks'][0]['end'] = 'different rule'
    (tmp_path / 'heating-v2.json').write_text(json.dumps(changed), encoding='utf-8')
    monkeypatch.setattr(spec, 'SPEC_DIR', tmp_path)
    with pytest.raises(ValueError, match='rules changed'):
        spec.resolve(snapshot)
    # Historical output remains interpretable using its original snapshot.
    assert spec.validate(answer(), snapshot, 4)['station_review']['spec_sha256'] == snapshot['sha256']


def test_spec_policy_does_not_change_generic_defaults_or_allow_technical_parts():
    generic = policy.resolve({'profile': 'teleop_arms'})
    selected = policy.resolve({'profile': 'teleop_arms'}, annotation_spec='heating-v2')
    assert 'annotation_spec' not in generic
    assert selected['sample_fps'] == 4 and selected['inspection'] == 'off' and selected['field_dictionary'] is False
    assert policy.resolve({'profile': 'teleop_arms', 'annotation_policy': selected}) == selected
    with pytest.raises(ValueError, match='unsplit'):
        policy.resolve({'profile': 'teleop_arms', 'piece': {'index': 1}}, annotation_spec='heating-v2')


def test_free_form_reply_cannot_be_reused_as_station_annotation():
    record = {'parse_ok': True, 'labels': {'task_summary': 'free-form result'},
              'annotation_policy': policy.resolve({'profile': 'teleop_arms'}, annotation_spec='heating-v2')}
    assert not harness.annotation_complete(record)


def complete_answer():
    value = answer()
    value['spec_id'] = 'heating-v2-complete'
    return value


@pytest.mark.parametrize('problem', ['missing_h5', 'gap', 'duplicate', 'out_of_order', 'two_cycles'])
def test_complete_mode_rejects_incomplete_or_disconnected_reply(problem):
    value = complete_answer()
    if problem == 'missing_h5':
        value['timeline'].pop(4)
        value['unobserved_subtasks'] = [{'cycle': 1, 'subtask_id': 'H5', 'reason': '杯子被遮挡'}]
    if problem == 'gap': value['timeline'][4]['start_s'] += .1
    if problem == 'duplicate': value['timeline'][4]['subtask_id'] = 'H4'
    if problem == 'out_of_order':
        value['timeline'][3]['subtask_id'], value['timeline'][4]['subtask_id'] = 'H5', 'H4'
    if problem == 'two_cycles': value['timeline'][-1]['cycle'] = 2
    snapshot = spec.resolve('heating-v2-complete')
    labels, ok = harness.parse_response(json.dumps(value), annotation_spec=snapshot, duration_s=4)
    assert not ok and '完整六段标注失败' in labels['_parse_error']
    assert not harness.annotation_complete({'parse_ok': True, 'labels': value,
               'annotation_policy': {'annotation_spec': snapshot}, 'config': {'annotation_duration_s': 4}})


def test_complete_mode_keeps_uncertain_boundaries_and_flags_without_exclusion():
    value = complete_answer()
    value['timeline'][4].update(boundary_status='uncertain', notes='遮挡，结合前后动作定位候选值')
    value['rule_flags'] = [{'code': 'simultaneous_arms', 'reason': '疑似双臂同时移动，待复核'}]
    labels = spec.validate(value, spec.resolve('heating-v2-complete'), 4)
    assert len(labels['timeline']) == 6
    assert labels['station_review']['status'] == 'needs_review'
    assert labels['timeline'][4]['boundary_status'] == 'uncertain'
    prompt = spec.instructions(spec.resolve('heating-v2-complete'))
    assert '恰好六段' in prompt and '允许缺失和重复' not in prompt
    assert '"subtask_id": "H6"' in prompt and '"unobserved_subtasks": []' in prompt


def test_code_only_flags_are_read_without_changing_times_or_inventing_evidence():
    value = complete_answer()
    value['rule_flags'] = ['uncertain_boundary', 'multiple_presses']
    snapshot = spec.resolve('heating-v2-complete')
    result = spec.validate(value, snapshot, 4)
    assert value['rule_flags'] == ['uncertain_boundary', 'multiple_presses']
    assert [(s['start_s'], s['end_s']) for s in result['timeline']] == [
        (s['start_s'], s['end_s']) for s in value['timeline']]
    assert all('未提供具体理由' in f['reason'] for f in result['rule_flags'])
    assert result['station_review']['status'] == 'needs_review'
    assert spec.validate(result, snapshot, 4) == result
    value['timeline'].pop(4)
    with pytest.raises(ValueError, match='完整六段标注失败'):
        spec.validate(value, snapshot, 4)


@pytest.mark.parametrize('flag', ['invented_flag', 42, {'code': 'multiple_presses', 'reason': ''}])
def test_flag_compatibility_does_not_accept_unknown_or_invalid_values(flag):
    value = complete_answer()
    value['rule_flags'] = [flag]
    with pytest.raises(ValueError):
        spec.validate(value, spec.resolve('heating-v2-complete'), 4)


@pytest.mark.parametrize('spec_id', ['heating-v2', 'heating-v2-complete'])
def test_deepseek_request_to_board_uses_fixed_rules_and_hides_old_answers(tmp_path, monkeypatch, spec_id):
    from test_label import _packed_episode
    from board.to_board import convert
    from label import providers, episode
    from label.request_attestation import only_implementation_changed
    monkeypatch.setenv('ARGUS_PROVIDER', 'deepseek')
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'sk-ds-test')
    ep, _ = _packed_episode(tmp_path)
    context = json.loads((ep / 'context.json').read_text())
    context['instruction'] = 'DO_NOT_SEND_KNOWN_TIMES'
    context['annotation_subtasks'] = [{'label': 'DO_NOT_SEND_KNOWN_TIMES', 'start_s': 1, 'end_s': 2}]
    (ep / 'context.json').write_text(json.dumps(context))
    identity_args = dict(model=providers.DEEPSEEK_MODEL, reasoning='high', max_tokens=2000, cell_w=128, example_dir=None)
    old = harness.input_identity(ep, **identity_args)
    policy.persist(ep.parent, annotation_spec=spec_id)
    current = harness.input_identity(ep, **identity_args)
    assert not only_implementation_changed(old, current)
    calls = []
    def provider(request, timeout):
        assert request.full_url == providers.DEEPSEEK_URL
        body = json.loads(request.data)
        text = '\n'.join(p.get('text', '') for p in body['messages'][0]['content'])
        assert 'DO_NOT_SEND_KNOWN_TIMES' not in text
        assert 'LEAN' not in text
        assert 'SENSOR MECHANICS ANALYSIS CONTRACT' not in text
        assert '奶茶：离开触控板' in text and '规范驱动标注' in text
        calls.append(body)
        response = answer() if spec_id == 'heating-v2' else complete_answer()
        return io.BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps(response)}}],
                                    'usage': {'prompt_tokens': 100, 'completion_tokens': 100}}).encode())
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    out = tmp_path / 'out'
    args = dict(keys=['sk-ds-test'], concurrency=1, force=False, model=providers.DEEPSEEK_MODEL,
                reasoning='high', max_tokens=2000, cell_w=128, timeout=1, max_spend=1)
    assert harness.run_batch([ep], out, **args) == 0
    result = json.loads((out / f'{ep.name}.json').read_text())
    assert result['parse_ok'] and len(calls) == 1
    assert harness.annotation_complete(result)
    board = convert(result, 'heating')
    assert len(board['event_labels']) == 6
    assert board['event_labels'][3]['verb_class'] == '奶茶：离开触控板'
    assert board['station_review']['spec_id'] == spec_id
    assert board['station_segments'][3]['start_evidence']
    assert harness.run_batch([ep], out, **args) == 0
    assert len(calls) == 1
