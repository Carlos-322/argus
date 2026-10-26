"""New visible boundaries must reach the model and survive validation unchanged."""
import copy
import hashlib
import json

import pytest

from label import annotation_spec as spec
from test_annotation_spec import answer

SPEC = 'heating-review-20261026'


def review_answer():
    value = answer()
    value['spec_id'] = SPEC
    value['timeline'][3]['end_evidence'] = 'chest_wide_cam：整个左夹爪最高点已低于显示屏下边缘'
    return value


@pytest.mark.parametrize('start', [2.25, 2.5, 2.75])
def test_h5_h6_can_overlap_touch_or_have_gap_without_moving_times(start):
    value = review_answer()
    value['timeline'][-1]['start_s'] = start
    if start < 2.5:
        value['rule_flags'] = [{'code': 'simultaneous_arms', 'reason': '两臂在2.25到2.5秒同时回退'}]
    result = spec.validate(value, spec.resolve(SPEC), 4)
    assert [(r['start_s'], r['end_s']) for r in result['timeline']] == [
        (r['start_s'], r['end_s']) for r in value['timeline']]
    assert result['timeline'][3]['boundary_status'] == 'observed'
    assert result['timeline'][4]['boundary_status'] == 'observed'
    assert result['station_review']['status'] == 'needs_review'
    assert spec.validate(result, spec.resolve(SPEC), 4) == result


def test_invisible_steps_are_explained_not_filled():
    value = review_answer()
    value['timeline'] = value['timeline'][:3]
    value['unobserved_subtasks'] = [{'cycle': 1, 'subtask_id': f'H{i}', 'reason': '录制在离板前结束'} for i in (4, 5, 6)]
    value['rule_flags'] = [{'code': 'incomplete_recording', 'reason': '只录到H3'}]
    result = spec.validate(value, spec.resolve(SPEC), 4)
    assert len(result['timeline']) == 3
    assert result['unobserved_subtasks'] == value['unobserved_subtasks']


@pytest.mark.parametrize('problem', ['h4_gap', 'h4_overlap', 'h6_over_h4', 'negative', 'range', 'nan', 'missing_reason', 'missing_evidence', 'different_cycle_overlap'])
def test_new_rules_still_reject_invalid_boundaries(problem):
    value = review_answer()
    if problem == 'h4_gap': value['timeline'][3]['start_s'] += .1
    if problem == 'h4_overlap': value['timeline'][3]['start_s'] -= .1
    if problem == 'h6_over_h4':
        value['timeline'][-1]['start_s'] = 1.8
        value['timeline'].sort(key=lambda r: r['start_s'])
    if problem == 'negative': value['timeline'][0]['start_s'] = -.1
    if problem == 'range': value['timeline'][-1]['end_s'] = 8
    if problem == 'nan': value['timeline'][-1]['end_s'] = float('nan')
    if problem == 'missing_reason': value['timeline'].pop()
    if problem == 'missing_evidence': value['timeline'][3]['end_evidence'] = ''
    if problem == 'different_cycle_overlap':
        value['timeline'][-1].update(cycle=2, start_s=2.25)
    with pytest.raises(ValueError):
        spec.validate(value, spec.resolve(SPEC), 4)


def test_new_source_is_in_prompt_and_labels_are_unchanged():
    current = spec.resolve(SPEC)['definition']
    old = spec.resolve('heating-v2')['definition']
    assert [r['name'] for r in current['subtasks']] == [r['name'] for r in old['subtasks']]
    source = (spec.ROOT / current['source']).read_bytes().replace(b'\r\n', b'\n')
    assert hashlib.sha256(source).hexdigest() == current['source_sha256']
    assert source.decode('utf-8-sig') == current['source_text']
    prompt = spec.instructions(SPEC)
    assert json.dumps(current['source_text'], ensure_ascii=False) in prompt
    for rule in ('不强行补齐六段', '整个左夹爪', '机位名称', 'H5/H6 不强制衔接', '不能按时长认定'):
        assert rule in prompt


def test_source_edits_block_current_requests_but_historical_snapshot_remains_readable(tmp_path, monkeypatch):
    snapshot = spec.resolve(SPEC)
    source = tmp_path / snapshot['definition']['source']
    source.parent.mkdir(parents=True)
    source.write_text('modified source', encoding='utf-8')
    monkeypatch.setattr(spec, 'ROOT', tmp_path)
    with pytest.raises(ValueError, match='Source document changed'):
        spec.resolve(snapshot)
    assert spec.validate(review_answer(), snapshot, 4)['station_review']['spec_sha256'] == snapshot['sha256']


def test_revised_rules_cannot_reuse_old_requests(tmp_path, monkeypatch):
    snapshot = spec.resolve(SPEC)
    changed = copy.deepcopy(snapshot['definition'])
    changed['subtasks'][3]['end'] = 'another boundary'
    (tmp_path / f'{SPEC}.json').write_text(json.dumps(changed), encoding='utf-8')
    monkeypatch.setattr(spec, 'SPEC_DIR', tmp_path)
    with pytest.raises(ValueError, match='rules changed'):
        spec.resolve(snapshot)


def test_markdown_line_endings_do_not_change_rule_identity(tmp_path, monkeypatch):
    snapshot = spec.resolve(SPEC)
    source = tmp_path / snapshot['definition']['source']
    source.parent.mkdir(parents=True)
    source.write_bytes(snapshot['definition']['source_text'].replace('\n', '\r\n').encode('utf-8'))
    monkeypatch.setattr(spec, 'ROOT', tmp_path)
    assert spec.resolve(snapshot) == snapshot
