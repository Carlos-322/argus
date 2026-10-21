"""Versioned station rules, direct constrained prompting and strict result validation."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SPEC_DIR = ROOT / 'configs' / 'annotation_specs'
SPEC_IDS = ('heating-v2', 'heating-v2-complete')


def digest(definition):
    return hashlib.sha256(json.dumps(definition, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()


def resolve(value, *, current=True):
    """Snapshot rules in the policy; changed rules cannot silently reuse old requests."""
    if isinstance(value, str):
        if value not in SPEC_IDS:
            raise ValueError(f'Unknown annotation spec: {value}')
        definition = json.loads((SPEC_DIR / f'{value}.json').read_text(encoding='utf-8'))
        value = {'id': value, 'sha256': digest(definition), 'definition': definition}
    if not isinstance(value, dict) or value.get('id') not in SPEC_IDS:
        raise ValueError('Invalid annotation spec snapshot')
    definition = value.get('definition')
    if not isinstance(definition, dict) or digest(definition) != value.get('sha256'):
        raise ValueError('Annotation spec snapshot hash does not match')
    tasks = definition.get('subtasks') or []
    if (definition.get('id') != value['id'] or [s.get('id') for s in tasks] != [f'H{i}' for i in range(1, 7)]
            or len({s.get('name') for s in tasks}) != 6):
        raise ValueError('Heating spec must contain the six distinct H1-H6 templates')
    for task in tasks:
        if any(not isinstance(task.get(k), str) or not task[k].strip()
               for k in ('name', 'description', 'start', 'end', 'arm', 'object')):
            raise ValueError('Incomplete subtask rule')
    if current:
        installed = json.loads((SPEC_DIR / f'{value["id"]}.json').read_text(encoding='utf-8'))
        if digest(installed) != value['sha256']:
            raise ValueError(f'Annotation rules changed; use a new --out with --annotation-spec {value["id"]}')
        if hashlib.sha256((ROOT / definition['source']).read_bytes()).hexdigest() != definition['source_sha256']:
            raise ValueError('Source PDF changed; update the annotation rules before labeling')
    return copy.deepcopy(value)


def instructions(snapshot):
    definition = resolve(snapshot)['definition']
    if definition.get('require_complete_sequence'):
        return complete_instructions(definition)
    return '''规范驱动标注。你正在处理已经过上游质检的加热工位录制。
只按下面给定的六种子任务、模板和判断规则定位时间区间。它们是标注标准；文件名、
原始数据自带描述及历史标注不是正确答案。不得自行生成标签，也不要先自由分段再改名。
只输出实际发生且可定位的片段。允许缺失和重复，不能为凑齐六段补造动作。
同一轮加热用同一个正整数 cycle；下一轮加热才递增。重复按键仍属于该轮，保留异常。
时间为本次提供的录制时间轴上的秒数，区间使用[start_s,end_s)，必须有正长度、按时间排列、
不能重叠、不能越过录制末尾。共享切点不会把同一时刻算进两段。不拉伸片段填满空白。
H3/H4共享切点选首次离开触控板的时刻，H3不包含该时刻，最后接触的画面仍属于H3。
画面经过抽样，边界只是基于可见证据的候选值；看不清时写uncertain，不能声称已逐帧验证。
H4结束及H5开始的歧义必须标uncertain并说明。不得虚构精确帧号。

返回一个JSON对象，结构如下（时间仅为格式示例，不要复制）：
{
  "spec_id": "heating-v2",
  "task_summary": "中文概述实际观察到的加热流程或不匹配情况",
  "timeline": [{"cycle": 1, "subtask_id": "H1", "start_s": 0.0, "end_s": 1.0,
    "start_evidence": "开始边界的可见动作或状态", "end_evidence": "结束边界的可见动作或状态",
    "boundary_status": "observed或uncertain", "notes": "边界不确定原因；没有可为空"}],
  "unobserved_subtasks": [{"cycle": 1, "subtask_id": "H2", "reason": "为何没有可定位片段"}],
  "rule_flags": [{"code": "下方rule_codes之一", "reason": "可见证据和涉及的时间"}]
}
timeline每行只代表一个完整业务子任务或录制截断的可见部分，使用subtask_id，不自行输出名称。
程序会从规范填入固定名称、描述和执行手臂。每轮H1-H6中未出现在timeline的标签，都必须在
unobserved_subtasks解释一次；已经标出的标签不能同时列为缺失。完全不匹配时timeline为空，
cycle=1的六个标签都列为未观察到，并给出task_mismatch原因。rule_flags没有异常时为空数组。
不要输出自由生成的tasks、completion、质量评分或instruction_variants。

本次启用的规范（固定模板、原文边界和判断规则）：
''' + json.dumps(definition, ensure_ascii=False, indent=2)


def complete_instructions(definition):
    example = {
        'spec_id': definition['id'], 'task_summary': '概述本次完整六步加热流程',
        'timeline': [dict(cycle=1, subtask_id=f'H{i}', start_s=i-1, end_s=i,
                          start_evidence='实际观察或推断起点的依据', end_evidence='实际观察或推断终点的依据',
                          boundary_status='uncertain', notes='说明抽样或遮挡导致的不确定性')
                     for i in range(1, 7)],
        'unobserved_subtasks': [], 'rule_flags': []}
    return '''规范驱动标注：已质检、完整六步的单次加热任务。
用户明确提供前提：本次MCAP由专业数采员采集且已通过上游质检，包含H1至H6的完整流程。
你的任务是定位给定六个子任务的时间边界，不是重新判断是否应当接收这条数据。
必须按H1、H2、H3、H4、H5、H6顺序输出恰好六段，每个标签一次，cycle固定为1。
unobserved_subtasks必须为空。不能因为抽样画面、遮挡或不确定就删掉H5或其他步骤。
先结合所有视角及前后动作定位整段流程，再检查六段是否齐全、每个切点是否符合定义。
看不清时结合相邻动作推断候选边界，boundary_status写uncertain，在notes中解释依据和不确定性；
不要把推断说成直接观察，不要虚构可见证据或声称已逐帧验证。
六段之间必须连续，共享同一个切点：前段end_s等于后段start_s；不能留下未分配的中间区间。
H1之前和H6之后可以有准备或收尾画面，不强制从0开始或在录制末尾结束。
不能均分时长、套用示例时间，或为了凑数随意插入极短片段。
rule_flags无异常时为[]；有异常时每项必须是包含code和reason的对象，例如
{"code":"uncertain_boundary","reason":"说明具体片段和不确定的依据"}，不要仅输出异常名称字符串。
时间使用本次录制时间轴上的秒数，区间[start_s,end_s)，各段正长度、不重叠、不越过录制末尾。
H3结束于首次离开触控板，H4从该切点开始，触控板上的等待归H3。
H5包含取回纸杯后的等待，直到右臂开始从加热位回退；不能因左臂暂时不动或杯子被遮挡而漏标。
H6从右臂开始回退定位，不能推迟到右臂接近Home时才开始。
H4/H5交界仍有原规范歧义，必须写uncertain并说明候选边界。
若观察与完整流程前提存在冲突，在notes和rule_flags保留冲突证据，供人工复核，不能删段或自动排除数据。
文件名、原始描述不是标注答案；只使用给定模板，不输出自由生成的tasks、completion、质量评分或instruction_variants。

只返回JSON对象。下面仅展示完整六行格式，时间是假设值，绝不能照抄：
''' + json.dumps(example, ensure_ascii=False, indent=2) + '\n本次规范：\n' + json.dumps(definition, ensure_ascii=False, indent=2)


def validate(labels, snapshot, duration_s=None):
    """Reject invented labels or invalid spans, rather than silently repairing a paid reply."""
    spec = resolve(snapshot, current=False)
    definition = spec['definition']
    templates = {s['id']: s for s in definition['subtasks']}
    # Accept a known flag code without inventing its missing justification. Keep the
    # raw provider reply in the receipt; six-segment and time validation stay strict.
    labels = copy.deepcopy(labels)
    missing_reason = '模型仅返回异常名称，未提供具体理由；此项需人工复核。'
    if isinstance(labels.get('rule_flags'), list):
        labels['rule_flags'] = [
            {'code': flag, 'reason': missing_reason} if isinstance(flag, str) and flag in definition['rule_codes'] else flag
            for flag in labels['rule_flags']]
    if labels.get('spec_id') != spec['id']:
        raise ValueError('spec_id does not match the selected annotation spec')
    if not isinstance(labels.get('task_summary'), str) or not labels['task_summary'].strip():
        raise ValueError('task_summary is required')
    for field in ('timeline', 'unobserved_subtasks', 'rule_flags'):
        if not isinstance(labels.get(field), list) or not all(isinstance(r, dict) for r in labels[field]):
            raise ValueError(f'{field} must be a list of objects')
    for field in ('tasks', 'instruction_variants', 'completion'):
        if labels.get(field):
            raise ValueError(f'{field} is not part of the station annotation format')
    complete = definition.get('require_complete_sequence', False)
    if complete:
        if (labels['unobserved_subtasks'] or len(labels['timeline']) != 6
                or [row.get('subtask_id') for row in labels['timeline']] != list(templates)
                or any(type(row.get('cycle')) is not int or row['cycle'] != 1 for row in labels['timeline'])):
            raise ValueError('完整六段标注失败：必须依次输出H1至H6各一次，cycle=1，不允许缺段或重复；未作为成功标注保存')
    if duration_s is not None and (type(duration_s) not in (int, float)
                                 or not math.isfinite(duration_s) or duration_s <= 0):
        raise ValueError('Invalid recording duration for annotation validation')
    previous_end, previous_cycle, previous_id = 0.0, 0, 0
    seen, absent, review = set(), set(), []
    rows = copy.deepcopy(labels['timeline'])
    for row in rows:
        key, cycle = row.get('subtask_id'), row.get('cycle')
        if key not in templates or type(cycle) is not int or cycle < 1:
            raise ValueError('Each segment needs an allowed subtask_id and positive integer cycle')
        start, end = row.get('start_s'), row.get('end_s')
        if any(type(t) not in (float, int) or not math.isfinite(t) for t in (start, end)):
            raise ValueError('Segment times must be finite numbers')
        if start < previous_end or end <= start or (duration_s is not None and end > duration_s + 1e-6):
            raise ValueError('Segments must be ordered, nonoverlapping and within the recording')
        if complete and previous_id and abs(start - previous_end) > 1e-6:
            raise ValueError('完整六段标注失败：相邻子任务必须共享切点，不能留下未标注区间')
        if cycle < previous_cycle:
            raise ValueError('Cycles must be in recording order')
        if row.get('boundary_status') not in ('observed', 'uncertain'):
            raise ValueError('boundary_status must be observed or uncertain')
        for field in ('start_evidence', 'end_evidence'):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f'{field} is required')
        if not isinstance(row.get('notes', ''), str):
            raise ValueError('Segment notes must be text')
        if key in ('H4', 'H5'):
            row['boundary_status'] = 'uncertain'
            review.append(f'第{cycle}轮{key}：H4/H5交界按原文保留候选值，需复核。')
        if row['boundary_status'] == 'uncertain':
            review.append(f'第{cycle}轮{key}边界待复核：{row.get("notes") or "见起止证据"}')
        if cycle == previous_cycle and int(key[1:]) < previous_id:
            review.append(f'第{cycle}轮出现顺序回退或重试，保留原始顺序。')
        previous_end, previous_cycle, previous_id = end, cycle, int(key[1:])
        seen.add((cycle, key))
        template = templates[key]
        row.update(action=template['name'], arm=template['arm'], object=template['object'],
                   description=template['description'])
    for row in labels['unobserved_subtasks']:
        key, cycle = row.get('subtask_id'), row.get('cycle')
        if key not in templates or type(cycle) is not int or cycle < 1 or not isinstance(row.get('reason'), str) or not row['reason'].strip():
            raise ValueError('Invalid unobserved_subtasks entry')
        pair = (cycle, key)
        if pair in seen or pair in absent:
            raise ValueError('A subtask cannot be both observed and missing, or missing twice')
        absent.add(pair)
    cycles = {cycle for cycle, _ in seen | absent} or {1}
    if seen | absent != {(cycle, key) for cycle in cycles for key in templates}:
        raise ValueError('Every cycle must account for all six subtask types, without inventing segments')
    flags = labels['rule_flags']
    for flag in flags:
        if flag.get('code') not in definition['rule_codes'] or not isinstance(flag.get('reason'), str) or not flag['reason'].strip():
            raise ValueError('Invalid rule flag')
        if flag['reason'] == missing_reason:
            review.append(f'异常{flag["code"]}仅有名称，未提供理由，已兼容读取，需复核。')
    if not rows and not flags:
        raise ValueError('An empty timeline needs an explanation in rule_flags')
    excluded = not complete and any(f['code'] in ('multiple_presses', 'simultaneous_arms', 'task_mismatch') for f in flags)
    result = copy.deepcopy(labels)
    result['timeline'] = rows
    result['station_review'] = {
        'spec_id': spec['id'], 'spec_sha256': spec['sha256'], 'title': definition['title'],
        'status': 'excluded_by_spec' if excluded else 'needs_review',
        'boundary_precision': 'sampled_video_candidate',
        'notes': list(dict.fromkeys(review)),
        'unobserved_subtasks': copy.deepcopy(labels['unobserved_subtasks']), 'rule_flags': copy.deepcopy(flags)}
    return result
