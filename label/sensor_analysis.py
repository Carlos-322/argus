"""Independent regional analysis using the same source-derived measurements as the local pass."""
from __future__ import annotations

import copy
import hashlib
import json

from label import regional_analysis, sensor_copy, sensor_evidence as se

VERSION = 1
PREFIX = 'mechanics:'
HEADER = '''Analyze the supplied regional sensor responses jointly with estimated configuration and camera evidence.
Find useful information about relative loading and its distribution beyond visible contact, pickup or holding. Compare coherent regional trajectories with the available configuration estimates and camera images. Assess what the responses add beyond configuration changes, preserving the distinction between recorded measurements and physical interpretation.
Use only the supplied source semantics, response convention, physical layout and calibration. Unknown polarity still permits raw rises and falls. An initial reference is not a verified unloaded baseline. Repeated pose estimates may be stale. Consider bending, preload, changing fit and tracking error when relevant; do not claim calibrated force, slip or an unseen manipulation outcome from response changes alone.
This independent pass receives regional and configuration derivations. Detailed component trajectories are analyzed separately. Do not invent or cite those omitted readings. Camera labels identify streams, not verified mounting. Use exact supplied measurement times and preserve the clock qualifications. Images provide discrete context, not continuous observation.
Return only a JSON object with sensor_findings. An empty sensor_findings array is valid. Preserve distinct supported comparisons, including different behavior across regions or response changes beyond estimated configuration changes. No visual timeline, contact verdicts or other annotation fields are requested in this pass.
'''
IMAGE_CAPTIONS = ('=== grid t=', '=== detail view,', '=== contact ',
                  'TACTILE ARRAY SNAPSHOTS |', 'RECORDED CAMERA CONTEXT PAIRED TO ARRAY SNAPSHOTS |')
LOCAL_SCOPE = '''LOCAL ARRAY ANALYSIS
Regional response and configuration comparisons have an independent analysis. For array sensor findings in this request, examine local redistribution, divergent neighboring components, and coherent localized sustained or pulsed responses that add information beyond broad regional loading. Regional trajectories remain useful comparison context. Preserve distinct supported local patterns without repeating a finding that only reports gross regional loading. Do not invent a local pattern or require a finding. All visual annotation and other modality contracts remain in effect.
'''


def _packed(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode()


def _digest(request):
    source = {key: request[key] for key in ('version', 'config', 'sensor_evidence', 'content_sha256')}
    return hashlib.sha256(_packed(source)).hexdigest()


def local_content(content):
    """Give the local pass its original independent objective, without another pass's answers."""
    result = copy.deepcopy(content)
    if result and result[0].get('type') == 'text':
        result[0]['text'] = LOCAL_SCOPE + result[0]['text']
    else:
        result.insert(0, {'type': 'text', 'text': LOCAL_SCOPE})
    return result


def build(content, sensor_doc, *, config, ep=None):
    """Separate regional reasoning when a dense local component view also exists."""
    if not sensor_copy.shared(config.get('sensor_copy_contract_version')):
        return None
    dense = any((s.get('array_evidence') or {}).get('constituents')
                or (s.get('array_evidence') or {}).get('native_constituents')
                for s in sensor_doc.get('sensors') or [])
    regional_ids = {d['id'] for d in sensor_doc.get('derivations') or [] if d.get('kind') == 'regional_response'}
    rows = [r for r in sensor_doc.get('series') or [] if r.get('derivation_id')]
    if not dense or not any(r.get('derivation_id') in regional_ids for r in rows):
        return None
    selected = copy.deepcopy(sensor_doc)
    sensor_ids = {r['sensor_id'] for r in rows}
    selected['series'] = copy.deepcopy(rows)
    selected['sensors'] = [copy.deepcopy(s) for s in sensor_doc['sensors'] if s['id'] in sensor_ids]
    for sensor in selected['sensors']:
        sensor.pop('array_evidence', None)
        sensor.pop('array_supplied_columns', None)
        sensor.pop('marker_motion', None)
        sensor.pop('times', None)
    selected['findings'], selected['unbound_findings'] = [], []
    selected['limitations'] = [*selected.get('limitations', []),
        'Independent regional pass: component matrices, individual-cell traces and raw inspection samples are not supplied. '
        'They remain available to the separate local analysis; their readings cannot support citations in this pass.']
    selected = regional_analysis.augment(selected, ep=ep)
    images, image_sources = [], []
    for index, part in enumerate(content):
        if part.get('type') != 'image_url' or not index:
            continue
        caption = content[index - 1]
        if caption.get('type') != 'text' or not caption.get('text', '').startswith(IMAGE_CAPTIONS):
            continue
        images.extend([copy.deepcopy(caption), copy.deepcopy(part)])
        image_sources.append({'source_part_index': index,
            'part_sha256': hashlib.sha256(_packed(part)).hexdigest(),
            'caption_sha256': hashlib.sha256(caption['text'].encode()).hexdigest()})
    if not images:
        return None
    parts = [{'type': 'text', 'text': HEADER}, {'type': 'text', 'text': se.prompt(selected)}, *images]
    settings = {key: copy.deepcopy(config[key]) for key in (
        'model', 'reasoning', 'max_tokens', 'input_identity', 'sensor_copy_contract_version') if key in config}
    request = {'version': VERSION, 'config': settings, 'sensor_evidence': selected,
               'content_sha256': hashlib.sha256(_packed(parts)).hexdigest()}
    return {**request, 'content': parts, 'digest': _digest(request),
            'copy_contract_version': config['sensor_copy_contract_version'],
            'source_images': image_sources,
            'omitted_series_ids': [r['id'] for r in sensor_doc.get('series') or [] if not r.get('derivation_id')]}


def bind(request, receipt):
    """Only the exact regional request may authorize this pass's findings."""
    if request.get('digest') != _digest(request) or receipt.get('digest') != request['digest']:
        raise ValueError('Regional sensor analysis request proof changed')
    if receipt.get('state') != 'received' or receipt.get('finish_reason') != 'stop':
        raise ValueError('Regional sensor analysis has no complete received response')
    text = receipt.get('response_text')
    if not isinstance(text, str):
        raise ValueError('Regional sensor analysis has no retained response text')
    text = text.strip()
    if text.startswith('```') and text.endswith('```'):
        text = text.split('\n', 1)[1].rsplit('```', 1)[0]
    labels = json.loads(text)
    if not isinstance(labels, dict) or not isinstance(labels.get('sensor_findings'), list):
        raise ValueError('Regional sensor analysis must return a sensor_findings array')
    result = se.bind(request['sensor_evidence'], labels['sensor_findings'],
                     copy_contract_version=request.get('copy_contract_version'))
    provenance = {'model': receipt.get('model'), 'generation_id': receipt.get('generation_id'),
                  'method': 'independent regional sensor analysis', 'request_digest': request['digest']}
    result['provenance'] = provenance
    for finding in result['findings']:
        finding['annotation_provenance'] = copy.deepcopy(provenance)
    return result


def attach(result, bound_stage):
    """Join independently bound findings without replacing either raw model output."""
    stage = copy.deepcopy(bound_stage)
    for sensor in stage.get('sensors') or []:
        sensor['id'] = PREFIX + sensor['id']
        if sensor.get('derivations'):
            sensor['derivations'] = [se._shift_derivation(d, PREFIX, 0) for d in sensor['derivations']]
    for row in stage.get('series') or []:
        row['id'], row['sensor_id'] = PREFIX + row['id'], PREFIX + row['sensor_id']
        if row.get('derivation_id'):
            row['derivation_id'] = PREFIX + row['derivation_id']
    stage['derivations'] = [se._shift_derivation(d, PREFIX, 0) for d in stage.get('derivations') or []]
    for finding in [*stage.get('findings', []), *stage.get('withheld_findings', [])]:
        for ref in finding.get('evidence') or []:
            ref['sensor_id'] = PREFIX + ref['sensor_id']
            if ref.get('series_id'):
                ref['series_id'] = PREFIX + ref['series_id']
    for row in stage.get('unbound_findings') or []:
        row['annotation_provenance'] = copy.deepcopy(stage['provenance'])
    target = result.setdefault('sensor_evidence', {'version': 1})
    for key in ('sensors', 'series', 'derivations', 'findings', 'withheld_findings', 'unbound_findings'):
        target.setdefault(key, []).extend(stage.get(key) or [])
    target.setdefault('provenance', {}).setdefault('analyses', []).append({
        **copy.deepcopy(stage['provenance']), 'limitations': copy.deepcopy(stage.get('limitations') or [])})


def without_attached(doc):
    """Reparse each received response against its own saved evidence, once."""
    result = copy.deepcopy(doc)
    for key, field in (('sensors', 'id'), ('series', 'sensor_id'), ('derivations', 'sensor_id')):
        if key in result:
            result[key] = [row for row in result[key] or [] if not str(row.get(field, '')).startswith(PREFIX)]
    for key in ('findings', 'withheld_findings'):
        if key in result:
            result[key] = [f for f in result[key] or []
                           if not any(str(ref.get('sensor_id', '')).startswith(PREFIX) for ref in f.get('evidence') or [])]
    if 'unbound_findings' in result:
        result['unbound_findings'] = [row for row in result['unbound_findings'] or []
                                     if (row.get('annotation_provenance') or {}).get('method') != 'independent regional sensor analysis']
    (result.get('provenance') or {}).pop('analyses', None)
    return result
