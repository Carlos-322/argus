"""Admit only source-owned images that are present in the independent request."""
from __future__ import annotations

import base64
import copy
import hashlib
import math
from pathlib import Path

from label import sensor_evidence as se
from label.dictionary_context import prepared_interpretation


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _images(parts):
    result = []
    for index in range(1, len(parts)):
        caption, part = parts[index - 1:index + 1]
        if caption.get('type') != 'text' or part.get('type') != 'image_url':
            continue
        url = (part.get('image_url') or {}).get('url', '')
        if not url.startswith('data:image/') or ';base64,' not in url:
            continue
        try:
            raw = base64.b64decode(url.split(',', 1)[1], validate=True)
        except (ValueError, TypeError):
            continue
        result.append({'caption': caption['text'], 'jpeg_sha256': _sha(raw),
                       'caption_sha256': _sha(caption['text'].encode())})
    return result


def _arrays(original, selected, images):
    from label import tactile_arrays
    supplied = {(row['caption_sha256'], row['jpeg_sha256']) for row in images}
    owners = {row['id']: row for row in original.get('sensors') or []}
    for sensor in selected.get('sensors') or []:
        owner = owners[sensor['id']]
        array = owner.get('array_evidence')
        if not array:
            continue
        frames, proofs = [], []
        # Re-render each canonical sheet independently. A renderer grouping change
        # fails closed unless the same sheet bytes and caption are actually sent.
        for start in range(0, len(array['frames']), 8):
            sheet = copy.deepcopy(owner)
            sheet['array_evidence']['frames'] = array['frames'][start:start + 8]
            rendered = tactile_arrays.render(sheet)
            if len(rendered) != 2:
                continue
            proof = _images(rendered)[0]
            if (proof['caption_sha256'], proof['jpeg_sha256']) in supplied:
                frames.extend(copy.deepcopy(sheet['array_evidence']['frames']))
                proofs.append(proof)
        if not frames:
            continue
        image_array = copy.deepcopy(array)
        image_array.pop('constituents', None)
        image_array.pop('native_constituents', None)
        image_array['reference'].pop('values', None)
        image_array['reference']['used_for_signed_change_images'] = True
        image_array['frames'] = frames
        for frame in frames:
            frame.pop('values', None)
        image_array['image_receipts'] = proofs
        sensor['array_evidence'] = image_array
        sensor['times'] = [frame['time_s'] for frame in frames if frame['valid_cells'] > 0]


def _baseline_capture(place, config, images):
    """A reused capture needs a supplied baseline grid and its recorded plan slot."""
    time = place.get('display_time_s')
    if type(time) not in (int, float) or not math.isfinite(time):
        return False
    times = config.get('timesteps_s') or []
    slots = [i for i, sample in enumerate(times) if sample == round(time, 3)]
    cols = config.get('grid_cols')
    if len(slots) != 1 or type(cols) is not int or cols < 1:
        return False
    grids = (config.get('request_images') or {}).get('grid_images') or []
    index = slots[0] // cols
    if index >= len(grids):
        return False
    grid = grids[index]
    start = index * cols
    if (grid.get('t0_s') != times[start] or grid.get('t1_s') != times[min(len(times), start + cols) - 1]
            or place.get('view') not in (config.get('views') or [])
            or place.get('camera_label') not in (config.get('cam_labels') or [])):
        return False
    return any(row['jpeg_sha256'] == grid.get('sha256') and row['caption'].startswith('=== grid t=')
               for row in images)


def _owned_place(ep, original, capture, place):
    from label import episode as me, sensor_scene
    view = place.get('view')
    if view not in ep.get('sources', {}) or view not in ep['context'].get('cameras', {}):
        return None
    source = ep['sources'].get(capture.get('view'))
    if source is None or capture.get('source_location_sha256') != _sha(str(Path(source['packed']).resolve()).encode()):
        return None
    owner_row, owner_frame = capture.get('row'), capture.get('source_frame')
    if (type(owner_row) is not int or type(owner_frame) is not int
            or not 0 <= owner_row < len(ep['state'])):
        return None
    owner_sample = me.camera_sample(ep, capture['view'], owner_row, source_frame=owner_frame)
    if (owner_sample is None or owner_sample['episode_alignment_verified'] is not True
            or any(capture.get(key) != value for key, value in owner_sample.items())):
        return None
    if place.get('camera_label') != me.cam_name(ep, view):
        return None
    role = sensor_scene._role(ep, view, original)
    if role is None or place.get('role') != role:
        return None
    row, frame = place.get('row'), place.get('source_frame')
    if type(row) is not int or type(frame) is not int or not 0 <= row < len(ep['state']):
        return None
    sample = me.camera_sample(ep, view, row, source_frame=frame)
    if sample is None or sample['episode_alignment_verified'] is not True:
        return None
    if any(place.get(key) != value for key, value in sample.items()):
        return None
    if sensor_scene._capture_key(ep, view, sample) != sensor_scene._capture_key(ep, capture['view'], owner_sample):
        return None
    time = sample['source_time_s']
    window = original.get('episode_window_s')
    if window and not window[0] <= time < window[1]:
        return None
    return sample


def _cameras(original, selected, images, config, ep):
    if ep is None:
        return
    pairing = config.get('sensor_scene_pairing') or {}
    hashes = {row['jpeg_sha256'] for row in images}
    grid_captures = {(row.get('view'), row.get('source_frame'))
                     for grid in pairing.get('grids') or [] if grid.get('jpeg_sha256') in hashes
                     for row in grid.get('captures') or []}
    admitted = {}
    for capture in pairing.get('captures') or []:
        for place in capture.get('placements') or []:
            if capture.get('status') == 'supplemental':
                present = (capture.get('view'), capture.get('source_frame')) in grid_captures
            elif capture.get('status') == 'reused':
                present = _baseline_capture(place, config, images)
            else:
                present = False
            sample = _owned_place(ep, original, capture, place) if present else None
            if sample is not None:
                view = place['view']
                entry = admitted.setdefault(view, {'labels': set(), 'captures': []})
                entry['labels'].add(place['camera_label'])
                if sample not in entry['captures']:
                    entry['captures'].append(sample)
    labels = {}
    for view, entry in admitted.items():
        for label in entry['labels']:
            labels.setdefault(label, set()).add(view)
    identifiers = {sensor['id'] for sensor in selected['sensors']}
    canonical_ids = {'camera:' + view for view in admitted}
    for view, entry in admitted.items():
        meta = ep['context']['cameras'][view]
        canonical = 'camera:' + view
        if canonical in identifiers:
            continue
        samples = sorted(entry['captures'], key=lambda row: row['source_time_s'])
        sensor = {'id': canonical, 'name': str(meta.get('key') or meta.get('name') or view),
                  'kind': 'image', 'quantity': 'camera_context', 'view': view,
                  'descriptor_digest': se._descriptor_digest(meta), 'source_descriptor': copy.deepcopy(meta),
                  'source': str(ep['sources'][view]['packed']),
                  'interpretation': copy.deepcopy(prepared_interpretation(ep['context'], view, 'camera')),
                  'capabilities': [], 'episode_alignment_verified': True,
                  'times': sorted({row['source_time_s'] for row in samples}), 'native_captures': samples,
                  'timing': {}, 'limitations': [
                      'Discrete supplied camera context only. Stored clock pairing has no measured physical synchronization error bound.',
                      'Camera context cannot independently support a mechanics or sensor-response claim.']}
        selected['sensors'].append(sensor)
        identifiers.add(canonical)
        for label in sorted(entry['labels']):
            if label and len(labels[label]) == 1 and label not in identifiers and label not in canonical_ids:
                alias = copy.deepcopy(sensor)
                alias.update(id=label, citation_alias_of=canonical,
                             citation_alias_authority='Exact unambiguous source camera label in supplied capture receipts.')
                selected['sensors'].append(alias)
                identifiers.add(label)


def admit(original, selected, parts, config, ep):
    """Retain image metadata without restoring omitted numeric component readings."""
    images = _images(parts)
    _arrays(original, selected, images)
    _cameras(original, selected, images, config, ep)
    return selected
