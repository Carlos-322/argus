"""Bounded camera context at admitted array snapshots, with separate capture clocks."""
from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path

import numpy as np

from label import frames as mf
from label.dictionary_context import (camera_content_role, prepared_interpretation,
                                      SENSOR_CONTENT_ROLES)

MAX_CAPTURES = 32
MAX_IMAGE_BYTES = 2_000_000
IMAGE_WIDTH = 384


def _sha(value):
    return hashlib.sha256(value).hexdigest()


def _role(ep, view, sensor_doc):
    from label import tactile_capture
    from label.signals import _role as normalized
    ctx = ep['context']
    content = camera_content_role(ctx, view)
    entry = prepared_interpretation(ctx, view, 'camera')
    if content is not None:
        return {**content, 'interpretation': copy.deepcopy(entry)} if content['role'] == 'scene' else None
    meta = (ctx.get('cameras') or {}).get(view) or {}
    role = normalized(entry.get('role'))
    sensory = SENSOR_CONTENT_ROLES | {'depth', 'depth_camera', 'depth_image', 'thermal', 'infrared'}
    if (role in sensory or normalized(meta.get('sensor_type')) in sensory
            or tactile_capture.eligible(ctx, view)
            or any(s.get('kind') == 'image' and s.get('view') == view for s in sensor_doc.get('sensors') or [])):
        return None
    return {'role': None, 'provenance': 'unclassified', 'interpretation': copy.deepcopy(entry)}


def _capture_key(ep, view, sample):
    from label import episode as me
    source = ep['sources'][view]
    path = Path(source['packed']).resolve()
    stat = path.stat()
    # Hardlinks and alternate slots for the same owned file are one capture.
    identity = (stat.st_dev, stat.st_ino)
    instant = ('pts', sample['source_pts']) if 'source_pts' in sample else (
        'frame', mf.base_frame(float(source.get('base_s') or 0), me.ep_fps(ep)) + sample['source_frame'])
    return identity, instant


def _spread(candidates, limit):
    """Fair per-view coverage, with preferred views receiving remainder slots."""
    if len(candidates) <= limit:
        return list(candidates)
    groups = {}
    for row in candidates:
        groups.setdefault(row['view'], []).append(row)
    result = []
    remaining = limit
    for i, rows in enumerate(groups.values()):
        quota = min(len(rows), remaining // (len(groups) - i))
        if remaining % (len(groups) - i):
            quota = min(len(rows), quota + 1)
        ids = np.unique(np.rint(np.linspace(0, len(rows) - 1, quota)).astype(int)) if quota else []
        result.extend(rows[int(j)] for j in ids)
        remaining -= len(ids)
    return result


def _parts(captures, tiles, *, width, grid_cols, quality, detail):
    parts, grids = [], []
    for view in dict.fromkeys(row['view'] for row in captures):
        rows = [row for row in captures if row['view'] == view]
        for start in range(0, len(rows), grid_cols):
            group = rows[start:start + grid_cols]
            label = group[0]['camera_label']
            raw = mf.compose_grid([(row['source_time_s'], {label: tiles[row['_key']]}) for row in group],
                                  [label], quality, 84, 30)
            capture_times = [row['source_time_s'] for row in group]
            sensor_times = [sorted({ref['time_s'] for ref in row['paired_sensor_refs']}) for row in group]
            display_times = [sorted({place['display_time_s'] for place in row['placements']}) for row in group]
            exact = all(all(t == row['source_time_s'] for t in [*sensor, *display])
                        for row, sensor, display in zip(group, sensor_times, display_times))
            caption = ('RECORDED CAMERA CONTEXT PAIRED TO ARRAY SNAPSHOTS | '
                       + group[0]['context_kind'] + ' from ' + json.dumps(label) + '. ')
            if exact:
                caption += ('Each source capture time equals its paired sensor snapshot and display time: '
                            + json.dumps(capture_times, separators=(',', ':')) + '. ')
            else:
                caption += ('Ordered source capture seconds: ' + json.dumps(capture_times, separators=(',', ':'))
                            + '; corresponding sensor seconds: ' + json.dumps(sensor_times, separators=(',', ':'))
                            + '; anchor display seconds: ' + json.dumps(display_times, separators=(',', ':')) + '. ')
            caption += ('Discrete captures, not continuous coverage. Stored clock pairing supplies no measured '
                        'physical synchronization error bound.')
            parts.extend([{'type': 'text', 'text': caption}, {'type': 'image_url', 'image_url': {
                'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii'), 'detail': detail}}])
            grids.append({'jpeg_sha256': _sha(raw), 'bytes': len(raw),
                          'captures': [{'view': row['view'], 'source_frame': row['source_frame']} for row in group]})
    return parts, grids, sum(grid['bytes'] for grid in grids)


def collect(ep, sensor_doc, *, shown=None, gate=None, width=IMAGE_WIDTH, grid_cols=6, quality=80, detail='high',
            max_captures=MAX_CAPTURES, max_bytes=MAX_IMAGE_BYTES):
    """Return supplemental parts, an exact source receipt and encoded image bytes."""
    from label import episode as me
    shown = shown or {}
    max_captures = max(0, min(MAX_CAPTURES, int(max_captures)))
    max_bytes = max(0, min(MAX_IMAGE_BYTES, int(max_bytes)))
    width = max(1, min(IMAGE_WIDTH, int(width)))
    grid_cols = max(1, min(int(grid_cols), 6, 1920 // width))
    omitted = dict.fromkeys(('unaligned_sensor_snapshots', 'ineligible_camera_pairs', 'unclassified_extra_camera_pairs',
                             'unaligned_camera_pairs', 'outside_capture_window', 'unavailable_captures',
                             'capture_budget', 'image_byte_budget'), 0)
    references = {}
    for sensor in sensor_doc.get('sensors') or []:
        frames = (sensor.get('array_evidence') or {}).get('frames') or []
        if sensor.get('episode_alignment_verified') is not True:
            omitted['unaligned_sensor_snapshots'] += len(frames)
            continue
        for frame in frames:
            time = frame.get('time_s')
            if type(time) in (int, float) and np.isfinite(time):
                ref = {'sensor_id': sensor['id'], 'time_s': time}
                if ref not in references.setdefault(time, []):
                    references[time].append(ref)
    receipt = {'version': 1, 'method': 'Camera captures paired to retained array snapshot times; no scene cadence change.',
               'max_supplemental_captures': max_captures, 'max_image_bytes': max_bytes,
               'snapshot_times': len(references), 'captures': [], 'grids': [], 'omissions': omitted,
               'supplemental_captures': 0, 'reused_captures': 0,
               'reuse_scope': 'Already supplied baseline grids and explicitly supplied detail captures; later inspection or contact-strip captures are not presumed present.',
               'coverage': 'Only retained captures supply camera context. Omitted pairs and discrete captures do not establish continuous coverage or physical synchronization accuracy.'}
    if not references:
        return [], receipt, 0
    available, unknown = [], []
    for view in me.views(ep):
        role = _role(ep, view, sensor_doc)
        if role is None:
            omitted['ineligible_camera_pairs'] += len(references)
        elif role['role'] == 'scene':
            available.append((view, role))
        elif shown.get(view):
            unknown.append((view, role))
    anchor = me.anchor(ep)
    # One ordinary unclassified baseline camera is useful context, without claiming its content or mounting.
    neutral = next((row for row in unknown if row[0] == anchor), unknown[0] if unknown else None)
    if neutral:
        available.append(neutral)
    omitted['unclassified_extra_camera_pairs'] += max(0, len(unknown) - int(neutral is not None)) * len(references)
    available.sort(key=lambda row: (row[0] != anchor, row[1]['role'] != 'scene', me.views(ep).index(row[0])))
    anchor_times = (ep.get('times') or {}).get(anchor)
    anchor_times = (np.asarray(anchor_times, dtype=float) if anchor_times is not None else
                    np.arange(len(ep['state'])) / me.ep_fps(ep))
    if not len(anchor_times) or not np.isfinite(anchor_times).all() or np.any(np.diff(anchor_times) <= 0):
        omitted['unaligned_camera_pairs'] += len(available) * len(references)
        return [], receipt, 0
    actual_start, actual_end = me.frame_window(ep)
    admitted = sensor_doc.get('episode_window_s', [actual_start, actual_end])
    if (not isinstance(admitted, (list, tuple)) or len(admitted) != 2
            or any(type(t) not in (int, float) or not np.isfinite(t) for t in admitted)):
        omitted['unaligned_camera_pairs'] += len(available) * len(references)
        return [], receipt, 0
    window_start, window_end = max(actual_start, admitted[0]), min(actual_end, admitted[1])
    receipt['episode_window_s'] = [window_start, window_end]
    anchor_rows = np.flatnonzero((anchor_times >= window_start) & (anchor_times < window_end))
    if not len(anchor_rows):
        omitted['outside_capture_window'] += len(available) * len(references)
        return [], receipt, 0
    existing = set()
    for view in me.views(ep):
        for k in shown.get(view) or []:
            sample = me.camera_sample(ep, view, k)
            if sample is not None and me.recording_at(ep, view, k):
                existing.add(_capture_key(ep, view, sample))
    grouped = {}
    for view, role in available:
        source = ep['sources'][view]
        own = (ep.get('recorded_times') or {}).get(view)
        if own is None:
            own = (ep.get('times') or {}).get(view)
        if own is None and ep.get('times') is None:
            own = np.arange(int(source['n_frames'])) / me.ep_fps(ep)
        own = np.asarray(own if own is not None else [], dtype=float)
        valid_clock = (own.ndim == 1 and len(own) and np.isfinite(own).all()
                       and not np.any(np.diff(own) < 0))
        if valid_clock and np.any(np.diff(own) == 0):
            pts = (ep.get('times') or {}).get(view + '_pts')
            explicit_mapping = view in (ep.get('kmap') or {}) or view in (ep.get('recorded_times') or {})
            valid_clock = (explicit_mapping and pts is not None and len(pts) == len(own)
                           and np.isfinite(pts).all() and np.all(np.diff(pts) > 0))
        for time, refs in sorted(references.items()):
            # Do not turn nearest-endpoint clamping into temporal coverage.
            if not valid_clock:
                omitted['unaligned_camera_pairs'] += 1
                continue
            if (not window_start <= time < window_end or not anchor_times[0] <= time <= anchor_times[-1]
                    or not own.min() <= time <= own.max()):
                omitted['outside_capture_window'] += 1
                continue
            k = int(anchor_rows[np.argmin(np.abs(anchor_times[anchor_rows] - time))])
            source_frame = int(np.argmin(np.abs(own - time)))
            sample = me.camera_sample(ep, view, k, source_frame=source_frame)
            if sample is None or not sample['episode_alignment_verified']:
                omitted['unaligned_camera_pairs'] += 1
                continue
            if not window_start <= sample['source_time_s'] < window_end:
                omitted['outside_capture_window'] += 1
                continue
            key = _capture_key(ep, view, sample)
            alias = {'view': view, 'camera_label': me.cam_name(ep, view), 'role': role, **sample}
            row = grouped.setdefault(key, {'_key': key, 'view': view, 'camera_label': me.cam_name(ep, view),
                'role': role, 'context_kind': 'scene context' if role['role'] == 'scene' else 'camera context',
                **sample, 'view_aliases': [], 'aliases': [], 'placements': [], 'paired_sensor_refs': [],
                'source_location_sha256': _sha(str(Path(source['packed']).resolve()).encode())})
            if view not in row['view_aliases']:
                row['view_aliases'].append(view)
                row['aliases'].append(alias)
            placement = {**copy.deepcopy(alias), 'paired_sensor_refs': copy.deepcopy(refs)}
            if placement not in row['placements']:
                row['placements'].append(placement)
            for ref in refs:
                if ref not in row['paired_sensor_refs']:
                    row['paired_sensor_refs'].append(ref)
    reused = [row for key, row in grouped.items() if key in existing]
    candidates = [row for key, row in grouped.items() if key not in existing]
    selected = _spread(candidates, max_captures)
    omitted['capture_budget'] = len(candidates) - len(selected)
    tiles = {}
    for view in dict.fromkeys(row['view'] for row in selected):
        rows = [row for row in selected if row['view'] == view]
        source = ep['sources'][view]
        pts = (ep.get('times') or {}).get(view + '_pts')
        held = set()
        mapping = (ep.get('kmap') or {}).get(view)
        for start, end in (ep['context'].get('placeholder_frames') or {}).get(view) or []:
            held.update(int(mapping[k]) if mapping is not None and k < len(mapping) else k
                        for k in range(start, end + 1))
        ids = sorted({row['source_frame'] for row in rows} - held)
        def decode(ids):
            def run():
                return mf.extract_frames(source['packed'], float(source.get('base_s') or 0),
                    int(source['n_frames']), ids, pts=pts, fps=me.ep_fps(ep),
                    keep=lambda k, image: mf.to_jpeg(image, width, 90), tail_ok=True)
            if gate is None:
                return run()
            with gate:
                return run()
        try:
            decoded = decode(ids)
        except Exception as error:
            if not me._decode_error(error):
                raise
            decoded = {}
            for source_frame in ids:
                try:
                    decoded.update(decode([source_frame]))
                except Exception as single:
                    if not me._decode_error(single):
                        raise
        for row in rows:
            raw = decoded.get(row['source_frame'])
            if raw is None:
                omitted['unavailable_captures'] += 1
                continue
            tiles[row['_key']] = raw
            row['tile_jpeg_sha256'] = _sha(raw)
    selected = [row for row in selected if row['_key'] in tiles]
    decoded_count = len(selected)
    parts, grids, size = _parts(selected, tiles, width=width, grid_cols=grid_cols, quality=quality, detail=detail)
    while size > max_bytes and selected:
        selected = _spread(selected, len(selected) - 1)
        parts, grids, size = _parts(selected, tiles, width=width, grid_cols=grid_cols, quality=quality, detail=detail)
    omitted['image_byte_budget'] = decoded_count - len(selected)
    for status, rows in (('reused', reused), ('supplemental', selected)):
        receipt['captures'].extend({**{key: copy.deepcopy(value) for key, value in row.items() if key != '_key'},
                                    'status': status} for row in rows)
    receipt.update(grids=grids, image_bytes=size, supplemental_captures=len(selected), reused_captures=len(reused))
    return parts, receipt, size
