"""Bounded native tactile capture selection, without semantic event or force claims."""
from __future__ import annotations

import base64
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from label.dictionary_context import camera_content_role, prepared_interpretation, TACTILE_CONTENT_ROLES
from prepare.formats import is_sensing

MAX_SCAN_FRAMES = 36000
MAX_ADDITIONAL_FRAMES = 24
MAX_IMAGE_BYTES = 2_000_000
THUMB_WIDTH = 128
IMAGE_WIDTH = 384
CHUNK = 256


def eligible(ctx, view):
    from label import signals as sg
    meta = (ctx.get('cameras') or {}).get(view) or {}
    role = sg._role(prepared_interpretation(ctx, view, 'camera').get('role'))
    content = camera_content_role(ctx, view)
    if content and content['role'] in ('scene', 'other'):
        return False
    recorded = bool(content and content['role'] == 'tactile') or meta.get('sensor_type') == 'optical_tactile'
    return (recorded or role in TACTILE_CONTENT_ROLES or not role and is_sensing(str(meta.get('name') or view))) \
        and (recorded or role not in sg.NON_TOUCH_ROLES)


def sample(ep, view, source_frame):
    """Native captures use their own clock, with a separately identified nearby scene placement."""
    from label import episode as me
    own = (ep.get('recorded_times') or {}).get(view)
    if own is None:
        own = (ep.get('times') or {}).get(view)
    capture = float(own[source_frame]) if own is not None and source_frame < len(own) else None
    if capture is None and ep.get('times') is None:
        capture = source_frame / me.ep_fps(ep)
    anchor_times = (ep.get('times') or {}).get(me.anchor(ep))
    if anchor_times is None:
        anchor_row = min(len(ep['state']) - 1, max(0, round((capture or 0.) * me.ep_fps(ep))))
    else:
        anchor_row = int(np.argmin(np.abs(np.asarray(anchor_times) - capture))) if capture is not None and np.isfinite(capture) else 0
    return me.camera_sample(ep, view, anchor_row, source_frame=int(source_frame))


def _selected(scores, already, limit):
    scores = [row for row in scores if row[2] > 0]
    if not scores:
        return []
    buckets = np.array_split(np.arange(len(scores)), min(8, len(scores)))
    maxima = [max((scores[int(i)] for i in bucket), key=lambda row: (row[2], -row[1])) for bucket in buckets]
    candidates = sorted(maxima, key=lambda row: (row[2], -row[1]), reverse=True)
    candidates += sorted(scores, key=lambda row: (row[2], -row[1]), reverse=True)
    selected = set()
    for before, after, score in candidates:
        extra = {before, after} - already - selected
        if len(selected) + len(extra) <= limit:
            selected.update(extra)
        if len(selected) >= limit:
            break
    return sorted(selected)


def _scan(ep, view, pl, gate, scan_limit, image_limit):
    from label import episode as me, frames as mf
    source = ep['sources'][view]
    count = int(source['n_frames'])
    ids = np.unique(np.rint(np.linspace(0, count - 1, min(count, scan_limit))).astype(int)).tolist() if count else []
    held = set()
    mapping = (ep.get('kmap') or {}).get(view)
    for start, end in (ep['context'].get('placeholder_frames') or {}).get(view) or []:
        held.update(int(mapping[k]) if mapping is not None and k < len(mapping) else k
                    for k in range(start, end + 1))
    ids = [row for row in ids if row not in held]
    pts = (ep.get('times') or {}).get(view + '_pts')
    seen, scores, previous = [], [], None
    failures = 0

    def keep(row, image):
        nonlocal previous
        pixels = np.asarray(mf.downscaled(image, THUMB_WIDTH), dtype=np.float32)
        if previous is not None and previous[1].shape == pixels.shape:
            difference = np.abs(pixels - previous[1])
            score = float(np.mean(difference) + .1 * np.percentile(difference, 99))
            scores.append((previous[0], int(row), score))
        previous = int(row), pixels
        seen.append(int(row))
        return None

    def decode(rows, callback):
        def run():
            return mf.extract_frames(source['packed'], source['base_s'], count, rows, pts=pts,
                                     fps=me.ep_fps(ep), keep=callback, tail_ok=True)
        if gate is None:
            return run()
        with gate:
            return run()
    for start in range(0, len(ids), CHUNK):
        try:
            decode(ids[start:start + CHUNK], keep)
        except Exception as error:
            if not me._decode_error(error):
                raise
            failures += 1
            previous = None
    already = {capture['source_frame'] for k in pl['ks']
               if (capture := me.camera_sample(ep, view, k)) is not None}
    selected = _selected(scores, already, image_limit)
    images = {}
    for row in selected:
        try:
            images.update(decode([row], lambda k, image: mf.Shrunk(image, [IMAGE_WIDTH])))
        except Exception as error:
            if not me._decode_error(error):
                raise
            failures += 1
    receipt = {'version': 1, 'method': 'Adjacent inspected RGB thumbnail changes ranked across temporal buckets; a selection proxy, not detected contact, slip or force.',
               'native_source_frames': count, 'inspected_captures': len(seen), 'scan_limit': scan_limit,
               'complete_native_scan': len(seen) == count, 'failed_chunks_or_captures': failures,
               'selected_source_frames': sorted(images),
               'coverage': 'Only selected supplied captures support image findings. Scan limits, invalid captures and ranking can miss brief or subtle changes; selection does not establish complete event coverage.'}
    return images, receipt


def collect(ep, pl, gate=None):
    views = [view for view in ep.get('sources') or {} if eligible(ep['context'], view)]
    if not views:
        return {}, {}
    quota = max(1, MAX_ADDITIONAL_FRAMES // len(views))
    scan = max(1, MAX_SCAN_FRAMES // len(views))
    with ThreadPoolExecutor(max_workers=len(views)) as executor:
        futures = {view: executor.submit(_scan, ep, view, pl, gate, scan, quota) for view in views}
        results = {view: future.result() for view, future in futures.items()}
    return ({view: images for view, (images, receipt) in results.items() if images},
            {view: receipt for view, (images, receipt) in results.items()})


def image_parts(ep, images, *, detail='high'):
    from label import episode as me, frames as mf
    parts = []
    for view, frames in images.items():
        ordered = sorted(frames)
        captures = {row: sample(ep, view, row) for row in ordered}
        steps = [(captures[row]['source_time_s'] if captures[row]['source_time_s'] is not None else row / me.ep_fps(ep),
                  [(me.cam_name(ep, view), mf.to_jpeg(frames[row], IMAGE_WIDTH, 80))]) for row in ordered]
        content, _, _ = mf.build_content('', '', steps, [me.cam_name(ep, view)], 4, detail, 80, 84, 30)
        for part in content[1:]:
            if part.get('type') == 'text':
                part['text'] += ' | native tactile captures; source frame indices and capture clocks are listed in sensor evidence'
                if any(not captures[row]['episode_alignment_verified'] for row in ordered):
                    part['text'] += '; clock alignment is unverified, so headers are source display placements, not synchronized episode events'
            parts.append(part)
    return parts


def bounded_parts(ep, images, *, detail='high'):
    """Rendered supplemental grids and supplied images always share one byte-bounded projection."""
    images = {view: dict(frames) for view, frames in images.items()}
    while True:
        parts = image_parts(ep, images, detail=detail)
        size = sum(len(base64.b64decode(part['image_url']['url'].split(',', 1)[1]))
                   for part in parts if part['type'] == 'image_url')
        if size <= MAX_IMAGE_BYTES:
            return images, parts, size
        view = max(images, key=lambda name: len(images[name]))
        images[view].pop(max(images[view]))
        if not images[view]:
            images.pop(view)
