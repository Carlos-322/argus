"""Whole-array tactile comparisons in source coordinates, without guessed anatomy or polarity."""
from __future__ import annotations

import base64
import io
import math

import numpy as np

MAX_ARRAYS = 4
MAX_VALUES = 8192
MAX_FRAMES = 8


def comparison(a, shape, times, ks, value_budget):
    """Keep complete snapshots, including extrema outside the ordinary RGB samples."""
    if (len(shape) != 2 or any(type(d) is not int or d <= 0 for d in shape)
            or math.prod(shape) != a.shape[1] or max(shape) > 32 or a.shape[1] > 1024):
        return None
    limit = min(MAX_FRAMES, len(a), value_budget // a.shape[1] - 1)
    if limit < min(3, len(a)):
        return None
    initial = np.asarray(a[:min(3, len(a))], dtype=float)
    # Incomplete initial cells have no reference; zero is a reading, not a validity mask.
    valid_reference = np.isfinite(initial).all(axis=0)
    reference = np.median(np.where(np.isfinite(initial), initial, 0), axis=0)
    reference[~valid_reference] = np.nan
    peak = np.zeros(len(a))
    mean = np.zeros(len(a))
    raw_low, raw_high, delta_high = np.inf, -np.inf, 0.0
    for start in range(0, len(a), 128):
        block = np.asarray(a[start:start + 128], dtype=float)
        valid = np.isfinite(block)
        if valid.any():
            raw_low = min(raw_low, float(block[valid].min()))
            raw_high = max(raw_high, float(block[valid].max()))
        delta = np.abs(block - reference)
        finite = np.isfinite(delta)
        values = np.where(finite, delta, 0)
        peak[start:start + len(block)] = values.max(axis=1)
        mean[start:start + len(block)] = values.sum(axis=1) / np.maximum(finite.sum(axis=1), 1)
        delta_high = max(delta_high, float(values.max()))
    if not math.isfinite(raw_low):
        return None
    selected = {0, len(a) - 1}
    priority = {0: 0, len(a) - 1: 0}
    # Preserve local excursions as well as large distributed changes. These select
    # source rows only; they are not claims of contact, force or failure.
    for score in (mean, peak):
        if len(selected) < limit:
            k = int(np.argmax(score))
            selected.add(k)
            priority.setdefault(k, 1)
    rgb_ids = {int(k) for k in ks if 0 <= k < len(a)}
    candidates = sorted(rgb_ids)
    candidates += list(range(len(a)))
    while len(selected) < limit:
        # Spread remaining samples through the recording, preferring RGB instants.
        available = [k for k in candidates if k not in selected]
        if not available:
            break
        rgb = [k for k in available if k in rgb_ids]
        pool = rgb or available
        k = max(pool, key=lambda k: min(abs(times[k] - times[j]) for j in selected))
        selected.add(k)
        priority[k] = 2
    def values(row):
        return [float(v) if np.isfinite(v) else None for v in row]
    return {'shape': list(shape), 'coordinates': 'Source array row and column indices; physical layout is not established.',
            'reference': {'times_s': [float(t) for t in times[:len(initial)]], 'values': values(reference),
                          'method': 'Per-cell median of the first up to three samples, requiring all to be finite. An initial reference, not a verified unloaded baseline.'},
            'raw_range': [raw_low, raw_high], 'signed_change_range': [-delta_high, delta_high],
            'selection': 'Endpoints, then whole-array and single-cell reference-change maxima as space permits, then spaced RGB instants. Unshown transients remain possible.',
            'native_samples': len(a),
            'frames': [{'time_s': float(times[k]), 'values': values(a[k]),
                        'valid_cells': int(np.isfinite(a[k]).sum()),
                        'signed_change_valid_cells': int(np.isfinite(a[k] - reference).sum()),
                        'selection_priority': priority[k],
                        'raw_quantiles': values(np.nanquantile(np.where(np.isfinite(a[k]), a[k], np.nan), [.1, .5, .9]))
                            if np.isfinite(a[k]).any() else [None] * 3,
                        'signed_change_quantiles': values(np.nanquantile(np.where(
                            np.isfinite(a[k] - reference), a[k] - reference, np.nan), [.1, .5, .9]))
                            if np.isfinite(a[k] - reference).any() else [None] * 3}
                       for k in sorted(selected)]}


def render(sensor):
    """Render raw and signed changes on scales shared across every supplied instant."""
    from PIL import Image, ImageDraw, ImageFont
    from label.frames import GRID_FONT
    array = sensor['array_evidence']
    rows, cols = array['shape']
    snapshots = array['frames']
    width, height = 210, 294
    per_row = min(4, len(snapshots))
    image = Image.new('RGB', (per_row * width, math.ceil(len(snapshots) / per_row) * height), '#101923')
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(GRID_FONT), 13)
    raw_low, raw_high = array['raw_range']
    delta_max = array['signed_change_range'][1]
    reference = np.array([np.nan if v is None else v for v in array['reference']['values']])
    for i, frame in enumerate(snapshots):
        x, y = (i % per_row) * width, (i // per_row) * height
        draw.text((x + 8, y + 3), f"{frame['time_s']:.4f} s", fill='white', font=font)
        raw = np.array([np.nan if v is None else v for v in frame['values']])
        for offset, label, values in ((24, 'Raw', raw), (157, 'Raw - initial reference', raw - reference)):
            draw.text((x + 8, y + offset), label, fill='white', font=font)
            for cell, value in enumerate(values):
                r, c = divmod(cell, cols)
                if not math.isfinite(value):
                    color = (105, 105, 105)
                elif offset == 24:
                    q = float(np.clip((value - raw_low) / (raw_high - raw_low or 1), 0, 1))
                    color = (int(25 + 230*q), int(35 + 180*q), int(55 + 50*q))
                else:
                    q = float(np.clip(value / (delta_max or 1), -1, 1))
                    color = (int(225 - 190*max(-q, 0)), int(225 - 160*abs(q)), int(225 - 190*max(q, 0)))
                left, top = x + 24 + c * 160 / cols, y + offset + 18 + r * 104 / rows
                draw.rectangle((left, top, left + 160/cols, top + 104/rows), fill=color)
            draw.text((x + 8, y + offset + 17), '0', fill='white', font=font)
            draw.text((x + 8, y + offset + 108), str(rows - 1), fill='white', font=font)
            draw.text((x + 24, y + offset + 122), f'columns 0 to {cols-1}', fill='white', font=font)
    stream = io.BytesIO()
    image.save(stream, format='JPEG', quality=88)
    raw = stream.getvalue()
    caption = (f"TACTILE ARRAY SNAPSHOTS | {sensor['id']} | raw scale {raw_low:g} to {raw_high:g}; "
               f"signed change scale {-delta_max:g} to {delta_max:g}. "
               'All frames use the same scales. Raw is dark to yellow; signed change is blue (negative), pale (zero), red (positive). '
               'Grey means missing. Axes are source array indices, not hand anatomy. '
               'Raw minus the initial per-cell reference is not calibrated force or a known load direction. '
               'The numeric summary gives 10th, 50th and 90th percentiles across finite cells. Read individual spatial patterns from these images; cite sensor_id without series_id for these full-array snapshots.')
    return [{'type': 'text', 'text': caption}, {'type': 'image_url', 'image_url': {
        'url': 'data:image/jpeg;base64,' + base64.b64encode(raw).decode('ascii'), 'detail': 'high'}}]
