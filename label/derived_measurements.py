"""Source-bound regional responses and point-set geometry, without inferred anatomy."""
from __future__ import annotations

import copy

import numpy as np


VERSION = 1


def _input(values, times, ndim):
    a = np.asarray(values)
    if a.dtype.kind not in 'biuf':
        raise ValueError('measurement values must be real numeric readings')
    a = np.asarray(a, dtype=float)
    ts = np.asarray(times, dtype=float)
    if a.ndim != ndim or not len(a) or any(d == 0 for d in a.shape[1:]):
        raise ValueError('measurement shape is empty or unsupported')
    if ts.ndim != 1 or len(ts) != len(a) or not np.isfinite(ts).all() or (np.diff(ts) <= 0).any():
        raise ValueError('measurement clock must match rows and increase strictly')
    return a, ts


def _indices(value, width):
    if (not isinstance(value, (list, tuple)) or not value
            or any(type(i) is not int or not 0 <= i < width for i in value)
            or len(set(value)) != len(value)):
        raise ValueError('indices must be distinct recorded indices within the source shape')
    return list(value)


def _numbers(values):
    return [float(v) if np.isfinite(v) else None for v in values]


def _metadata(kind, semantics):
    return {'version': VERSION, 'kind': kind, 'source': copy.deepcopy(semantics.get('source')),
            'provenance': copy.deepcopy(semantics.get('provenance')),
            'semantics': copy.deepcopy(semantics), 'limitations': []}


def _units(semantics, width):
    units = semantics.get('units', semantics.get('unit'))
    if isinstance(units, (list, tuple)):
        if len(units) != width or any(u is not None and not isinstance(u, str) for u in units):
            raise ValueError('units must name every recorded component')
        return list(units)
    if units is not None and not isinstance(units, str):
        raise ValueError('units must be a string or component list')
    return [units] * width


def regional_response(values, times, semantics):
    """Compare explicit sparse regions with the first up to three sample rows.

    Missing cells do not contribute. Zero remains a valid reading. A separately
    requested sensitivity comparison excludes cells that ever reach zero.
    """
    a, ts = _input(values, times, 2)
    width = a.shape[1]
    units = _units(semantics, width)
    valid_indices = semantics.get('valid_indices')
    allowed = set(range(width)) if valid_indices is None else set(_indices(valid_indices, width))
    groups = semantics.get('regions')
    if not isinstance(groups, list) or not groups:
        raise ValueError('explicit regions are required')
    checked, names = [], set()
    for group in groups:
        if not isinstance(group, dict) or not isinstance(group.get('name'), str) or not group['name'].strip():
            raise ValueError('each region needs a recorded name and indices')
        if group['name'] in names:
            raise ValueError('region names must be distinct')
        names.add(group['name'])
        ids = [i for i in _indices(group.get('indices'), width) if i in allowed]
        component_units = {units[i] for i in ids}
        if len(component_units) > 1:
            raise ValueError('region units are mixed or unresolved')
        checked.append((group['name'], ids, next(iter(component_units), None)))
    direction = semantics.get('response_direction')
    if direction not in (None, 'up', 'down'):
        raise ValueError('response_direction must be up, down or unknown')
    initial = a[:min(3, len(a))]
    reference = np.full(width, np.nan)
    counts = np.isfinite(initial).sum(axis=0)
    for i in np.flatnonzero(counts):
        reference[i] = np.median(initial[np.isfinite(initial[:, i]), i])
    result = _metadata('regional_response', semantics)
    result.update(times_s=ts.tolist(), response_direction=direction,
                  operation='reference-relative regional response; no physical force conversion',
                  reference={'times_s': ts[:len(initial)].tolist(), 'values': _numbers(reference),
                             'valid_counts': counts.tolist(),
                             'method': 'Per-cell median of finite readings in the first up to three rows; not a verified unloaded baseline.'},
                  traces=[])
    if direction is None:
        result['limitations'].append('Response direction is unknown; values are signed raw minus initial reference.')

    def trace(ids):
        raw = a[:, ids]
        finite_raw = np.isfinite(raw)
        delta = raw - reference[ids]
        finite = np.isfinite(delta)
        valid_counts = finite.sum(axis=1)
        def mean(data, mask):
            count = mask.sum(axis=1)
            total = np.where(mask, data, 0).sum(axis=1)
            return _numbers(np.divide(total, count, out=np.full(len(a), np.nan), where=count > 0))
        return {'indices': ids, 'valid_counts': valid_counts.tolist(),
                'raw_valid_counts': finite_raw.sum(axis=1).tolist(),
                'raw_means': mean(raw, finite_raw), 'zero_counts': ((raw == 0) & finite_raw).sum(axis=1).tolist(),
                'values': mean(delta if direction is None else np.maximum(delta * (1 if direction == 'up' else -1), 0), finite),
                'signed_values': mean(delta, finite),
                'positive_values': mean(np.maximum(delta, 0), finite),
                'negative_values': mean(np.minimum(delta, 0), finite)}

    for name, ids, unit in checked:
        row = {'name': name, 'unit': unit,
               'operation': ('mean signed raw minus initial reference' if direction is None else
                             'mean positive response in declared direction from initial reference'), **trace(ids)}
        if semantics.get('zero_exclusion_sensitivity'):
            nonzero = [i for i in ids if not np.any(a[:, i] == 0)]
            row['sensitivity'] = {'operation': 'Separate sensitivity comparison excluding cells that ever read zero; zero remains valid in the primary trace.',
                                  'unit': unit, **trace(nonzero)}
        result['traces'].append(row)
    return result


def pointset_shape(values, times, semanticmetadata):
    """Measure rigidly aligned point-set deformation on one fixed reference scale.

    Point indices and optional chain names come only from explicit metadata.
    Adjacent distances compare the two poses directly, never their distances
    from the first pose. Work is linear in frames and points.
    """
    a, ts = _input(values, times, 3)
    frames, points, dims = a.shape
    if dims not in (2, 3):
        raise ValueError('point coordinates must have two or three dimensions')
    semantics = semanticmetadata
    stated_units = semantics.get('units', semantics.get('unit'))
    if isinstance(stated_units, (list, tuple)):
        if len(stated_units) not in (dims, points * dims) or len(set(stated_units)) != 1:
            raise ValueError('coordinate units must be common across all points and axes')
        unit = stated_units[0]
    else:
        unit = stated_units
    if unit is not None and not isinstance(unit, str):
        raise ValueError('coordinate units must be a string or component list')
    result = _metadata('pointset_shape', semantics)
    finite_points = np.isfinite(a).all(axis=2)
    if semantics.get('valid_indices') is not None:
        allowed = np.zeros(points * dims, dtype=bool)
        allowed[_indices(semantics['valid_indices'], points * dims)] = True
        finite_points &= allowed.reshape(points, dims).all(axis=1)[None, :]
    first = a[0, finite_points[0]]
    pair = semantics.get('normalization_pair')
    if pair is not None:
        pair = _indices(pair, points)
        if len(pair) != 2:
            raise ValueError('normalization_pair must contain two point indices')
        scale = float(np.linalg.norm(a[0, pair[1]] - a[0, pair[0]])) if finite_points[0, pair].all() else np.nan
        method = 'Fixed distance between declared point pair in the first frame'
    else:
        scale = float(np.sqrt(np.mean(np.sum((first - first.mean(axis=0)) ** 2, axis=1)))) if len(first) >= dims else np.nan
        method = 'Fixed first-frame RMS point spread about its centroid'
    usable_scale = bool(np.isfinite(scale) and scale > 0)
    result.update(times_s=ts.tolist(), operation='Rigid alignment without reflection; RMS point distance divided by fixed first-frame scale',
                  normalization={'method': method, 'scale': scale if usable_scale else None,
                                 'unit': unit, 'indices': pair, 'reference_time_s': float(ts[0])},
                  unit='reference-scale fractions', first_rms=[], adjacent_rms=[],
                  pair_valid_counts={'first': [], 'adjacent': []},
                  repeated_previous=[False] * frames, stale_intervals=[], curls=[])
    if not usable_scale:
        result['limitations'].append('First-frame normalization is missing or degenerate; shape distances are unavailable.')
    result['limitations'].append('Rigid alignment removes point-set translation and rotation; it does not establish unchanged contact, force or tracking accuracy.')

    def compare(i, j):
        keep = finite_points[i] & finite_points[j]
        count = int(keep.sum())
        if not usable_scale or count < dims:
            return None, count
        x, y = a[i, keep], a[j, keep]
        x, y = x - x.mean(axis=0), y - y.mean(axis=0)
        if not np.any(x) or not np.any(y):
            return None, count
        u, _, vt = np.linalg.svd(x.T @ y)
        if np.linalg.det(u @ vt) < 0:
            u[:, -1] *= -1
        distance = float(np.sqrt(np.mean(np.sum((x @ (u @ vt) - y) ** 2, axis=1))) / scale)
        return distance if np.isfinite(distance) else None, count

    for j in range(frames):
        first_distance, first_count = compare(0, j)
        adjacent_distance, adjacent_count = compare(j - 1, j) if j else (None, 0)
        result['first_rms'].append(first_distance)
        result['adjacent_rms'].append(adjacent_distance)
        result['pair_valid_counts']['first'].append(first_count)
        result['pair_valid_counts']['adjacent'].append(adjacent_count)
        if j and adjacent_count >= dims:
            result['repeated_previous'][j] = bool(np.array_equal(a[j - 1], a[j], equal_nan=True))
    result['coverage'] = {'frames': frames,
                          'usable_first_comparisons': sum(v is not None for v in result['first_rms']),
                          'usable_adjacent_comparisons': sum(v is not None for v in result['adjacent_rms']),
                          'repeated_pairs': sum(result['repeated_previous']),
                          'incomplete_frames': int((~finite_points.all(axis=1)).sum())}
    start = None
    for j in range(1, frames + 1):
        repeated = j < frames and result['repeated_previous'][j]
        if repeated and start is None:
            start = j - 1
        if not repeated and start is not None:
            result['stale_intervals'].append({'start_s': float(ts[start]), 'end_s': float(ts[j - 1]),
                                              'frames': j - start,
                                              'interpretation': 'Exactly repeated estimates; possible stale tracking or unchanged input, not proof of no movement.'})
            start = None
    chains = semantics.get('chains') or []
    if not isinstance(chains, list):
        raise ValueError('chains must be explicitly named point-index chains')
    for chain in chains:
        if not isinstance(chain, dict) or not isinstance(chain.get('name'), str) or not chain['name'].strip():
            raise ValueError('chains need explicit names and indices')
        ids = _indices(chain.get('indices'), points)
        if len(ids) < 3:
            raise ValueError('chain indices need at least three points')
        q = a[:, ids]
        valid_chain_points = finite_points[:, ids]
        complete = valid_chain_points.all(axis=1)
        length = np.linalg.norm(np.diff(q, axis=1), axis=2).sum(axis=1)
        chord = np.linalg.norm(q[:, -1] - q[:, 0], axis=1)
        usable = complete & (length > 0) & np.isfinite(length)
        curl = np.full(frames, np.nan)
        curl[usable] = np.clip(1 - chord[usable] / length[usable], 0, 1)
        result['curls'].append({'name': chain['name'], 'indices': ids, 'values': _numbers(curl),
                                'unit': 'dimensionless', 'valid_counts': valid_chain_points.sum(axis=1).tolist(),
                                'operation': 'One minus endpoint distance divided by the sum of consecutive chain segment lengths'})
    return result
