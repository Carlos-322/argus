"""Card traces for each sensor finding, derived only from the finding's own citations and its bound sensor document.

A finding cites recorded series: regional derivations, single array components, or other numeric fields. The card
shows at most two rows that support the finding at a glance, so each row is named by what it measures and points the
way the recording declares a stronger response. A cited region becomes that region's response; cited components of an
array that declares regions are grouped by region and then by how differently they move during the finding, so
opposite changes inside one region are not averaged away. Nothing here reads the finding's words or chooses
components the finding did not cite.
"""
from __future__ import annotations

import math
import re

VERSION = 1
MAX_ROWS = 2


def _columns(value):
    if isinstance(value, dict) and type(value.get('start')) is int and type(value.get('count')) is int:
        return list(range(value['start'], value['start'] + value['count']))
    return [int(c) for c in value] if isinstance(value, list) and all(type(c) is int for c in value) else []


def _finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _plot(row):
    """The row's full-rate native samples when present, else its supplied samples."""
    plot = row.get('plot')
    if (isinstance(plot, dict) and plot.get('kind') == 'native_samples' and isinstance(plot.get('times'), list)
            and isinstance(plot.get('values'), list) and len(plot['times']) == len(plot['values']) > 1):
        return plot['times'], plot['values']
    times, values = row.get('times'), row.get('values')
    if isinstance(times, list) and isinstance(values, list) and len(times) == len(values) > 1:
        return times, values
    return None, None


def _region_name(name):
    name = str(name).strip()
    return name[:1].upper() + name[1:]


def _part(sensor_id):
    """The stitched part prefix of an id (part2:...), or '' for a recording labelled in one request."""
    match = re.match(r'part\d+:', str(sensor_id))
    return match.group(0) if match else ''


def _direction(sensor, derivation):
    """The declared stronger-response direction, from the array, its regional derivation or the sensor itself."""
    array = (sensor or {}).get('array_evidence') or {}
    for value in ((array.get('response_convention') or {}).get('direction'), (derivation or {}).get('response_direction'),
                  (sensor or {}).get('response_direction')):
        if value in ('up', 'down'):
            return value
    return None


class _Source:
    """One array sensor's declared layout, response direction, reference and full-rate component values.

    Every card row is the oriented change of some cited cells from the same starting level the regional derivation
    uses (its per-cell median of the first readings), so a region row and an area row of that region agree, and a
    response weaker than the start (a hand easing off) draws below zero instead of flat.
    """

    def __init__(self, sensor, derivation):
        array = sensor.get('array_evidence') or {}
        semantics = (derivation or {}).get('semantics') or {}
        self.direction = _direction(sensor, derivation)
        self.zero_censored = bool(semantics.get('zero_exclusion_sensitivity'))
        self.regions = [(r.get('name'), set(_columns(r.get('indices')))) for r in semantics.get('regions') or []
                        if r.get('name') and _columns(r.get('indices'))]
        reference = ((derivation or {}).get('reference') or {}).get('values')
        if not isinstance(reference, list):
            reference = (array.get('reference') or {}).get('values')
        self.reference = reference if isinstance(reference, list) else None
        native = array.get('native_constituents') or {}
        self.columns = {c: i for i, c in enumerate(_columns(native.get('source_columns')))}
        self.times = native.get('times_s') if isinstance(native.get('times_s'), list) else None
        self.values = native.get('values') if isinstance(native.get('values'), list) else None
        self.sample_period_s = sensor.get('sample_period_s')

    def region_of(self, column):
        return next((name for name, ids in self.regions if column in ids), None)

    def response(self, column):
        """Change from the reference, oriented so a stronger declared response is positive. A zero the source flags
        as suspect is left blank."""
        if self.times is None or self.values is None or column not in self.columns or self.reference is None \
                or not 0 <= column < len(self.reference) or not _finite(self.reference[column]):
            return None
        i, ref, sign = self.columns[column], self.reference[column], -1.0 if self.direction == 'down' else 1.0
        out = []
        for row in self.values:
            raw = row[i] if isinstance(row, list) and i < len(row) else None
            if not _finite(raw) or (self.zero_censored and raw == 0):
                out.append(None)
            else:
                out.append(sign * (raw - ref))
        return out

    def mean(self, cells):
        """The oriented mean over whichever of these cells has a reading at each sample, or None without any."""
        rows = [r for r in (self.response(c) for c in cells) if r is not None]
        if not rows:
            return None
        out = []
        for values in zip(*rows):
            finite = [v for v in values if _finite(v)]
            out.append(sum(finite) / len(finite) if finite else None)
        return out


def _window(times, values, start, end):
    return [v for t, v in zip(times, values) if start - 1e-9 <= t <= end + 1e-9]


def _distance(a, b):
    pairs = [(x, y) for x, y in zip(a, b) if _finite(x) and _finite(y)]
    return math.sqrt(sum((x - y) ** 2 for x, y in pairs) / len(pairs)) if pairs else None


def _rms(a):
    finite = [x for x in a if _finite(x)]
    return math.sqrt(sum(x * x for x in finite) / len(finite)) if finite else 0.0


# Two cited areas are drawn apart when their trajectories inside the finding differ by at least this fraction of the
# largest cited response; closer components are one area.
SPLIT_AT = 0.5


def _split(cells, responses, times, start, end):
    """Group cited components by their oriented trajectories inside the finding's interval.

    The first cited component seeds one group and the component farthest from it seeds the other; every component
    joins the nearer seed. The distance compares level and timing together, so a patch that presses harder while its
    neighbour eases, or a response that moves from one patch to the next, stays two areas instead of one flat average.
    """
    windows = {c: _window(times, responses[c], start, end) for c in cells}
    first = cells[0]
    others = [(c, _distance(windows[first], windows[c])) for c in cells[1:]]
    others = [(c, d) for c, d in others if d is not None]
    if not others:
        return [cells]
    second, farthest = max(others, key=lambda item: item[1])
    if farthest < SPLIT_AT * max(_rms(windows[c]) for c in cells):
        return [cells]
    groups = ([first], [second])
    for c in cells[1:]:
        if c == second:
            continue
        a, b = _distance(windows[first], windows[c]), _distance(windows[second], windows[c])
        groups[0 if (a if a is not None else math.inf) <= (b if b is not None else math.inf) else 1].append(c)
    return [g for g in groups if g]


def rows_for(doc, finding):
    """At most MAX_ROWS card rows for one bound finding: {label, times, values, sample_period_s, source_ids}, or a
    {series_id} reference for a cited series the board draws as it always does."""
    series = {r['id']: r for r in doc.get('series') or [] if isinstance(r, dict) and isinstance(r.get('id'), str)}
    sensors = {s['id']: s for s in doc.get('sensors') or [] if isinstance(s, dict) and isinstance(s.get('id'), str)}
    derivations = {d['id']: d for d in doc.get('derivations') or [] if isinstance(d, dict) and isinstance(d.get('id'), str)}
    regional = {d.get('sensor_id'): d for d in derivations.values() if d.get('kind') == 'regional_response'}
    start, end = finding.get('start_s'), finding.get('end_s')
    if not _finite(start) or not _finite(end):
        return []
    items, seen = [], set()
    sources = {}

    # An inspected read of an array field belongs to the array with its descriptor digest in the same stitched part;
    # an independent analysis's copy of the array (mechanics:) is never the owner.
    arrays = {(_part(s['id']), s.get('descriptor_digest')): s['id'] for s in sensors.values()
              if s.get('array_evidence') and s.get('descriptor_digest')
              and not s['id'][len(_part(s['id'])):].startswith('mechanics:')}

    def resolve(sensor_id):
        sensor = sensors.get(sensor_id) or {}
        if sensor.get('array_evidence'):
            return sensor_id
        return arrays.get((_part(sensor_id), sensor.get('access_descriptor_digest')), sensor_id)

    def source(sensor_id):
        if sensor_id not in sources:
            sensor = sensors.get(sensor_id)
            sources[sensor_id] = _Source(sensor, regional.get(sensor_id)) if sensor and sensor.get('array_evidence') else None
        return sources[sensor_id]

    for ref in finding.get('evidence') or []:
        sid = ref.get('series_id') if isinstance(ref, dict) else None
        row = series.get(sid)
        if row is None:
            continue
        derivation = derivations.get(row.get('derivation_id'))
        if derivation and derivation.get('kind') == 'regional_response':
            columns = set(_columns(row.get('source_columns')))
            regions = [r for r in (derivation.get('semantics') or {}).get('regions') or [] if set(_columns(r.get('indices'))) >= columns]
            if columns and regions:
                region = min(regions, key=lambda r: len(_columns(r.get('indices'))))
                key = ('region', row['sensor_id'], region['name'])
                if key not in seen:
                    seen.add(key)
                    # The cited trace's own cells: the region's cells that the recording counts as valid.
                    items.append(key + (sid, tuple(sorted(columns))))
                continue
        cells = _columns(row.get('columns'))
        owner = resolve(row.get('sensor_id'))
        src = source(owner)
        if src is not None and len(cells) == 1 and src.response(cells[0]) is not None:
            region = src.region_of(cells[0])
            key = ('cells', owner, region)
            if key not in seen:
                seen.add(key)
                items.append(key + ([],))
            next(i for i in items if i[:3] == key)[3].append(cells[0])
            continue
        key = ('series', sid)
        if key not in seen and _plot(row)[0] is not None:
            seen.add(key)
            items.append(key)

    out = []
    for item in items:
        if len(out) >= MAX_ROWS:
            break
        if item[0] == 'region':
            _, sensor_id, name, cited, indices = item
            src = source(sensor_id)
            values = src.mean([c for c in indices if c in src.columns]) if src is not None else None
            if values is not None:
                out.append({'label': _region_name(name), 'times': src.times, 'values': values,
                            'sample_period_s': src.sample_period_s,
                            'source_ids': [f'{sensor_id}:array:{c}' for c in indices if c in src.columns]})
                continue
            # Without the array's own readings (an independent analysis was given only its regional traces), draw
            # the cited trace itself, turned so a stronger declared response points up.
            row = series[cited]
            times, values = _plot(row)
            if times is None:
                continue
            direction = _direction(sensors.get(sensor_id), derivations.get(row.get('derivation_id')))
            if direction == 'down' and str(row.get('operation', '')).startswith('mean signed'):
                values = [-v if _finite(v) else None for v in values]
            out.append({'label': _region_name(name), 'times': times, 'values': values,
                        'sample_period_s': row.get('sample_period_s'), 'source_ids': [cited]})
        elif item[0] == 'cells':
            _, sensor_id, region, cells = item
            src = source(sensor_id)
            responses = {c: src.response(c) for c in cells}
            groups = _split(cells, responses, src.times, start, end)
            base = _region_name(region) if region else 'Sensor'
            for n, group in enumerate(groups, 1):
                if len(out) >= MAX_ROWS:
                    break
                label = f'{base} area {n}' if len(groups) > 1 else f'{base} area'
                out.append({'label': label, 'times': src.times, 'values': src.mean(group),
                            'sample_period_s': src.sample_period_s,
                            'source_ids': [f'{sensor_id}:array:{c}' for c in group]})
        else:
            # Any other cited series is drawn by the board as it always is; the row only keeps its place in order.
            out.append({'series_id': item[1]})
    return [r for r in out if 'series_id' in r or (r['values'] and any(_finite(v) for v in r['values']))]


def attach(doc):
    """Add a display block to every finding in a bound sensor document, in place. Returns the document."""
    if not isinstance(doc, dict):
        return doc
    for finding in doc.get('findings') or []:
        if not isinstance(finding, dict):
            continue
        rows = rows_for(doc, finding)
        if any('series_id' not in r for r in rows):
            finding['display'] = {'version': VERSION, 'rows': rows}
        else:
            finding.pop('display', None)
    return doc
