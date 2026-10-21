"""Validate explicitly supplied sensor layouts without inferring physical meaning."""
import json
import math
from numbers import Integral
from urllib.parse import urlsplit

VERSION = 1
MAX_BYTES = 64 * 1024
MAX_COMPONENTS = 65536
MAX_GROUPS = 128
MAX_INDICES = 8192


def validate(value, shape):
    """Return a detached v1 descriptor, or None when its bounds or declarations are unusable.

    Regions address flattened source components. Explicit landmark chains and normalization pairs address
    points of a declared [points, coordinates] shape. Unknown finite JSON fields remain attributed source data.
    Zero exclusion requests a sensitivity comparison; it never marks every zero reading invalid.
    """
    try:
        if not isinstance(shape, (list, tuple)) or len(shape) > 8 or any(
                isinstance(n, bool) or not isinstance(n, Integral) or n <= 0 for n in shape):
            return None
        shape = [int(n) for n in shape]
        width = math.prod(shape)
        if width > MAX_COMPONENTS:
            return None
        if isinstance(value, bytes):
            value = value.decode('utf-8')
        if isinstance(value, str):
            if len(value.encode('utf-8')) > MAX_BYTES:
                return None
            value = json.loads(value)
        if not isinstance(value, dict) or type(value.get('version')) is not int or value['version'] != VERSION:
            return None
        nodes = 0

        def finite_json(item, level=0):
            nonlocal nodes
            nodes += 1
            if nodes > 16384 or level > 12:
                return False
            if isinstance(item, dict):
                return all(isinstance(k, str) and len(k) <= 512 and finite_json(v, level + 1)
                           for k, v in item.items())
            if isinstance(item, list):
                return len(item) <= MAX_INDICES and all(finite_json(v, level + 1) for v in item)
            if isinstance(item, str):
                return len(item.encode('utf-8')) <= MAX_BYTES
            if item is None or type(item) is bool:
                return True
            return type(item) in (int, float) and math.isfinite(item)

        if not finite_json(value):
            return None
        encoded = json.dumps(value, allow_nan=False, separators=(',', ':'), ensure_ascii=False)
        if len(encoded.encode('utf-8')) > MAX_BYTES:
            return None
        references = 0

        def indices(items, bound, minimum=1):
            nonlocal references
            if not isinstance(items, list) or not minimum <= len(items) <= MAX_INDICES:
                return False
            references += len(items)
            return (references <= MAX_INDICES and all(type(i) is int and 0 <= i < bound for i in items)
                    and len(set(items)) == len(items))

        landmark = 'normalization_pair' in value or 'chains' in value
        if landmark and not (len(shape) == 2 and shape[1] in (2, 3)):
            return None
        if 'valid_indices' in value and not indices(value['valid_indices'], width, minimum=0):
            return None
        for key, bound, minimum in (('regions', width, 1), ('chains', shape[0] if landmark else width, 3)):
            if key not in value:
                continue
            groups = value[key]
            if not isinstance(groups, list) or len(groups) > MAX_GROUPS:
                return None
            names = set()
            for group in groups:
                if (not isinstance(group, dict) or not isinstance(group.get('name'), str)
                        or not group['name'].strip() or len(group['name']) > 256 or group['name'] in names
                        or not indices(group.get('indices'), bound, minimum)):
                    return None
                names.add(group['name'])
        if 'normalization_pair' in value and (not indices(value['normalization_pair'], shape[0], 2)
                                              or len(value['normalization_pair']) != 2):
            return None
        if 'positions' in value:
            positions = value['positions']
            if (not isinstance(positions, list) or len(positions) != width or not positions
                    or not isinstance(positions[0], list) or len(positions[0]) not in (2, 3)
                    or any(not isinstance(p, list) or len(p) != len(positions[0])
                           or any(type(x) not in (int, float) for x in p) for p in positions)):
                return None
        if 'response_direction' in value and value['response_direction'] not in ('up', 'down'):
            return None
        if 'zero_exclusion_sensitivity' in value and type(value['zero_exclusion_sensitivity']) is not bool:
            return None
        unit = value.get('unit')
        if 'unit' in value and (not isinstance(unit, str) or not unit.strip() or len(unit) > 128):
            return None
        if 'units' in value:
            units = value['units']
            if isinstance(units, str):
                units = [units]
            elif not isinstance(units, list) or len(units) not in ({width, shape[-1]} if landmark else {width}):
                return None
            if (not units or any(not isinstance(u, str) or not u.strip() or len(u) > 128 for u in units)
                    or unit is not None and any(u != unit for u in units)):
                return None
        if 'provenance' in value:
            provenance = value['provenance']
            if (not isinstance(provenance, dict) or provenance.get('authority') not in
                    ('recorded', 'publisher_candidate', 'recording_verified')):
                return None
            if 'source' in provenance:
                if not isinstance(provenance['source'], str):
                    return None
                source = urlsplit(provenance['source'])
                if source.scheme not in ('http', 'https') or not source.netloc:
                    return None
            if any(not isinstance(provenance[k], str) or not provenance[k].strip() or len(provenance[k]) > 512
                   for k in ('revision', 'hash') if k in provenance):
                return None
        return json.loads(encoded)
    except (TypeError, ValueError, OverflowError, RecursionError):
        return None
