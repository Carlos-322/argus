"""Apply bounded publisher data contracts only to complete matching recording schemas."""
import copy
import hashlib
import io
import json
from pathlib import Path

from prepare.sensor_semantics import MAX_BYTES, validate

SCHEMA_DIR = Path(__file__).resolve().parent.parent / 'configs' / 'sensor_schemas'


def _contracts():
    for path in sorted(SCHEMA_DIR.glob('*.json')):
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
            raw = path.read_bytes()
            contract = json.loads(raw)
            if (isinstance(contract, dict) and type(contract.get('version')) is int and contract['version'] == 1
                    and isinstance(contract.get('id'), str)
                    and isinstance(contract.get('required_numeric'), dict)
                    and isinstance(contract.get('signals'), dict)):
                yield contract, hashlib.sha256(raw).hexdigest()
        except (OSError, ValueError, TypeError):
            continue


def _jpeg_rgb(dataset):
    from PIL import Image
    try:
        if len(dataset.shape) > 2:
            return False
        raw = dataset[0]
        raw = raw.tobytes() if hasattr(raw, 'tobytes') else bytes(raw)
        if len(raw) > 16 * 1024 * 1024 or not raw.startswith(b'\xff\xd8\xff'):
            return False
        with Image.open(io.BytesIO(raw)) as image:
            matched = image.format == 'JPEG' and image.mode == 'RGB'
            image.verify()
            return matched
    except (OSError, TypeError, ValueError, IndexError):
        return False


def hdf5_candidates(source, streams):
    """Return source-path candidates from complete, colocated field shapes and an actual RGB JPEG stream.

    File names, episode IDs and caller dataset labels never participate in matching. The existing HDF inventory
    supplies admitted source objects; no extra source paths or neighboring recordings are searched.
    """
    groups = {}
    for stream in streams.get('signal') or []:
        path = stream['path']
        parent, _, leaf = path.rpartition('/')
        groups.setdefault(parent, {})[leaf] = path
    candidates = {}
    for contract, digest in _contracts():
        required = contract['required_numeric']
        if not required or contract.get('required_image') != {'encoding': 'JPEG', 'mode': 'RGB'}:
            continue
        for parent, members in groups.items():
            if not all(name in members for name in required):
                continue
            try:
                datasets = {name: source[members[name]] for name in required}
                lengths = {len(dataset) for dataset in datasets.values()}
                if (len(lengths) != 1 or next(iter(lengths)) < 2 or any(
                        dataset.dtype.kind not in 'iuf' or list(dataset.shape[1:]) != required[name]
                        for name, dataset in datasets.items())):
                    continue
                count = next(iter(lengths))
                images = [source[stream['path']] for stream in streams.get('camera') or []
                          if (not parent or stream['path'].startswith(parent + '/'))
                          and stream.get('n') == count]
                if not any(_jpeg_rgb(image) for image in images):
                    continue
                for name, template in contract['signals'].items():
                    if name not in datasets:
                        continue
                    descriptor = validate(template, list(datasets[name].shape[1:]))
                    if descriptor is not None:
                        descriptor['publisher_schema'] = {'id': contract['id'], 'version': 1, 'hash': digest}
                        candidates[members[name]] = descriptor
            except (OSError, KeyError, TypeError, ValueError):
                continue
    return candidates


def with_candidate(metadata, candidate, shape):
    """Add qualified topology without replacing explicit recorded semantics, units, calibration or polarity."""
    if not candidate or 'sensor_semantics' in metadata:
        return metadata
    descriptor = copy.deepcopy(candidate)
    overrides = []
    incompatible_units = False
    if any(key in metadata for key in ('unit', 'units')):
        candidate_units = descriptor.get('units', descriptor.get('unit'))
        recorded_units = metadata.get('units', metadata.get('unit'))
        candidate_units = candidate_units if isinstance(candidate_units, list) else [candidate_units]
        recorded_units = recorded_units if isinstance(recorded_units, (list, tuple)) else [recorded_units]
        # A convention for raw ADC codes does not describe a converted force or
        # pressure export. Recorded units can replace units without transferring
        # the publisher's polarity to that different representation.
        incompatible_units = (candidate_units != recorded_units and not (
            len(candidate_units) == 1 and all(unit == candidate_units[0] for unit in recorded_units) or
            len(recorded_units) == 1 and all(unit == recorded_units[0] for unit in candidate_units)))
        descriptor.pop('unit', None)
        descriptor.pop('units', None)
        for key in ('unit', 'units'):
            if key in metadata:
                descriptor[key] = metadata[key]
                overrides.append(key)
    if incompatible_units or 'response_direction' in metadata or 'calibration' in metadata:
        descriptor.pop('response_direction', None)
        if metadata.get('response_direction') in ('up', 'down'):
            descriptor['response_direction'] = metadata['response_direction']
        overrides.extend(key for key in ('response_direction', 'calibration') if key in metadata)
    if incompatible_units:
        limits = descriptor.get('limitations')
        limits = list(limits) if isinstance(limits, list) else []
        limits.append('Recorded units differ from the publisher representation; its response direction is not applied unless explicitly declared by this recording.')
        descriptor['limitations'] = limits
    if overrides:
        descriptor['recorded_overrides'] = overrides
    descriptor = validate(descriptor, shape)
    return {**metadata, 'sensor_semantics': descriptor} if descriptor is not None else metadata
