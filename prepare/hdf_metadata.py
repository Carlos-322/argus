"""Bounded reads of original HDF5 metadata, without duplicating source arrays."""
from pathlib import Path
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

MAX_VALUES = 4096
_SOURCES = ContextVar('hdf_sources', default=None)


MAX_SCANNED_OBJECTS = 2_000_000
_SCANS = {}


class ForeignStorage(ValueError):
    """An HDF5 file whose links or storage name files outside the upload. It is never opened for reading."""


def _allowed_now(path):
    allowed = _SOURCES.get()
    return frozenset(allowed) if allowed is not None else frozenset({Path(path).resolve()})


def foreign_references(path, allowed=None, _active=None):
    """Why reading this HDF5 file could touch a file outside `allowed`, or [] when it cannot.

    HDF5 resolves a link's or virtual dataset's file next to the HDF5 file, but falls back to the working directory
    when that file is missing, and it resolves external raw storage against the working directory and never
    against the file. An absolute name ignores every prefix. So a reader that opens such a file can be steered to
    any file on the host by a fixed dataset name, whatever the walker admitted. One scan of every object at open
    is the only boundary every read passes through."""
    import h5py
    path = Path(path).resolve()
    allowed = frozenset(allowed) if allowed is not None else _allowed_now(path)
    stat = path.stat()
    key = (path, stat.st_size, stat.st_mtime_ns, allowed)
    if key in _SCANS:
        return _SCANS[key]
    active = set() if _active is None else _active
    if path in active:
        return []
    active.add(path)
    problems = []
    try:
        with h5py.File(path, 'r') as f:
            seen, pending, count = set(), [f], 0
            while pending and not problems:
                group = pending.pop()
                try:
                    keys = list(group)
                except (KeyError, OSError, RuntimeError, ValueError):
                    continue          # an unreadable group cannot be read through by any later reader either
                for key_name in keys:
                    count += 1
                    if count > MAX_SCANNED_OBJECTS:
                        problems.append(f'more than {MAX_SCANNED_OBJECTS:,} objects to verify')
                        break
                    name = group.name.rstrip('/') + '/' + key_name
                    try:
                        link = group.get(key_name, getlink=True)
                    except (KeyError, OSError, RuntimeError, ValueError):
                        continue
                    if isinstance(link, h5py.ExternalLink):
                        target = (path.parent / link.filename).resolve()
                        if target not in allowed:
                            problems.append(f'{name} links to {link.filename}')
                        elif target != path:
                            nested = foreign_references(target, allowed, active)
                            if nested:
                                problems.append(f'{name} links to {link.filename}, which {nested[0]}')
                        continue
                    if isinstance(link, h5py.SoftLink):
                        continue      # names an object in this file, which the hard links reach
                    try:
                        obj = group[key_name]
                    except (KeyError, OSError, RuntimeError, ValueError):
                        continue
                    if isinstance(obj, h5py.Group):
                        address = h5py.h5o.get_info(obj.id).addr
                        if address not in seen:
                            seen.add(address)
                            pending.append(obj)
                        continue
                    if not isinstance(obj, h5py.Dataset):
                        continue
                    try:
                        if obj.external:
                            problems.append(f'{name} stores its values in {obj.external[0][0]}')
                            break
                        sources = obj.virtual_sources() if obj.is_virtual else []
                    except (KeyError, OSError, RuntimeError, ValueError):
                        continue
                    for source in sources:
                        target = path if source.file_name == '.' else (path.parent / source.file_name).resolve()
                        if target not in allowed:
                            problems.append(f'{name} reads values from {source.file_name}')
                            break
                        if target != path and target.exists():
                            nested = foreign_references(target, allowed, active)
                            if nested:
                                problems.append(f'{name} reads values from {source.file_name}, which {nested[0]}')
                                break
                    if problems:
                        break
    finally:
        active.discard(path)
    _SCANS[key] = problems
    return problems


def open_hdf5(path, allowed=None):
    """Open an uploaded HDF5 file for reading, or raise ForeignStorage when anything in it names another file."""
    import h5py
    problems = foreign_references(path, allowed)
    if problems:
        raise ForeignStorage(f'{Path(path).name} was not read because it names storage outside the upload '
                             f'({problems[0]})')
    return h5py.File(path, 'r')


@contextmanager
def authorized_sources(paths):
    """Only files selected by the upload may supply linked HDF objects or storage."""
    token = _SOURCES.set(frozenset(Path(p).resolve() for p in paths))
    try:
        yield
    finally:
        _SOURCES.reset(token)


def object_id(obj):
    import h5py
    return (str(Path(obj.file.filename).resolve()), h5py.h5o.get_info(obj.id).addr)


def storage_admitted(obj, allowed, seen=None):
    """Resolve virtual source datasets too, since their paths can hide external links."""
    import h5py
    path = Path(obj.file.filename).resolve()
    if path not in allowed:
        return False
    if not isinstance(obj, h5py.Dataset):
        return True
    if any((path.parent / name).resolve() not in allowed for name, _, _ in obj.external or []):
        return False
    if not obj.is_virtual:
        return True
    seen = set() if seen is None else seen
    identity = object_id(obj)
    if identity in seen:
        return False
    seen.add(identity)
    try:
        for source in obj.virtual_sources():
            filename = path if source.file_name == '.' else (path.parent / source.file_name).resolve()
            if filename not in allowed:
                return False
            with h5py.File(filename, 'r') as linked:
                if not storage_admitted(linked[source.dset_name], allowed, seen):
                    return False
        return True
    finally:
        seen.remove(identity)


def children(group, issues=None):
    import h5py
    allowed = _SOURCES.get()
    if allowed is None:
        allowed = {Path(group.file.filename).resolve()}

    def admitted(path):
        return Path(path).resolve() in allowed

    def excluded(name):
        if issues is not None:
            issue = {'kind': 'external_source_unread',
                     'what': f'HDF5 object {name} references storage outside the selected upload; '
                             'its external values were not read.'}
            if issue not in issues:
                issues.append(issue)

    for key in group:
        name = group.name.rstrip('/') + '/' + key
        try:
            link = group.get(key, getlink=True)
            if isinstance(link, h5py.ExternalLink):
                target = Path(group.file.filename).resolve().parent / link.filename
                if not admitted(target):
                    excluded(name)
                    continue
            obj = group[key]
            if not storage_admitted(obj, allowed):
                excluded(name)
                continue
            yield key, obj
        except (KeyError, OSError):
            continue


def objects(group, prefix='', issues=None):
    """Walk each owned group once, including legal hard-link and soft-link cycles."""
    import h5py
    seen = {object_id(group)}
    pending = [(prefix, iter(children(group, issues)))]
    while pending:
        base, entries = pending[-1]
        try:
            key, obj = next(entries)
        except StopIteration:
            pending.pop()
            continue
        name = f'{base}/{key}' if base else key
        yield name, obj
        if isinstance(obj, h5py.Group) and object_id(obj) not in seen:
            seen.add(object_id(obj))
            pending.append((name, iter(children(obj, issues))))


class NumericRows:
    """Read healthy numeric rows when an individual HDF chunk fails its checksum."""
    def __init__(self, dataset, issues):
        self.dataset, self.issues = dataset, issues
        self.shape = dataset.shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        try:
            return self.dataset[key]
        except OSError:
            rows = range(len(self))[key if isinstance(key, slice) else slice(None)]
            result = np.full((len(rows), *self.shape[1:]), np.nan)
            missing = []
            for destination, row in enumerate(rows):
                try:
                    result[destination] = self.dataset[row]
                except OSError:
                    missing.append(row)
            if missing:
                self.issues.append({'kind': 'signal_bad_cells', 'signal': self.dataset.name,
                                    'what': f'{self.dataset.name}: {len(missing)} numeric rows could not '
                                            'be read from damaged HDF5 chunks and remain missing.'})
            return result


def value(x):
    if isinstance(x, bytes):
        return x.decode('utf-8', 'replace')
    if isinstance(x, np.ndarray):
        if x.size > MAX_VALUES:
            return {'shape': list(x.shape), 'dtype': str(x.dtype), 'truncated': True,
                    'preview': [value(y) for y in x.ravel()[:50]],
                    'note': 'The complete metadata array remains in the original HDF5 file.'}
        return value(x[()]) if x.ndim == 0 else [value(y) for y in x]
    if isinstance(x, np.void) and x.dtype.names:
        return {name: value(x[name]) for name in x.dtype.names}
    if isinstance(x, np.generic):
        return value(x.item())
    if isinstance(x, (list, tuple)):
        return [value(y) for y in x]
    if x is None or isinstance(x, (str, bool, int, float)):
        return x
    return {'unsupported_type': type(x).__name__}


def inventory(f, group, recording_paths, shared_paths, episode_groups):
    path = Path(f.filename).resolve()
    stat = path.stat()
    fields = []
    datasets = {p: 'recording' for p in recording_paths}
    datasets.update({p: 'file' for p in shared_paths})
    for name, scope in sorted(datasets.items()):
        ds = f[name]
        fields.append({'name': f'HDF5 {scope} {name}', 'object': name, 'scope': scope,
                       'shape': list(ds.shape) if ds.shape is not None else None, 'dtype': str(ds.dtype)})

    def attributes(name, obj):
        local = not group or name == group or name.startswith(group + '/')
        other_episode = any(e and (name == e or name.startswith(e + '/')) for e in episode_groups)
        if not local and other_episode:
            return
        scope = 'recording' if local else 'file'
        for key in obj.attrs:
            raw = obj.attrs.get_id(key)
            fields.append({'name': f'HDF5 {scope} {name or "/"} attribute {key}',
                           'object': name, 'attribute': key, 'scope': scope,
                           'shape': list(raw.shape) if raw.shape is not None else None, 'dtype': str(raw.dtype)})

    attributes('', f)
    # h5o.visit can traverse a damaged dataset B-tree even when its useful peers and attributes are readable.
    for name, obj in objects(f):
        attributes(name, obj)
    return {'source_path': str(path), 'source_size': stat.st_size, 'source_mtime_ns': stat.st_mtime_ns,
            'source_required_for_inspection': True, 'fields': fields}


def check_source(source):
    path = Path(source['source_path'])
    try:
        stat = path.stat()
    except OSError as error:
        raise ValueError('Original HDF5 metadata source is unavailable') from error
    if stat.st_size != source['source_size'] or stat.st_mtime_ns != source['source_mtime_ns']:
        raise ValueError('Original HDF5 metadata source changed after preparation')
    return path


def read(source, field, pointer):
    import h5py
    path = check_source(source)
    tokens = [t.replace('~1', '/').replace('~0', '~') for t in pointer.split('/')[1:]]
    with open_hdf5(path, {Path(path).resolve()}) as f:
        obj = f[field['object'] or '/']
        if Path(obj.file.filename).resolve() != path.resolve():
            raise ValueError('External HDF5 metadata requires its own source inventory')
        if 'attribute' in field:
            info = obj.attrs.get_id(field['attribute'])
            if info.shape is None:
                raise ValueError('Original HDF5 attribute has a null dataspace and contains no values')
            if np.prod(info.shape, dtype=object) * info.dtype.itemsize > 256 * 1024 * 1024:
                raise ValueError('Original HDF5 attribute exceeds the metadata read budget')
            selected = obj.attrs[field['attribute']]
        else:
            if obj.shape is None:
                raise ValueError('Original HDF5 field has a null dataspace and contains no values')
            if obj.is_virtual or obj.external:
                raise ValueError('External HDF5 storage requires its own source inventory')
            indices = []
            while tokens and len(indices) < obj.ndim and tokens[0].isdecimal():
                index = int(tokens.pop(0))
                if index >= obj.shape[len(indices)]:
                    raise ValueError('metadata pointer is not present')
                indices.append(index)
            if np.prod(obj.shape[len(indices):], dtype=object) > MAX_VALUES:
                raise ValueError('Metadata selection exceeds 4096 values; request a deeper JSON pointer')
            selected = obj[tuple(indices) if indices else ()]
        try:
            for token in tokens:
                if isinstance(selected, (dict, np.void)):
                    selected = selected[token]
                else:
                    if not token.isdecimal():
                        raise ValueError('metadata array pointer requires a nonnegative index')
                    selected = selected[int(token)]
        except (IndexError, KeyError, TypeError, ValueError) as error:
            raise ValueError('metadata pointer is not present') from error
        if isinstance(selected, np.ndarray) and selected.size > MAX_VALUES:
            raise ValueError('Metadata selection exceeds 4096 values; request a deeper JSON pointer')
        return value(selected)
