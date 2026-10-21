"""Reusable MCAP conversion, independent of prompts, sampling and paid results.

Entries contain private copies, never hardlinks to a previous run. A complete
manifest is published only after every video, clock and sidecar has been hashed.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import uuid

from label.atomic import write_atomic

VERSION = 1
# Preparation uses these helpers, but not the annotation prompt/model modules.
HELPERS = ('label/atomic.py', 'label/presented.py', 'label/frames.py',
           'label/signals.py', 'label/depth.py', 'label/source_eligibility.py')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def episode_conversion_identity(root):
    """Hash the episode helpers used by readers, excluding request construction."""
    tree = ast.parse((root / 'label/episode.py').read_text(encoding='utf-8'))
    definitions = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            definitions[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    definitions[target.id] = node
    pending = {'order_views', 'actors', 'frame_runs', 'load', 'anchor', 'frame_time', 'runs_within'}
    selected = {}
    while pending:
        name = pending.pop()
        if name in selected or name not in definitions:
            continue
        node = definitions[name]
        selected[name] = ast.dump(node, include_attributes=False)
        pending.update(n.id for n in ast.walk(node) if isinstance(n, ast.Name)
                       and n.id in definitions and n.id not in selected)
    return hashlib.sha256(json.dumps(selected, sort_keys=True).encode()).hexdigest()


def producer(root):
    paths = list((root / 'prepare').rglob('*.py'))
    paths += [root / name for name in HELPERS]
    paths += list((root / 'configs/sensor_schemas').glob('*.json'))
    result = {p.relative_to(root).as_posix(): sha256(p) for p in paths if p.is_file()}
    result['episode_conversion_helpers'] = episode_conversion_identity(root)
    return result


def source_proof(data, members):
    """Hash full input contents, including files above the old size threshold."""
    return {name: {'sha256': sha256(data.parent / name),
                   'size': (data.parent / name).stat().st_size}
            for name in sorted(members)}


def artifact_proof(root):
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Preparation cache cannot contain symbolic links')
        if path.is_file() and path.name != 'cache.json':
            result[path.relative_to(root).as_posix()] = {'size': path.stat().st_size, 'sha256': sha256(path)}
    return result


def relocate(value, old, new):
    if isinstance(value, dict):
        return {key: relocate(item, old, new) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, old, new) for item in value]
    if isinstance(value, str) and (value == old or value.startswith(old + '/')):
        return new + value[len(old):]
    return value


def copy_episodes(source, destination, *, clean=False):
    """Copy clocks/media as bytes; rebase JSON paths without sharing mutable files."""
    from label.dictionary import DERIVED_CONTEXT
    old, new = str(source.resolve()), str(destination.resolve())
    for path in sorted(source.rglob('*')):
        if path.is_symlink():
            raise ValueError('Prepared episode contains a symbolic link')
        if not path.is_file():
            continue
        relative = path.relative_to(source)
        if any(part.startswith('.') for part in relative.parts):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == '.json':
            document = json.loads(path.read_text(encoding='utf-8'))
            if clean and path.name == 'context.json':
                document = {k: v for k, v in document.items()
                            if k not in DERIVED_CONTEXT | {'annotation_policy', 'pieces'}}
            write_atomic(target, relocate(document, old, new))
        else:
            shutil.copy2(path, target)


@contextmanager
def legacy_lock(job):
    """Read only a stopped old job; do not create or alter its lock file."""
    import fcntl
    with (job / '.execution.lock').open('rb') as stream:
        fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        yield


def import_legacy(destination, current, root, candidates, verify_prepared, verify_file):
    """Adopt existing, intact local runs whose recorded conversion code matches.

    Legacy receipts use their original size/mtime proof for large source files;
    new cache entries additionally hash the full source and every copied artifact.
    """
    for job in candidates:
        try:
            saved = json.loads((job / 'preparation_receipt.json').read_text())
            if any(saved.get(k) != current[k] for k in ('resource', 'source', 'settings')):
                continue
            compatible = {k: v for k, v in current['producer'].items()
                          if k.startswith('prepare/') or k in HELPERS}
            if not compatible or any(saved.get('producer', {}).get(k) != v for k, v in compatible.items()):
                continue
            # Old snapshots do not record helper-level fingerprints. Require their
            # full episode module to match; never guess after reader helper edits.
            if saved.get('producer', {}).get('label/episode.py') != current['producer'].get('label/episode.py'):
                continue
            with legacy_lock(job):
                if saved['prepared'] != verify_prepared(job) or saved['report'] != verify_file(job / 'report.json'):
                    continue
                report = json.loads((job / 'report.json').read_text())
                if not report.get('episodes') or (job / 'episodes_unclipped').exists():
                    continue
                copy_episodes(job / 'episodes', destination / 'episodes', clean=True)
                write_atomic(destination / 'report.json', report)
                return str(job)
        except (OSError, ValueError, KeyError, TypeError):
            continue
    return None


def prepare(job, data, args, dataset, current, *, root, convert, verify_prepared, verify_file):
    """Return report and provenance; None means caching is disabled for this input."""
    cache_root = getattr(args, 'preparation_cache', None)
    if cache_root is None or not data.is_file() or data.suffix.lower() != '.mcap':
        return None
    cache_root = Path(cache_root).resolve()
    if cache_root == job or cache_root.is_relative_to(job) or job.is_relative_to(cache_root):
        raise ValueError('Preparation cache and result directory must be separate')
    identity = {'version': VERSION, 'resource': current['resource'], 'settings': current['settings'],
                'source': source_proof(data, current['source']), 'producer': producer(root)}
    key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    entry = cache_root / key
    from label.harness import execution_lock
    with execution_lock(cache_root / '.locks' / key):
        valid = False
        if (entry / 'cache.json').is_file():
            try:
                saved = json.loads((entry / 'cache.json').read_text())
                valid = saved['identity'] == identity and saved['artifacts'] == artifact_proof(entry)
            except (OSError, ValueError, KeyError):
                pass
        if not valid:
            print('== convert cache miss (preparing or checking existing local results)', flush=True)
            # Keep corrupt entries for diagnosis; only publish fully prepared data.
            if entry.exists():
                entry.rename(cache_root / f'.invalid-{key}-{uuid.uuid4().hex}')
            with tempfile.TemporaryDirectory(prefix='.building-', dir=cache_root) as directory:
                staging = Path(directory)
                candidates = []
                explicit = getattr(args, 'reuse_prepared', None)
                if explicit:
                    candidates.append(Path(explicit).resolve())
                for pattern in ('*/preparation_receipt.json', '*/*/preparation_receipt.json'):
                    candidates.extend(p.parent for p in sorted((root / 'data/review').glob(pattern), reverse=True)
                                      if p.parent.resolve() != job)
                imported = import_legacy(staging, current, root, candidates, verify_prepared, verify_file)
                if not imported:
                    print('== convert (new conversion)', flush=True)
                    # The converter may stage symlinks to the upload beside episodes.
                    # Only its independent prepared artifacts belong in the cache.
                    with tempfile.TemporaryDirectory(prefix='.converting-', dir=cache_root) as work:
                        converted = Path(work) / 'episodes'
                        report = convert(data, args.rig, converted, dataset,
                                         args.max_minutes * 60 if args.max_minutes else float('inf'), grouping=args.grouping)
                        if not report.get('episodes'):
                            raise ValueError('No readable episodes; conversion was not cached')
                        copy_episodes(converted, staging / 'episodes', clean=True)
                        report = relocate(report, str(converted.parent), str(staging))
                    write_atomic(staging / 'report.json', report)
                if identity['source'] != source_proof(data, current['source']) or identity['producer'] != producer(root):
                    raise ValueError('Source or conversion code changed during preparation; cache was not published')
                # Paths must reference the permanent location, not the temporary one.
                for path in staging.rglob('*.json'):
                    write_atomic(path, relocate(json.loads(path.read_text()), str(staging), str(entry)))
                write_atomic(staging / 'cache.json', {'identity': identity, 'artifacts': artifact_proof(staging),
                                                     'imported_from': imported})
                staging.rename(entry)
                print('== convert cache saved' + (f' (imported {imported})' if imported else ''), flush=True)
        print('== convert (reused cached videos and time mappings)', flush=True)
        copy_episodes(entry / 'episodes', job / 'episodes', clean=True)
        report = relocate(json.loads((entry / 'report.json').read_text()), str(entry), str(job))
        return report, {'key': key, 'path': str(entry), 'hit': valid,
                        'imported_from': json.loads((entry / 'cache.json').read_text()).get('imported_from')}
