"""Explicit offline reconstruction of the frame grids shown to the model.

    python -m board grids BOARD [--jobs 4] [--dataset NAME]

Missing request hashes, changed inputs or differing image bytes leave the episode
without grids. Supplemental detail and sensor images are counted but not rebuilt.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import re
from pathlib import Path

from label import grids as lg


def grid_dir(grids: Path, file: str) -> Path:
    if not file or Path(file).name != file or '\\' in file or file.startswith('.'):
        raise ValueError('invalid episode grid name')
    folder = Path(grids) / Path(file).stem
    if not folder.resolve().is_relative_to(Path(grids).resolve()):
        raise ValueError('episode grid path escapes its root')
    return folder


def matches(idx: dict, record: dict) -> bool:
    expected = record.get('request_images') or (record.get('config') or {}).get('request_images') or {}
    saved = (idx or {}).get('request_images') or {}
    if not isinstance(expected, dict) or not isinstance(saved, dict):
        return False
    if not isinstance(idx.get('grids'), list) or not all(isinstance(g, dict) for g in idx['grids']):
        return False
    if not isinstance(expected.get('grid_images'), list) or not all(
            isinstance(g, dict) for g in expected['grid_images']):
        return False
    hashes = [g.get('sha256') for g in expected.get('grid_images') or []]
    return (idx.get('check') == 'sha256' and bool(hashes)
            and isinstance(expected.get('fingerprint'), str)
            and re.fullmatch(r'[0-9a-f]{64}', expected['fingerprint']) is not None
            and expected.get('fingerprint') == saved.get('fingerprint')
            and hashes == [g.get('sha256') for g in saved.get('grid_images') or []]
            and hashes == [g.get('sha256') for g in idx.get('grids') or []]
            and all(all(actual.get(k) == wanted[k] for k in
                        ('sha256', 'bytes', 'width', 'height', 't0_s', 't1_s', 'part') if k in wanted)
                    for actual, wanted in zip(idx.get('grids') or [], expected['grid_images'])))


def image_path(grids: Path, file: str, entry: dict) -> Path:
    folder = grid_dir(grids, file)
    if not isinstance(entry, dict):
        raise ValueError('invalid grid image receipt')
    name = entry.get('file') or ''
    if not re.fullmatch(r'grid_[0-9]+\.jpg', name):
        raise ValueError('invalid grid image name')
    path = folder / name
    if not path.resolve().is_relative_to(folder.resolve()) or not path.is_file():
        raise ValueError('grid image is outside its episode folder')
    raw = path.read_bytes()
    if len(raw) != entry.get('bytes') or lg.digest(raw) != entry.get('sha256'):
        raise ValueError('grid image does not match the verified request')
    import io
    from PIL import Image
    with Image.open(io.BytesIO(raw)) as image:
        if image.size != (entry.get('width'), entry.get('height')):
            raise ValueError('grid dimensions do not match the verified image')
    return path


def read(grids: Path | None, file: str, record: dict) -> dict | None:
    if grids is None:
        return None
    try:
        idx = json.loads((grid_dir(grids, file) / 'grids.json').read_text())
        if not isinstance(idx, dict) or not matches(idx, record):
            return None
        for g in idx['grids']:
            image_path(grids, file, g)
        # Dispatch status can change after the exact same input has been reconstructed.
        return {**idx, 'request_images': lg.public_receipt(record.get('request_images')
                or (record.get('config') or {}).get('request_images'))}
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def view(idx: dict, src) -> dict:
    receipt = idx['request_images']
    rows = []
    for i, g in enumerate(idx['grids']):
        actual = (receipt.get('grid_images') or [])[i]
        status = actual.get('dispatch_status', receipt.get('dispatch_status', 'unknown'))
        rows.append({**{k: g[k] for k in ('t0_s', 't1_s', 'width', 'height', 'bytes', 'part') if k in g},
                     'dispatch_status': status, 'src': src(i, g)})
    return {'fingerprint': receipt['fingerprint'],
            'supplemental_images': receipt.get('supplemental_images'),
            **{k: receipt[k] for k in ('dispatch_status', 'coverage', 'parts', 'unavailable_parts') if k in receipt},
            'grids': rows}


def build(board: Path, jobs: int = 4, datasets: list | None = None, quiet: bool = False,
          files: list[str] | None = None) -> dict:
    from board import build as bb
    board = Path(board)
    manifest = json.loads((board / 'manifest.json').read_text())
    pending, kept = [], 0
    for entry in manifest.get('datasets') or []:
        if datasets and entry['dataset'] not in datasets:
            continue
        episodes = bb._path(entry['episodes'], board.resolve())
        for name, (episode, _source, result, _run) in sorted(bb.entry_labels(entry, board.resolve())[1].items()):
            if files is not None and name not in files:
                continue
            if read(board / 'grids', name, result):
                kept += 1
            else:
                pending.append((name, result, episodes / episode))
    summary = {'rebuilt': 0, 'kept': kept, 'failed': {}, 'bytes': 0}
    def one(job):
        name, result, ep = job
        try:
            idx = lg.write(grid_dir(board / 'grids', name), result, lg.rebuild(result, ep))
            return name, sum(g['bytes'] for g in idx['grids']), None
        except (OSError, ValueError, KeyError, TypeError, AttributeError, ArithmeticError, RuntimeError):
            return name, 0, 'Original frame-grid inputs could not be verified.'
    with cf.ThreadPoolExecutor(max(1, jobs)) as executor:
        for name, size, error in executor.map(one, pending):
            if error:
                summary['failed'][name] = error
            else:
                summary['rebuilt'] += 1
                summary['bytes'] += size
    if not quiet:
        print(json.dumps(summary, indent=2))
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('board', type=Path)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--dataset', action='append')
    parser.add_argument('--file', action='append', help='only this episode JSON file')
    args = parser.parse_args()
    result = build(args.board, args.jobs, args.dataset, files=args.file)
    return int(bool(result['failed']))
