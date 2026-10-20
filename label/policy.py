"""Resolve annotation settings once for requests, parts, resume and upload callers."""
from __future__ import annotations

import math

VERSION = 1
ASTRA = 'openai/gpt-6-astra'
SOL61 = 'openai/gpt-6.1-sol'
PERIODS = {'teleop_arms': 1.5, 'handheld_gripper': 1.0, 'ego_head': 0.5}
TASK_MODES = ('episode', 'session')


def resolve(ctx: dict, *, model: str | None = None, reasoning: str | None = None,
            sample_fps: float | None = None, task_mode: str | None = None,
            dataset_note: str | None = None) -> dict:
    """Source declarations and caller overrides, independent of dataset names or absent task text."""
    saved = ctx.get('annotation_policy') or {}
    if saved and (not isinstance(saved, dict) or saved.get('version') != VERSION):
        raise ValueError('annotation policy version is not supported')
    rig = ctx.get('profile') or 'teleop_arms'
    if rig not in PERIODS:
        raise ValueError(f'unknown recording rig {rig!r}')
    model = model or saved.get('model') or ASTRA
    if not isinstance(model, str) or not model.strip():
        raise ValueError('model must be a nonempty identifier')
    sol = model.split('/')[-1] == SOL61.split('/')[-1]
    prior_reasoning = saved.get('reasoning') if saved.get('model') == model else None
    reasoning = reasoning or prior_reasoning or ('high' if sol else 'medium')
    if reasoning not in ('none', 'minimal', 'low', 'medium', 'high', 'xhigh'):
        raise ValueError('unsupported reasoning effort')
    if sample_fps is None:
        sample_fps = saved.get('sample_fps')
    if sample_fps is not None:
        if type(sample_fps) not in (float, int) or not math.isfinite(sample_fps) or sample_fps <= 0:
            raise ValueError('sample FPS must be finite and positive')
        period = 1.0 / sample_fps
    elif saved.get('model') == model and saved.get('sample_every_s') is not None:
        period = saved['sample_every_s']
        if type(period) not in (float, int) or not math.isfinite(period) or period <= 0:
            raise ValueError('saved sampling cadence is invalid')
    else:
        period = 1.0 if sol else PERIODS[rig]
    if sol and period > 1.0:
        raise ValueError('GPT-6.1 Sol requires at least 1 sample per second')
    declared = ctx.get('task_mode')
    if declared is None and ctx.get('sessions') is True:
        declared = 'session'
    mode = task_mode or saved.get('task_mode') or declared or 'episode'
    if mode not in TASK_MODES:
        raise ValueError('task mode must be episode or session')
    note = dataset_note if dataset_note is not None else saved.get('dataset_note', ctx.get('collection_note', ''))
    if not isinstance(note, str):
        raise ValueError('dataset description must be text')
    # Preserve Astra's request budget. Denser sampling reduces the duration allowance proportionally.
    baseline = 1.0 if sol else PERIODS[rig]
    maximum = 450.0 if period >= baseline else 330.0 * min(1.0, period / 0.5)
    if sample_fps is None and saved.get('model') == model and saved.get('part_max_s') is not None:
        previous = saved['part_max_s']
        if type(previous) not in (float, int) or not math.isfinite(previous) or not 0 < previous <= maximum:
            raise ValueError('saved part allowance is incompatible; rebuild with an explicit policy')
        maximum = previous
    return {'version': VERSION, 'rig': rig, 'model': model, 'reasoning': reasoning,
            'sample_fps': float(sample_fps) if sample_fps is not None else None,
            'sample_every_s': period, 'part_max_s': maximum, 'task_mode': mode,
            'dataset_note': note}


def effective(ctx: dict) -> dict:
    return resolve(ctx)


def add_arguments(parser) -> None:
    """Shared optional controls; omitted values inherit source policy or defaults."""
    parser.add_argument('--model', default=None, help='model identifier (default openai/gpt-6-astra)')
    parser.add_argument('--reasoning', default=None, help='reasoning effort (Astra medium; GPT-6.1 Sol high)')
    parser.add_argument('--sample-fps', type=float, default=None,
                        help='optional sampling override; GPT-6.1 Sol defaults to at least 1 FPS')
    parser.add_argument('--task-mode', choices=TASK_MODES, default=None,
                        help='optional task-scope override; source declarations otherwise apply')
    parser.add_argument('--dataset-note', default=None, help='dataset description supplied as recording context')


def persist(episodes, **overrides) -> dict:
    """Write resolved policy into prepared sidecars before cuts or dispatch."""
    import json
    from pathlib import Path
    from label.atomic import write_atomic
    results = {}
    for directory in sorted(Path(episodes).glob('episode_*')):
        path = directory / 'context.json'
        if not path.is_file():
            continue
        ctx = json.loads(path.read_text())
        selected = resolve(ctx, **overrides)
        if ctx.get('annotation_policy') != selected:
            ctx['annotation_policy'] = selected
            write_atomic(path, ctx)
        results[directory.name] = selected
    return results


def require_request_window(ctx: dict, seconds: float) -> None:
    """Refuse an oversized request before routing or final dispatch."""
    maximum = effective(ctx)['part_max_s']
    original = (ctx.get('piece') or {}).get('max_s')
    if original is not None:
        if type(original) not in (float, int) or not math.isfinite(original) or original <= 0:
            raise ValueError('saved request-part allowance is invalid')
        maximum = min(maximum, original)
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError('recording duration is invalid')
    if seconds > maximum + 1e-8:
        raise ValueError(f'Recording window {seconds:g} s exceeds the {maximum:g} s request allowance. '
                         'Prepare request parts with the selected policy before labeling.')
