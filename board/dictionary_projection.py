"""Current public dictionary and evidence projection over an unchanged saved label."""
from __future__ import annotations

import copy
import json
from pathlib import Path


from label.dictionary_context import camera_content_role, TACTILE_CONTENT_ROLES


def current_camera_observations(record: dict, ctx: dict, episode_dir: Path) -> list:
    """Revalidate saved observation receipts against current native inputs and owned request parts."""
    if not isinstance(record, dict) or record.get('version') != 1:
        return []
    from label.harness import bind_camera_observations, camera_observation_sources_match
    from label.evidence_access import same_source_proof, resolve_piece
    try:
        checked = bind_camera_observations(record.get('observations'), {
            'config': {'views': record.get('views'), 'camera_observation_views': record.get('views'),
                       'schema_fields': ['camera_observations']},
            'input_identity': {'source_proof': record.get('source_proof')}})
        valid = (not checked['withheld_observations'] and
                 same_source_proof(record.get('source_proof'), ctx, episode_dir))
        for part in record.get('parts') or []:
            owner = part.get('source_piece') or {}
            directory = resolve_piece(episode_dir, part.get('source_location'), owner, ctx)
            piece_ctx = json.loads((directory / 'context.json').read_text()) if directory else None
            child = part.get('record')
            if not isinstance(piece_ctx, dict) or not isinstance(child, dict):
                valid = False
                continue
            current_camera_observations(child, piece_ctx, directory)
            valid = (valid and child.get('source_current') is True and
                     camera_observation_sources_match(record.get('source_proof'), child.get('source_proof'),
                                                      child.get('views')))
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        valid = False
    record['source_current'] = bool(valid)
    if not valid:
        withheld = record.setdefault('withheld_observations', [])
        for row in record.get('observations') or []:
            item = {'observation': copy.deepcopy(row), 'why': 'Saved camera observations use changed or unverified source inputs.'}
            if item not in withheld:
                withheld.append(item)
        record['observations'] = []
    return record.get('observations') or []


def camera_presentation(ctx: dict, observations: list = ()) -> dict:
    """Display roles from exact bindings or explicit preparation facts, never camera names."""
    from board.clips import unshown_views
    from label.dictionary_context import prepared_interpretation, field_interpretation
    from label import signals as sg
    entries = list((ctx.get('cameras') or {}).items()) + unshown_views(ctx)
    retained_index = {id(entry): index for index, entry in enumerate(ctx.get('unshown_cameras') or [])}
    presentation = {}
    observed = {row['view']: row for row in observations if row.get('content') in ('scene', 'sensor', 'unclear')}
    for view, entry in entries:
        authoritative = camera_content_role(ctx, view, entry)
        if authoritative is not None:
            presentation[view] = {'role': authoritative['role']}
            continue
        if view in observed:
            presentation[view] = {'role': 'scene' if observed[view]['content'] == 'scene' else
                                  'sensor' if observed[view]['content'] == 'sensor' else 'other',
                                  'provenance': 'observed'}
            continue
        if view in (ctx.get('cameras') or {}):
            interpretation = prepared_interpretation(ctx, view, 'camera')
        else:
            path = 'unshown_cameras/' + str(retained_index[id(entry)])
            interpretation = field_interpretation(ctx, entry.get('name'), 'camera', context_path=path)
        role = sg._role(interpretation.get('role') if 'role' in interpretation else entry.get('role'))
        if role:
            presentation[view] = {'role': 'tactile' if role in
                TACTILE_CONTENT_ROLES else 'sensor' if role in ('sensor', 'sensor_view') else 'scene' if role in
                ('scene', 'scene_camera', 'rgb_camera', 'head_camera', 'ego_camera', 'wrist_camera', 'external_camera')
                else 'other'}
    return presentation


def project(label: dict, episode_dir: Path, owner: str | None = None) -> dict:
    """Return the current interpretation without writing to the label or prepared source."""
    shown = copy.deepcopy(label)
    episode_dir = Path(episode_dir)
    try:
        ctx = json.loads((episode_dir / 'context.json').read_text())
    except (OSError, ValueError):
        ctx = {}
    return apply(shown, ctx, episode_dir, owner)


def apply(shown: dict, ctx: dict, episode_dir: Path, owner: str | None = None) -> dict:
    """Apply the same projection while board/build already has the prepared context."""
    from label.dictionary_editor import context_dictionary, for_episode

    owner = owner or (ctx.get('piece') or {}).get('of') or ctx.get('episode_id') or episode_dir.name
    dictionary = for_episode(episode_dir, owner)
    if dictionary is None and isinstance(ctx.get('data_dictionary'), dict):
        dictionary = context_dictionary(ctx['data_dictionary'])
    old = shown.get('data_dictionary') or ctx.get('data_dictionary') or {}
    if dictionary is not None:
        shown['data_dictionary'] = dictionary
    effective_ctx = {**ctx, **({'data_dictionary': dictionary} if dictionary is not None else {})}
    # Camera presentation follows current exact bindings, including human overrides.
    # It does not reassign cameras or alter the clock on which labels were produced.
    observations = current_camera_observations(shown.get('camera_observations'), effective_ctx, episode_dir)
    presentation = camera_presentation(effective_ctx, observations)
    shown.pop('camera_presentation', None)
    if presentation:
        shown['camera_presentation'] = presentation
    if isinstance(shown.get('evidence_inspection'), dict):
        from label.evidence_access import same_source_proof, withhold_stale_untimed, resolve_piece
        inspection = shown['evidence_inspection']
        withhold_stale_untimed(inspection, effective_ctx, episode_dir)
        if inspection.get('inspections') or inspection.get('parts'):
            inspection['source_current'] = same_source_proof(
                inspection.get('source_proof'), effective_ctx, episode_dir)
            if not inspection['source_current']:
                inspection.setdefault('limitations', []).append(
                    'Saved inspection regions use changed or missing source data; receipts withheld pending refresh.')
        for part in inspection.get('parts') or []:
            if not isinstance(part, dict) or not isinstance(part.get('record'), dict):
                continue
            record = part['record']
            directory = resolve_piece(episode_dir, part.get('source_location'), {'index': part.get('part')}, ctx)
            try:
                piece_ctx = json.loads((directory / 'context.json').read_text()) if directory else None
            except (OSError, ValueError):
                piece_ctx = None
            record['source_current'] = bool(
                inspection.get('source_current') and isinstance(piece_ctx, dict)
                and same_source_proof(record.get('source_proof'), piece_ctx, directory))
    if isinstance(shown.get('sensor_evidence'), dict) or isinstance(shown.get('grip_evidence'), dict):
        from label.sensor_evidence import compatible
    if isinstance(shown.get('sensor_evidence'), dict):
        if not compatible(shown['sensor_evidence'], effective_ctx, episode_dir):
            saved = shown['sensor_evidence']
            withheld = saved.setdefault('withheld_findings', [])
            withheld.extend(f for f in saved.get('findings') or [] if f not in withheld)
            saved['findings'] = []
            note = 'Sensor source, interpretation or timing changed since labeling; saved findings withheld pending refresh.'
            if note not in saved.setdefault('limitations', []):
                saved['limitations'].append(note)
    if isinstance(shown.get('grip_evidence'), dict) and shown['grip_evidence'].get('version') == 1:
        if not compatible(shown['grip_evidence'], effective_ctx, episode_dir):
            shown['withheld_grip_evidence'] = shown.pop('grip_evidence')
            shown.setdefault('dictionary_projection_limitations', []).append(
                'Saved legacy grip findings have no verifiable current source and interpretation proof.')
    if isinstance(shown.get('jaw_evidence'), dict):
        from label.evidence_access import same_source_proof, resolve_piece
        for view, receipt in list(shown['jaw_evidence'].items()):
            if not isinstance(receipt, dict) or receipt.get('status') == 'abstained':
                continue
            current = same_source_proof(receipt.get('source_proof'), effective_ctx, episode_dir)
            if not current:
                shown.setdefault('withheld_jaw_evidence', {})[view] = shown['jaw_evidence'].pop(view)
                continue
            receipt['source_current'] = True
            for part in receipt.get('parts') or []:
                if not isinstance(part, dict) or not isinstance(part.get('record'), dict):
                    continue
                directory = resolve_piece(episode_dir, part.get('source_location'), {'index': part.get('part')}, ctx)
                try:
                    piece_ctx = json.loads((directory / 'context.json').read_text()) if directory else None
                except (OSError, ValueError):
                    piece_ctx = None
                valid = isinstance(piece_ctx, dict) and same_source_proof(part.get('source_proof'), piece_ctx, directory)
                if not valid:
                    part['withheld_record'] = part.pop('record')
                    part['withheld_events'] = part.pop('events', [])
                    part['source_current'] = False
                else:
                    part['source_current'] = part['record']['source_current'] = True
        if shown.get('withheld_jaw_evidence') or any(
                part.get('source_current') is False for receipt in shown['jaw_evidence'].values()
                if isinstance(receipt, dict) for part in receipt.get('parts') or [] if isinstance(part, dict)):
            shown.setdefault('dictionary_projection_limitations', []).append(
                'Saved jaw measurements use changed or missing source data and are withheld pending refresh.')
    changed = (isinstance(old.get('entries'), dict) and
               (dictionary is None or old['entries'] != dictionary.get('entries')))
    if changed:
        for key in ('contacts', 'contacts_missing', 'tactile_qc'):
            if shown.get(key):
                shown['withheld_' + key] = shown.pop(key)
        checks = shown.get('dataset_checks')
        if isinstance(checks, dict) and 'contact_checks' in checks:
            shown['withheld_contact_checks'] = checks.pop('contact_checks')
        if any(key in shown for key in ('withheld_contacts', 'withheld_contacts_missing',
                                         'withheld_contact_checks', 'withheld_tactile_qc')):
            shown.setdefault('dictionary_projection_limitations', []).append(
                'Saved contact claims use an earlier sensor interpretation and are withheld pending refresh.')
    return shown
