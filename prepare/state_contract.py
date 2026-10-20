"""Qualify prepared physical state before new computation without changing saved labels or readings."""
from __future__ import annotations

VERSION = 1
ACTION_ROLES = frozenset({'command', 'target', 'observation', 'action/target_unknown'})
GENERIC_PRODUCERS = frozenset({'prepare.formats:lerobot', 'prepare.formats:hdf5',
                              'prepare.formats:mcap-joints', 'prepare.mcap_pose:pose'})
# Recorded adapter identity is preparation provenance, not a dataset-name heuristic.
NATIVE_KINDS = {'abc130k': 'joints', 'galaxea': 'joints', 'molmo': 'joints',
                'habit': 'ee_pose', 'realomin': 'ee_pose', 'fastumi': 'ee_pose'}
REBUILD = ('Rebuild this generic preparation from its original source before new physical-state computation: '
           'the saved context does not prove the supported state roles and units. '
           'Existing readings and historical labels must remain unchanged.')


def declared_contract(producer, kind, names, units, roles=None, action=None, unit_conversion=None):
    """A producer's checked role/unit facts in prepared column order, with the record of any columns it converted from
    degrees (prepare.formats.qualified_state), whose names may still say degrees."""
    from prepare.formats import physical_state_layout
    names, units = list(names), list(units)
    roles = list(roles) if roles else [None] * len(names)
    actual, _ = physical_state_layout(len(names), 'teleop_arms', names, metadata={'units': units},
                                      value_roles=roles, conversion=unit_conversion)
    if producer not in GENERIC_PRODUCERS or actual != kind or kind == 'none':
        raise ValueError('Cannot stamp an unqualified physical state contract')
    contract = {'version': VERSION, 'producer': producer, 'kind': kind,
                'names': names, 'units': units, 'value_roles': roles}
    if unit_conversion is not None:
        contract['unit_conversion'] = unit_conversion
    if action is not None:
        contract['action'] = action
    return contract


def native_contract(adapter, kind, action_metadata=None):
    """Stamp an adapter selected by the preparation caller, never by an input dataset name."""
    if NATIVE_KINDS.get(adapter) != kind:
        return None
    result = {'version': VERSION, 'producer': 'prepare.' + adapter, 'kind': kind,
              'native_contract': adapter}
    if adapter in ('habit', 'molmo'):
        result['action_semantics'] = declared_action_semantics('prepare.' + adapter, 'command')
    elif adapter == 'fastumi':
        result['action_semantics'] = declared_action_semantics('prepare.fastumi', 'target', relationship={
            'kind': 'next_observation', 'field': 'state', 'row_offset': 1, 'terminal': 'repeat_last',
            'scope': 'source_recording'})
    publisher = publisher_action_semantics(action_metadata, 'prepare.' + adapter + ':publisher')
    if publisher is not None:
        result['publisher_action_semantics'] = publisher
    return result


def declared_action_semantics(producer, role, *, relationship=None):
    """Source facts about an action array, independent of its physical column layout."""
    if not isinstance(producer, str) or not producer or role not in ACTION_ROLES:
        raise ValueError('Invalid source action semantics')
    result = {'version': VERSION, 'producer': producer, 'role': role}
    if relationship is not None:
        if (role != 'target' or not isinstance(relationship, dict) or relationship.get('kind') != 'next_observation'
                or relationship.get('field') != 'state' or type(relationship.get('row_offset')) is not int
                or relationship['row_offset'] != 1 or relationship.get('terminal') != 'repeat_last'
                or relationship.get('scope') != 'source_recording'):
            raise ValueError('Invalid source action relationship')
        result['relationship'] = dict(relationship)
    return result


def source_action_semantics(metadata, producer):
    """Read an explicit publisher declaration, never infer semantics from names or values."""
    import json
    declaration = (metadata or {}).get('action_semantics')
    try:
        if isinstance(declaration, str):
            declaration = json.loads(declaration)
        if not isinstance(declaration, dict) or declaration.get('version') != VERSION:
            return None
        return declared_action_semantics(producer, declaration.get('role'),
                                         relationship=declaration.get('relationship'))
    except (ValueError, TypeError):
        return None


def publisher_action_semantics(metadata, producer):
    """Retain explicit source declarations, including invalid ones, for conflict qualification."""
    if not isinstance(metadata, dict) or 'action_semantics' not in metadata:
        return None
    return source_action_semantics(metadata, producer) or {
        'producer': producer, 'source_declaration': metadata['action_semantics']}


def action_semantics(context):
    """Resolve checked preparation provenance; dictionary guesses cannot declare control commands."""
    proof = context.get('state_contract') or {}
    proof = proof if isinstance(proof, dict) else {}
    action_proof = proof.get('action') or {}
    action_proof = action_proof if isinstance(action_proof, dict) else {}
    candidates = [context.get('action_semantics'), proof.get('action_semantics'),
                  action_proof.get('action_semantics'), context.get('publisher_action_semantics'),
                  proof.get('publisher_action_semantics')]
    adapter = proof.get('native_contract')
    if (isinstance(adapter, str) and adapter in NATIVE_KINDS and proof.get('version') == VERSION
            and proof.get('producer') == 'prepare.' + adapter
            and proof.get('kind') == context.get('state_kind') == NATIVE_KINDS[adapter]):
        candidates.append(native_contract(adapter, proof['kind']).get('action_semantics'))
    source = context.get('source') or {}
    source_adapter = source.get('adapter') if isinstance(source, dict) else None
    if (isinstance(source_adapter, str) and source_adapter in NATIVE_KINDS
            and context.get('state_kind') == NATIVE_KINDS[source_adapter]):
        if adapter and (proof.get('version') != VERSION or proof.get('producer') != 'prepare.' + str(adapter)
                        or proof.get('kind') != context.get('state_kind')):
            return {'role': 'action/target_unknown'}
        candidates.append(native_contract(source_adapter, context['state_kind']).get('action_semantics'))
    validated = []
    for declaration in candidates:
        if declaration is None:
            continue
        if not isinstance(declaration, dict) or declaration.get('version') != VERSION:
            return {'role': 'action/target_unknown'}
        try:
            validated.append(declared_action_semantics(declaration.get('producer'), declaration.get('role'),
                                                       relationship=declaration.get('relationship')))
        except (ValueError, TypeError):
            return {'role': 'action/target_unknown'}
    if validated:
        facts = [{k: v for k, v in declaration.items() if k != 'producer'} for declaration in validated]
        if all(item == facts[0] for item in facts):
            return validated[0]
    return {'role': 'action/target_unknown'}


def redundant_observation_target(episode):
    """Verify an explicitly declared copy on available rows; similarity never supplies the role."""
    import numpy as np
    context = episode['context']
    declaration = action_semantics(context)
    if (declaration.get('relationship') or {}).get('kind') != 'next_observation':
        return False
    state, action = episode.get('state'), episode.get('action')
    if (state is None or action is None or state.ndim != 2 or state.shape != action.shape
            or state.size == 0 or len(state) < 2 or context.get('state_unaligned')):
        return False
    from label.episode import anchor
    if (episode.get('sources') or {}).get(anchor(episode), {}).get('n_frames') != len(state):
        return False
    if not np.array_equal(action[:-1], state[1:], equal_nan=True):
        return False
    # A cut's final target can refer outside this request. Only an uncut source verifies its terminal row here.
    if not context.get('piece') and not context.get('trimmed'):
        return bool(np.array_equal(action[-1], state[-1], equal_nan=True))
    return True


def _generic(ctx):
    source = ctx.get('source') or {}
    fmt = source.get('format')
    return isinstance(fmt, str) and (fmt.startswith(('lerobot ', 'mcap ('))
                                    or fmt in ('hdf5', 'video files'))


def _declared_problem(proof, kind, dims):
    from prepare.formats import physical_state_layout
    if not isinstance(proof, dict) or proof.get('version') != VERSION:
        return REBUILD
    names, units, roles = proof.get('names'), proof.get('units'), proof.get('value_roles')
    if (proof.get('producer') not in GENERIC_PRODUCERS or proof.get('kind') != kind
            or not isinstance(names, list) or not isinstance(units, list)
            or len(names) != len(units) or (dims is not None and len(names) != dims)
            or not isinstance(roles, list) or len(roles) != len(names)):
        return REBUILD
    actual, _ = physical_state_layout(len(names), 'teleop_arms', names, metadata={'units': units},
                                      value_roles=roles, conversion=proof.get('unit_conversion'))
    return None if actual == kind and kind != 'none' else REBUILD


def prepared_state_problem(ctx, dims=None, *, has_action=False):
    """Return a rebuild reason or None before new state checks/requests; never mutate any artifact.

    Pass the loaded state's column count and whether a canonical action array is present when available.
    Historical labels can still be displayed. A reason prohibits new physical computation and cache reuse;
    callers should request preparation again, retaining the old directory and its annotations.
    """
    kind = ctx.get('state_kind')
    if kind not in ('joints', 'ee_pose'):
        return None
    source = ctx.get('source') or {}
    if not isinstance(source, dict):
        return REBUILD
    proof = ctx.get('state_contract')
    adapter = source.get('adapter')
    if isinstance(proof, dict) and proof.get('native_contract'):
        adapter = proof['native_contract'] if proof.get('producer') == 'prepare.' + str(proof['native_contract']) \
            and proof.get('version') == VERSION and proof.get('kind') == kind else None
    if isinstance(adapter, str) and adapter in NATIVE_KINDS:
        return None if NATIVE_KINDS[adapter] == kind and (dims is None or dims in (7, 14)) else REBUILD
    if isinstance(proof, dict) and 'version' in proof:
        problem = _declared_problem(proof, kind, dims)
        if problem:
            return problem
        return _declared_problem(proof.get('action'), kind, dims) if has_action else None
    if not _generic(ctx):
        # Older named/direct producers have no generic format marker. Do not reinterpret their contracts.
        return None
    # A small compatible migration: the retained native column names themselves prove canonical units.
    # Anonymous columns, ordinary names without units, and commands lacking their own proof require a rebuild.
    identities = ctx.get('state_identities') or ([ctx['state_identity']] if ctx.get('state_identity') else [])
    if identities and not has_action and all(isinstance(item, dict) and isinstance(item.get('names'), list)
                                            for item in identities):
        from prepare.formats import physical_state_layout
        names = [name for item in identities for name in item['names']]
        actual, _ = physical_state_layout(len(names), 'teleop_arms', names,
                                          value_roles=ctx.get('state_value_roles'))
        if actual == kind and (dims is None or len(names) == dims):
            return None
    return REBUILD
