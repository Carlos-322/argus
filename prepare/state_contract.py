"""Qualify prepared physical state before new computation without changing saved labels or readings."""
from __future__ import annotations

VERSION = 1
GENERIC_PRODUCERS = frozenset({'prepare.formats:lerobot', 'prepare.formats:hdf5',
                              'prepare.formats:mcap-joints', 'prepare.mcap_pose:pose'})
# Recorded adapter identity is preparation provenance, not a dataset-name heuristic.
NATIVE_KINDS = {'abc130k': 'joints', 'galaxea': 'joints', 'molmo': 'joints',
                'habit': 'ee_pose', 'realomin': 'ee_pose', 'fastumi': 'ee_pose'}
REBUILD = ('Rebuild this generic preparation from its original source before new physical-state computation: '
           'the saved context does not prove the supported state roles and units. '
           'Existing readings and historical labels must remain unchanged.')


def declared_contract(producer, kind, names, units, roles=None, action=None):
    """A producer's checked role/unit facts in prepared column order."""
    from prepare.formats import physical_state_layout
    names, units = list(names), list(units)
    roles = list(roles) if roles else [None] * len(names)
    actual, _ = physical_state_layout(len(names), 'teleop_arms', names,
                                      metadata={'units': units}, value_roles=roles)
    if producer not in GENERIC_PRODUCERS or actual != kind or kind == 'none':
        raise ValueError('Cannot stamp an unqualified physical state contract')
    contract = {'version': VERSION, 'producer': producer, 'kind': kind,
                'names': names, 'units': units, 'value_roles': roles}
    if action is not None:
        contract['action'] = action
    return contract


def native_contract(adapter, kind):
    """Stamp an adapter selected by the preparation caller, never by an input dataset name."""
    if NATIVE_KINDS.get(adapter) != kind:
        return None
    return {'version': VERSION, 'producer': 'prepare.' + adapter, 'kind': kind,
            'native_contract': adapter}


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
    actual, _ = physical_state_layout(len(names), 'teleop_arms', names,
                                      metadata={'units': units}, value_roles=roles)
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
