"""One source-owned auxiliary call within the existing inspection invoice: the regional mechanics pass before the
final annotation, or the video card pass after it. Both save a claim and the raw response before interpreting it."""
from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import math
from pathlib import Path

from label.atomic import private_path
from label.errors import DispatchError, SpendCap

CAP_NOTE = 'Sensor analysis exceeds the remaining batch spend cap.'


def _parse_analysis(text):
    from label.sensor_analysis import parse_response
    return parse_response(text)


def _parse_cards(text):
    from label.sensor_cards import parse_response
    return parse_response(text)


# What differs between the two calls. A regional pass is part of the inspection, so a pause or an unknown outcome
# leaves the inspection incomplete. Video cards come after the annotation is paid for and bound, so they never
# change the inspection's status, and a newer request for changed findings supersedes a settled older one.
STAGES = {
    'sensor_analysis': {'name': 'Sensor analysis', 'suffix': 'analysis', 'parse': _parse_analysis,
                        'cap_note': CAP_NOTE, 'part_of_inspection': True, 'supersede': False},
    'sensor_cards': {'name': 'Sensor cards', 'suffix': 'cards', 'parse': _parse_cards,
                     'cap_note': 'Sensor cards exceed the remaining batch spend cap.',
                     'part_of_inspection': False, 'supersede': True},
}


def _received(receipt, stage=STAGES['sensor_analysis']):
    name = stage['name']
    if receipt.get('cost_is_conservative_estimate'):
        raise ValueError(f'{name} charge is unresolved; no automatic retry.')
    if receipt.get('finish_reason') != 'stop':
        raise ValueError(f'{name} response is incomplete; no automatic retry.')
    try:
        stage['parse'](receipt.get('response_text'))
    except (ValueError, TypeError) as error:
        raise ValueError(f'{name} response is not a complete JSON object; no automatic retry.') from error
    return copy.deepcopy(receipt)


def execute(request, doc, cache_path, *, call, reserve_cost, model,
            reserve_dispatch=None, settle_dispatch=None, verify_inputs=None, key='sensor_analysis'):
    """Save a claim and raw response before interpreting it. Paid inputs are never retried implicitly."""
    from label import evidence_access as ea
    from label.harness import _verified_final_cost, persist_final_claim
    stage = STAGES[key]
    name, history = stage['name'], key + '_history'
    content_hash = hashlib.sha256(json.dumps(request['content'], sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if content_hash != request.get('content_sha256'):
        raise ValueError(f'{name} request body changed before dispatch.')
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with private_path(path, '.', f'.{stage["suffix"]}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f'{name} dispatch is already running.') from error
        if verify_inputs is not None:
            verify_inputs()
        if path.exists():
            saved_doc = json.loads(path.read_text())
            if saved_doc.get('digest') != doc.get('digest'):
                raise ValueError(f'{name} inspection owner changed; no automatic retry.')
            if saved_doc.get(key) is not None:
                for field in (key, history, 'cost_usd', 'cost_is_conservative_estimate'):
                    if field in saved_doc:
                        doc[field] = copy.deepcopy(saved_doc[field])
                    else:
                        doc.pop(field, None)
        ea.recover_selection_reservations(doc)
        saved = doc.get(key)
        if saved is not None and stage['supersede'] and isinstance(saved, dict) and saved.get('digest') != request['digest'] \
                and (saved.get('state') in ('received', 'not dispatched') or ea.rejected_selection_proof(saved)) \
                and not saved.get('cost_is_conservative_estimate'):
            # Settled and billed into the invoice already; kept for the record, never charged again.
            doc.setdefault(history, []).append(copy.deepcopy(saved))
            doc.pop(key)
            saved = None
        if saved is not None:
            if not isinstance(saved, dict) or saved.get('digest') != request['digest']:
                raise ValueError(f'{name} source or settings changed; no automatic retry.')
            if saved.get('state') == 'received':
                return _received(saved, stage)
            if saved.get('state') != 'not dispatched' and not ea.rejected_selection_proof(saved):
                raise ValueError(f'{name} dispatch outcome is unresolved; no automatic retry.')
        if doc.get('cost_is_conservative_estimate'):
            raise ValueError(f'Inspection charges are unresolved; {name[0].lower() + name[1:]} was not dispatched.')
        rejected = (list(saved.get('rejected_attempts') or []) +
                    [{k: copy.deepcopy(v) for k, v in saved.items() if k != 'rejected_attempts'}]
                    if saved and ea.rejected_selection_proof(saved) else [])
        if saved and saved.get('pause_reason') == 'spend_cap' and stage['part_of_inspection']:
            doc.pop('pause_reason', None)
            doc['status'] = 'complete'
            doc['limitations'] = [note for note in doc['limitations'] if note != stage['cap_note']]
        reserve = reserve_cost(request['content'])
        if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve <= 0:
            raise ValueError(f'{name} reservation must be finite and positive.')
        if reserve_dispatch is not None and not reserve_dispatch(reserve):
            # A stage after the annotation keeps its pause on its own receipt: the inspection's status and notes
            # were part of the paid request, so changing them would make that finished request look stale.
            if stage['part_of_inspection']:
                doc.update(status='incomplete', pause_reason='spend_cap')
                if stage['cap_note'] not in doc['limitations']:
                    doc['limitations'].append(stage['cap_note'])
            doc[key] = {'digest': request['digest'], 'state': 'not dispatched',
                        'pause_reason': 'spend_cap', 'note': stage['cap_note'], 'cost_usd': 0.0}
            persist_final_claim(path, doc)
            raise SpendCap(stage['cap_note'])
        try:
            if verify_inputs is not None:
                verify_inputs()
            token = hashlib.sha256((str(doc['digest']) + request['digest']).encode()).hexdigest()[:16]
            claim = private_path(path, suffix=f'.{stage["suffix"]}.{token}.{len(rejected)}.claim')
            with claim.open('x') as stream:
                stream.write(request['digest'])
        except BaseException:
            if settle_dispatch is not None:
                settle_dispatch(reserve, 0.0)
            raise
        receipt = {'digest': request['digest'], 'model': model,
                   'request': copy.deepcopy({k: value for k, value in request.items() if k != 'content'}),
                   'state': 'dispatched', 'reserved_cost_usd': reserve,
                   'cost_usd': reserve, 'reservation_accounted': True, 'cost_is_conservative_estimate': True,
                   **({'rejected_attempts': rejected} if rejected else {})}
        doc[key] = receipt
        doc['cost_usd'] += reserve
        doc['cost_is_conservative_estimate'] = True
        persist_final_claim(path, doc)
        try:
            response = call(request['content'])
        except Exception as error:
            if isinstance(error, DispatchError) and error.verified_unpaid:
                receipt.update(state='rejected unpaid', http_status=error.http_status, http_attempt=1,
                               verified_unpaid=True, cost_usd=0.0, cost_is_conservative_estimate=False,
                               reservation_accounted=False,
                               **({'rejection_code': error.rejection_code} if error.rejection_code else {}))
                doc['cost_usd'] -= reserve
                doc.pop('cost_is_conservative_estimate', None)
                persist_final_claim(path, doc)
                if settle_dispatch is not None:
                    settle_dispatch(reserve, 0.0)
                raise
            if stage['part_of_inspection']:
                doc['status'] = 'incomplete'
            persist_final_claim(path, doc)
            raise ValueError(f'{name} dispatch outcome is unresolved; no automatic retry.') from error
        envelope = response if isinstance(response, dict) else {}
        choices = envelope.get('choices')
        choice = choices[0] if isinstance(choices, list) and choices and isinstance(choices[0], dict) else {}
        message = choice.get('message') if isinstance(choice.get('message'), dict) else {}
        usage = envelope.get('usage') or {}
        cost, cost_source = _verified_final_cost(usage, model)
        receipt.update(state='received', response=copy.deepcopy(response), response_text=message.get('content'),
                       finish_reason=choice.get('finish_reason') or choice.get('native_finish_reason'),
                       usage=copy.deepcopy(usage), cost_source=cost_source, generation_id=envelope.get('id'),
                       model_served=envelope.get('model'), provider_name=envelope.get('provider'),
                       cost_usd=reserve if cost is None else cost, cost_is_conservative_estimate=cost is None)
        if cost is not None:
            doc['cost_usd'] += cost - reserve
            doc.pop('cost_is_conservative_estimate', None)
        elif stage['part_of_inspection']:
            doc['status'] = 'incomplete'
        persist_final_claim(path, doc)
        if cost is not None and settle_dispatch is not None:
            settle_dispatch(reserve, cost)
        if verify_inputs is not None:
            verify_inputs()
        return _received(receipt, stage)
