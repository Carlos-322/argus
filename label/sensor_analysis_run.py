"""One source-owned mechanics call within the existing inspection invoice."""
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


def _received(receipt):
    if receipt.get('cost_is_conservative_estimate'):
        raise ValueError('Sensor analysis charge is unresolved; no automatic retry.')
    if receipt.get('finish_reason') != 'stop':
        raise ValueError('Sensor analysis response is incomplete; no automatic retry.')
    try:
        answer = json.loads(receipt.get('response_text') or '')
    except (ValueError, TypeError) as error:
        raise ValueError('Sensor analysis response is not a complete JSON object; no automatic retry.') from error
    if not isinstance(answer, dict):
        raise ValueError('Sensor analysis response is not a JSON object; no automatic retry.')
    return copy.deepcopy(receipt)


def execute(request, doc, cache_path, *, call, reserve_cost, model,
            reserve_dispatch=None, settle_dispatch=None, verify_inputs=None):
    """Save a claim and raw response before interpreting it. Paid inputs are never retried implicitly."""
    from label import evidence_access as ea
    from label.harness import _verified_final_cost, persist_final_claim
    content_hash = hashlib.sha256(json.dumps(request['content'], sort_keys=True, separators=(',', ':'),
                                             ensure_ascii=True, allow_nan=False).encode()).hexdigest()
    if content_hash != request.get('content_sha256'):
        raise ValueError('Sensor analysis request body changed before dispatch.')
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with private_path(path, '.', '.analysis.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Sensor analysis dispatch is already running.') from error
        if verify_inputs is not None:
            verify_inputs()
        if path.exists():
            saved_doc = json.loads(path.read_text())
            if saved_doc.get('digest') != doc.get('digest'):
                raise ValueError('Sensor analysis inspection owner changed; no automatic retry.')
            if saved_doc.get('sensor_analysis') is not None:
                for key in ('sensor_analysis', 'cost_usd', 'cost_is_conservative_estimate'):
                    if key in saved_doc:
                        doc[key] = copy.deepcopy(saved_doc[key])
                    else:
                        doc.pop(key, None)
        ea.recover_selection_reservations(doc)
        saved = doc.get('sensor_analysis')
        if saved is not None:
            if not isinstance(saved, dict) or saved.get('digest') != request['digest']:
                raise ValueError('Sensor analysis source or settings changed; no automatic retry.')
            if saved.get('state') == 'received':
                return _received(saved)
            if saved.get('state') != 'not dispatched' and not ea.rejected_selection_proof(saved):
                raise ValueError('Sensor analysis dispatch outcome is unresolved; no automatic retry.')
        if doc.get('cost_is_conservative_estimate'):
            raise ValueError('Inspection charges are unresolved; sensor analysis was not dispatched.')
        rejected = (list(saved.get('rejected_attempts') or []) +
                    [{k: copy.deepcopy(v) for k, v in saved.items() if k != 'rejected_attempts'}]
                    if saved and ea.rejected_selection_proof(saved) else [])
        if saved and saved.get('pause_reason') == 'spend_cap':
            doc.pop('pause_reason', None)
            doc['status'] = 'complete'
            doc['limitations'] = [note for note in doc['limitations'] if note != CAP_NOTE]
        reserve = reserve_cost(request['content'])
        if type(reserve) not in (int, float) or not math.isfinite(reserve) or reserve <= 0:
            raise ValueError('Sensor analysis reservation must be finite and positive.')
        if reserve_dispatch is not None and not reserve_dispatch(reserve):
            doc.update(status='incomplete', pause_reason='spend_cap')
            doc['sensor_analysis'] = {'digest': request['digest'], 'state': 'not dispatched',
                                      'pause_reason': 'spend_cap', 'cost_usd': 0.0}
            if CAP_NOTE not in doc['limitations']:
                doc['limitations'].append(CAP_NOTE)
            persist_final_claim(path, doc)
            raise SpendCap(CAP_NOTE)
        try:
            if verify_inputs is not None:
                verify_inputs()
            token = hashlib.sha256((str(doc['digest']) + request['digest']).encode()).hexdigest()[:16]
            claim = private_path(path, suffix=f'.analysis.{token}.{len(rejected)}.claim')
            with claim.open('x') as stream:
                stream.write(request['digest'])
        except BaseException:
            if settle_dispatch is not None:
                settle_dispatch(reserve, 0.0)
            raise
        receipt = {'digest': request['digest'], 'model': model,
                   'request': copy.deepcopy({key: value for key, value in request.items() if key != 'content'}),
                   'state': 'dispatched', 'reserved_cost_usd': reserve,
                   'cost_usd': reserve, 'reservation_accounted': True, 'cost_is_conservative_estimate': True,
                   **({'rejected_attempts': rejected} if rejected else {})}
        doc['sensor_analysis'] = receipt
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
            doc['status'] = 'incomplete'
            persist_final_claim(path, doc)
            raise ValueError('Sensor analysis dispatch outcome is unresolved; no automatic retry.') from error
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
        else:
            doc['status'] = 'incomplete'
        persist_final_claim(path, doc)
        if cost is not None and settle_dispatch is not None:
            settle_dispatch(reserve, cost)
        if verify_inputs is not None:
            verify_inputs()
        return _received(receipt)
