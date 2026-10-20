"""Video cards: one call chooses and writes the cards from bound findings; the card names its finding by content."""
import copy
import json

import pytest

from label import harness, sensor_cards as sc
from label.errors import SpendCap


def finding(start, headline, adds=True, **extra):
    return {'start_s': start, 'end_s': start + .4, 'headline': headline, 'detail': headline + ' detail',
            'observation': 'observed', 'claim': 'claimed', 'confidence': 'medium', 'adds_beyond_video': adds,
            'claim_type': 'response_change', 'evidence': [], 'copy_contract_version': 3, **extra}


def doc():
    return {'version': 1, 'sensors': [{'id': 'signal:p', 'name': 'right_pressure', 'quantity': 'sensor_response',
                                       'response_direction': 'down'}],
            'series': [], 'findings': [finding(2.0, 'Later'), finding(0.5, 'Earlier'), finding(1.0, 'Video only', adds=False)]}


def reply(cards, finish='stop'):
    return {'digest': None, 'state': 'received', 'finish_reason': finish, 'response_text': json.dumps({'cards': cards}),
            'model': harness.DEFAULT_MODEL, 'generation_id': 'cards-1'}


def received(request, cards):
    return {**reply(cards), 'digest': request['digest'],
            'request': {k: v for k, v in request.items() if k != 'content'}}


def test_request_lists_only_findings_the_viewer_can_show_in_playback_order():
    request = sc.build(doc(), task='pick up a cup', config={'model': 'm', 'reasoning': 'medium'})
    listed = json.loads(request['content'][0]['text'].split('\nRECORDING\n', 1)[1])
    assert [f['headline'] for f in listed['findings']] == ['Earlier', 'Later']
    assert listed['task'] == 'pick up a cup' and listed['sensors'][0]['stronger_response'] == 'down'
    assert request['config'] == {'model': 'm', 'reasoning': 'medium', 'max_tokens': sc.MAX_TOKENS}
    assert sc.build({'findings': [finding(0, 'x', adds=False)]}, config={}) is None


def test_cards_attach_by_content_and_a_changed_finding_loses_its_card():
    d = doc()
    request = sc.build(d, config={'model': 'm', 'reasoning': 'r'})
    receipt = received(request, [{'finding': 1, 'headline': 'Thumb pressure eases', 'detail': 'The thumb eases as the cup lowers.'}])
    assert sc.apply(d, receipt['request'], receipt) == 1
    assert d['findings'][0]['video_card'] == {'version': 1, 'headline': 'Thumb pressure eases',
                                              'detail': 'The thumb eases as the cup lowers.'}
    assert 'video_card' not in d['findings'][1] and d['video_cards']['cards'] == 1
    changed = doc()
    changed['findings'][0]['claim'] = 'a different claim'
    assert sc.apply(changed, receipt['request'], receipt) == 0
    assert not any('video_card' in f for f in changed['findings']) and changed['video_cards']['cards'] == 0
    reviewed = doc()
    reviewed['findings'][0]['review_status'] = 'accepted'
    assert sc.apply(reviewed, receipt['request'], receipt) == 1


def test_only_the_exact_request_authorizes_cards():
    d = doc()
    request = sc.build(d, config={'model': 'm', 'reasoning': 'r'})
    receipt = received(request, [])
    tampered = copy.deepcopy(receipt['request'])
    tampered['finding_keys'].reverse()
    with pytest.raises(ValueError):
        sc.apply(d, tampered, receipt)
    with pytest.raises(ValueError):
        sc.apply(d, receipt['request'], {**receipt, 'finish_reason': 'length'})


def test_a_card_that_names_no_finding_or_is_not_short_plain_copy_is_left_out():
    cards = sc.parse_response(json.dumps({'cards': [
        {'finding': 0, 'headline': 'Index finger presses harder', 'detail': 'The index finger presses harder than the thumb.'},
        {'finding': 0, 'headline': 'Duplicate', 'detail': 'Same finding twice.'},
        {'finding': 5, 'headline': 'Out of range', 'detail': 'No such finding.'},
        {'finding': '1', 'headline': 'Not an index', 'detail': 'A string index.'},
        {'finding': 1, 'headline': 'Pressure eases — then returns', 'detail': 'A typographic dash.'},
        {'finding': 1, 'headline': ' '.join(['word'] * 12), 'detail': 'Too long a headline.'}]}), count=2)
    assert [c['headline'] for c in cards] == ['Index finger presses harder']
    assert sc.parse_response('```json\n{"cards": []}\n```') == []
    with pytest.raises(ValueError):
        sc.parse_response('{"findings": []}')


def ledger():
    return {'version': 1, 'digest': 'inspection-one', 'status': 'complete', 'rounds': [], 'cost_usd': .01,
            'limitations': []}


def run(request, d, path, call, **kwargs):
    from label.sensor_analysis_run import execute
    return execute(request, d, path, call=call, reserve_cost=lambda parts: .2, model=harness.DEFAULT_MODEL,
                   key='sensor_cards', **kwargs)


def provider_for(cards, cost=.05):
    calls = []
    def provider(parts):
        calls.append(parts)
        return {'id': 'cards', 'usage': {'cost': cost},
                'choices': [{'finish_reason': 'stop', 'message': {'content': json.dumps({'cards': cards})}}]}
    return provider, calls


def test_cards_bill_into_the_inspection_invoice_and_a_new_request_supersedes_a_settled_one(tmp_path):
    path, d = tmp_path / '.evidence' / 'episode_a.json.inspection.json', ledger()
    first = sc.build(doc(), config={'model': 'm', 'reasoning': 'r'})
    provider, calls = provider_for([])
    assert run(first, d, path, provider)['state'] == 'received'
    assert d['cost_usd'] == pytest.approx(.06) and len(calls) == 1
    assert run(first, ledger(), path, provider)['state'] == 'received' and len(calls) == 1
    changed = doc()
    changed['findings'][0]['headline'] = 'Changed'
    second = sc.build(changed, config={'model': 'm', 'reasoning': 'r'})
    again = ledger()
    assert run(second, again, path, provider)['digest'] == second['digest'] and len(calls) == 2
    assert again['cost_usd'] == pytest.approx(.11)
    assert [r['digest'] for r in again['sensor_cards_history']] == [first['digest']]


def test_a_spend_cap_pause_leaves_the_inspection_complete(tmp_path):
    path, d = tmp_path / 'cache.json', ledger()
    request = sc.build(doc(), config={'model': 'm', 'reasoning': 'r'})
    provider, calls = provider_for([])
    with pytest.raises(SpendCap):
        run(request, d, path, provider, reserve_dispatch=lambda amount: False)
    assert d['status'] == 'complete' and 'pause_reason' not in d and not calls
    assert d['sensor_cards']['state'] == 'not dispatched'
    assert run(request, d, path, provider, reserve_dispatch=lambda amount: True)['state'] == 'received'
    assert d['limitations'] == []


def test_an_unknown_card_charge_is_held_and_never_resent(tmp_path):
    path, d = tmp_path / 'cache.json', ledger()
    request = sc.build(doc(), config={'model': 'm', 'reasoning': 'r'})
    calls = []
    def lost(parts):
        calls.append(parts)
        raise TimeoutError('response lost')
    with pytest.raises(ValueError):
        run(request, d, path, lost)
    assert d['cost_is_conservative_estimate'] and d['cost_usd'] == pytest.approx(.21)
    changed = doc()
    changed['findings'][0]['headline'] = 'Changed'
    with pytest.raises(ValueError):
        run(sc.build(changed, config={'model': 'm', 'reasoning': 'r'}), d, path, lost)
    assert len(calls) == 1
