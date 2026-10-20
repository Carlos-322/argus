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
        {'finding': 1, 'headline': 'Pressure eases \u2014 then returns', 'detail': 'A typographic dash.'},
        {'finding': 1, 'headline': ' '.join(['word'] * 12), 'detail': 'Too long a headline.'}]}), count=2)
    assert [c['headline'] for c in cards['cards']] == ['Index finger presses harder'] and cards['list'] == []
    assert sc.parse_response('```json\n{"cards": []}\n```') == {'cards': [], 'list': []}
    with pytest.raises(ValueError):
        sc.parse_response('{"findings": []}')
    both = sc.parse_response(json.dumps({
        'cards': [{'finding': 1, 'headline': 'Thumb pressure eases', 'detail': 'The thumb eases first.'}],
        'list': [{'finding': 1, 'headline': 'Also listed', 'detail': 'A card wins over a list entry.'},
                 {'finding': 0, 'headline': 'Index finger presses', 'detail': 'The index finger presses harder.'}]}), count=2)
    assert [e['finding'] for e in both['cards']] == [1] and [e['finding'] for e in both['list']] == [0]


def test_every_written_finding_reads_in_one_voice_and_only_cards_reach_the_video():
    d = doc()
    request = sc.build(d, config={'model': 'm', 'reasoning': 'r'})
    receipt = {**received(request, [{'finding': 0, 'headline': 'Index finger presses', 'detail': 'It presses harder first.'}]),
               'response_text': json.dumps({
                   'cards': [{'finding': 0, 'headline': 'Index finger presses', 'detail': 'It presses harder first.'}],
                   'list': [{'finding': 1, 'headline': 'Thumb pressure eases', 'detail': 'The thumb eases later.'}]})}
    assert sc.apply(d, receipt['request'], receipt) == 1
    earlier, later = d['findings'][1], d['findings'][0]
    assert earlier['video_card']['headline'] == 'Index finger presses' and 'list_copy' not in earlier
    assert later['list_copy'] == {'version': 1, 'headline': 'Thumb pressure eases', 'detail': 'The thumb eases later.'}
    assert 'video_card' not in later and d['video_cards']['cards'] == 1


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


def test_rejoined_parts_keep_their_cards_and_list_copy_and_play_only_chosen_cards():
    from label import sensor_evidence as se
    from label.dictionary_context import field_interpretation
    meta = {'name': 'p', 'key': 'p', 'shape': [2]}
    ctx = {'fps': 30, 'signals': [meta]}
    sensor = {'id': 'signal:p', 'name': 'p', 'kind': 'numeric', 'capabilities': ['response_change'],
              'descriptor_digest': se._descriptor_digest(meta), 'interpretation': field_interpretation(ctx, 'p')}
    def part(card, carded):
        f = {'start_s': 0.1, 'end_s': 0.5, 'headline': 'h', 'adds_beyond_video': True,
             'evidence': [{'sensor_id': 'signal:p', 'series_id': 'signal:p:columns:0', 'time_s': [0.1]}], **card}
        doc = {'version': 1, 'sensors': [copy.deepcopy(sensor)], 'findings': [f],
               'series': [{'id': 'signal:p:columns:0', 'sensor_id': 'signal:p', 'times': [0.1, 0.5], 'values': [1, 2]}]}
        if carded:
            doc['video_cards'] = {'version': 1, 'cards': 1 if 'video_card' in card else 0}
        return doc
    chosen = {'video_card': {'version': 1, 'headline': 'Thumb presses harder', 'detail': 'The thumb presses first.'}}
    listed = {'list_copy': {'version': 1, 'headline': 'Palm eases', 'detail': 'The palm eases later.'}}
    parts = [({'piece': {'index': 0, 't0_s': 0.0}}, {'sensor_evidence': part(chosen, True)}),
             ({'piece': {'index': 1, 't0_s': 10.0}}, {'sensor_evidence': part(listed, True)})]
    merged = se.merge(parts, ctx)
    assert [f.get('video_card', {}).get('headline') for f in merged['findings']] == ['Thumb presses harder', None]
    assert merged['findings'][1]['list_copy']['headline'] == 'Palm eases'
    assert merged['video_cards'] == {'version': 1, 'parts': 2, 'cards': 1}
    uncarded = [parts[0], ({'piece': {'index': 1, 't0_s': 10.0}}, {'sensor_evidence': part({}, False)})]
    assert 'video_cards' not in se.merge(uncarded, ctx), 'a part without a card choice plays every finding'


def test_a_context_citation_need_not_span_the_finding_on_its_own():
    from label import sensor_evidence as se
    times = [i * 0.5 for i in range(21)]
    doc = {'version': 1, 'episode_window_s': [0, 10.5], 'limitations': [], 'coverage': [],
           'sensors': [{'id': 'signal:p', 'name': 'p', 'kind': 'numeric', 'capabilities': ['response_change'], 'timing': {}},
                       {'id': 'camera:tac', 'name': 'tac', 'kind': 'image', 'view': 'tac', 'capabilities': ['image_response_change'],
                        'times': [3.0, 4.0], 'timing': {}, 'episode_alignment_verified': True}],
           'series': [{'id': 'signal:p:columns:0', 'sensor_id': 'signal:p', 'columns': [0], 'times': times,
                       'values': [float(i) for i in range(21)], 'operation': 'recorded value', 'native_samples': 21}]}
    trace = {'sensor_id': 'signal:p', 'series_id': 'signal:p:columns:0', 'time_s': [2.0, 5.0]}
    image = {'sensor_id': 'camera:tac', 'time_s': [3.0]}
    finding = {'start_s': 2.0, 'end_s': 5.0, 'headline': 'pad pressure rises while the image shows contact',
               'claim_type': 'response_change', 'evidence': [trace, image]}
    assert len(se.bind(doc, [finding])['findings']) == 1
    alone = {**finding, 'evidence': [image]}
    assert se.bind(doc, [alone])['unbound_findings'][0]['reason'] == 'finding exceeds the evidence clock'


def test_the_card_prompt_asks_for_the_same_plain_english_as_the_annotation():
    from label import sensor_copy
    assert sensor_copy.PLAIN_ENGLISH in sc.PROMPT and 'at most six words' not in sc.PROMPT
    assert f'at most {sc.HEADLINE_WORDS} words' in sc.PROMPT and f'under {sc.DETAIL_WORDS} words' in sc.PROMPT
    assert sc.HEADLINE_LIMIT >= sc.HEADLINE_WORDS and sc.DETAIL_LIMIT >= sc.DETAIL_WORDS
    eight = 'Two spots on the thumb press harder again'
    assert sc.parse_response(json.dumps({'cards': [{'finding': 0, 'headline': eight,
        'detail': 'A third spot on the thumb stays the same as the hand pulls back.'}]}), count=1)['cards']
