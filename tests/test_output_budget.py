"""Reasoning tokens count inside max_completion_tokens, so every reasoning call's limit leaves room to reason."""
import json

import pytest

from label import dictionary, harness, policy, route, sensor_cards
from label import evidence_access as ea
from test_dispatch_integrity_v5 import options, provider_kind, response
from test_label import _packed_episode
from test_sensor_cards_harness import install


@pytest.fixture(autouse=True)
def _reader_in_process(monkeypatch):
    monkeypatch.setenv('ARGUS_READER_ISOLATION', '0')


@pytest.mark.parametrize('effort', ['medium', 'high'])
def test_selection_and_card_calls_send_and_reserve_room_to_reason_at_the_episode_effort(tmp_path, monkeypatch, effort):
    # Evidence selection sent 2,000 tokens at the episode's effort. A medium or high
    # selection could spend them reasoning and be cut off before its JSON.
    ep, _ = _packed_episode(tmp_path)
    calls = []
    install(monkeypatch, calls, lambda: response('{"cards": []}'))
    answered = harness.urllib.request.urlopen
    sent = []
    def provider(request, timeout):
        sent.append((provider_kind(request), json.loads(request.data)['max_completion_tokens']))
        return answered(request, timeout)
    monkeypatch.setattr(harness.urllib.request, 'urlopen', provider)
    reserved = []
    bound = harness.final_cost_bound
    def spy(content, max_tokens, **kw):
        reserved.append(max_tokens)
        return bound(content, max_tokens, **kw)
    monkeypatch.setattr(harness, 'final_cost_bound', spy)
    assert harness.run_batch([ep], tmp_path / 'out', **{**options(), 'reasoning': effort, 'max_spend': 20.}) == 0
    limits = {}
    for kind, limit in sent:
        limits.setdefault(kind, set()).add(limit)
    assert min(limits['selection']) > 2000, limits
    room = policy.REASONING_TOKENS[effort]
    assert limits['selection'] == {ea.SELECTION_ANSWER_TOKENS + room}
    assert limits['cards'] == {sensor_cards.ANSWER_TOKENS + room}
    # The spend reservation is the limit actually sent, so the cap still covers the worst case.
    assert {ea.SELECTION_ANSWER_TOKENS + room, sensor_cards.ANSWER_TOKENS + room} <= set(reserved)


def test_routing_and_field_interpretation_leave_room_to_reason_at_low_effort(monkeypatch):
    assert route.ROUTE_REASONING == 'low' and dictionary.REASONING == 'low'
    room = policy.REASONING_TOKENS['low']
    # The routing answer is one short JSON object; its limit adds the low-effort room.
    assert route.ROUTE_MAX_TOKENS == 500 + room
    for count in (0, 10, 400):
        supplied = [f'field_{i}' for i in range(count)]
        monkeypatch.setattr(dictionary, '_request', lambda inventory, supplied=supplied: ([], supplied, []))
        answer = max(dictionary.MAX_TOKENS, min(dictionary.MAX_OUTPUT_TOKENS, 1000 + 160 * count))
        assert dictionary.request_plan({'digest': 'd'})['max_tokens'] == answer + room


def test_reasoning_room_grows_with_effort_and_refuses_an_unknown_effort():
    efforts = ['none', 'minimal', 'low', 'medium', 'high', 'xhigh']
    assert list(policy.REASONING_TOKENS) == efforts
    assert all(policy.REASONING_TOKENS[a] < policy.REASONING_TOKENS[b] for a, b in zip(efforts, efforts[1:]))
    with pytest.raises(ValueError):
        policy.output_limit(100, 'r')
