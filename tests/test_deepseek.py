"""Offline coverage for provider selection, wire format and episode processing."""
import io
import json
import subprocess
import sys
import urllib.error

import pytest
from label import harness, policy, providers


@pytest.fixture
def deepseek(monkeypatch):
    monkeypatch.setenv("ARGUS_PROVIDER", "deepseek")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-test")
    monkeypatch.setenv("OPENROUTER_API_KEYS", "sk-or-test")


def test_provider_never_falls_back_to_another_key(deepseek, monkeypatch):
    assert harness.get_keys() == ["sk-ds-test"]
    monkeypatch.delenv("DEEPSEEK_API_KEY")
    assert harness.get_keys() == []
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-or-test")
    assert harness.get_keys() == []
    monkeypatch.delenv("ARGUS_PROVIDER")
    assert harness.get_keys() == ["sk-or-test"]


def test_process_start_selects_main_and_auxiliary_models(deepseek):
    code = ("from label import policy, dictionary, route; import json; "
            "print(json.dumps([policy.resolve({})['model'], dictionary.MODEL, route.ROUTE_MODEL]))")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == [providers.DEEPSEEK_MODEL] * 3
    assert policy.resolve({})["reasoning"] == "high"


def test_wrong_model_or_key_rejected_before_network(deepseek, monkeypatch):
    monkeypatch.setattr(harness, "_post", lambda *a, **kw: pytest.fail("network attempted"))
    with pytest.raises(ValueError, match="requires --model"):
        harness.call_model_once([], "openai/gpt-6-astra", "medium", "sk-ds-test", 100, 5)
    with pytest.raises(ValueError, match="DEEPSEEK_API_KEY"):
        harness.call_model_once([], providers.DEEPSEEK_MODEL, "high", "sk-openai-test", 100, 5)


@pytest.mark.parametrize("effort,expected", [("none","none"), ("low","low"), ("medium","high"), ("high","high")])
def test_wire_format_and_cache_pricing(deepseek, monkeypatch, effort, expected):
    sent = []
    content = [{"type":"text", "text":"Return JSON."},
               {"type":"image_url", "image_url":{"url":"data:image/png;base64,fixture","detail":"high"}}]
    def answer(req, timeout):
        sent.append((req.full_url, json.loads(req.data), req.get_header("Authorization")))
        return io.BytesIO(json.dumps({"choices":[{"message":{"content":"{}"}}],
            "usage":{"prompt_tokens":1000,"prompt_cache_hit_tokens":600,"completion_tokens":200}}).encode())
    monkeypatch.setattr(harness.urllib.request, "urlopen", answer)
    result = harness.call_model_once(content, providers.DEEPSEEK_MODEL, effort, "sk-ds-test", 400, 5)
    url, body, auth = sent[0]
    assert len(sent) == 1 and url == providers.DEEPSEEK_URL and auth == "Bearer sk-ds-test"
    assert body["model"] == "deepseek-flash" and body["max_tokens"] == 400
    assert "max_completion_tokens" not in body and "usage" not in body
    assert body["messages"][0]["content"] == content
    assert body["reasoning_effort"] == expected
    assert body["thinking"]["type"] == ("disabled" if expected == "none" else "enabled")
    assert body["response_format"] == {"type":"json_object"}
    assert result["provider"] == "DeepSeek"
    cost, source = harness._verified_final_cost(result["usage"], providers.DEEPSEEK_MODEL)
    assert cost == pytest.approx((400*.30 + 600*.006 + 200*1.2) / 1e6)
    assert source == "peak_list_estimate"


def test_timeout_does_not_retry_paid_dispatch(deepseek, monkeypatch):
    calls = []
    def fail(req, timeout):
        calls.append(req.full_url)
        # urllib propagates response timeouts directly; URLError is a pre-dispatch connection failure.
        raise TimeoutError("timeout after dispatch")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fail)
    monkeypatch.setattr(harness.time, "sleep", lambda *a: pytest.fail("retried"))
    with pytest.raises(RuntimeError):
        harness.call_model_once([], providers.DEEPSEEK_MODEL, "high", "sk-ds-test", 100, 5)
    assert calls == [providers.DEEPSEEK_URL]


@pytest.mark.parametrize("usage", [{}, {"prompt_tokens":True,"completion_tokens":2},
    {"prompt_tokens":10,"completion_tokens":2,"prompt_cache_hit_tokens":11}])
def test_invalid_usage_does_not_get_a_list_cost(usage):
    providers.normalize_usage(usage)
    assert "list_cost" not in usage


def test_provider_limits():
    with pytest.raises(ValueError, match="600 images"):
        providers.request_body([{"type":"image_url"}]*601, providers.DEEPSEEK_MODEL, "high", 100)
    with pytest.raises(ValueError, match="48 MiB"):
        providers.request_body([{"type":"text","text":"x"*(48*1024*1024)}], providers.DEEPSEEK_MODEL, "high", 100)


def test_budget_uses_deepseek_peak_rates(deepseek):
    assert harness.model_prices(providers.DEEPSEEK_MODEL) == providers.DEEPSEEK_PRICES
    assert harness.final_cost_bound([{"type":"text","text":"hello"}], 100) == pytest.approx(.000429)


def test_episode_including_router_uses_only_deepseek(deepseek, monkeypatch, tmp_path):
    from test_route_accounting import observed_packed_episode
    from label import route, evidence_access as ea
    ep = observed_packed_episode(tmp_path)
    monkeypatch.setattr(route, "ROUTE_MODEL", providers.DEEPSEEK_MODEL)
    monkeypatch.setattr(route, "_CACHE", {})
    sent = []
    def answer(req, timeout):
        body = json.loads(req.data)
        assert req.full_url == providers.DEEPSEEK_URL and body["model"] == "deepseek-flash"
        texts = "\n".join(p.get("text","") for p in body["messages"][0]["content"])
        if route.ROUTE_PROMPT in texts:
            kind, reply = "route", {"fine_detail":False,"why":"whole objects"}
        elif ea.SELECTION in texts:
            kind, reply = "inspect", {"requests":[]}
        else:
            kind, reply = "label", {"timeline":[], "task_summary":"Wait beside the visible workspace."}
        sent.append(kind)
        return io.BytesIO(json.dumps({"model":"deepseek-flash","choices":[{"finish_reason":"stop",
            "message":{"content":json.dumps(reply)}}],"usage":{"prompt_tokens":100,"completion_tokens":10}}).encode())
    monkeypatch.setattr(harness.urllib.request, "urlopen", answer)
    out = tmp_path / "out"
    assert harness.run_batch([ep], out, keys=["sk-ds-test"], concurrency=1, force=False, max_spend=1,
        model=providers.DEEPSEEK_MODEL, reasoning="high", max_tokens=100, timeout=5) == 0
    result = json.loads((out / f"{ep.name}.json").read_text())
    assert "route" in sent and "label" in sent
    assert result["parse_ok"] and result["provider"] == "deepseek"
    assert result["usage"]["cost_source"] == "peak_list_estimate"
    assert "sk-ds-test" not in json.dumps(result)


def test_deepseek_body_changes_invalidate_saved_requests(deepseek, monkeypatch, tmp_path):
    from test_route_accounting import observed_packed_episode
    from label.request_attestation import only_implementation_changed

    ep = observed_packed_episode(tmp_path)
    kwargs = dict(model=providers.DEEPSEEK_MODEL, reasoning="high", max_tokens=100,
                  cell_w=128, example_dir=None)
    saved = harness.input_identity(ep, **kwargs)
    assert saved["settings"]["request_body_sha256"] != harness.FIRST_BODY_SHA256
    original = providers.request_body
    monkeypatch.setattr(providers, "request_body", lambda *a, **kw:
                        {**original(*a, **kw), "response_format": {"type": "json_schema"}})
    current = harness.input_identity(ep, **kwargs)
    assert not only_implementation_changed(saved, current)


def test_deepseek_connection_failure_retries_without_losing_billing_state(deepseek, monkeypatch):
    from label.errors import DispatchError

    sent = []
    def fail(req, timeout):
        sent.append(req.full_url)
        raise urllib.error.URLError("connection refused")
    monkeypatch.setattr(harness.urllib.request, "urlopen", fail)
    monkeypatch.setattr(harness.time, "sleep", lambda *_: None)
    with pytest.raises(DispatchError) as error:
        harness.call_model_once([], providers.DEEPSEEK_MODEL, "high", "sk-ds-test", 100, 5)
    assert error.value.verified_unpaid and error.value.stage == "not sent"
    assert sent == [providers.DEEPSEEK_URL] * harness.UNPAID_ATTEMPTS
