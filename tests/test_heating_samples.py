"""Check interactive selection and dispatch without reading MCAP or calling an API."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / 'local-dev/heating-samples.py'
    spec = importlib.util.spec_from_file_location('heating_samples', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def samples():
    return [dict(sample_id=f'{i:03}', episode_id=f'episode_{i}', duration_seconds=30,
                 mcap_file=f'{i:03}.mcap') for i in range(1, 21)]


@pytest.mark.parametrize('value,expected', [
    ('17', ['017']), ('1,3，5', ['001', '003', '005']),
    ('1-3 2 5', ['001', '002', '003', '005']),
    ('all', [f'{i:03}' for i in range(1, 21)]),
])
def test_selection(runner, samples, value, expected):
    assert [s['sample_id'] for s in runner.select_samples([value], samples)] == expected


@pytest.mark.parametrize('value', ['', '0', '21', '5-1', '1-99999999', 'all 1', 'abc'])
def test_invalid_selection(runner, samples, value):
    with pytest.raises(ValueError):
        runner.select_samples([value], samples)


@pytest.mark.parametrize('answers,live,expected', [
    (['21', '1,3', ''], False, ['001', '003']),
    (['17', '2', 'nan', '-1', '0.5', 'y'], True, ['017']),
])
def test_dispatch_only_selected_mcap(runner, samples, tmp_path, monkeypatch, answers, live, expected):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'samples': samples}), encoding='utf-8')
    for sample in samples:
        (tmp_path / sample['mcap_file']).touch()
    replies = iter(answers)
    monkeypatch.setattr('builtins.input', lambda _: next(replies))
    monkeypatch.setenv('DEEPSEEK_API_KEY', 'test-key')
    monkeypatch.setattr(runner.sys, 'argv', ['heating-samples.py', '--interactive',
                        '--manifest', str(manifest), '--out', str(tmp_path / 'out')])
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run',
                        lambda cmd, **kw: calls.append((cmd, kw)) or SimpleNamespace(returncode=0))
    assert runner.main() == 0
    assert [Path(cmd[cmd.index('--data') + 1]).stem for cmd, _ in calls] == expected
    for cmd, kw in calls:
        assert ('--free' in cmd) is not live
        assert ('DEEPSEEK_API_KEY' in kw['env']) is live
        assert cmd[cmd.index('--annotation-spec') + 1] == 'heating-v2-complete'
        if live:
            assert cmd[cmd.index('--cap') + 1] == '0.5'


@pytest.mark.parametrize('answers', [['q'], ['1', 'q'], ['1', '2', 'q'], ['1', '2', '', 'n']])
def test_cancel_before_key_or_processing(runner, samples, tmp_path, monkeypatch, answers):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text(json.dumps({'samples': samples}), encoding='utf-8')
    replies = iter(answers)
    monkeypatch.setattr('builtins.input', lambda _: next(replies))
    monkeypatch.setattr(runner.sys, 'argv', ['heating-samples.py', '--interactive',
                        '--manifest', str(manifest), '--out', str(tmp_path / 'out')])
    def unexpected(*args, **kwargs):
        pytest.fail('Cancellation must not request a key or process data')
    monkeypatch.setattr(runner.getpass, 'getpass', unexpected)
    monkeypatch.setattr(runner.subprocess, 'run', unexpected)
    assert runner.main() == 0
    assert not (tmp_path / 'out').exists()
