"""Conversion reuse must never reuse answers or change the original time axis."""
import base64
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from review import __main__ as review
from review import preparation_cache as cache
from prepare import formats


def make_mcap(path):
    from mcap.writer import Writer
    with path.open('wb') as stream:
        writer = Writer(stream)
        writer.start()
        schema = writer.register_schema(name='foxglove.CompressedImage', encoding='jsonschema', data=b'{}')
        channels = [writer.register_channel(topic=topic, message_encoding='json', schema_id=schema)
                    for topic in ('/head/image', '/left_arm/image', '/right_arm/image')]
        for i in range(12):
            image = io.BytesIO()
            Image.new('RGB', (64, 48), (i * 10, 40, 90)).save(image, format='JPEG')
            body = json.dumps({'format': 'jpeg', 'data': base64.b64encode(image.getvalue()).decode()}).encode()
            for j, channel in enumerate(channels):
                ns = 1_790_000_000_000_000_000 + i * 33_333_333 + j * 1_000_000
                writer.add_message(channel, log_time=ns, publish_time=ns, data=body)
        writer.finish()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    source = tmp_path / 'recording.mcap'
    make_mcap(source)
    args = SimpleNamespace(data=str(source), rig='teleop_arms', max_minutes=0, grouping={},
                           preparation_cache=tmp_path / 'cache', reuse_prepared=None)
    conversions = []
    original = formats.convert
    def counted(*args, **kwargs):
        conversions.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(formats, 'convert', counted)
    def run(name, **overrides):
        job = tmp_path / name
        job.mkdir()
        report, receipt = review.verified_preparation(job, source, args, 'fixture', overrides)
        return job, job / 'episodes' / report['episodes'][0]['episode_id'], receipt
    return source, args, conversions, run


def test_changed_model_prompt_rules_and_sampling_reuse_identical_media(setup):
    _, _, conversions, run = setup
    first, ep1, _ = run('first', model='openai/gpt-6-astra', annotation_spec='heating-v2', sample_fps=4)
    (first / 'run/out').mkdir(parents=True)
    (first / 'run/out/answer.json').write_text('{"old_paid_answer": true}')
    second, ep2, receipt = run('second', model='deepseek/deepseek-flash',
                              annotation_spec='heating-v2-complete', sample_fps=2, dataset_note='new prompt')
    assert len(conversions) == 1
    assert receipt['conversion_cache']['hit']
    assert not (second / 'run').exists()
    context = json.loads((ep2 / 'context.json').read_text())
    policy = context['annotation_policy']
    assert policy['model'] == 'deepseek/deepseek-flash'
    assert policy['sample_every_s'] == .5
    assert policy['annotation_spec']['id'] == 'heating-v2-complete'
    assert policy['dataset_note'] == 'new prompt'
    for path in ep1.iterdir():
        if path.suffix in ('.mp4', '.npz', '.npy'):
            assert cache.sha256(path) == cache.sha256(ep2 / path.name)
            assert not os.path.samefile(path, ep2 / path.name)
    for source in json.loads((ep2 / 'sources.json').read_text()).values():
        assert Path(source['packed']).is_relative_to(ep2)
        assert Path(source['packed']).is_file()
    assert json.loads((ep1 / 'context.json').read_text())['annotation_policy']['sample_every_s'] == .25
    with np.load(ep1 / 'times.npz') as first_times, np.load(ep2 / 'times.npz') as second_times:
        assert first_times.files == second_times.files
        for name in first_times.files:
            assert np.array_equal(first_times[name], second_times[name])
    from label import episode
    request = episode.build_request(ep2, read_only=True)
    assert 'new prompt' in request['prompt']
    assert 'heating-v2-complete' in request['prompt']
    assert request['image_bytes'] > 0


def test_new_run_cannot_modify_cached_media_or_another_run(setup):
    _, _, conversions, run = setup
    _, first_ep, receipt = run('first')
    entry = Path(receipt['conversion_cache']['path'])
    cached_video = next((entry / 'episodes').rglob('exo.mp4'))
    expected = cache.sha256(cached_video)
    (first_ep / 'exo.mp4').write_bytes(b'changed in one job')
    _, second_ep, _ = run('second')
    assert len(conversions) == 1
    assert cache.sha256(cached_video) == cache.sha256(second_ep / 'exo.mp4') == expected


def test_interrupted_conversion_does_not_publish_a_cache(setup, monkeypatch):
    _, args, _, run = setup
    def interrupted(*call, **kwargs):
        episode = call[2] / 'episode_partial'
        episode.mkdir(parents=True)
        (episode / 'exo.mp4').write_bytes(b'partial')
        raise RuntimeError('simulated conversion failure')
    monkeypatch.setattr(formats, 'convert', interrupted)
    with pytest.raises(RuntimeError, match='simulated conversion failure'):
        run('interrupted')
    assert not list(args.preparation_cache.rglob('cache.json'))


@pytest.mark.parametrize('change', ['source', 'settings', 'producer', 'artifact', 'missing_clock'])
def test_changed_or_damaged_inputs_cannot_reuse_conversion(setup, monkeypatch, change):
    source, args, conversions, run = setup
    _, _, receipt = run('first')
    entry = Path(receipt['conversion_cache']['path'])
    if change == 'source':
        # Even unchanged size/mtime must not disguise a changed MCAP byte.
        stamp = source.stat()
        data = bytearray(source.read_bytes())
        data[-1] ^= 1
        source.write_bytes(data)
        os.utime(source, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        # This test checks the cache boundary; avoid decoding deliberately corrupt input.
        from shutil import copytree
        saved_ep = entry / 'episodes'
        def converted(*call, **kwargs):
            conversions.append(1)
            copytree(saved_ep, call[2])
            return json.loads((entry / 'report.json').read_text())
        monkeypatch.setattr(formats, 'convert', converted)
    elif change == 'settings':
        args.max_minutes = .01
    elif change == 'producer':
        original = cache.producer
        monkeypatch.setattr(cache, 'producer', lambda root: {**original(root), 'prepare/new.py': 'changed'})
    elif change == 'artifact':
        next((entry / 'episodes').rglob('*.mp4')).write_bytes(b'corrupt')
    else:
        next((entry / 'episodes').rglob('times.npz')).unlink()
    run('second')
    assert len(conversions) == 2


def test_existing_prepared_job_import_is_private_and_keeps_old_results(setup):
    _, args, conversions, run = setup
    cache_path = args.preparation_cache
    args.preparation_cache = None
    old, old_ep, _ = run('legacy', annotation_spec='heating-v2', sample_fps=4)
    (old / '.execution.lock').touch()
    old_context = (old_ep / 'context.json').read_bytes()
    args.preparation_cache = cache_path
    args.reuse_prepared = old
    new, new_ep, receipt = run('new', annotation_spec='heating-v2-complete', sample_fps=2)
    assert len(conversions) == 1
    assert receipt['conversion_cache']['imported_from'] == str(old)
    assert (old_ep / 'context.json').read_bytes() == old_context
    assert json.loads((new_ep / 'context.json').read_text())['annotation_policy']['sample_every_s'] == .5
    assert cache.sha256(old_ep / 'exo.mp4') == cache.sha256(new_ep / 'exo.mp4')


def test_busy_legacy_job_is_not_imported(setup):
    from label.harness import execution_lock
    _, args, conversions, run = setup
    cache_path = args.preparation_cache
    args.preparation_cache = None
    old, _, _ = run('legacy')
    args.preparation_cache, args.reuse_prepared = cache_path, old
    with execution_lock(old):
        _, _, receipt = run('new')
    assert len(conversions) == 2
    assert receipt['conversion_cache']['imported_from'] is None


def test_prompt_code_is_outside_conversion_key(tmp_path):
    (tmp_path / 'label').mkdir()
    episode = tmp_path / 'label/episode.py'
    episode.write_text('VIEW_ORDER = ("exo", "left")\ndef order_views(x): return VIEW_ORDER\ndef build_prompt(): return "old"\n')
    before = cache.producer(tmp_path)
    episode.write_text(episode.read_text().replace('"old"', '"new prompt"'))
    assert cache.producer(tmp_path) == before
    episode.write_text(episode.read_text().replace('("exo", "left")', '("left", "exo")'))
    assert cache.producer(tmp_path) != before
