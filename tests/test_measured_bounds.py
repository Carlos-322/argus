"""Bounds and measurements that must hold for every input of their kind, not for one example.

An image reservation is an upper bound for every published tokenizer, a frame rate is replaced only by times that
leave the declared grid, and a saved unpaid failure is read back with the verdict that was made when it was sent.
"""
import numpy as np
import pytest

from label.evidence_access import rejected_selection_proof
from label.pricing import image_token_bound
from prepare.formats import measured_rate


def openai_mini_tokens(width, height):
    # gpt-4o mini high detail: fit 2048, shorter side to 768 (downscale only), 2,833 + 5,667 per 512 px tile
    scale = min(1.0, 2048 / max(width, height))
    width, height = width * scale, height * scale
    scale = min(1.0, 768 / min(width, height))
    width, height = width * scale, height * scale
    return 2833 + 5667 * int(np.ceil(width / 512)) * int(np.ceil(height / 512))


def anthropic_tokens(width, height):
    scale = min(1.0, 1568 / max(width, height))
    return int(np.ceil(width * scale * height * scale / 750))


@pytest.mark.parametrize('width', [16, 128, 370, 512, 513, 640, 768, 1024, 1892, 2048, 4096])
@pytest.mark.parametrize('height', [16, 240, 370, 480, 602, 768, 1080, 2048])
def test_the_image_bound_is_at_least_every_published_tokenizer(width, height):
    bound = image_token_bound(width, height)
    assert bound >= openai_mini_tokens(width, height)
    assert bound >= anthropic_tokens(width, height)
    assert bound >= 258 * int(np.ceil(width / 768)) * int(np.ceil(height / 768))      # Gemini tiles


@pytest.mark.parametrize('declared, present', [(30, [1, 3, 5]), (30, [0, 1, 2, 5, 6, 9]), (25, [0, 2, 4, 6, 8]),
                                               (15, [0, 1, 3, 4, 6, 7])])
def test_frames_missing_from_the_declared_grid_keep_the_declared_rate(declared, present):
    extra = {}
    times = np.asarray(present, dtype=np.float64) / declared
    assert measured_rate(extra, 'front', times, float(declared)) == declared
    assert not extra.get('reader_issues')


@pytest.mark.parametrize('declared, actual', [(15, 30), (60, 24), (30, 25), (10, 29.97)])
def test_times_off_the_declared_grid_replace_the_rate_with_an_issue(declared, actual):
    extra = {}
    times = np.arange(90) / actual
    rate = measured_rate(extra, 'front', times, float(declared))
    assert abs(rate - actual) < .05 * actual
    assert [i['kind'] for i in extra['reader_issues']] == ['declared_rate_wrong']


@pytest.mark.parametrize('status', [None, 400, 401, 402, 403, 404, 429, 500, 502, 503])
def test_every_failure_saved_as_unpaid_is_read_back_as_unpaid(status):
    row = {'state': 'rejected unpaid', 'http_status': status, 'http_attempt': 1, 'verified_unpaid': True,
           'cost_usd': 0.0}
    assert rejected_selection_proof(row)
    assert not rejected_selection_proof({**row, 'verified_unpaid': False})
    assert not rejected_selection_proof({**row, 'cost_usd': .01})


@pytest.mark.parametrize('width', [16, 513, 1892, 2048, 4096, 9000])
@pytest.mark.parametrize('height', [16, 769, 1080, 2048, 9000])
def test_an_image_of_unknown_size_is_bounded_by_the_most_any_image_costs(width, height):
    from label.pricing import LARGEST_CHARGED_SIZE, image_sizes
    assert image_sizes([{'type': 'image_url'}]) == [LARGEST_CHARGED_SIZE]
    assert image_token_bound(*LARGEST_CHARGED_SIZE) >= image_token_bound(width, height)
