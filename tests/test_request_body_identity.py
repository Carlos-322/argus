"""A label requested with another request body is never taken as current.

The body's own settings (its response format, the usage it asks for) are code constants, so a change to them was
an implementation change that attestation ignored. A body that differs from the one every saved label was
requested with is named in the input identity.
"""
import numpy as np

from label import harness
from label.request_attestation import only_implementation_changed
from test_camera_timing import recording


def identity(ep):
    return harness.input_identity(ep, model=harness.DEFAULT_MODEL, reasoning='medium', max_tokens=100, cell_w=128,
                                  example_dir=None)


def test_the_body_every_saved_label_used_adds_nothing_to_the_identity(tmp_path):
    ep = recording(tmp_path, np.arange(12) / 30)
    assert harness.request_body_sha256() == harness.FIRST_BODY_SHA256
    assert 'request_body_sha256' not in identity(ep)['settings']


def test_a_changed_body_makes_saved_labels_stale(tmp_path, monkeypatch):
    ep = recording(tmp_path, np.arange(12) / 30)
    saved = identity(ep)
    monkeypatch.setattr(harness, 'OPENROUTER_BODY', {**harness.OPENROUTER_BODY,
                                                     'response_format': {'type': 'json_schema'}})
    current = identity(ep)
    assert current['settings']['request_body_sha256'] != harness.FIRST_BODY_SHA256
    assert not only_implementation_changed(saved, current)
