"""Pins on the readout thresholds: for each one, a case just inside it and a case just outside, so moving the threshold
changes an outcome here. label/signals.py resolution and tracking; label/episode.py camera capture mapping."""
from __future__ import annotations

import json

import numpy as np

from label import episode as me
from label import signals as sg
from tests.test_readout_duplicates import FPS, N, _path

T = np.arange(N) / FPS
KS = list(range(0, N, FPS))      # one instant a second for 60 s


def _later(x: np.ndarray, frames: int = 3) -> np.ndarray:
    return np.concatenate([np.full(frames, x[0]), x[:-frames]])


def test_a_reading_that_half_stands_for_its_second_is_not_resolved():
    """Random levels held for a second each, plus a 2.7 Hz wave: the readings stand for their seconds with a share of
    about 0.68 (between 0.5 and RESOLVED_STAND) and do not follow one another, so the row is dropped; with a smaller
    wave the share is about 0.86 and the row is kept."""
    rng = np.random.default_rng(5)
    rng.normal(size=N)           # the same draw order as the fixture search
    levels = np.repeat(rng.normal(size=N // FPS + 1), FPS)[FPS // 2:FPS // 2 + N]
    for amp, share, kept in ((1.0, (0.6, 0.75), False), (0.6, (0.855, 0.87), True)):
        follow, stand = sg.resolution((levels + amp * np.sin(2 * np.pi * 2.7 * T))[:, None], KS)
        assert share[0] < stand[0] < share[1] and follow[0] < sg.RESOLVED_FOLLOW
        assert bool(sg.is_resolved(follow, stand)[0]) is kept


def test_twenty_instants_are_enough_to_drop_an_unresolved_row():
    """Noise read at 20 instants (more than RESOLVED_MIN_INSTANTS, fewer than 40) is judged, and dropped."""
    noise = np.random.default_rng(3).normal(size=(20 * FPS, 1))
    assert sg.resolved(noise, list(range(0, 20 * FPS, FPS))).tolist() == [False]


def test_the_level_threshold_sits_between_these_two_pairs():
    """y follows x 0.1 s later plus a slow wave of its own: with the wave at 0.25 the level fit is about 0.961 and the
    pair tracks, at 0.30 it is about 0.945 and it does not (TRACK_LEVEL 0.95); the change fit stays above 0.99."""
    x = _path(4)
    slow = np.sin(2 * np.pi * T / 37 + 1.0)
    assert sg.tracking(x, _later(x) + 0.25 * slow, KS, FPS)[0] is True
    assert sg.tracking(x, _later(x) + 0.30 * slow, KS, FPS)[0] is False


def test_the_change_threshold_sits_between_these_two_pairs():
    """y follows x plus white noise: at 0.16 the change fit is about 0.912 and the pair tracks, at 0.18 about 0.893 and
    it does not (TRACK_CHANGE 0.9); the level fit stays above 0.96."""
    x = _path(4)
    noise = np.random.default_rng(5).normal(size=N)
    assert sg.tracking(x, _later(x) + 0.16 * noise, KS, FPS)[0] is True
    assert sg.tracking(x, _later(x) + 0.18 * noise, KS, FPS)[0] is False


def _bump(height: float, secs: float, at: float = 30.0) -> tuple[np.ndarray, np.ndarray]:
    """x and an exact copy of it 0.1 s later, except for a step of `height` of x's range held for `secs` seconds."""
    x = _path(4)
    y = _later(x)
    y[int(at * FPS):int((at + secs) * FPS)] += height * np.ptp(x)
    return x, y


def _apart(x, y):
    same, lag = sg.tracking(x, y, KS, FPS)
    assert same
    return sg.apart_instants(x, y, KS, lag, FPS)


def test_the_apart_threshold_sits_between_these_two_steps():
    """A 2 s step of 0.025 of the range comes apart (TRACK_APART 0.02); one of 0.015 does not."""
    assert 30 in _apart(*_bump(0.025, 2.0))
    assert _apart(*_bump(0.015, 2.0)) == []


def test_the_hold_window_is_one_second():
    """A step of 0.03 of the range held 1 s keeps a one second mean of 0.03 and comes apart, but a two second window
    would average it to 0.015; held 0.5 s its one second mean is 0.015 and it does not come apart, though a half second
    window would keep 0.03 (TRACK_HOLD_S 1)."""
    assert 30 in _apart(*_bump(0.03, 1.0))
    assert _apart(*_bump(0.03, 0.5)) == []


def _capture(own_times, kmap, anchor_fps=30.0, n=90):
    ep = {"context": {"profile": "teleop_arms", "state_kind": "none", "fps": anchor_fps,
                      "cameras": {"exo": {"name": "scene"}, "left": {"name": "left"}}},
          "state": np.zeros((n, 0)), "sources": {"exo": {"n_frames": n}, "left": {"n_frames": len(own_times)}},
          "kmap": {"left": np.asarray(kmap)},
          "times": {"exo": np.arange(n) / anchor_fps, "left": np.asarray(own_times, dtype=float)}}
    text = me._camera_capture_text(ep, {"ks": [0, 30, 60, 88]})
    return json.loads(text.splitlines()[-1]) if "CAMERA CAPTURE MAPPING" in text else {}


def test_an_offset_between_half_a_frame_and_a_frame_is_listed():
    """25 ms at 30 fps is more than half a frame (16.7 ms) and less than a whole one (33.3 ms): listed."""
    mapping = _capture(np.arange(90) / 30.0 + 0.025, np.arange(90))
    assert [row[3] for row in mapping["left"]] == [0.025] * 4


def test_with_mixed_frame_rates_the_shorter_frame_interval_decides():
    """A 15 fps camera (66.7 ms frames) paired to a 30 fps anchor (33.3 ms), 25 ms late: more than half the anchor's
    frame though less than half its own, so it is listed."""
    mapping = _capture(np.arange(45) / 15.0 + 0.025, np.arange(90) // 2)
    assert [row[3] for row in mapping["left"]] == [0.025] * 4
