"""The per-instant readout sends a value's row only when its readings at the sampled instants resolve it, and the
camera capture mapping lists only offsets of at least half a frame interval; what each leaves out is named."""
from __future__ import annotations

import json

import numpy as np

from label import episode as me
from label import signals as sg
from tests.test_prompt_blocks import CASES, _pl


def _arm(n=1800, fps=30):
    """60 s of one arm: a joint that sweeps, a gripper that closes three times for a few seconds (a few large steps,
    so its movement score is low), a velocity of a joint that oscillates 2.7 times a second (aliased when read
    once a second) and a noisy effort with large swings (a high movement score)."""
    t = np.arange(n) / fps
    rng = np.random.default_rng(0)
    joint = 0.8 * np.sin(0.3 * t)
    gripper = np.ones(n)
    for a, b in ((8, 12), (25, 31), (44, 47)):
        gripper[(t >= a) & (t < b)] = 0.04
    vel = 0.5 * 2 * np.pi * 2.7 * np.cos(2 * np.pi * 2.7 * t)
    effort = 20 * rng.normal(0, 1, n)
    return t, joint, gripper, vel, effort


def _readout_case():
    ep, _ = CASES["teleop_video_only"]()
    n = 1800
    _t, joint, gripper, vel, effort = _arm(n)
    ep["state"] = np.zeros((n, 0))
    ep["signals"] = {"arm.joint": joint[:, None], "arm.gripper": gripper[:, None], "arm.vel": vel[:, None],
                     "arm.effort": effort[:, None], "arm.still_vel": np.zeros((n, 1))}
    ep["signal_meta"] = {k: {} for k in ep["signals"]}
    return ep, _pl(n, every=1.0)


def test_resolution_separates_positions_from_aliased_rates():
    _t, joint, gripper, vel, effort = _arm()
    ks = list(range(0, 1800, 30))
    a = np.stack([joint, gripper, vel, effort], axis=1)
    assert sg.resolved(a, ks).tolist() == [True, True, False, False]
    # a vibration sampled exactly at its own period reads the same at every instant while the frames around swing
    t = np.arange(1800) / 30.0
    beat = np.cos(2 * np.pi * 3 * t) + 0.01 * np.sin(0.2 * t)
    assert sg.resolved(beat[:, None], ks).tolist() == [False]
    # movement alone ranks the gripper below the noisy effort, which is why it does not decide the budget
    mv = sg.movements(a)
    assert mv[1] < mv[3]


def test_a_gripper_position_row_survives_and_an_aliased_velocity_row_is_dropped_with_its_range_kept():
    ep, pl = _readout_case()
    table = me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1]
    table = table.split("OTHER RECORDED SIGNALS")[1].split("BETWEEN INSTANTS")[0]
    readout = table.split("at every instant you receive")[1]
    assert "    arm.gripper: " in readout and "    arm.joint: " in readout
    assert "    arm.vel: " not in readout and "    arm.effort: " not in readout
    # every signal keeps its range line, and the rows left out are named with why
    assert "  arm.vel (1 value): " in table and "  arm.effort (1 value): " in table
    left = [l for l in table.splitlines() if "because one instant every 1.00 s does not resolve their readings" in l]
    assert len(left) == 1 and "arm.vel" in left[0] and "arm.effort" in left[0]
    assert "arm.gripper" not in left[0] and "arm.joint" not in left[0]
    assert left[0].endswith("Their ranges are given above, and when each reached its largest and smallest reading.")
    assert "arm.vel (max " in left[0] and " s, min " in left[0]
    # a velocity that never changes gets no row and is named once with its value
    assert "    arm.still_vel: " not in readout and "arm.still_vel 0" in table


def test_under_a_tight_budget_the_gripper_row_outranks_a_row_that_only_moves_more(monkeypatch):
    """A smooth effort that moves far (high movement score) and the gripper both resolve; a budget with room for the
    times row and one value row keeps the gripper, whose readings stand for their instants better."""
    ep, pl = _readout_case()
    t = np.arange(1800) / 30.0
    ep["signals"] = {"arm.gripper": ep["signals"]["arm.gripper"],
                     "arm.load": (5 * np.sin(0.05 * t) + 0.3 * np.sin(2 * np.pi * 0.4 * t))[:, None]}
    ep["signal_meta"] = {k: {} for k in ep["signals"]}
    assert sg.movements(ep["signals"]["arm.load"])[0] > sg.movements(ep["signals"]["arm.gripper"])[0]
    lines, _ = me._signal_readout(ep, pl)
    head = next(l for l in lines if l.startswith("    at: "))
    grip = next(l for l in lines if l.startswith("    arm.gripper: "))
    monkeypatch.setattr(me, "SIGNAL_TABLE_MAX_CHARS", len(head) + len(grip) + 1)
    lines, whole = me._signal_readout(ep, pl)
    shown = [l.split(":")[0].strip() for l in lines if l.startswith("    ") and not l.startswith("    at: ")]
    assert shown == ["arm.gripper"] and "arm.load" in lines[-1]


def _camera_case(offsets, fps=30.0, n=90):
    """An anchor at fps and a wrist camera on its own clock, stamped offsets[i] after the anchor at frame i."""
    ep = {"context": {"profile": "teleop_arms", "state_kind": "none", "fps": fps,
                      "cameras": {"exo": {"name": "scene"}, "left": {"name": "left"}}},
          "state": np.zeros((n, 0)), "sources": {"exo": {"n_frames": n}, "left": {"n_frames": n}},
          "kmap": {"left": np.arange(n)}}
    anchor = np.arange(n) / fps
    ep["times"] = {"exo": anchor, "left": anchor + np.asarray(offsets, dtype=float)}
    return ep


def test_sub_frame_capture_offsets_are_not_listed_but_their_size_is_said():
    ep = _camera_case(np.full(90, 0.009810447692871094))
    text = me._camera_capture_text(ep, {"ks": [0, 30, 60, 89]})
    assert "CAMERA CAPTURE MAPPING" not in text and "{" not in text
    assert text.strip() == ("CAMERA CAPTURE TIMES: every camera's captures are within half a frame interval of their "
                            "column times, at most 9.8 ms for left.")
    # the frame heading does not point to a mapping that is not there
    assert "capture mapping below" not in me._frames_head(ep, 240, 180, (640, 480),
                                                           capture_mapped="CAMERA CAPTURE MAPPING" in text)


def test_an_offset_of_half_a_frame_or_more_is_listed_to_the_millisecond():
    offsets = np.full(90, 0.002)
    offsets[30] = 0.0412345
    ep = _camera_case(offsets)
    text = me._camera_capture_text(ep, {"ks": [0, 30, 60, 89]})
    assert "CAMERA CAPTURE MAPPING" in text
    assert "the largest left out is 2.0 ms for left." in text
    mapping = json.loads(text.splitlines()[-1])
    assert mapping == {"left": [[1.0, 30, 1.041, 0.041]]}
    heading_mapped = "CAMERA CAPTURE MAPPING" in text
    assert heading_mapped


def test_a_camera_with_no_offsets_adds_no_capture_text():
    ep = _camera_case(np.zeros(90))
    assert me._camera_capture_text(ep, {"ks": [0, 30, 60, 89]}) == ""


def test_with_few_instants_every_row_is_kept():
    """Fewer than RESOLVED_MIN_INSTANTS readings cannot estimate either score, so nothing is dropped as unresolved."""
    _t, joint, gripper, vel, effort = _arm()
    ks = list(range(0, 30 * (sg.RESOLVED_MIN_INSTANTS - 1), 30))
    assert sg.resolved(np.stack([joint, vel, effort], axis=1), ks).all()
