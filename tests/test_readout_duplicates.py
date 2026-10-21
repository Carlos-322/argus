"""A per-instant row whose readings track a row already sent is sent once, named as tracking it; its range line stays."""
from __future__ import annotations

import numpy as np

from label import episode as me
from label import signals as sg
from tests.test_prompt_blocks import CASES, _pl

N, FPS = 1800, 30


def _path(seed: int) -> np.ndarray:
    """60 s of one joint moving smoothly and irregularly: a few slow waves of random period and phase."""
    rng = np.random.default_rng(seed)
    t = np.arange(N) / FPS
    return sum(rng.uniform(0.2, 0.8) * np.sin(2 * np.pi * t / rng.uniform(4, 15) + rng.uniform(0, 6)) for _ in range(4))


def _later(x: np.ndarray, frames: int, noise: float = 0.002, seed: int = 9) -> np.ndarray:
    """x reached `frames` later, with a little noise: a measured joint following its command."""
    y = np.concatenate([np.full(frames, x[0]), x[:-frames]])
    return y + np.random.default_rng(seed).normal(0, noise, len(x))


def _case(signals: dict, names: dict | None = None) -> tuple[dict, dict]:
    ep, _ = CASES["teleop_video_only"]()
    ep["state"] = np.zeros((N, 0))
    ep["signals"] = {k: (v if v.ndim == 2 else v[:, None]) for k, v in signals.items()}
    ep["signal_meta"] = {k: ({"names": names[k]} if names and k in names else {}) for k in signals}
    return ep, _pl(N, every=1.0)


def _readout(ep, pl):
    lines, whole = me._signal_readout(ep, pl)
    shown = [l.split(":")[0].strip() for l in lines if l.startswith("    ") and not l.startswith("    at: ")]
    left = [l for l in lines if "so each is sent once" in l]
    return shown, left, whole, lines


def test_a_commanded_arm_and_the_arm_that_follows_it_send_the_following_rows_once():
    """A 7-value command (six joints and a gripper) and the measured arm that follows it 0.1 s later, recorded as six
    joints and a separate gripper: each command value is matched to its measured value column by column, the measured
    rows are kept whatever the names say, and each command row is named as tracking its measured row."""
    cmd = np.stack([_path(s) for s in range(7)], axis=1)
    cmd[:, 6] = (np.sin(2 * np.pi * np.arange(N) / FPS / 9) > 0.3).astype(float)
    arm = np.stack([_later(cmd[:, j], 3, seed=j) for j in range(6)], axis=1)
    grip = _later(cmd[:, 6], 3, seed=6)
    names = {"zz.a": [f"q{j}" for j in range(6)] + ["grip"], "aa.b": [f"q{j}" for j in range(6)]}
    ep, pl = _case({"zz.a": cmd, "aa.b": arm, "aa.b_grip": grip}, names)
    shown, left, whole, _ = _readout(ep, pl)
    assert shown == [f"aa.b q{j}" for j in range(6)] + ["aa.b_grip"]
    assert len(left) == 1
    assert left[0].startswith("  The values at each instant leave out zz.a q0, q1, q2, q3, q4 and q5 (tracking aa.b "
                              "q0, q1, q2, q3, q4 and q5) and zz.a grip (tracking aa.b_grip)")
    assert left[0].endswith("Their ranges are given above, where values that come apart still show.")
    assert "zz.a" not in whole and {"aa.b", "aa.b_grip"} <= whole
    # every range line stays
    table = me.build_prompt(ep, pl, cell_w=448, cell_h=252)[1].split("OTHER RECORDED SIGNALS")[1]
    assert "  zz.a (7 values (q0, " in table and "  aa.b (6 values (q0, " in table and "  aa.b_grip (1 value): " in table


def test_two_unrelated_joints_both_keep_their_rows():
    ep, pl = _case({"j.one": _path(1), "j.two": _path(2)})
    shown, left, _, _ = _readout(ep, pl)
    assert shown == ["j.one", "j.two"] and not left


def test_a_sign_flipped_scaled_copy_is_caught():
    """The same joint in degrees and mirrored (-57.3 x + 10) tracks its reading in radians."""
    x = _path(3)
    same, lag = sg.tracking(x, -57.2958 * x + 10.0, list(range(0, N, 30)), FPS)
    assert same and lag == 0
    ep, pl = _case({"rad": x, "deg_mirrored": -57.2958 * x + 10.0})
    shown, left, _, _ = _readout(ep, pl)
    assert len(shown) == 1 and len(left) == 1 and "tracking" in left[0]


def test_a_pair_that_comes_apart_halfway_keeps_both_rows():
    """A measured joint that follows its command for 30 s and then goes its own way (a tracking fault) is no copy."""
    cmd = _path(4)
    arm = _later(cmd, 3)
    arm[N // 2:] = _path(5)[N // 2:] + arm[N // 2 - 1] - _path(5)[N // 2 - 1]
    ep, pl = _case({"cmd": cmd, "arm": arm})
    shown, left, _, _ = _readout(ep, pl)
    assert sorted(shown) == ["arm", "cmd"] and not left


def test_few_instants_never_track():
    x = _path(6)
    assert sg.tracking(x, x, list(range(0, 30 * (sg.RESOLVED_MIN_INSTANTS - 1), 30)), FPS) == (False, 0)


def _stalled(secs: int, seed: int = 4):
    """A command and the measured joint that follows it 0.1 s later, except that the joint stalls for `secs` seconds
    from 20 s while the command keeps moving (a tracking fault)."""
    cmd = _path(seed)
    arm = _later(cmd, 3)
    s0 = 20 * FPS
    arm[s0:s0 + secs * FPS] = arm[s0]
    return cmd, arm



def _apart_lines(lines):
    """The lines of rows that track another except where they come apart: (row, partner, [reading times])."""
    out = []
    for l in lines:
        if " tracks " in l and " except " in l:
            row, rest = l.strip().split(" tracks ", 1)
            other = rest.split(" except ", 1)[0]
            times = [float(p.split()[0]) for p in rest.split("given: ", 1)[1].rstrip(".").split(", ")]
            out.append((row, other, times))
    return out


def test_a_two_second_stall_sends_the_command_only_around_the_stall():
    """The joint stalls from 20 to 22 s while its command moves 12% of its range: the joint's row is sent, and the
    command's readings are given only at the instants around the stall, on a line naming the row it tracks."""
    cmd, arm = _stalled(2)
    assert np.ptp(cmd[20 * FPS:22 * FPS]) > 0.1 * np.ptp(cmd)
    shown, left, _, lines = _readout(*_case({"cmd": cmd, "arm": arm}))
    assert shown == ["arm"] and not left
    (row, other, times), = _apart_lines(lines)
    assert (row, other) == ("cmd", "arm")
    assert 20.0 in times and 21.0 in times and min(times) >= 18.0 and max(times) <= 24.0
    assert any(l.strip().startswith("These rows follow another row") for l in lines)


def test_a_six_second_stall_sends_the_command_only_around_the_stall():
    cmd, arm = _stalled(6, seed=13)
    shown, _, _, lines = _readout(*_case({"cmd": cmd, "arm": arm}))
    assert shown == ["arm"]
    (row, other, times), = _apart_lines(lines)
    assert (row, other) == ("cmd", "arm")
    assert {21.0, 23.0, 25.0} <= set(times) and min(times) >= 18.0 and max(times) <= 28.0


def test_an_offset_that_opens_late_is_read_where_it_is_open():
    """The measured joint drifts 0.2 sd away from its command from 40 s to 50 s of 60 and stays there (a calibration
    slip): the command is read only over the late stretch where the two have come apart."""
    cmd = _path(7)
    t = np.arange(N) / FPS
    arm = _later(cmd, 3) + 0.2 * cmd.std() * np.clip((t - 40) / 10, 0, 1)
    shown, _, _, lines = _readout(*_case({"cmd": cmd, "arm": arm}))
    assert shown == ["arm"]
    (row, other, times), = _apart_lines(lines)
    assert row == "cmd" and min(times) >= 35.0 and max(times) >= 55.0


def test_a_pair_apart_most_of_the_time_keeps_both_full_rows():
    """An offset of 0.3 sd that opens at 20 s and stays: the two still fit closely over the whole request, but they are
    apart at more than half the instants, so both full rows are sent."""
    cmd = _path(7)
    t = np.arange(N) / FPS
    arm = _later(cmd, 3) + 0.3 * cmd.std() * (t >= 20)
    ks = list(range(0, N, FPS))
    same, lag = sg.tracking(cmd, arm, ks, FPS)
    assert same and len(sg.apart_instants(cmd, arm, ks, lag, FPS)) > sg.TRACK_APART_MOST * len(ks)
    shown, left, _, lines = _readout(*_case({"cmd": cmd, "arm": arm}))
    assert sorted(shown) == ["arm", "cmd"] and not left and not _apart_lines(lines)


def test_a_clean_copy_gives_no_readings():
    x = _path(3)
    shown, left, _, lines = _readout(*_case({"rad": x, "deg": 57.2958 * x}))
    assert len(shown) == 1 and len(left) == 1 and not _apart_lines(lines)


def test_a_stall_between_instants_is_seen_on_the_frames():
    """A one second stall that starts on an instant and ends before the next one changes no reading at the instants,
    yet the command moved 12% of its range meanwhile: the frames show it, so the instants on both sides are read."""
    cmd, arm = _stalled(1)
    ks = list(range(0, N, FPS))
    assert np.allclose(arm[ks], _later(cmd, 3)[ks])
    same, lag = sg.tracking(cmd, arm, ks, FPS)
    apart = sg.apart_instants(cmd, arm, ks, lag, FPS)
    assert same and {20, 21} <= set(apart) and len(apart) <= 5
