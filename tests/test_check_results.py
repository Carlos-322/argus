"""checks/results.py: one shape for a check that could not run or does not apply. checks/capture_qc.py wrote
not_applicable where checks/sensors.py wrote na, a capture check that errored had only why where a sensor check had
only error, and an episode whose capture checks could not run at all had no error of its own."""
from __future__ import annotations

import json

from board import serve
from checks import capture_qc, results, sensors, stream_pairing


def test_an_episode_whose_checks_cannot_run_reads_the_same_in_every_check_module(tmp_path):
    missing = str(tmp_path / "episode_000000")
    _, capture, _ = capture_qc._safe(missing)
    _, sensor, _ = sensors._safe(missing)
    _, pairing, _ = stream_pairing._safe("pairing", missing)
    for result in (capture, sensor, pairing):
        assert isinstance(result["error"], str) and result["error"], result
        assert serve.withheld_status(result) == {"status": "errored", "error": result["error"]}
    for result in (capture, sensor):
        assert {results.status(row) for row in result["checks"]} == {results.ERRORED}
        assert all(row["error"] and row["why"].startswith("the check stopped with an error") for row in result["checks"])
    assert pairing["crossed"] is False


def test_a_check_with_nothing_to_run_on_is_not_applicable_in_every_module(tmp_path):
    from test_board_sensors import _episode
    ep = _episode(tmp_path / "eps")                  # signals, no depth: the depth checks have nothing to run on
    ctx = json.loads((ep / "context.json").read_text())
    (ep / "context.json").write_text(json.dumps({**ctx, "state_kind": "none"}))
    rows = {row["check"]: row for row in sensors.run_episode(ep)["checks"]}
    assert {rows[c]["status"] for c in sensors.DEPTH_CHECKS} == {results.NOT_APPLICABLE}
    assert not results.rows_need_check(list(rows.values()), sensors.NAMES)
    capture = {row["status"] for row in capture_qc.run_episode(ep)["checks"]}
    assert results.NOT_APPLICABLE in capture and capture <= {*results.FINISHED, results.ERRORED}


def test_a_sensor_result_saved_with_na_needs_no_rerun_and_reads_as_not_applicable():
    rows = [{"check": c, "name": n, "status": "na" if c in sensors.DEPTH_CHECKS else "clear"}
            for c, n in sensors.NAMES.items()]
    ctx = {"check_versions": {"sensor_checks": sensors.VERSION},
           "sensor_checks": {"notes": [], "flagged": False, "checks": rows}}
    assert not sensors.needs_check(json.loads(json.dumps(ctx)))
    assert {results.status(r) for r in rows if r["check"] in sensors.DEPTH_CHECKS} == {results.NOT_APPLICABLE}
    now = [{**r, "status": results.status(r)} for r in rows]
    assert serve.withheld_status({"checks": rows}) == serve.withheld_status({"checks": now})
    rows[0]["status"] = "errored"
    assert sensors.needs_check(ctx)
