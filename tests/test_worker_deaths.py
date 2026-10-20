"""A worker process that dies (killed by a signal, out of memory) costs its own episode only, in every check that runs
episodes on a process pool (checks/workers.py)."""
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

from checks import EXIT_PARTIAL
from checks.workers import run_each

REPO = Path(__file__).resolve().parent.parent


def _dies_on(item: str) -> str:
    """item back, except that the worker on "victim" kills itself, and the worker on a "once:<marker>" item kills
    itself the first time only."""
    if item == "victim":
        os.kill(os.getpid(), signal.SIGKILL)
    if item.startswith("once:"):
        marker = Path(item[len("once:"):])
        if not marker.exists():
            marker.touch()
            os.kill(os.getpid(), signal.SIGKILL)
    return item


def test_a_worker_that_dies_costs_only_its_own_item_and_every_item_is_recorded_once():
    """One dead worker broke the whole pool, so every item still waiting was lost with it. The others are run again
    and recorded; only the item whose worker dies again on its own retry is recorded through died, with the error."""
    recorded = []
    retried = run_each(_dies_on, ["a", "b", "victim", "c", "d"], 2, recorded.append,
                       lambda item, e: f"errored {item} {type(e).__name__}")
    assert sorted(recorded) == ["a", "b", "c", "d", "errored victim BrokenProcessPool"], recorded
    assert "victim" in retried


def test_an_item_whose_worker_dies_once_is_recorded_from_its_retry(tmp_path):
    """A worker killed once (memory short while two large episodes run together) costs nothing: the retry alone
    records the item's own answer."""
    once = f"once:{tmp_path / 'died'}"
    recorded = []
    run_each(_dies_on, [once], 1, recorded.append, lambda item, e: f"errored {item}")
    assert recorded == [once] and (tmp_path / "died").exists()


SENSORS_KILLER = '''
import os, signal
from pathlib import Path
from checks import sensors

VICTIM = os.environ["VICTIM"]


def run_episode(ep_dir):
    if Path(ep_dir).name == VICTIM:
        os.kill(os.getpid(), signal.SIGKILL)
    return {"notes": [], "flagged": False,
            "checks": [{"check": c, "name": sensors.NAMES[c], "status": "clear"} for c in sensors.NAMES]}


def constant_values(ep_dir):
    if Path(ep_dir).name == VICTIM:
        os.kill(os.getpid(), signal.SIGKILL)
    return {}


sensors.run_episode = run_episode
sensors.constant_values = constant_values
if __name__ == "__main__":
    raise SystemExit(sensors.main())
'''

PAIRING_KILLER = '''
import os, signal
from checks import stream_pairing

VICTIM = os.environ["VICTIM"]


def assess(ep):
    if ep.name == VICTIM:
        os.kill(os.getpid(), signal.SIGKILL)
    return {"crossed": False}


key, _, field = stream_pairing.MODES["pairing"]
stream_pairing.MODES["pairing"] = (key, assess, field)
if __name__ == "__main__":
    raise SystemExit(stream_pairing.main())
'''


def _run(tmp_path, script: str, victim: str):
    names = [f"episode_{i:06d}" for i in range(6)]
    for n in names:
        (tmp_path / "eps" / n).mkdir(parents=True)
        (tmp_path / "eps" / n / "context.json").write_text(json.dumps({"fps": 30}))
    (tmp_path / "killer.py").write_text(script)
    r = subprocess.run([sys.executable, str(tmp_path / "killer.py"), "--jobs", "2", "--force", str(tmp_path / "eps")],
                       cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO), "VICTIM": victim},
                       capture_output=True, text=True, timeout=300)
    return r, {n: json.loads((tmp_path / "eps" / n / "context.json").read_text()) for n in names}


def test_the_sensor_checks_record_every_other_episode_when_a_worker_dies(tmp_path):
    """A worker killed in either of the sensor checks' pools (the constant values, then the checks) stopped the whole
    step with BrokenProcessPool and no result for the episodes still waiting, and every rerun stopped the same way.
    Now the other episodes are checked and the dead one is recorded with every check errored, so the step exits
    EXIT_PARTIAL and a rerun checks it again."""
    from checks import sensors
    r, ctx = _run(tmp_path, SENSORS_KILLER, "episode_000002")
    assert r.returncode == EXIT_PARTIAL, r.stdout + r.stderr
    for n, c in ctx.items():
        statuses = {x["status"] for x in c["sensor_checks"]["checks"]}
        if n == "episode_000002":
            assert statuses == {"errored"} and "BrokenProcessPool" in c["sensor_checks"]["error"], c
            assert sensors.needs_check(c)
        else:
            assert statuses == {"clear"} and not sensors.needs_check(c), (n, c)
    assert "ERRORED episode_000002" in r.stdout and "failed=1" in r.stdout, r.stdout


def test_stream_pairing_records_every_other_episode_when_a_worker_dies(tmp_path):
    """A worker killed while pairing streams stopped the whole step with BrokenProcessPool. Now the other episodes get
    their result and the dead one gets an error result, so the step exits EXIT_PARTIAL and a rerun checks it again."""
    from checks import stream_pairing
    r, ctx = _run(tmp_path, PAIRING_KILLER, "episode_000004")
    assert r.returncode == EXIT_PARTIAL, r.stdout + r.stderr
    for n, c in ctx.items():
        if n == "episode_000004":
            assert "BrokenProcessPool" in c["stream_pairing"]["error"] and c["stream_pairing"]["crossed"] is False, c
            assert stream_pairing.needs_check(c, "pairing")
        else:
            assert c["stream_pairing"] == {"crossed": False}, (n, c)
            assert not stream_pairing.needs_check(c, "pairing")
    assert "ERRORED episode_000004" in r.stdout and "failed=1" in r.stdout, r.stdout
