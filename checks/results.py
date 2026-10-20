"""The shape every deterministic check gives a result that could not run or does not apply, shared by
checks/capture_qc.py, checks/sensors.py and checks/stream_pairing.py.

A check module that lists its checks (capture_qc, sensor_checks) gives each one a row with one of four statuses:

  fired            the check found something
  clear            the check ran and found nothing
  not_applicable   the check has nothing to run on in this episode, with why when there is a reason to give
  errored          the check stopped with an error: "error" is the error, "why" says it in words

An episode none of whose checks could run at all has "error" at the top of its result as well. A module that is one
check per result (stream_pairing's three modes) gives {"error": ...} when it stopped with an error and
{"not_assessed": why} when it does not apply.

checks/sensors.py wrote "na" for not_applicable until 2026-10-20. status() reads it as not_applicable, so a result
saved before then needs no rerun and reads the same everywhere."""
from __future__ import annotations

from pathlib import Path

FIRED, CLEAR, NOT_APPLICABLE, ERRORED = "fired", "clear", "not_applicable", "errored"
FINISHED = (CLEAR, FIRED, NOT_APPLICABLE)   # a row with any of these needs no rerun
LEGACY = {"na": NOT_APPLICABLE}


def status(row: dict) -> str | None:
    """A row's status, with a word saved before the statuses were shared read as the one it is now."""
    s = row.get("status") if isinstance(row, dict) else None
    return LEGACY.get(s, s)


def error_text(e: BaseException) -> str:
    """An error as every check records it: its type and message, at most 300 characters."""
    return f"{type(e).__name__}: {e}"[:300]


def errored_row(row: dict, error: str) -> dict:
    """A listed check that stopped with an error."""
    return {**row, "status": ERRORED, "error": error, "why": f"the check stopped with an error ({error})"}


def not_applicable_row(row: dict, why: str | None = None) -> dict:
    """A listed check with nothing to run on, with why when there is a reason to give."""
    return {**row, "status": NOT_APPLICABLE, **({"why": why} if why else {})}


def rows_need_check(rows, names) -> bool:
    """Whether a stored list of check rows has to be run again: it is not a list of rows, it does not list each of
    names exactly once, or one of its checks did not finish."""
    if not isinstance(rows, list) or not rows or not all(isinstance(r, dict) for r in rows):
        return True
    listed = [r.get("check") for r in rows]
    if not all(isinstance(n, str) for n in listed) or len(listed) != len(set(listed)) or set(listed) != set(names):
        return True
    return any(status(r) not in FINISHED for r in rows)


def stale_version(ctx: dict, key: str, version: int) -> bool:
    """Whether context.json holds no result under key, or one an older version of its check wrote
    (context.json check_versions)."""
    versions = ctx.get("check_versions")
    return (not isinstance(versions, dict) or type(versions.get(key)) is not int or versions[key] != version
            or key not in ctx)


def run_safely(d: str, run, errored) -> tuple:
    """(d, run(d)) for one episode folder, or (d, errored(e)) when run raises, so an episode whose checks cannot run
    still gets a result, recorded on its checks, never silently skipped. The third item is the error of a worker that
    stopped outside run, which only a worker pool gives (checks/workers.py)."""
    try:
        return d, run(Path(d)), None
    except Exception as e:  # noqa: BLE001 - recorded as the result, never silently skipped
        return d, errored(e), None
