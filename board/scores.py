"""Episode and subgoal quality scores, computed from the labels a board already holds, never asked of the model.

A model reads one episode at a time, so a number it writes for one episode is not comparable with the number it writes
for another. These scores are measured instead: each one is built from fields the annotation already records (the
timeline's advancing, wasteful and idle steps, the key events, the operator mistakes and recoveries, the completion and
the goal alignment), and the ones that only mean something next to other demonstrations of the same task (how long the
goal took, how much of the work hesitated) are ranked within the episode's task group on the board.

Four dimensions, each 0 to 10, each with the numbers it came from (inputs), so the board can say why:

| Dimension | Question | Built from | Compared |
|---|---|---|---|
| outcome | How much of the labelled goal was achieved | completion, the progress reached | absolute |
| cleanliness | How free the work is of mistakes and wasted motion | operator mistakes by severity, failed attempts, wasteful time | absolute |
| efficiency | How quickly the goal was reached, for this task | time from the first action to the goal | within the task group |
| fluency | How steadily the work went, for this task | hesitation inside the work, actions per unit of progress | within the task group |

execution, the quality of what the operator actually did, is cleanliness, efficiency and fluency together; outcome is
kept apart, so a clean demonstration of a mislabelled task reads as a good execution with relation "different", never
as a bad one. A dimension with too few peers to compare against is None, never a guess.
"""
from __future__ import annotations

import re
import statistics

VERSION = 1
MIN_PEERS = 5                     # a relative dimension needs this many other demonstrations of the same task
SEVERITY_POINTS = {"low": 0.5, "medium": 1.5, "high": 3.0}
FAILED_ATTEMPT_POINTS = 1.0       # a key event that failed, beyond the mistake that may describe it
OUTCOME_CAPS = {"success": 1.0, "success_then_undone": 0.7, "partial": 0.9, "failure": 0.4}
EXECUTION_WEIGHTS = {"cleanliness": 0.5, "efficiency": 0.25, "fluency": 0.25}
STOP = {"the", "a", "an", "and", "then", "with", "its", "it", "of", "on", "to", "into", "in", "from", "onto", "by", "at",
        "for", "both", "each", "one", "that", "this", "is", "are", "be", "so", "as", "up", "down", "out", "back", "over"}


def _num(x):
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) else None


def _union(spans):
    total, end = 0.0, None
    for a, b in sorted((a, b) for a, b in spans if b > a):
        if end is None or a > end:
            total, end = total + b - a, b
        elif b > end:
            total, end = total + b - end, b
    return total


def _subtract(spans, cut):
    """The parts of spans not covered by any span in cut."""
    out = []
    for a, b in spans:
        pieces = [(a, b)]
        for c, d in cut:
            pieces = [p for x, y in pieces for p in ((x, min(y, c)), (max(x, d), y)) if p[1] > p[0]]
        out += pieces
    return out


def facts(d: dict) -> dict:
    """What one episode's labels measure, before any comparison: its work window, its time split, its attempts."""
    steps = [e for e in d.get("event_labels") or [] if _num(e.get("t_s")) is not None and _num(e.get("end_s")) is not None]
    spans = {c: [(e["t_s"], e["end_s"]) for e in steps if e.get("contribution") == c]
             for c in ("advancing", "wasteful", "idle")}
    work = spans["advancing"] + spans["wasteful"]
    start = min((a for a, _ in work), default=None)
    comp = d.get("completion") or {}
    goal = _num(comp.get("completed_at_s"))
    if goal is None:
        goal = _num(comp.get("goal_reached_at_s"))
    end = goal if goal is not None else max((b for _, b in work), default=None)
    active = _union(work)
    wasteful = _union(spans["wasteful"])
    # hesitation: stretches inside the work window where no arm advances or wastes, ie every arm waits
    window = [(start, end)] if start is not None and end is not None and end > start else []
    hesitation = _union(_subtract(window, work)) if window else 0.0
    progress = [p for p in (_num(e.get("progress")) for e in steps) if p is not None]
    reached = max(progress, default=0.0)
    mistakes = [m for m in d.get("operator_mistakes") or [] if isinstance(m, dict) and m.get("deliberate") != "yes"]
    failed = [k for k in d.get("key_events") or [] if isinstance(k, dict) and k.get("outcome") == "failure"]
    return {"start_s": start, "goal_s": goal, "end_s": end, "active_s": active, "wasteful_s": wasteful,
            "hesitation_s": hesitation, "window_s": (end - start) if window else None,
            "actions": sum(1 for e in steps if e.get("contribution") == "advancing"), "progress_reached": reached,
            "outcome": comp.get("task_completed"), "mistakes": mistakes, "failed_attempts": len(failed),
            "relation": (d.get("goal_alignment") or {}).get("relation")}


def task_key(d: dict) -> frozenset:
    text = (d.get("episode_prompt") or "").strip() or " ".join(d.get("task_label") or [])
    return frozenset(w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOP)


def task_groups(records: dict) -> dict:
    """{file: group id}: demonstrations of one task, joined when their instructions share most of their words
    (Jaccard of content words at least 0.6), within one dataset. A paraphrase of an instruction joins its group; a
    different task does not."""
    groups, members = {}, []
    for f, d in sorted(records.items()):
        key, ds = task_key(d), d.get("dataset")
        for gid, (gds, gkey) in enumerate(members):
            if gds == ds and key and gkey and len(key & gkey) / len(key | gkey) >= 0.6:
                groups[f] = gid
                break
        else:
            groups[f] = len(members)
            members.append((ds, key))
    return groups


def _ratio_score(value, peers):
    """0 to 10 for a lower-is-better measure against its peers' median: the median or better is 10, twice the median
    is 5, and so on (10 x median / value). A rank would put half of any group below 5 even when every demonstration
    is fine; against the median, typical work scores full and only work slower or busier than typical loses points.
    None with too few peers."""
    if value is None or len(peers) < MIN_PEERS:
        return None
    median = statistics.median(peers)
    if value <= median or value <= 0:
        return 10.0
    return round(10 * median / value, 1)


def _outcome(f: dict):
    cap = OUTCOME_CAPS.get(f["outcome"])
    if cap is None:
        return None, {"outcome": f["outcome"]}
    value = 1.0 if f["outcome"] == "success" else min(cap, f["progress_reached"])
    return round(10 * value, 1), {"outcome": f["outcome"], "progress_reached": f["progress_reached"]}


def _cleanliness(f: dict):
    penalties = [{"mistake": m.get("issue"), "severity": m.get("severity"), "recovered": bool(m.get("recovered")),
                  "points": SEVERITY_POINTS.get(m.get("severity"), 1.5) * (0.5 if m.get("recovered") else 1.0)}
                 for m in f["mistakes"]]
    attempts = FAILED_ATTEMPT_POINTS * max(0, f["failed_attempts"] - len(f["mistakes"]))
    share = f["wasteful_s"] / f["active_s"] if f["active_s"] else 0.0
    score = max(0.0, 10 * (1 - share) - sum(p["points"] for p in penalties) - attempts)
    return round(score, 1), {"wasteful_share": round(share, 3), "penalties": penalties,
                             "unexplained_failed_attempts": attempts}


def subgoals(d: dict, f: dict) -> list:
    """The stretches of work that end in the episode's subgoals: from the start of the work (or the previous subgoal)
    to each key event tagged subgoal_complete, and a last stretch to the goal, or to the end of the work when the goal
    is never reached ("reached": False). The model records a subgoal only once it is achieved, so a subgoal's own
    outcome is always success and is not scored; how cleanly, steadily and quickly its stretch went is. A subgoal is a
    lasting change the goal depends on, so the failed attempts between two subgoals are attempts at the later one. Each
    carries its duration, wasteful share, hesitation, and the mistakes and failed attempts inside it."""
    events = sorted((k for k in d.get("key_events") or [] if isinstance(k, dict) and k.get("kind") == "subgoal_complete"
                     and _num(k.get("t_s")) is not None), key=lambda k: k["t_s"])
    if f["start_s"] is None or not events:
        return []
    steps = [e for e in d.get("event_labels") or [] if _num(e.get("t_s")) is not None and _num(e.get("end_s")) is not None]
    # an episode of several subtasks (human ego, sessions) has no single goal: each subtask is scored on its own, from
    # its start through its subgoals to its own end-state, and the time between subtasks belongs to none of them
    tasks = sorted((t for t in d.get("tasks") or [] if isinstance(t, dict) and _num(t.get("start_s")) is not None
                    and _num(t.get("end_s")) is not None), key=lambda t: t["start_s"])
    spans = ([(t["start_s"], _num(t.get("completed_at_s")) if _num(t.get("completed_at_s")) is not None else t["end_s"],
               t.get("task"), t.get("outcome"), t["end_s"]) for t in tasks] if tasks
             else [(f["start_s"], f["end_s"], (d.get("completion") or {}).get("success_predicate"), f["outcome"], None)])
    bounds = []
    for a, end, goal_label, goal_outcome, last in spans:
        for k in events:
            if k["t_s"] > a and (last is None or k["t_s"] <= last + 0.5):
                bounds.append((a, k["t_s"], k.get("label"), k.get("outcome"), False))
                a = k["t_s"]
        if end is not None and end > a + 0.5:
            bounds.append((a, end, goal_label, goal_outcome, True))
    out = []
    for a, b, label, outcome, goal in bounds:
        clip = lambda spans: [(max(a, x), min(b, y)) for x, y in spans if min(b, y) > max(a, x)]
        work = clip([(e["t_s"], e["end_s"]) for e in steps if e.get("contribution") in ("advancing", "wasteful")])
        waste = clip([(e["t_s"], e["end_s"]) for e in steps if e.get("contribution") == "wasteful"])
        active = _union(work)
        share = _union(waste) / active if active else 0.0
        hes = _union(_subtract([(a, b)], work)) / (b - a)
        ms = [m for m in f["mistakes"] if _num(m.get("t_s")) is not None and a <= m["t_s"] < b]
        failed = [k for k in d.get("key_events") or [] if isinstance(k, dict) and k.get("outcome") == "failure"
                  and _num(k.get("t_s")) is not None and a <= k["t_s"] < b]
        clean = max(0.0, 10 * (1 - share) - sum(SEVERITY_POINTS.get(m.get("severity"), 1.5)
                                               * (0.5 if m.get("recovered") else 1.0) for m in ms)
                    - FAILED_ATTEMPT_POINTS * max(0, len(failed) - len(ms)))
        out.append({"start_s": a, "end_s": b, "label": label, "outcome": outcome, "goal": goal,
                    "reached": outcome == "success",
                    "duration_s": round(b - a, 2),
                    "cleanliness": round(clean, 1), "steady": round(10 * max(0.0, 1 - 2 * hes), 1),
                    "wasteful_share": round(share, 3), "mistakes": [m.get("issue") for m in ms],
                    "failed_attempts": [{"t_s": k["t_s"], "label": k.get("label"), "kind": k.get("kind")}
                                        for k in sorted(failed, key=lambda k: k["t_s"])]})
    return out


def score_board(records: dict) -> dict:
    """{file: scores} for every episode of a board: each record's own measures, and the relative dimensions ranked
    within its task group among the episodes that reached the goal."""
    fs = {f: facts(d) for f, d in records.items()}
    groups = task_groups(records)
    by_group = {}
    for f, g in groups.items():
        by_group.setdefault(g, []).append(f)
    out = {}
    for f, d in records.items():
        x = fs[f]
        peers = [fs[p] for p in by_group[groups[f]] if p != f and fs[p]["outcome"] == "success"]
        outcome, o_in = _outcome(x)
        clean, c_in = _cleanliness(x)
        took = x["goal_s"] - x["start_s"] if x["goal_s"] is not None and x["start_s"] is not None else None
        peer_took = [p["goal_s"] - p["start_s"] for p in peers if p["goal_s"] is not None and p["start_s"] is not None]
        efficiency = _ratio_score(took, peer_took) if x["outcome"] == "success" else None
        hes = x["hesitation_s"] / x["window_s"] if x["window_s"] else None
        peer_hes = [p["hesitation_s"] / p["window_s"] for p in peers if p["window_s"]]
        per_progress = x["actions"] / x["progress_reached"] if x["progress_reached"] else None
        peer_pp = [p["actions"] / p["progress_reached"] for p in peers if p["progress_reached"]]
        # hesitation is absolute (half the work spent with every arm waiting scores 0); busyness is against peers
        steady = round(10 * max(0.0, 1 - 2 * hes), 1) if hes is not None else None
        parts = [s for s in (steady, _ratio_score(per_progress, peer_pp)) if s is not None]
        fluency = round(statistics.mean(parts), 1) if parts else None
        dims = {"cleanliness": clean, "efficiency": efficiency, "fluency": fluency}
        have = {k: v for k, v in dims.items() if v is not None}
        weight = sum(EXECUTION_WEIGHTS[k] for k in have)
        execution = round(sum(EXECUTION_WEIGHTS[k] * v for k, v in have.items()) / weight, 1) if weight else None
        subs = subgoals(d, x)
        # a subgoal's speed is judged against the same subgoal of peers that split the task the same way
        peer_subs = [subgoals(records[p], fs[p]) for p in by_group[groups[f]] if p != f and fs[p]["outcome"] == "success"]
        for i, st in enumerate(subs):
            same = [ps[i]["duration_s"] for ps in peer_subs if len(ps) == len(subs)]
            st["efficiency"] = _ratio_score(st["duration_s"], same)
            st["peers"] = len(same)
            st["peer_median_s"] = statistics.median(same) if len(same) >= MIN_PEERS else None
            parts = [v for v in (st["cleanliness"], st["steady"], st["efficiency"]) if v is not None]
            st["execution"] = round(statistics.mean(parts), 1)
        out[f] = {"version": VERSION, "execution": execution, "outcome": outcome, **dims, "relation": x["relation"],
                  "subgoals": subs,
                  "peers": len(peers), "group": groups[f],
                  "inputs": {**o_in, **c_in, "time_to_goal_s": took, "peer_median_time_to_goal_s":
                             statistics.median(peer_took) if peer_took else None,
                             "hesitation_share": round(hes, 3) if hes is not None else None,
                             "actions_per_progress": round(per_progress, 2) if per_progress else None}}
    return out
