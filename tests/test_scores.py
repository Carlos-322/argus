"""Episode and subgoal quality scores (board/scores.py): measured from the labels, compared within a task group."""
from board import scores


def _record(prompt, goal_s, steps, subgoals=(), mistakes=(), outcome="success"):
    return {"dataset": "ds", "episode_prompt": prompt, "duration_s": steps[-1][1],
            "event_labels": [{"t_s": a, "end_s": b, "contribution": c, "progress": p} for a, b, c, p in steps],
            "key_events": [{"t_s": t, "kind": "subgoal_complete", "outcome": "success", "label": label}
                           for t, label in subgoals],
            "operator_mistakes": list(mistakes),
            "completion": {"task_completed": outcome, "completed_at_s": goal_s, "success_predicate": "done"}}


def _board(n_peers=6, peer_goal=10.0):
    peers = {f"p{i}.json": _record("open the left door then the right door", peer_goal,
                                   [(0, peer_goal / 2, "advancing", 0.5), (peer_goal / 2, peer_goal, "advancing", 1.0)],
                                   [(peer_goal / 2, "left door open")]) for i in range(n_peers)}
    return peers


def test_a_clean_demo_at_the_typical_time_scores_full_and_a_slow_one_loses_speed_only():
    recs = _board()
    recs["slow.json"] = _record("open the left door, then open the right door", 20.0,
                                [(0, 10, "advancing", 0.5), (10, 20, "advancing", 1.0)], [(10, "left door open")])
    s = scores.score_board(recs)
    assert s["p0.json"]["efficiency"] == 10.0 and s["p0.json"]["cleanliness"] == 10.0
    slow = s["slow.json"]
    assert slow["peers"] == 6 and slow["efficiency"] == 5.0          # twice the median time
    assert slow["cleanliness"] == 10.0 and slow["outcome"] == 10.0
    assert slow["inputs"]["peer_median_time_to_goal_s"] == 10.0


def test_mistakes_and_wasted_time_lower_cleanliness_and_a_recovered_mistake_costs_half():
    recs = _board()
    recs["messy.json"] = _record("open the left door then the right door", 10.0,
                                 [(0, 4, "advancing", 0.3), (4, 6, "wasteful", 0.3), (6, 10, "advancing", 1.0)],
                                 [(5, "left door open")],
                                 [{"issue": "dropped the handle", "severity": "medium", "recovered": True, "t_s": 4}])
    s = scores.score_board(recs)["messy.json"]
    assert s["inputs"]["wasteful_share"] == 0.2
    assert s["cleanliness"] == round(10 * 0.8 - 1.5 * 0.5, 1)


def test_too_few_peers_leaves_the_relative_scores_empty_never_guessed():
    recs = _board(n_peers=3)
    s = scores.score_board(recs)["p0.json"]
    assert s["efficiency"] is None and s["cleanliness"] == 10.0 and s["execution"] == 10.0


def test_subgoals_split_the_work_and_a_goal_never_reached_is_its_own_stretch():
    recs = _board()
    recs["fail.json"] = _record("open the left door then the right door", None,
                                [(0, 5, "advancing", 0.5), (5, 12, "advancing", 0.7)], [(5, "left door open")],
                                outcome="failure")
    s = scores.score_board(recs)
    ok = s["p0.json"]["subgoals"]
    assert [(st["start_s"], st["end_s"], st["goal"]) for st in ok] == [(0, 5.0, False), (5.0, 10.0, True)]
    assert ok[0]["peers"] == 5 and ok[0]["execution"] == 10.0
    failed = s["fail.json"]
    assert failed["subgoals"][-1]["goal"] and not failed["subgoals"][-1]["reached"]
    assert failed["outcome"] == 4.0 and failed["efficiency"] is None   # failure is capped and has no time to goal


def test_a_paraphrase_joins_its_task_group_and_another_task_does_not():
    recs = _board()
    recs["para.json"] = _record("open the left door, then open the right door", 10.0,
                                [(0, 10, "advancing", 1.0)])
    recs["other.json"] = _record("stack the red cube on the blue cube", 10.0, [(0, 10, "advancing", 1.0)])
    g = scores.task_groups(recs)
    assert g["para.json"] == g["p0.json"] and g["other.json"] != g["p0.json"]


def test_failed_attempts_belong_to_the_subgoal_they_precede_and_cost_only_when_no_mistake_explains_them():
    recs = _board()
    rec = _record("open the left door then the right door", None,
                  [(0, 5, "advancing", 0.5), (5, 12, "wasteful", 0.5)], [(5, "left door open")], outcome="failure")
    rec["key_events"] += [{"t_s": 3.0, "kind": "grasp", "outcome": "failure", "label": "handle slips"},
                          {"t_s": 8.0, "kind": "grasp", "outcome": "failure", "label": "right handle missed"},
                          {"t_s": 10.0, "kind": "pickup", "outcome": "success", "label": "handle held"}]
    rec["operator_mistakes"] = [{"issue": "slipped grasp", "severity": "low", "recovered": True, "t_s": 3.0}]
    recs["tries.json"] = rec
    first, last = scores.score_board(recs)["tries.json"]["subgoals"]
    assert [k["label"] for k in first["failed_attempts"]] == ["handle slips"]
    assert first["cleanliness"] == round(10.0 - 0.5 * 0.5, 1)         # the mistake explains the failed grasp
    assert [k["label"] for k in last["failed_attempts"]] == ["right handle missed"]
    assert last["goal"] and not last["reached"]
    assert last["cleanliness"] == 0.0                                 # all wasteful, then one unexplained failure
