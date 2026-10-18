"""The recording's contacts checked against what the model saw at them, after labelling.

The touch signals time each contact exactly (label/contacts.py). The model is shown frames around each contact's begin
and end (label/episode.py contact_image) and says, per contact, whether the frames show the hand touching, the first
frame of the begin strip that shows touch and the last of the end strip, and which hand; and it lists the moments a
hand clearly takes hold of something that no contact covers. The two sources are compared here:

  clock_offset       where touch first shows in the frames against where the signal says it begins (and the same at
                     the end), over every contact whose strip brackets it: the median is how far the touch sensor's
                     clock runs from the cameras'. Reported when at least MIN_CONTACTS contacts agree on it and it is
                     more than OFFSET_FRAMES camera frames
  touch_not_seen     contacts the signal records but the frames show no touch at (a sensor that fires on its own, a
                     contact on the other hand, a clock far off)
  hand_mismatch      contacts the model sees on the other hand than the signal's name says; when every contact with a
                     hand disagrees, the gloves look swapped
  contact_missing    moments the model sees a hand take hold of or press something that no recorded contact covers (a
                     sensor that missed it, or one that was off)

A contact with an alignment assumption (its "aligned_by", label/contacts.py mark_aligned) has times that are the
placement's, not measured contact times. It never counts toward clock_offset or touch_not_seen. The result groups
such contacts by their particular placement assumption.

The result is context-free numbers and sentences, written by `python -m board build` into the episode's
dataset_checks["contact_checks"]. Like checks/sensors.py they are notes, not counted issues, until the datasets they
fire on are checked on the frames.
"""
from __future__ import annotations

import numpy as np

from prepare.signal_alignment import (PLACEMENT_FIELDS, placement_text, recorder_arrival,
                                      RECORDER_ARRIVAL, RECORDER_ARRIVAL_LIMITATION)

MIN_CONTACTS = 3
OFFSET_FRAMES = 1.0


def _between(strip: list, f, first: bool) -> float | None:
    """The time touch begins (first) or ends between two frames of a strip, from the frame number the model gave: the
    middle of the gap before the first frame that shows touch, or after the last one. None when the strip does not
    bracket it (touch already showing at frame 1, or still at frame 5) or no frame was given."""
    if not isinstance(f, int) or not strip or not 1 <= f <= len(strip):
        return None
    i = f - 1
    pair = (i - 1, i) if first else (i, i + 1)
    if not 0 <= pair[0] < pair[1] < len(strip):
        return None
    a, b = strip[pair[0]], strip[pair[1]]
    # a strip time that is not a finite number places nothing: a NaN offset would make every median after it NaN
    if not all(isinstance(x, (int, float, np.number)) and not isinstance(x, bool) and np.isfinite(x) for x in (a, b)):
        return None
    return (a + b) / 2


def check(labels: dict, contacts: list[dict], strips: dict, fps: float, instrumented_hands=None,
          *, timing_context=None) -> dict | None:
    """dataset_checks["contact_checks"] for one episode: the model's contacts (labels["contacts"], by id), the
    recording's (contacts), the strip times each contact picture was built with ({id: {"begin": [...], "end": [...]}})
    and the camera's frame rate. instrumented_hands names the hands with touch recordings, including quiet sensors.
    Legacy callers without coverage can check missing contacts only on recorded-contact sides. Raw labels are kept.
    None when no contacts, missing observations or actual instrumented coverage exist."""
    missing_observations = [(x, x['hand'].strip().lower() if isinstance(x.get('hand'), str) else None)
                            for x in (labels.get('contacts_missing') or []) if isinstance(x, dict)]
    if not contacts and not instrumented_hands and not missing_observations:
        return None
    seen = {c.get("id"): c for c in (labels.get("contacts") or []) if isinstance(c, dict) and c.get("review_status") != "rejected"}
    by_id = {c["id"]: c for c in contacts if c.get("review_status") != "rejected"}
    covered = ({c.get('hand') for c in by_id.values()} if instrumented_hands is None else set(instrumented_hands))
    covered &= {'left', 'right'}
    notes, offsets = [], []
    # a contact with an alignment assumption (label/contacts.py mark_aligned) has the placement's times, so where the frames
    # show its touch says how far off the placement is, not that the sensor's clock is, nor that the sensor fired on
    # its own: it is left out of clock_offset and touch_not_seen and named by its placement
    placed = [cid for cid, rec in by_id.items() if rec.get("aligned_by")]
    arrival = {cid for cid, rec in by_id.items() if rec.get('aligned_by') == RECORDER_ARRIVAL}
    arrival_clock = recorder_arrival(timing_context or {})
    if arrival_clock:
        placed = list(by_id)
        arrival |= {cid for cid, rec in by_id.items() if not rec.get('aligned_by')}
    physically_unverified = set(by_id) if arrival_clock else arrival
    for cid, rec in by_id.items():
        m = seen.get(cid)
        if not m or cid in placed:
            continue
        st = strips.get(cid) or {}
        b = _between(st.get("begin") or [], m.get("first_touch_frame"), True)
        e = _between(st.get("end") or [], m.get("last_touch_frame"), False)
        if b is not None:
            offsets.append(("begin", cid, b - rec["start_s"]))
        if e is not None:
            offsets.append(("end", cid, e - rec["end_s"]))
    shown = [cid for cid in by_id if cid in seen]
    # a frame rate that is missing, zero or not a finite number says nothing; the cameras' usual 30 is taken instead
    rate = float(fps) if isinstance(fps, (int, float, np.number)) and np.isfinite(fps) and fps > 0 else 30.0
    frame_ms = 1000.0 / rate
    out = {"contacts": len(by_id), "checked": len(shown), "notes": []}
    if arrival_clock or arrival:
        out.update(timing_basis='mcap_log_time', timing_limitation=RECORDER_ARRIVAL_LIMITATION)
    if arrival_clock or arrival and len(arrival) == len(by_id):
        out['not_assessed'] = {name: RECORDER_ARRIVAL_LIMITATION for name in
                               ('clock_offset', 'touch_not_seen', 'hand_mismatch', 'contact_missing')}
    for cid in placed:
        by = RECORDER_ARRIVAL if cid in arrival else by_id[cid]['aligned_by']
        field = PLACEMENT_FIELDS.get(by, "placed_with_unspecified_alignment")
        out.setdefault(field, []).append(cid)
    if offsets:
        ms = np.array([o for _, _, o in offsets]) * 1000.0
        out["offset_ms"] = {"median": round(float(np.median(ms)), 1), "spread": round(float(np.std(ms)), 1),
                            "n": len(ms)}
        if len({cid for _, cid, _ in offsets}) >= MIN_CONTACTS and abs(float(np.median(ms))) > OFFSET_FRAMES * frame_ms:
            lead = "after" if np.median(ms) > 0 else "before"
            notes.append({"check": "clock_offset", "evidence": (
                f"the frames show touch begin and end about {abs(float(np.median(ms))):.0f} ms {lead} the touch "
                f"signal says (median of {len(ms)} measurements over {len({c for _, c, _ in offsets})} contacts, "
                f"spread {float(np.std(ms)):.0f} ms), more than a camera frame of {frame_ms:.0f} ms")})
    timed = [cid for cid in shown if cid not in placed]
    not_seen = [cid for cid in timed if str(seen[cid].get("touch_seen")).lower() == "no"]
    if not_seen:
        notes.append({"check": "touch_not_seen", "evidence": (
            f"{len(not_seen)} of the {len(timed)} contacts checked show no touch in the frames "
            f"({', '.join(not_seen)})")})
    pairs = [(cid, by_id[cid].get("hand"), str(seen[cid].get("hand") or "").lower())
             for cid in shown if cid not in physically_unverified]
    known = [(cid, h, m) for cid, h, m in pairs if h in ("left", "right") and m in ("left", "right")]
    wrong = [(cid, h, m) for cid, h, m in known if h != m]
    if wrong:
        swapped = len(wrong) == len(known) and len(known) >= 2
        notes.append({"check": "hand_mismatch", "evidence": (
            ("every contact with a hand is seen on the other hand than its signal's name says "
             f"({', '.join(c for c, _, _ in wrong)}): the left and right sensors look swapped") if swapped else
            f"{len(wrong)} of {len(known)} contacts are seen on the other hand than the signal's name says "
            f"({', '.join(f'{c} recorded {h}, seen {m}' for c, h, m in wrong)})")})
    arrival_hands = {by_id[cid].get('hand') for cid in arrival}
    verified_hands = {rec.get('hand') for cid, rec in by_id.items() if cid not in arrival}
    missing = [x for x, hand in missing_observations if not arrival_clock
               and hand not in arrival_hands - verified_hands and x.get('review_status') != 'rejected'
               and not x.get('withheld_from_final') and hand in covered]
    if missing:
        notes.append({"check": "contact_missing", "evidence": (
            f"{len(missing)} moment{'s' if len(missing) != 1 else ''} where a hand takes hold of or presses something "
            "with no recorded contact: " + "; ".join(
                f"{float(x.get('t_s') or 0):.1f} s, {x.get('hand') or 'a hand'}, {x.get('object') or 'an object'}"
                for x in missing[:6])
            + ("; the contacts " + " or ".join(dict.fromkeys(
                placement_text(by_id[cid]["aligned_by"]) for cid in placed))
                + " can miss a grasp by their placement alone" if placed else ""))})
    out["notes"] = notes
    return out
