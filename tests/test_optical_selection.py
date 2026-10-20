"""Keep clock context and quiet comparisons when native captures add detail."""
from label import optical_motion as om
from tests.test_optical_motion import pad


def _pairs(motion):
    return {(row['from_s'], row['to_s']): row for row in motion['pairs']}


def test_native_captures_preserve_anchor_comparisons_and_their_measurements():
    anchors = {10 * i: pad(2 * (i % 2)) for i in range(40)}
    times = {key: key / 10 for key in anchors}
    baseline = om.measure(anchors, times, anchor_frames=anchors)
    frames = dict(anchors)
    expanded_times = dict(times)
    for key in range(1, 390, 17):
        if key not in frames:
            frames[key] = pad(.5)
            expanded_times[key] = key / 10
    expanded = om.measure(frames, expanded_times, anchor_frames=anchors)
    before, after = _pairs(baseline), _pairs(expanded)
    assert len(before) == len(anchors) - 1
    assert before.keys() <= after.keys()
    assert all(after[key] == row for key, row in before.items())
    assert any(row['selection'] == 'native_detail' for row in after.values())
    assert len(after) <= om.MAX_COMPARISONS + om.MAX_NATIVE_COMPARISONS


def test_small_valid_comparison_survives_activity_elsewhere_in_its_time_bucket():
    frames = {i: pad(2 * (i % 2)) for i in range(40)}
    frames[24] = frames[23]
    motion = om.measure(frames, {i: float(i) for i in frames}, anchor_frames=frames)
    quiet = _pairs(motion)[23., 24.]
    assert quiet['selection'] == 'anchor_context'
    assert quiet['p95_px'] < .02
    assert quiet['correspondence']['ambiguous'] is False
    assert any(row['p95_px'] > 1 for row in motion['pairs'])


def test_context_selection_uses_elapsed_time_not_capture_density():
    # Dense early captures must not crowd the later context out of the overview.
    times = {i: float(i) / 100 for i in range(150)}
    times.update({150 + i: float(i) for i in range(2, 32)})
    frames = {key: pad(2 * (key % 2)) for key in times}
    motion = om.measure(frames, times, anchor_frames=frames)
    context = [row for row in motion['pairs'] if row['selection'] == 'anchor_context']
    starts = [row['from_s'] for row in context]
    assert sum(start >= 1 for start in starts) >= 6
    assert max(starts) == 30.
    assert len(motion['pairs']) == om.MAX_COMPARISONS
    assert 'anchor comparisons omitted' in motion['coverage']


def test_ambiguous_context_is_a_quality_gap_and_cannot_be_quiet(monkeypatch):
    original = om._correspondence
    def reject_one(row, first, second):
        result = original(row, first, second)
        if row.get('from_s') == 0:
            return {**result, 'p95_px': None, 'quality_gap': 'ambiguous test pattern',
                    'correspondence': {'ambiguous': True}}
        return result
    monkeypatch.setattr(om, '_correspondence', reject_one)
    frames = {i: pad(2 * (i % 2)) for i in range(12)}
    motion = om.measure(frames, {i: float(i) for i in frames}, anchor_frames=frames)
    gap = _pairs(motion)[0., 1.]
    assert gap['selection'] == 'anchor_context'
    assert gap['p95_px'] is None
    assert gap['quality_gap']
    assert all(row['p95_px'] is not None for row in motion['pairs']
               if row['selection'] == 'quiet_context')


def test_native_capture_before_first_anchor_does_not_change_anchor_reference():
    anchors = {i: pad(2 * (i % 2)) for i in range(1, 8)}
    times = {i: float(i) for i in anchors}
    before = om.measure(anchors, times, anchor_frames=anchors)
    after = om.measure({0: pad(), **anchors}, {0: 0., **times}, anchor_frames=anchors)
    assert before['reference_s'] == after['reference_s'] == 1.
    assert all(_pairs(after)[key] == value for key, value in _pairs(before).items())


def test_repeated_pattern_translation_stays_unavailable_beside_valid_stability():
    frames = {0: pad(), 1: pad(20), 2: pad(), 3: pad()}
    motion = om.measure(frames, {i: float(i) for i in frames}, anchor_frames=frames)
    pairs = _pairs(motion)
    assert pairs[0., 1.]['p95_px'] is None
    assert pairs[1., 2.]['p95_px'] is None
    assert pairs[2., 3.]['p95_px'] < .02


def test_complete_anchor_sequence_keeps_repeated_quiet_and_modest_changes():
    shifts = [0, 2, 2.5, 2.5, 0, 0] * 12
    anchors = {10 * i: pad(shift) for i, shift in enumerate(shifts)}
    frames = {**anchors, **{10 * i + 1: pad(1) for i in range(0, len(shifts) - 1, 7)}}
    motion = om.measure(frames, {key: key / 10 for key in frames}, anchor_frames=anchors)
    pairs = _pairs(motion)
    assert all((float(i), float(i + 1)) in pairs for i in range(len(shifts) - 1))
    for start in range(0, len(shifts) - 1, 6):
        assert pairs[float(start), float(start + 1)]['p95_px'] > 1
        assert .2 < pairs[float(start + 1), float(start + 2)]['p95_px'] < 1
        assert pairs[float(start + 2), float(start + 3)]['p95_px'] < .02
