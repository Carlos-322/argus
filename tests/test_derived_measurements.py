"""Derived evidence must separate measured contrasts from physical assumptions."""
import importlib

import numpy as np
import pytest


def module():
    return importlib.import_module('label.derived_measurements')


def regional(**extra):
    return {'regions': [{'name': 'branch A', 'indices': [0, 2]},
                        {'name': 'branch B', 'indices': [1, 3]}],
            'unit': 'ADC', 'provenance': 'publisher_candidate', 'source': 'signal:pad', **extra}


def test_sparse_regional_response_keeps_opposing_changes_and_source_semantics():
    values = np.full((5, 4), 100.)
    values[3] = [120, 90, 80, 70]
    result = module().regional_response(values, np.arange(5.), regional())
    a, b = result['traces']
    assert a['indices'] == [0, 2]
    assert a['signed_values'][3] == 0
    assert a['positive_values'][3] == 10
    assert a['negative_values'][3] == -10
    assert b['values'][3] == -20
    assert result['response_direction'] is None
    assert result['provenance'] == 'publisher_candidate'
    assert result['source'] == 'signal:pad'
    assert result['semantics']['regions'] == regional()['regions']
    assert a['unit'] == 'ADC'


def test_declared_direction_valid_mask_zero_sensitivity_and_missing_reference():
    values = np.array([[100, 7, 100, np.nan], [100, 7, 100, np.nan],
                       [100, 7, 100, np.nan], [0, 7, 80, 25], [90, 7, np.nan, 30.]])
    result = module().regional_response(values, np.arange(5.), regional(
        response_direction='down', valid_indices=[0, 2, 3], zero_exclusion_sensitivity=True))
    a, b = result['traces']
    assert a['values'][3] == 60
    assert a['zero_counts'][3] == 1
    assert a['sensitivity']['indices'] == [2]
    assert a['sensitivity']['values'][3] == 20
    assert a['valid_counts'] == [2, 2, 2, 2, 1]
    assert b['values'] == [None] * 5
    assert b['valid_counts'] == [0] * 5
    assert result['reference']['values'][3] is None


def test_regional_partial_reference_is_explicit_and_clock_is_checked():
    values = np.array([[np.nan], [10], [12], [15.]])
    sem = {'regions': [{'name': 'pad', 'indices': [0]}]}
    result = module().regional_response(values, [0, 1, 2, 3], sem)
    assert result['reference']['values'] == [11]
    assert result['reference']['valid_counts'] == [2]
    assert result['traces'][0]['values'] == [None, -1, 1, 4]
    with pytest.raises(ValueError, match='clock'):
        module().regional_response(values, [0, 1, 1, 3], sem)
    with pytest.raises(ValueError, match='units'):
        module().regional_response(np.zeros((4, 2)), np.arange(4.), {
            'regions': [{'name': 'mixed', 'indices': [0, 1]}], 'units': ['N', 'C']})


def test_pointset_shape_ignores_rigid_motion_but_measures_deformation_directly():
    base = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = base @ rotation + [4, 5, 6]
    a, b = base.copy(), base.copy()
    a[1, 0] += .5
    b[1, 0] -= .5
    result = module().pointset_shape(np.array([base, moved, a, b]), [0, 1, 2, 3], {
        'unit': 'm', 'provenance': 'recorded', 'source': 'signal:points'})
    assert result['first_rms'][1] == pytest.approx(0, abs=1e-12)
    assert result['first_rms'][2] > 0
    assert result['adjacent_rms'][3] > abs(result['first_rms'][3] - result['first_rms'][2]) + .01
    assert result['normalization']['scale'] > 0
    assert result['source'] == 'signal:points'
    assert result['provenance'] == 'recorded'
    assert result['curls'] == []


def test_pointset_gaps_repeats_declared_normalization_and_explicit_curl_chains():
    base = np.array([[0., 0.], [1., 0.], [2., 0.], [3., 0.]])
    bent = np.array([[0., 0.], [1., 0.], [1., 1.], [0., 1.]])
    gap = np.full_like(base, np.nan)
    result = module().pointset_shape(np.array([base, base, gap, bent, bent]), np.arange(5.), {
        'normalization_pair': [0, 3], 'chains': [{'name': 'recorded branch', 'indices': [0, 1, 2, 3]}]})
    assert result['normalization']['scale'] == 3
    assert result['repeated_previous'] == [False, True, False, False, True]
    assert result['adjacent_rms'][2:4] == [None, None]
    assert result['first_rms'][2] is None
    assert result['pair_valid_counts']['adjacent'][2:4] == [0, 0]
    assert len(result['stale_intervals']) == 2
    assert result['curls'][0]['name'] == 'recorded branch'
    assert result['curls'][0]['values'][0] == 0
    assert result['curls'][0]['values'][3] == pytest.approx(2/3)


def test_degenerate_pointsets_and_mixed_coordinates_do_not_create_pose_evidence():
    values = np.ones((3, 4, 3))
    result = module().pointset_shape(values, [0, 1, 2], {})
    assert result['first_rms'] == [None, None, None]
    assert result['normalization']['scale'] is None
    assert result['limitations']
    with pytest.raises(ValueError, match='units'):
        module().pointset_shape(values, [0, 1, 2], {'units': ['m', 'm', 'pixels']})
    with pytest.raises(ValueError, match='indices'):
        module().regional_response(np.zeros((4, 2)), np.arange(4), {
            'regions': [{'name': 'bad', 'indices': [2]}]})


def test_pointset_missing_subset_uses_shared_points_without_bridging_gap():
    base = np.array([[0., 0., 0.], [2., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    moved = base + [20, -10, 2]
    moved[3] = np.nan
    too_few = moved.copy()
    too_few[2] = np.nan
    result = module().pointset_shape(np.array([base, moved, too_few, base]), np.arange(4.), {})
    assert result['first_rms'][1] == pytest.approx(0, abs=1e-12)
    assert result['pair_valid_counts']['first'] == [4, 3, 2, 4]
    assert result['first_rms'][2] is None
    assert result['adjacent_rms'][2:4] == [None, None]


def test_declared_scale_is_fixed_so_growth_is_deformation_not_normalized_away():
    base = np.array([[0., 0.], [1., 0.], [0., 1.]])
    result = module().pointset_shape(np.array([base, base * 2]), [0, 1], {'normalization_pair': [0, 1]})
    assert result['normalization']['scale'] == 1
    assert result['first_rms'][1] == pytest.approx(2/3)


def test_complex_readings_are_rejected_instead_of_discarding_imaginary_parts():
    with pytest.raises(ValueError, match='real'):
        module().regional_response(np.array([[1+2j], [2+3j]]), [0, 1], {
            'regions': [{'name': 'pad', 'indices': [0]}]})


def test_missing_and_repeated_pose_coverage_and_strict_json_are_explicit():
    import json
    base = np.array([[0., 0.], [1., 0.], [0., 1.]])
    result = module().pointset_shape(np.array([base, base, base * np.nan]), [0, 1, 2], {})
    assert result['coverage'] == {'frames': 3, 'usable_first_comparisons': 2,
                                  'usable_adjacent_comparisons': 1, 'repeated_pairs': 1,
                                  'incomplete_frames': 1}
    json.dumps(result, allow_nan=False)


def test_pointset_invalid_components_cannot_create_shape_or_chain_changes():
    base = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
    changed = base.copy()
    changed[3] = 100
    result = module().pointset_shape(np.array([base, changed]), [0, 1], {
        'valid_indices': list(range(9)), 'chains': [{'name': 'branch', 'indices': [0, 1, 2, 3]}]})
    assert result['first_rms'][1] == pytest.approx(0, abs=1e-12)
    assert result['pair_valid_counts']['first'] == [3, 3]
    assert result['curls'][0]['values'] == [None, None]
