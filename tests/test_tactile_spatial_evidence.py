"""Local contrasts must remain visible when regional means cancel them."""
import base64
import io

import numpy as np
import pytest
from PIL import Image

from label import tactile_arrays as ta


def spatial_case():
    times = np.arange(186) / 30
    raw = np.full((len(times), 256), 1000.)
    # Opposing patches share one region, so no regional mean identifies the event.
    for row, time in enumerate(times):
        if 1.8 <= time <= 3.2:
            raw[row, [10, 26]] = 600
            raw[row, [90, 106]] = 1400
    raw[103, :8] = 50
    positions = [[float(i % 16), float(i // 16)] for i in range(256)]
    positions[10], positions[26] = [30., 2.], [31., 2.]
    positions[90], positions[106] = [30., 8.], [31., 8.]
    semantics = {'version': 1, 'positions': positions, 'valid_indices': [10, 26, 90, 106],
                 'regions': [{'name': 'branch A', 'indices': [10, 26, 90, 106]}],
                 'provenance': {'authority': 'publisher_candidate'},
                 'limitations': ['Recording hardware match is unverified.']}
    return raw, times, semantics


def test_native_spatial_snapshots_expose_opposing_patches_between_rgb_instants():
    raw, times, semantics = spatial_case()
    assert np.all(raw[54:97, [10, 26, 90, 106]].mean(axis=1) == 1000)
    array = ta.comparison(raw, [16, 16], times, [0, 103, 185], 16384, semantics=semantics)
    frames = array['frames']
    assert max(np.diff([frame['time_s'] for frame in frames])) <= .25
    event = [frame for frame in frames if 1.8 <= frame['time_s'] <= 2.3]
    assert len(event) >= 2
    assert all(frame['values'][10] == 600 and frame['values'][90] == 1400 for frame in event)
    assert any(frame['time_s'] == times[103] for frame in frames)
    assert array['physical_layout']['authority'] == 'publisher_candidate'
    assert array['physical_layout']['positions'] == [
        {'index': i, 'x': semantics['positions'][i][0], 'y': semantics['positions'][i][1], 'region': 'branch A'}
        for i in [10, 26, 90, 106]]
    assert len(frames) <= ta.MAX_FRAMES
    assert (len(frames) + 1) * 256 <= 16384


def test_positioned_maps_show_source_cell_ids_and_keep_candidate_qualification():
    raw, times, semantics = spatial_case()
    array = ta.comparison(raw, [16, 16], times, [0, 103, 185], 16384, semantics=semantics)
    parts = ta.render({'id': 'signal:pad', 'array_evidence': array})
    captions = '\n'.join(part['text'] for part in parts if part['type'] == 'text')
    assert 'publisher_candidate' in captions
    assert 'cell IDs' in captions
    assert 'branch A' in captions
    assert 'unverified' in captions
    images = [Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',')[1])))
              for part in parts if part['type'] == 'image_url']
    assert images and max(image.height for image in images) <= 1600
    assert all(image.width <= 1600 for image in images)
    # Physical coordinates put the opposing patches vertically apart, despite
    # their unrelated source-grid rows and columns.
    event = next(frame for frame in array['frames'] if 1.8 <= frame['time_s'] <= 2.3)
    single = {**array, 'frames': [event]}
    part = next(part for part in ta.render({'id': 'signal:pad', 'array_evidence': single})
                if part['type'] == 'image_url')
    image = Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',')[1])))
    lower_code, higher_code = image.getpixel((30, 404)), image.getpixel((30, 664))
    assert lower_code[0] > lower_code[2] + 30
    assert higher_code[2] > higher_code[0] + 30


def test_unattributed_or_incomplete_positions_do_not_replace_source_grid():
    raw, times, semantics = spatial_case()
    for descriptor in ({'version': 1, 'positions': semantics['positions']},
                       {**semantics, 'positions': semantics['positions'][:3]},
                       {**semantics, 'positions': [[0., 0.]] * 256}):
        array = ta.comparison(raw, [16, 16], times, [0, 103, 185], 8192, semantics=descriptor)
        assert 'physical_layout' not in array
        assert 'Source array row and column' in array['coordinates']


@pytest.mark.parametrize(('direction', 'word'), [('down', 'Lower'), ('up', 'Higher')])
def test_map_caption_preserves_candidate_response_direction_and_its_limits(direction, word):
    raw, times, semantics = spatial_case()
    semantics['response_direction'] = direction
    array = ta.comparison(raw, [16, 16], times, [], 8192, semantics=semantics)
    array['frames'] = array['frames'][:1]
    caption = ta.render({'id': 'signal:pad', 'array_evidence': array})[0]['text']
    assert word + ' raw values indicate stronger sensor response' in caption
    assert 'publisher_candidate' in caption and 'unverified' in caption
    assert 'not a physical force calibration' in caption
    assert 'not calibrated force or a known physical loading direction' not in caption


def test_fixed_physical_scale_uses_complete_valid_source_period_and_declares_clipping():
    raw, times, semantics = spatial_case()
    # One valid spike and a larger erased-cell endpoint must not wash out the
    # persistent opposing patches. Neither changes the underlying readings.
    raw[75, 10] = -10000
    raw[100, 0] = -100000
    before = raw.copy()
    array = ta.comparison(raw, [16, 16], times, [0, 103, 185], 16384, semantics=semantics)
    changes = raw[:, semantics['valid_indices']] - np.median(raw[:3], axis=0)[semantics['valid_indices']]
    scale = array['physical_scale']
    assert scale['maximum'] == float(np.percentile(abs(changes), 99.5)) == 400
    assert scale['valid_cells'] == 4
    assert scale['finite_values'] == changes.size
    assert scale['clipped_values'] == 1
    assert np.array_equal(raw, before)
    for frame in array['frames']:
        row = int(np.argmin(abs(times - frame['time_s'])))
        assert frame['values'] == before[row].tolist()
    assert array['signed_change_range'][1] == 101000
    assert next(frame for frame in array['frames'] if frame['time_s'] == times[100])['values'][0] == -100000
    parts = ta.render({'id': 'signal:pad', 'array_evidence': array})
    captions = '\n'.join(part['text'] for part in parts if part['type'] == 'text')
    assert '99.5' in captions and 'clipped' in captions
    assert 'complete' in captions and 'valid' in captions


def test_large_sources_keep_a_declared_bounded_scale_fallback(monkeypatch):
    raw, times, semantics = spatial_case()
    monkeypatch.setattr(ta, 'MAX_SCALE_VALUES', raw.size - 1)
    array = ta.comparison(raw, [16, 16], times, [0, 103, 185], 16384, semantics=semantics)
    assert array['physical_scale']['maximum'] == array['signed_change_range'][1]
    assert 'source-value budget' in array['physical_scale']['method']
    assert 'percentile' not in array['physical_scale']


def test_later_raw_cells_remain_visible_when_the_initial_reference_is_missing():
    raw = np.tile(np.arange(256.) + 100, (10, 1))
    raw[:3] = np.nan
    semantics = {'version': 1, 'positions': [[i % 16, i // 16] for i in range(256)],
                 'provenance': {'authority': 'recorded'}}
    array = ta.comparison(raw, [16, 16], np.arange(10) / 30, [], 8192, semantics=semantics)
    assert array['reference']['values'] == [None] * 256
    frame = next(frame for frame in array['frames'] if frame['time_s'] == .1)
    parts = ta.render({'id': 'signal:pad', 'array_evidence': {**array, 'frames': [frame]}})
    captions = '\n'.join(part['text'] for part in parts if part['type'] == 'text')
    assert 'Raw scale 100 to 355' in captions
    assert 'missing reference' in captions
    part = next(part for part in parts if part['type'] == 'image_url')
    image = Image.open(io.BytesIO(base64.b64decode(part['image_url']['url'].split(',')[1]))).convert('RGB')
    # The last cell is at the lower right of both physical maps. Its later raw
    # value is present even though no reference change can be computed.
    observed = image.getpixel((290, 304))
    unavailable_change = image.getpixel((290, 664))
    assert observed[0] > observed[2] + 60
    assert max(observed) - min(observed) > 80
    assert unavailable_change[2] > unavailable_change[1] + 15
    missing_part = next(part for part in ta.render({'id': 'signal:pad', 'array_evidence': {
        **array, 'frames': [array['frames'][0]]}}) if part['type'] == 'image_url')
    missing_image = Image.open(io.BytesIO(base64.b64decode(missing_part['image_url']['url'].split(',')[1]))).convert('RGB')
    missing_raw = missing_image.getpixel((290, 304))
    assert max(missing_raw) - min(missing_raw) < 15


def test_missing_samples_do_not_inflate_the_number_of_clipped_display_values():
    raw = np.arange(80, dtype=float).reshape(20, 4)
    raw[:3] = 0
    raw[3, 0] = np.nan
    semantics = {'version': 1, 'positions': [[0, 0], [1, 0], [0, 1], [1, 1]],
                 'provenance': {'authority': 'recorded'}}
    array = ta.comparison(raw, [2, 2], np.arange(20), [], 1024, semantics=semantics)
    changes = abs(raw - np.median(raw[:3], axis=0))
    expected = int(np.count_nonzero(changes > np.nanpercentile(changes, 99.5)))
    assert expected == 1
    assert array['physical_scale']['clipped_values'] == expected
