"""Evidence selection sees every primary scene grid, including late action phases."""
import io

from PIL import Image

from label import evidence_access as ea, frames, grids
from tests.test_evidence_access import access


def scene_content():
    stream = io.BytesIO()
    Image.new('RGB', (16, 16), 'white').save(stream, 'JPEG')
    times = [i * 1.5 for i in range(45)] + [66.3]
    steps = [(time, [('scene', stream.getvalue())]) for time in times]
    content, _, _ = frames.build_content('', '', steps, ['scene'], 6, 'high', 80, 24, 30)
    image = next(part for part in content if part['type'] == 'image_url')
    for i in range(4):
        content.extend([{'type': 'text', 'text': f'=== detail view {i} ==='}, image.copy()])
    return content


def test_selection_preserves_all_primary_scene_grids_and_late_phase(tmp_path):
    content = scene_content()
    original = grids.receipt(content, {})
    assert len(original['grid_images']) == 8
    selected = access(tmp_path).selection_content(content)
    receipt = grids.receipt(selected, {})
    assert receipt['grid_images'] == original['grid_images']
    assert any(row['t0_s'] <= 39 <= row['t1_s'] for row in receipt['grid_images'])


def test_complete_scene_selection_still_obeys_existing_image_limit(tmp_path):
    record, _ = ea.discover(access(tmp_path), scene_content(),
        lambda _: (_ for _ in ()).throw(AssertionError('selection image cap was bypassed')),
        tmp_path / 'inspection.json', reserve_cost=.01, image_limit=7)
    assert record['status'] == 'incomplete'
    assert record['rounds'] == []
    assert any('image count or byte budget' in note for note in record['limitations'])
