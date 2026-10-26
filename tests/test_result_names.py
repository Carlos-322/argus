"""Repeated sample labels must stay distinguishable in the local result picker."""
import importlib.util
import json
import os
from pathlib import Path


def viewer():
    spec = importlib.util.spec_from_file_location('view_results', Path(__file__).resolve().parents[1] / 'local-dev/view-results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def job(root, stamp, sample='001', recording='20260923_155221_765848', spec='heating-v2-complete'):
    path = root / f'heating_v2_api_{stamp}' / f'{sample}_{recording}'
    path.mkdir(parents=True)
    (path / 'annotation_policy.json').write_text(json.dumps({f'episode_{sample}_{recording}': {'annotation_spec': {'id': spec}}}))
    (path / 'manifest.json').write_text('{}')
    return path


def test_rebuild_and_list_order_do_not_change_annotation_number(tmp_path):
    old = job(tmp_path, '20261021_160505_752770', spec='heating-v2')
    new = job(tmp_path, '20261021_172949_177724')
    os.utime(old / 'manifest.json', (2000000000, 2000000000))
    titles = viewer().result_titles([new, old])
    assert titles[old] == '加热 001｜第 1 次标注｜2026-10-21 16:05:05｜旧规则'
    assert titles[new] == '加热 001｜第 2 次标注｜2026-10-21 17:29:49｜完整六段'
    assert viewer().result_titles([old, new]) == titles


def test_recordings_are_numbered_independently_and_old_names_survive(tmp_path):
    first = job(tmp_path, '20261021_160505_752770')
    third = job(tmp_path, '20261021_164327_760466', '003', '20260923_155452_541868')
    cooking = tmp_path / 'realman_cooking_20261009_180214_deepseek_live'
    titles = viewer().result_titles([first, third, cooking])
    assert '第 1 次标注' in titles[first] and '第 1 次标注' in titles[third]
    assert titles[cooking] == '炒菜任务'


def test_missing_policy_still_has_unique_names_and_printed_titles(tmp_path, capsys):
    one = job(tmp_path, '20261021_160505_752770')
    two = job(tmp_path, '20261021_160505_752771')
    (two / 'annotation_policy.json').write_text('incomplete')
    module = viewer()
    module.ROOT = tmp_path
    module.show_results([two, one])
    output = capsys.readouterr().out
    assert '第 1 次标注' in output and '第 2 次标注' in output
    assert module.result_titles([two, one])[two].endswith('2026-10-21 16:05:05')


def test_new_rule_title_remains_distinct_from_old_results(tmp_path):
    old = job(tmp_path, '20261021_160505_752770')
    new = job(tmp_path, '20261026_160505_752770', spec='heating-review-20261026')
    titles = viewer().result_titles([new, old])
    assert titles[old].endswith('完整六段')
    assert titles[new].endswith('10月26日分段规范')
    assert '第 2 次标注' in titles[new]
