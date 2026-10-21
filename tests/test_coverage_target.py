"""A manifest target turns the coverage strip into labelled against target; without one the strip is unchanged."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from board import serve

HERE = Path(__file__).parent
pytestmark_node = pytest.mark.skipif(not shutil.which('node'), reason='no node')

# The real coverage functions, with the page's globals stubbed: two datasets, one episode each.
SCRIPT = r'''
const {piece} = require(process.argv[1])();
const targets = JSON.parse(process.argv[2]);
const stub = `
const BOARD = {targets: ${JSON.stringify(targets)}};
const DS_SHORT = {};
const ALL_EPS = [{dataset: 'yam', duration_s: 3600 * 45.1}, {dataset: 'umi', duration_s: 1800}];
const railEps = () => ALL_EPS;
const presentDatasets = eps => [...new Set(eps.map(e => e.dataset))];
const datasetOf = e => e.dataset;
const esc = String, dsLabel = String;
const document = {body: {classList: {contains: () => false}}};
const requestAnimationFrame = f => f();
const coverageEl = {innerHTML: '', querySelectorAll: () => [], querySelector: () => null};
let currentDataset = 'yam';
`;
const page = require('fs').readFileSync(require('path').join(process.argv[3], 'board', 'index.html'), 'utf8');
const line = name => page.slice(page.indexOf(name), page.indexOf('\n', page.indexOf(name)) + 1);
const code = stub + line('const TARGETS =') + line('const coverageWas =')
  + ['fmtHours', 'fmtSpan', 'sumDur', 'fmtOf', 'coverageBar', 'renderCoverage'].map(piece).join('')
  + 'renderCoverage("yam", ALL_EPS.filter(e => e.dataset === "yam")); coverageEl.innerHTML';
console.log(JSON.stringify(eval(code)));
'''


def render(targets):
    p = subprocess.run(['node', '-e', SCRIPT, str(HERE / 'page_functions.js'), json.dumps(targets), str(HERE.parent)],
                       text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    return json.loads(p.stdout)


@pytestmark_node
def test_a_target_shows_labelled_of_target_with_a_bar():
    html = render({'yam': {'episodes': 2682, 'hours': 101.9}, 'umi': {'episodes': 10, 'hours': 2}})
    assert '1 of 2,682<span class="u"> eps</span> &middot; 45.1 of 101.9 h' in html
    assert '0.5 of 2.0 h' in html                       # each dataset in its own target's unit
    assert 'of 2,692 eps' in html and 'of 103.9 h' in html     # the totals add the targets
    assert html.count('role="progressbar"') == 3
    assert 'aria-valuenow="44"' in html                 # 45.1 of 101.9 hours


@pytestmark_node
def test_without_a_target_for_every_dataset_the_totals_are_unchanged():
    html = render({'yam': {'episodes': 2682, 'hours': 101.9}})
    assert '45.1 of 101.9 h' in html
    assert 'of 2,682 eps</small>' not in html and html.count('role="progressbar"') == 1
    assert '<small>episodes</small>' in html


@pytestmark_node
def test_no_targets_leave_the_strip_as_it_was():
    html = render({})
    assert 'progressbar' not in html and ' of ' not in html
    assert '1<span class="u"> ep</span> &middot; 45.1 h' in html


def test_targets_read_only_positive_numbers_from_the_manifest(tmp_path):
    (tmp_path / 'manifest.json').write_text(json.dumps({'datasets': [
        {'dataset': 'yam', 'target': {'episodes': 2682, 'hours': 101.9}},
        {'dataset': 'umi', 'target': {'episodes': 0, 'hours': 5}},
        {'dataset': 'ego', 'target': {'episodes': '9', 'hours': 5}},
        {'dataset': 'abc'},
        'not an entry']}))
    assert serve.targets(tmp_path) == {'yam': {'episodes': 2682, 'hours': 101.9}}
    assert serve.targets(tmp_path / 'missing') == {}
