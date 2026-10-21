"""An open page follows a board that is rebuilt while a run is in progress, without a reload."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from board import serve

HERE = Path(__file__).parent


def test_the_board_version_changes_when_the_board_is_rebuilt(tmp_path):
    (tmp_path / 'BUILT.json').write_text('{"counts": {}}')
    (tmp_path / 'manifest.json').write_text('{"datasets": []}')
    first = serve.board_version(tmp_path)
    assert first == serve.board_version(tmp_path)          # two stats, the same answer while nothing changes
    os.utime(tmp_path / 'BUILT.json', ns=(1, 1))
    assert serve.board_version(tmp_path) != first
    second = serve.board_version(tmp_path)
    (tmp_path / 'manifest.json').write_text('{"datasets": [{"dataset": "a", "target": {"episodes": 9, "hours": 1}}]}')
    assert serve.board_version(tmp_path) != second
    assert serve.board_version(tmp_path / 'missing')        # a board with no build yet still has a version


# The real liveCheck and liveVersion, with the page's globals stubbed: each call answers from a script of versions.
SCRIPT = r'''
const {piece} = require(process.argv[1])();
const steps = JSON.parse(process.argv[2]);
const page = require('fs').readFileSync(require('path').join(process.argv[3], 'board', 'index.html'), 'utf8');
const line = name => page.slice(page.indexOf(name), page.indexOf('\n', page.indexOf(name)) + 1);
const run = new Function('steps', `
  const STATIC = false, BY = null;
  let TARGETS = {}, ALL_EPS = [{file: 'a.json'}], currentDataset = 'ds', _activeFile = 'a.json';
  const document = {hidden: false, body: {classList: {contains: () => false}}};
  const epListEl = {scrollTop: 340};
  const issueFilterEl = {classList: {contains: () => false}};
  const calls = [];
  let k = 0;
  const okJson = async r => r;
  const fetch = async url => ({version: steps[Math.min(k++, steps.length - 1)].v, targets: {ds: {episodes: 9, hours: 2}}});
  const loadAllEpisodes = async () => [{file: 'a.json'}, {file: 'b.json'}];
  const buildIssueFilter = ds => calls.push(['filter', ds]);
  const renderRail = (ds, keep, fromSearch) => { calls.push(['rail', ds, keep, fromSearch]); epListEl.scrollTop = 0; };
  ${line('let _liveVersion')}
  ${piece('liveVersion')}${piece('liveCheck')}
  return (async () => {
    await liveVersion(true);
    const out = [];
    for (const s of steps.slice(1)) {
      document.hidden = !!s.hidden;
      calls.length = 0;
      await liveCheck();
      out.push({calls: calls.slice(), n: ALL_EPS.length, scroll: epListEl.scrollTop, targets: TARGETS});
    }
    return out;
  })();`);
run(steps).then(r => console.log(JSON.stringify(r)));
'''


@pytest.mark.skipif(not shutil.which('node'), reason='no node')
def test_a_rebuild_reloads_the_list_in_place_and_keeps_the_open_episode():
    steps = [{'v': 'b1'}, {'v': 'b1'}, {'v': 'b2', 'hidden': True}, {'v': 'b2'}, {'v': 'b2'}]
    p = subprocess.run(['node', '-e', SCRIPT, str(HERE / 'page_functions.js'), json.dumps(steps), str(HERE.parent)],
                       text=True, capture_output=True)
    assert p.returncode == 0, p.stderr
    same, hidden, rebuilt, after = json.loads(p.stdout)
    assert same['calls'] == [] and same['n'] == 1               # the same build: nothing is fetched or drawn
    assert hidden['calls'] == [] and hidden['n'] == 1           # a hidden tab does not ask
    assert rebuilt['calls'] == [['filter', 'ds'], ['rail', 'ds', 'a.json', True]]
    assert rebuilt['n'] == 2 and rebuilt['scroll'] == 340       # new episodes, the open one and the scroll kept
    assert rebuilt['targets'] == {'ds': {'episodes': 9, 'hours': 2}}
    assert after['calls'] == []                                 # and once seen, that build is not drawn again
