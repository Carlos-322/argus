"""Unpaired table depth replays original pixels and source regions without RGB registration."""
import io
import json
import shutil
import subprocess
from pathlib import Path

import av
import numpy as np
import pandas as pd
import pytest
from PIL import Image

from board import clips, sensors, serve, static
from label import depth, episode, evidence_access
from prepare import formats
from tests.test_label import _packed_mp4


def table_depth(tmp_path):
    home = tmp_path / 'episode_depth'
    home.mkdir()
    rgb = home / 'rgb.mp4'
    _packed_mp4(rgb, 4)
    cells = []
    for value in (1000, 2000, None, 3000):
        if value is None:
            cells.append(None)
            continue
        buffer = io.BytesIO()
        Image.fromarray(np.full((48, 64), value, np.uint16)).save(buffer, format='PNG')
        cells.append({'bytes': buffer.getvalue()})
    context = {'episode_id': home.name, 'fps': 10, 'n_state_frames': 4, 'state_kind': 'none',
               'cameras': {'exo': {'name': 'scene'}}, 'profile': 'ego_head'}
    root = {'fps': 10, 'image_depths': ['observation.depth'], 'features': {
        'observation.depth': {'dtype': 'image', 'sensor_type': 'depth', 'units': 'mm'},
        'timestamp': {'units': 's'}}}
    table = pd.DataFrame({'observation.depth': cells, 'timestamp': np.arange(4) / 10, 'frame_index': np.arange(4)})
    formats.lerobot_image_depth(home, context, root, table, formats.Signals(), image_anchor=True,
                                anchor_times=np.arange(4) / 10)
    (home / 'context.json').write_text(json.dumps(context))
    (home / 'sources.json').write_text(json.dumps({'exo': {'packed': str(rgb), 'n_frames': 4}}))
    np.savez(home / 'times.npz', exo=np.arange(4) / 10, exo_pts=formats.probe(rgb)['pts'])
    return home


def test_table_depth_has_independent_source_clip_and_sensor_metadata(tmp_path):
    home = table_depth(tmp_path)
    doc = sensors.episode_doc(home)
    assert list(doc['depth']) == ['depth_stream1']
    descriptor = doc['depth']['depth_stream1']
    assert descriptor['paired_camera'] is None
    assert descriptor['playback']['kind'] == 'source_frames'
    assert descriptor['n_frames'] == 3
    assert descriptor['units'] == 'metres'
    assert sensors.summary(doc)['unpaired_depth'] == ['depth_stream1']
    output = tmp_path / 'clips'
    (job,) = clips.depth_jobs(home, output, False)
    assert job[1] == 'depth_stream1' and job[2] is None
    assert clips.extract_depth(*job[:4], fps=job[4]) == []
    with av.open(str(job[3])) as container:
        frames = [frame.to_ndarray(format='rgb24') for frame in container.decode(video=0)]
    assert len(frames) == 3
    original = depth.at_anchor({}, depth.load(home), 'depth_stream1', [0, 1, 2, 3])
    assert list(original) == [0, 1, 3]
    for actual, row in zip(frames, (0, 1, 3)):
        expected = np.asarray(depth.picture(original[row], {'scale_m': .001}))
        assert np.abs(actual.astype(float) - expected).mean() < 5
    assert clips.depth_jobs(home, output, False) == []
    assert serve.clip_path(output, home.name, 'depth_depth_stream1') == job[3]
    assert serve.depth_clips(output, home.name) == {'depth_stream1': job[3]}


def test_depth_legend_cannot_label_disparity_as_metric_distance():
    doc = sensors.depth_doc({'kind': 'disparity', 'scale_m': .001}, {'units': 'metres'})
    assert doc['units'] == 'relative'


def test_static_media_plan_includes_unpaired_depth_without_a_matching_rgb_camera(tmp_path):
    home = table_depth(tmp_path)
    media = tmp_path / 'clips'
    path = clips.depth_clip(media, home.name, 'depth_stream1')
    path.parent.mkdir(parents=True)
    _packed_mp4(path, 3)
    qa = tmp_path / 'qa'
    qa.mkdir()
    (qa / (home.name + '.json')).write_text(json.dumps({'dataset': 'fixture', 'episode_prompt': 'inspect', 'duration_s': .4,
        '_meta': {'episode_id': home.name}, 'camera_views': ['exo'], 'timesteps_s': [0., .3]}))
    (planned,) = static.plan(qa, media)
    assert planned['media']['depth_depth_stream1']['src'] == path
    assert planned['media']['depth_depth_stream1']['rel'].startswith('v/depth_depth_stream1/')


def test_unpaired_regions_use_source_frame_addresses_and_never_episode_times(tmp_path):
    node = shutil.which('node')
    if not node:
        pytest.skip('node is not installed')
    home = table_depth(tmp_path)
    access = evidence_access.Access(episode.load(home))
    field = next(field for field in access.inventory() if field['kind'] == 'depth')
    receipt = access.inspect({'field_id': field['id'], 'mode': 'regions', 'rows': [0, 3],
                              'regions': [[.5, 0, 1, 1]]})
    assert 'times_s' not in receipt
    label = {'evidence_inspection': {'version': 1, 'source_current': True, 'inspections': [receipt]}}
    script = r'''
const assert = require('assert');
const page = require('./tests/page_functions')();
const esc = value => String(value).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const code = ['depthRoiFrames','depthRoiMarkup','unpairedDepthHtml','setupUnpairedDepth'].map(name => page.piece(name)).join('\n');
const api = new Function('esc','videoSrc','window',code+'return {depthRoiFrames,depthRoiMarkup,unpairedDepthHtml,setupUnpairedDepth}')(
  esc, () => 'source.mp4', {_epCleanup:[]});
const label = JSON.parse(process.argv[1]);
const frames = api.depthRoiFrames(label, 'depth_stream1', true);
assert.deepEqual(frames.map(frame => frame.source_frame), [0, 2]);
assert.deepEqual(api.depthRoiFrames(label, 'exo'), []);
assert(api.depthRoiMarkup(frames[1]).includes('source frame 2'));
const html = api.unpairedDepthHtml({depth: {depth_stream1: {paired_camera:null,name:'depth sensor',n_frames:3,
playback:{kind:'source_frames',fps:30},width:64,height:48,units:'metres',bar:['#f00','#00f'],ticks:[[0,'0.1 m'],[1,'10 m']]}}},label,'episode_depth');
assert(html.includes('data-depth-sample="1"'));
assert(html.includes('Source frame 2'));
assert(html.includes('Distance in metres') && html.includes('0.1 m'));
assert(!html.includes('Compare RGB') && !html.includes('NaN') && !html.includes('0.0s'));
const video={readyState:0,currentTime:0,seeking:false,pause(){}}, slider={}, address={}, play={}, overlay={};
const panel={dataset:{depthView:'depth_stream1'},querySelector(selector){return ({video,
  '[data-depth-row]':slider,'[data-depth-address]':address,'[data-depth-play]':play,'.source-depth-roi':overlay})[selector];},
  querySelectorAll(){return [];}};
const on=(element,event,callback)=>{(element.events ||= {})[event]=callback;};
api.setupUnpairedDepth({querySelectorAll(){return [panel];}},
  {depth:{depth_stream1:{n_frames:3,playback:{fps:30}}}},label,on);
assert(overlay.hidden,'unloaded pixels cannot receive measurement outlines');
video.readyState=4;video.currentTime=2/30;video.events.loadeddata();
assert(!overlay.hidden && overlay.innerHTML.includes('source frame 2'));
video.seeking=true;video.events.seeking();
assert(overlay.hidden,'outlines wait for the requested source picture');
video.error={message:'missing clip'};video.events.error();
assert(overlay.hidden && play.disabled && slider.disabled,'unavailable source pixels cannot appear inspected');
label.evidence_inspection.source_current = false;
assert.deepEqual(api.depthRoiFrames(label,'depth_stream1',true),[]);
'''
    subprocess.run([node, '-e', script, json.dumps(label)], cwd=Path(__file__).resolve().parent.parent,
                   check=True, capture_output=True, text=True)
