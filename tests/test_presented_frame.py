"""Annotations follow decoded presentation, including paused and overlapping seeks."""
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.test_overlay_stack import CARD_DOM


HERE = Path(__file__).parent
SOURCE = HERE.parent / 'board/index.html'
pytestmark = pytest.mark.skipif(not shutil.which('node'), reason='no node')


def test_pending_seek_cannot_show_future_sensor_finding():
    script = r'''
const assert=require('assert'),{src}=require(process.argv[1])();
const begin=src.indexOf('// ================= sensor evidence:'),end=src.indexOf('// ================= touch:',begin);
const {strip,createElement}=cardDocument();
const button={dataset:{gripFinding:'0',evidenceT:'55',evidencePause:'true'},setAttribute(){}};
const doc={createElement,getElementById:id=>id==='grip-finding-overlay'?strip:null,
 querySelectorAll:q=>q==='[data-evidence-t]'?[button]:[]};
const api=new Function('esc','fmtT','document',src.slice(begin,end)
 +'return {sensorEvidence,setupSensorEvidence};')(String,String,doc);
const callbacks=new Map(),media={currentTime:0,readyState:1,seeking:true,ended:false};
const evidence=api.sensorEvidence({sensor_evidence:{version:1,findings:[
 {start_s:54,end_s:56,headline:'Later insertion',adds_beyond_video:true}]}});
const wire=api.setupSensorEvidence(evidence,()=>{media.currentTime=55;},
 (el,event,fn)=>callbacks.set(el,fn),()=>{},null,null,()=>{},media);
callbacks.get(button)();
assert(strip.hidden,'a target-time finding must not describe the old decoded picture during a blocked seek');
wire.dispose();
'''
    subprocess.run(['node', '-e', CARD_DOM + script, str(HERE / 'page_functions.js')], check=True)


def test_presented_clock_waits_for_frame_and_rejects_stale_seek_callbacks():
    script = r'''
const assert=require('assert'),{piece}=require(process.argv[1])();
const setup=new Function(piece('setupPresentedVideoClock')+'return setupPresentedVideoClock;')();
const events={},callbacks=new Map(),cancelled=new Map(),rendered=[],pending=[];
let next=0;
const video={currentTime:0,readyState:1,seeking:false,paused:true,ended:false,error:null,
 requestVideoFrameCallback(fn){callbacks.set(++next,fn);return next;},
 cancelVideoFrameCallback(id){cancelled.set(id,callbacks.get(id));callbacks.delete(id);}};
const on=(el,event,fn)=>{events[event]=fn;};
const clock=setup(video,on,t=>rendered.push(t),value=>pending.push(value));
const present=t=>{const [id,fn]=callbacks.entries().next().value;callbacks.delete(id);fn(0,{mediaTime:t});};
assert.equal(clock.time(),null,'metadata alone does not establish presented pixels');
video.readyState=4;events.loadeddata();present(0);
assert.equal(clock.time(),0);
clock.begin(55.2);video.currentTime=55.2;video.seeking=true;events.seeking();
assert.equal(clock.time(),null);assert.equal(pending.at(-1),true);
video.readyState=1;events.timeupdate();
assert.deepEqual(rendered,[0],'currentTime cannot advance annotations while an old frame remains');
video.readyState=4;video.seeking=false;events.seeked();
assert.equal(clock.time(),null,'seeked can precede compositor presentation');
present(55);
assert.equal(clock.time(),55.2,'a native frame may precede the requested time without changing the sensor clock');
assert.equal(pending.at(-1),false);
const count=rendered.length;clock.begin(55.2);
assert.equal(clock.time(),55.2,'an unchanged paused seek keeps its already presented frame');
assert.equal(rendered.length,count);
clock.begin(20);video.currentTime=20;video.seeking=true;events.seeking();
const stale=[...callbacks.values()][0];
clock.begin(40);video.currentTime=40;events.seeking();
video.seeking=false;events.seeked();stale(0,{mediaTime:20});
assert.equal(clock.time(),null,'a callback from the previous seek cannot release the latest seek');
present(40);assert.equal(clock.time(),40);
// Presentation may precede dispatch of seeked, after native seeking becomes false.
clock.begin(12);video.currentTime=12;video.seeking=true;events.seeking();
video.seeking=false;present(12);events.seeked();assert.equal(clock.time(),12);
// Some browsers present the target while seeking is still true, then emit seeked without another frame.
clock.begin(13.1);video.currentTime=13.1;video.seeking=true;video.readyState=1;events.seeking();
present(13);assert.equal(clock.time(),null,'a frame callback alone cannot bypass native readiness');
video.seeking=false;video.readyState=4;events.seeked();assert.equal(clock.time(),13.1);
video.currentTime=12;
video.paused=false;events.play();video.currentTime=12.1;events.timeupdate();video.currentTime=12.2;events.timeupdate();
assert.deepEqual(rendered.slice(-3),[12,12.1,12.2]);
video.ended=true;events.ended();assert.equal(rendered.at(-1),12.2);
const late=[...callbacks.values()][0];clock.dispose();
if(late)late(0,{mediaTime:30});
assert.equal(rendered.at(-1),12.2,'disposed episode callbacks cannot update the new episode');
assert.equal(callbacks.size,0);
'''
    subprocess.run(['node', '-e', script, str(HERE / 'page_functions.js')], check=True)


def test_presented_clock_without_video_frame_callback_uses_ready_seek_lifecycle():
    script = r'''
const assert=require('assert'),{piece}=require(process.argv[1])();
const setup=new Function(piece('setupPresentedVideoClock')+'return setupPresentedVideoClock;')();
const events={},rendered=[];
const video={currentTime:0,readyState:0,seeking:false,paused:true,ended:false,error:null};
const clock=setup(video,(el,event,fn)=>events[event]=fn,t=>rendered.push(t),()=>{});
assert.equal(clock.time(),null);
video.readyState=2;events.loadeddata();assert.equal(clock.time(),0);
clock.begin(20);video.currentTime=20;video.seeking=true;events.seeking();
events.timeupdate();assert.deepEqual(rendered,[0]);
video.readyState=1;video.seeking=false;events.seeked();assert.equal(clock.time(),null);
video.readyState=2;events.loadeddata();assert.equal(clock.time(),20);
video.currentTime=20.4;events.timeupdate();assert.equal(clock.time(),20.4);
video.error={code:3};events.error();assert.equal(clock.time(),null);
clock.dispose();video.error=null;events.loadeddata();assert.equal(clock.time(),null);
'''
    subprocess.run(['node', '-e', script, str(HERE / 'page_functions.js')], check=True)


def test_presented_sensor_selection_retains_cards_and_minimum_display_window():
    script = r'''
const assert=require('assert'),{src,piece}=require(process.argv[1])();
const begin=src.indexOf('// ================= sensor evidence:'),end=src.indexOf('// ================= touch:',begin);
const {strip,createElement}=cardDocument(),events=new Map(),frames=new Map();let next=0;
const buttons=[0,1].map(i=>({dataset:{gripFinding:String(i),evidenceT:'55',evidencePause:'true'},
 setAttribute(k,v){this[k]=v;}}));
const doc={createElement,getElementById:id=>id==='grip-finding-overlay'?strip:null,
 querySelectorAll:q=>['[data-evidence-t]','[data-grip-finding]'].includes(q)?buttons:[]};
const api=new Function('esc','fmtT','document',src.slice(begin,end)+piece('setupPresentedVideoClock')
 +'return {sensorEvidence,setupSensorEvidence,setupPresentedVideoClock};')(String,String,doc);
const video={currentTime:0,readyState:4,seeking:false,paused:true,ended:false,error:null,
 requestVideoFrameCallback(fn){frames.set(++next,fn);return next;},cancelVideoFrameCallback(id){frames.delete(id);}};
const on=(el,event,fn)=>{const key=el===video?event:el;(events.get(key)||events.set(key,[]).get(key)).push(fn);};
const emit=event=>(events.get(event)||[]).forEach(fn=>fn());
const present=t=>{const [id,fn]=frames.entries().next().value;frames.delete(id);fn(0,{mediaTime:t});};
let clock,wire;
const seek=t=>{clock.begin(t);wire.reset(t);video.currentTime=t;video.seeking=true;emit('seeking');};
wire=api.setupSensorEvidence(api.sensorEvidence({sensor_evidence:{version:1,findings:[
 {start_s:54,end_s:56,headline:'Earlier concurrent finding',adds_beyond_video:true},
 {start_s:54.9,end_s:55.1,headline:'Brief local change',adds_beyond_video:true}]}}),seek,on,()=>{},
 null,null,()=>{},video,()=>clock.time());
clock=api.setupPresentedVideoClock(video,on,t=>wire.sync(t),()=>{});present(0);
clock.begin(60);video.currentTime=60;video.seeking=true;emit('seeking');
video.seeking=false;emit('seeked');present(60);
events.get(buttons[0])[0]();
assert(strip.hidden,'backward manual inspection remains hidden until its picture arrives');
video.seeking=false;emit('seeked');present(55);
assert.equal(buttons[0]['aria-pressed'],'true','the pending explicit choice survives backward presentation');
assert.equal(buttons[1]['aria-pressed'],'false');
const nodes=strip.querySelectorAll();
video.paused=false;emit('play');video.currentTime=55.2;emit('timeupdate');video.currentTime=55.3;emit('timeupdate');
assert.deepEqual(strip.querySelectorAll(),nodes,'normal playback retains card DOM identity');
assert(!strip.hidden,'a brief source event retains its two-second presentation window');
video.currentTime=57;emit('timeupdate');assert(strip.hidden);
clock.dispose();wire.dispose();
'''
    subprocess.run(['node', '-e', CARD_DOM + script, str(HERE / 'page_functions.js')], check=True)


def test_presented_low_fps_frame_does_not_quantize_native_sensor_clock():
    script = r'''
const assert=require('assert'),{src,piece}=require(process.argv[1])();
const begin=src.indexOf('// ================= sensor evidence:'),end=src.indexOf('// ================= touch:',begin);
const {strip,createElement}=cardDocument(),events={},callbacks=new Map();let next=0;
const doc={createElement,getElementById:id=>id==='grip-finding-overlay'?strip:null,querySelectorAll:()=>[]};
const api=new Function('esc','fmtT','document',src.slice(begin,end)+piece('setupPresentedVideoClock')
 +piece('snPlaybackTime')+'return {sensorEvidence,setupSensorEvidence,setupPresentedVideoClock,snPlaybackTime};')(String,String,doc);
const video={currentTime:0,readyState:4,seeking:false,paused:true,ended:false,error:null,
 requestVideoFrameCallback(fn){callbacks.set(++next,fn);return next;},cancelVideoFrameCallback(id){callbacks.delete(id);}};
const on=(el,event,fn)=>(events[event] ||= []).push(fn);
let clock;
const capture={playback:{starts:[0,.2,.4,1],captures:[0,.31,.72,1.3]}};
const wire=api.setupSensorEvidence(api.sensorEvidence({sensor_evidence:{version:1,findings:[
 {start_s:.3,end_s:.35,headline:'Native sub-frame response',adds_beyond_video:true}]}}),()=>{},()=>{},()=>{},
 null,null,()=>{},video,()=>{const t=clock.time();return t==null?null:video.paused?t:api.snPlaybackTime(capture,t);});
clock=api.setupPresentedVideoClock(video,on,t=>wire.sync(t),()=>{});
const first=[...callbacks.entries()][0];callbacks.delete(first[0]);first[1](0,{mediaTime:0});
assert(strip.hidden);
// RGB holds its 0s picture until 1s. Sensor capture .31 occurs in its valid presentation interval.
video.paused=false;video.currentTime=.3;events.timeupdate.forEach(fn=>fn());
assert.equal(clock.time(),.3,'presentation readiness must not turn a 100Hz sensor clock into RGB PTS');
assert(!strip.hidden,'an intermediate native-clock response cannot disappear between low-FPS RGB frames');
// Pausing retains the exact selected sensor clock instead of applying the playback capture map.
video.paused=true;video.currentTime=.1;events.timeupdate.forEach(fn=>fn());assert(strip.hidden);
clock.dispose();wire.dispose();
'''
    subprocess.run(['node', '-e', CARD_DOM + script, str(HERE / 'page_functions.js')], check=True)


def test_native_identical_paused_seek_reuses_only_its_already_presented_clock():
    script = r'''
const assert=require('assert'),{piece}=require(process.argv[1])();
const setup=new Function(piece('setupPresentedVideoClock')+'return setupPresentedVideoClock;')();
const events={},callbacks=new Map();let next=0;
const video={currentTime:55,readyState:4,seeking:false,paused:true,ended:false,error:null,
 requestVideoFrameCallback(fn){callbacks.set(++next,fn);return next;},cancelVideoFrameCallback(id){callbacks.delete(id);}};
const clock=setup(video,(el,event,fn)=>events[event]=fn,()=>{},()=>{});
let [id,fn]=[...callbacks.entries()][0];callbacks.delete(id);fn(0,{mediaTime:55});
video.seeking=true;video.readyState=1;events.seeking();assert.equal(clock.time(),null);
video.seeking=false;video.readyState=4;events.seeked();
assert.equal(clock.time(),55,'native identical-clock seeks may not submit a new compositor frame');
clock.begin(20);video.currentTime=20;video.seeking=true;events.seeking();
assert.equal(video._annotationFrame,null,'a genuinely new seek cannot retain the old cache across re-render');
clock.dispose();
'''
    subprocess.run(['node', '-e', script, str(HERE / 'page_functions.js')], check=True)


def test_seek_callback_is_not_armed_for_an_old_submitted_playback_frame():
    script = r'''
const assert=require('assert'),{piece}=require(process.argv[1])();
const setup=new Function(piece('setupPresentedVideoClock')+'return setupPresentedVideoClock;')();
const events={},callbacks=new Map();let next=0;
const video={currentTime:55,readyState:4,seeking:false,paused:false,ended:false,error:null,
 requestVideoFrameCallback(fn){callbacks.set(++next,fn);return next;},cancelVideoFrameCallback(id){callbacks.delete(id);}};
const clock=setup(video,(el,event,fn)=>events[event]=fn,()=>{},()=>{});
let [id,fn]=[...callbacks.entries()][0];callbacks.delete(id);fn(0,{mediaTime:55});
clock.begin(9);
assert.equal(callbacks.size,0,'arming before currentTime assignment can observe an already submitted old playback frame');
video.currentTime=9;video.readyState=1;video.seeking=true;video.paused=true;events.pause();
assert.equal(callbacks.size,0,'pause/timeupdate cannot arm the seek before its native seeking lifecycle');
events.seeking();assert.equal(callbacks.size,1);
[id,fn]=[...callbacks.entries()][0];callbacks.delete(id);fn(0,{mediaTime:9});
video.readyState=4;video.seeking=false;events.seeked();assert.equal(clock.time(),9);
clock.dispose();
'''
    subprocess.run(['node', '-e', script, str(HERE / 'page_functions.js')], check=True)


@pytest.mark.parametrize('frame_callbacks', [False, True])
def test_cold_zero_default_start_does_not_require_a_native_seek_event(frame_callbacks):
    script = r'''
const assert=require('assert'),{piece}=require(process.argv[1])();
const setup=new Function(piece('setupPresentedVideoClock')+'return setupPresentedVideoClock;')();
const events={},callbacks=new Map();let next=0;
const video={currentTime:0,readyState:0,seeking:false,paused:true,ended:false,error:null};
if(process.argv[2]==='true'){
 video.requestVideoFrameCallback=fn=>{callbacks.set(++next,fn);return next;};
 video.cancelVideoFrameCallback=id=>callbacks.delete(id);
}
const clock=setup(video,(el,event,fn)=>events[event]=fn,()=>{},()=>{});
clock.begin(0);video.currentTime=0;
// Loading at default start zero does not emit a seeking event.
video.readyState=4;events.loadeddata();
for(const [id,fn]of callbacks){callbacks.delete(id);fn(0,{mediaTime:0});break;}
assert.equal(clock.time(),0,'default start zero must release through its initial presentation lifecycle');
video.paused=false;video.currentTime=.3;events.play();events.timeupdate();assert.equal(clock.time(),.3);
clock.dispose();
'''
    subprocess.run(['node', '-e', script, str(HERE / 'page_functions.js'), str(frame_callbacks).lower()], check=True)
