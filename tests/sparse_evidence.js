'use strict';
const fs=require('fs'), assert=require('assert');
const source = require('./page_functions')(process.argv[2]).src;
const begin=source.indexOf('// ================= sensor evidence:');
const end=source.indexOf('// ================= touch:',begin);
const api=new Function('esc','fmtT',source.slice(begin,end)+'return {sensorEvidence,genericSensorSampleTimes,genericSensorSeries,genericSensorPanelHtml,setupSensorEvidence,syncGripOverlayPressure};')(String,t=>t+'s');
const saved=JSON.parse(fs.readFileSync(process.argv[3],'utf8'));
const before=JSON.stringify(saved),E=api.sensorEvidence(saved),f=E.insights[0];
assert.deepEqual(api.genericSensorSampleTimes(E.grip,f),[27,31.5,36,40.5]);
const html=api.genericSensorPanelHtml(E.grip,f);
assert(html.includes('data-sensor-sample-t="31.5"'),'exact sample is directly inspectable');
assert.equal((html.match(/data-sensor-sample-t=/g)||[]).length,4,'duplicate citations share controls');
const traces=api.genericSensorSeries(E.grip,f);
assert(traces.every(v=>v.low>0),'depth region scales use measured range, not zero');
assert.equal(traces[0].values[1],578);
assert(!/<path d="[^"]*L/.test(html),'sampled summaries cannot imply continuous measurements');
assert.equal((html.match(/data-sensor-recorded-point/g)||[]).length,8,'all measured points stay visible');
assert(html.includes('559')&&html.includes('637'),'recorded range is visible');
const panel={listeners:{},querySelectorAll(){return [];}};
const document={getElementById(id){return id==='grip-evidence-panel'?panel:null;},querySelectorAll(){return [];}};
global.document=document;
let sought=null;
const wire=api.setupSensorEvidence(E,(...args)=>sought=args,(el,type,fn)=>el.listeners[type]=fn,()=>{});
panel.listeners.click({target:{closest(){return {dataset:{sensorSampleT:'31.5'}};}}});
assert.deepEqual(sought,[31.5,false],'inspection seeks paused through the existing seek path');
const dot={dataset:{gripMiniDot:'0'},setAttribute(k,v){this[k]=v}},value={dataset:{gripMiniValue:'0'}};
const scope={querySelectorAll(q){return q==='[data-grip-mini-dot]'?[dot]:q==='[data-grip-mini-value]'?[value]:[];}};
api.syncGripOverlayPressure(E.grip,f,31.5,scope);
assert.equal(dot.visibility,'visible');assert.equal(value.textContent,'578');
api.syncGripOverlayPressure(E.grip,f,31.656,scope);
assert.equal(dot.visibility,'hidden');assert.equal(value.textContent,'');
const partial=JSON.parse(before);partial.sensor_evidence.findings[0].evidence.forEach(r=>r.time_s=[27]);
assert.deepEqual(api.genericSensorSampleTimes(api.sensorEvidence(partial).grip,f),[27,31.5,36,40.5],'full supplied samples survive abbreviated citations');
const stale=JSON.parse(before);stale.evidence_inspection.source_current=false;
const S=api.sensorEvidence(stale);assert.deepEqual(api.genericSensorSampleTimes(S.grip,S.insights[0]),[]);
assert.equal(api.genericSensorSeries(S.grip,S.insights[0]).length,0,'stale receipt cannot produce a current trace');
const rejected=JSON.parse(before);rejected.evidence_inspection.inspections.forEach(r=>r.review_status='rejected');
assert.deepEqual(api.genericSensorSampleTimes(api.sensorEvidence(rejected).grip,f),[]);
const withheld=JSON.parse(before);withheld.evidence_inspection.inspections.forEach(r=>r.withheld_from_final=true);
assert.deepEqual(api.genericSensorSampleTimes(api.sensorEvidence(withheld).grip,f),[]);
const clipped={...f,start:31,end:37};assert.deepEqual(api.genericSensorSampleTimes(E.grip,clipped),[31.5,36]);
const unrelated=JSON.parse(before);unrelated.sensor_evidence.series.push({...unrelated.sensor_evidence.series[1],id:'uncited',times:[30],values:[999]});
assert.deepEqual(api.genericSensorSampleTimes(api.sensorEvidence(unrelated).grip,f),[27,31.5,36,40.5]);
const piece=JSON.parse(before),origin=10;
piece.evidence_inspection={version:1,source_current:true,parts:[{part:2,time_origin_s:origin,record:piece.evidence_inspection}]};
for(const s of piece.sensor_evidence.sensors){s.id='part2:'+s.id;s.times=s.times.map(t=>t+origin);}
for(const r of piece.sensor_evidence.series){r.id='part2:'+r.id;r.sensor_id='part2:'+r.sensor_id;r.times=r.times.map(t=>t+origin);}
for(const finding of piece.sensor_evidence.findings){finding.start_s+=origin;finding.end_s+=origin;for(const r of finding.evidence){r.sensor_id='part2:'+r.sensor_id;if(r.series_id)r.series_id='part2:'+r.series_id;r.time_s=r.time_s.map(t=>t+origin);}}
const P=api.sensorEvidence(piece);assert.deepEqual(api.genericSensorSampleTimes(P.grip,P.insights[0]),[37,41.5,46,50.5],'receipt origin applied exactly once');
const rgb=api.sensorEvidence({});assert.equal(api.genericSensorPanelHtml(rgb.grip,null),'');
const dense={generic:true,sensors:[{id:'recorded',kind:'numeric'}],series:[{id:'recorded:0',sensor_id:'recorded',label:'Signal',times:[0,.03,.06],values:[1,2,3]}]};
const denseF={start:0,end:.06,evidence:[{sensor_id:'recorded',series_id:'recorded:0'}]};
assert.deepEqual(api.genericSensorSampleTimes(dense,denseF),[0,.03,.06]);
assert(api.genericSensorSeries(dense,denseF)[0].low<0,'non-depth scale remains zero-inclusive');
const region={label:'Thumb',times:[0,.03,.06],values:[0,4,9]};
const mixed=api.genericSensorSeries(dense,{...denseF,display:[region,{series_id:'recorded:0'},{series_id:'uncited'}]},3);
assert.deepEqual(mixed.map(r=>r.label),['Thumb','Signal'],'a derived row keeps its place and a cited series is drawn as always');
assert(mixed[0].displayRow&&!mixed[1].displayRow);
assert.deepEqual(api.genericSensorSeries(dense,{...denseF,display:[{series_id:'recorded:0'}]},3),api.genericSensorSeries(dense,denseF,3),
  'references alone change nothing');
assert.deepEqual(api.genericSensorSeries(dense,{...denseF,display:[region]},Infinity).map(r=>r.label),['Signal'],
  'the expanded panel keeps the raw cited series');
const many={...dense,series:[{...dense.series[0],times:Array.from({length:12},(_,i)=>i*.03),values:Array.from({length:12},(_,i)=>i)}]};
const manyF={...denseF,end:1};
assert(api.genericSensorPanelHtml(many,manyF).includes('data-sensor-sample-select'),'larger sample lists stay compact');
panel.listeners.change({target:{value:'.06',matches(){return true;}}});
assert.deepEqual(sought,[.06,false],'sample dropdown shares paused inspection behavior');
const roiBegin=source.indexOf('function depthRoiFrames('),roiEnd=source.indexOf('// each camera with depth:',roiBegin);
const roi=new Function(source.slice(roiBegin,roiEnd)+'return {depthRoiFrames,depthRoiMarkup};')();
const frames=roi.depthRoiFrames(saved,'exo');
assert.deepEqual(frames.map(frame=>frame.time_s),[27,31.5,36,40.5]);
for(const t of api.genericSensorSampleTimes(E.grip,f)){
 const frame=frames.find(frame=>frame.time_s===t);
 assert.equal((roi.depthRoiMarkup(frame).match(/<rect /g)||[]).length,2,'each exact control has both measured depth regions');
}
assert.equal(frames[1].regions[0].percentiles_stored[1],578,'readout equals sampled ROI median');
for(const label of [stale,rejected,withheld]) assert.deepEqual(roi.depthRoiFrames(label,'exo'),[],'ineligible receipts cannot leave depth boxes visible');
assert.deepEqual(roi.depthRoiFrames(piece,'exo').map(frame=>frame.time_s),[37,41.5,46,50.5],'stitched depth boxes share the control clock');
for(const status of ['rejected','withheld','stale']){
 const label=JSON.parse(JSON.stringify(piece)),record=label.evidence_inspection.parts[0].record;
 if(status==='stale') record.source_current=false;
 else for(const receipt of record.inspections){if(status==='rejected') receipt.review_status='rejected';else receipt.withheld_from_final=true;}
 assert.deepEqual(roi.depthRoiFrames(label,'exo'),[],'stitched ineligible receipts cannot leave depth boxes visible');
}
assert.equal(JSON.stringify(saved),before,'viewer does not mutate exported evidence');
const finding=(start,headline,card)=>({start_s:start,end_s:start+.5,headline,detail:headline+' in full',adds_beyond_video:true,evidence:[],
  ...(card?{video_card:{version:1,headline:card,detail:card+' on the video.'}}:{})});
const cardsDoc=chosen=>({sensor_evidence:{version:1,sensors:[],series:[],findings:[finding(0,'Loading builds','Thumb presses harder'),
  finding(.2,'Local redistribution'),finding(2,'Thumb loading eases','Thumb pressure eases')],...(chosen?{video_cards:{version:1}}:{})}});
const C=api.sensorEvidence(cardsDoc(true));
assert.deepEqual(C.overlayFindings.map(f=>f.headline),['Thumb presses harder','Thumb pressure eases'],'the video plays only the chosen cards, in their words');
assert.equal(C.overlayFindings[1].overlayDetail,'Thumb pressure eases on the video.');
assert.deepEqual(C.insights.map(f=>f.headline),['Loading builds','Local redistribution','Thumb loading eases'],'every finding stays in the list');
assert.deepEqual(C.overlayFindings.map(f=>f.findingIndex),[0,2],'a card keeps its finding identity');
assert.equal(api.sensorEvidence(cardsDoc(false)).overlayFindings.length,3,'a record without chosen cards plays every finding');
const none=cardsDoc(true);none.sensor_evidence.findings.forEach(f=>delete f.video_card);
assert.equal(api.sensorEvidence(none).overlayFindings.length,0,'a run that chose no cards shows none');
delete global.document;
console.log('Sparse evidence inspection passed');
