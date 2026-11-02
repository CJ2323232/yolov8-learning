// SPDX-License-Identifier: AGPL-3.0-only
// Optional developer check: node tests/browser_detector.test.mjs
import assert from 'node:assert/strict';
import {roundEven,letterboxGeometry,rgbaToCHW,inspectImageHeader,validateDimensions,
  classAwareNms,decodeDetections} from '../deployment/browser/detector.mjs';

export const passed = [];
function check(name, callback) { callback(); passed.push(name); }
check('Python-compatible rounding, including signed zero',()=>{
  for(const [value,expected] of [[.5,0],[1.5,2],[2.5,2],[-.5,0],[-1.5,-2]])
    assert.strictEqual(roundEven(value),expected);
});
check('Odd landscape letterbox matches fixed-square geometry',()=>{
  const g=letterboxGeometry(1280,721);
  assert.deepStrictEqual([g.resizedWidth,g.resizedHeight,g.left,g.top],[640,360,0,140]);
});
check('Portrait letterbox preserves original dimensions',()=>{
  const g=letterboxGeometry(513,769);
  assert.deepStrictEqual([g.resizedWidth,g.resizedHeight,g.left,g.top],[427,640,106,0]);
});
check('RGB normalization is NCHW, not BGR or interleaved',()=>{
  assert.deepStrictEqual([...rgbaToCHW(new Uint8Array([255,0,0,255,0,255,0,255]),2,1)], [1,0,0,1,0,0]);
});
check('Decoded pixel limit and degenerate sizes are rejected',()=>{
  assert.throws(()=>validateDimensions(10000,10000));
  assert.throws(()=>validateDimensions(0,10));
  assert.throws(()=>letterboxGeometry(16384,1));
});
check('SVG and truncated images are rejected before decoding',()=>{
  assert.throws(()=>inspectImageHeader(new TextEncoder().encode('<svg width="1"></svg>')));
  assert.throws(()=>inspectImageHeader(new Uint8Array([255,216])));
});
check('PNG dimensions read from header before image allocation',()=>{
  const bytes=new Uint8Array(33),v=new DataView(bytes.buffer);
  bytes.set([137,80,78,71,13,10,26,10]);v.setUint32(8,13);bytes.set([73,72,68,82],12);
  v.setUint32(16,123);v.setUint32(20,45);
  assert.deepStrictEqual(inspectImageHeader(bytes),{width:123,height:45,format:'PNG'});
  v.setUint32(16,200000);assert.throws(()=>inspectImageHeader(bytes));
});
const candidates=[{anchor:0,classId:0,confidence:.9,box:[0,0,10,10]},
  {anchor:1,classId:0,confidence:.8,box:[0,0,10,10]},
  {anchor:2,classId:1,confidence:.7,box:[0,0,10,10]}];
check('NMS removes same-class duplicate but keeps other class',()=>{
  assert.deepStrictEqual(classAwareNms(candidates).map(x=>x.anchor),[0,2]);
  assert.strictEqual(classAwareNms(candidates,.7,1).length,1);
});
check('Output shape mismatch and NaN fail closed',()=>{
  assert.throws(()=>decodeDetections(new Float32Array(6),[1,6,1],letterboxGeometry(640,640)));
  const data=new Float32Array(6*34000);data[1]=NaN;
  assert.throws(()=>decodeDetections(data,[1,6,34000],letterboxGeometry(640,640)));
});
check('Confidence is strict, class argmax and coordinate restoration',()=>{
  const n=34000,data=new Float32Array(6*n);
  data[0]=320;data[n]=320;data[2*n]=100;data[3*n]=100;data[4*n]=.25;
  assert.strictEqual(decodeDetections(data,[1,6,n],letterboxGeometry(1280,640)).length,0);
  data[5*n]=.75;
  const rows=decodeDetections(data,[1,6,n],letterboxGeometry(1280,640));
  assert.strictEqual(rows.length,1);assert.strictEqual(rows[0].classId,1);
  assert.deepStrictEqual(rows[0].box,[540,220,740,420]);
});
check('Invalid class score is rejected',()=>{
  const data=new Float32Array(6*34000);data[4*34000]=2;
  assert.throws(()=>decodeDetections(data,[1,6,34000],letterboxGeometry(640,640)));
});
console.log(`Browser detector: ${passed.length} checks passed`);
