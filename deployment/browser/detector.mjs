// SPDX-License-Identifier: AGPL-3.0-only
// Pure preprocessing/postprocessing shared by the page and lightweight tests.
export const INPUT_SIZE = 640;
export const OUTPUT_SHAPE = Object.freeze([1, 6, 34000]);
export const CLASS_NAMES = Object.freeze(['hat', 'person']);
export const CLASS_LABELS = Object.freeze(['戴安全帽头部', '裸头']);
export const MAX_FILE_BYTES = 12 * 1024 * 1024;
export const MAX_PIXELS = 24_000_000;
export const MAX_SIDE = 16384;

export function roundEven(value) {
  if (!Number.isFinite(value)) throw new Error('Invalid rounding input');
  const base = Math.floor(value), fraction = value - base;
  const rounded = fraction === 0.5 ? base + (base % 2 === 0 ? 0 : 1) : Math.round(value);
  return rounded === 0 ? 0 : rounded;
}
export function validateDimensions(width, height) {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1) throw new Error('无法读取有效的图片尺寸。');
  if (width * height > MAX_PIXELS || Math.max(width, height) > MAX_SIDE) throw new Error('图片过大：最多2400万像素，单边最多16384像素，请先缩小图片。');
  return {width, height};
}
// Inspect dimensions before decoding. Filenames and EXIF are never retained.
export function inspectImageHeader(bytes) {
  if (!(bytes instanceof Uint8Array) || bytes.length < 12 || bytes.length > MAX_FILE_BYTES) throw new Error('请选择不超过12 MiB的JPEG、PNG或WebP图片。');
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  let width, height, format;
  if ([137,80,78,71,13,10,26,10].every((n,i)=>bytes[i]===n)) {
    if (bytes.length<33 || view.getUint32(8)!==13 || String.fromCharCode(...bytes.slice(12,16))!=='IHDR') throw new Error('PNG文件头无效。');
    width=view.getUint32(16);height=view.getUint32(20);format='PNG';
  } else if (bytes[0]===255 && bytes[1]===216) {
    let offset=2;
    const sof=new Set([0xC0,0xC1,0xC2,0xC3,0xC5,0xC6,0xC7,0xC9,0xCA,0xCB,0xCD,0xCE,0xCF]);
    while(offset<bytes.length) {
      if(bytes[offset++]!==255) throw new Error('JPEG文件头无效。');
      while(offset<bytes.length && bytes[offset]===255)offset++;
      const marker=bytes[offset++];
      if(marker===0xDA || marker===0xD9)break;
      if(marker===0x01 || (marker>=0xD0 && marker<=0xD8))continue;
      if(offset+2>bytes.length)break;
      const length=view.getUint16(offset);
      if(length<2 || offset+length>bytes.length)throw new Error('JPEG文件头不完整。');
      if(sof.has(marker)) {
        if(length<8)throw new Error('JPEG尺寸信息无效。');
        height=view.getUint16(offset+3);width=view.getUint16(offset+5);format='JPEG';break;
      }
      offset+=length;
    }
  } else if(String.fromCharCode(...bytes.slice(0,4))==='RIFF' && String.fromCharCode(...bytes.slice(8,12))==='WEBP') {
    const chunk=String.fromCharCode(...bytes.slice(12,16));
    const uint24=(p)=>bytes[p]|(bytes[p+1]<<8)|(bytes[p+2]<<16);
    if(chunk==='VP8X' && bytes.length>=30) {
      if(bytes[20]&2)throw new Error('请使用静态WebP图片，暂不支持动画。');
      width=1+uint24(24);height=1+uint24(27);
    } else if(chunk==='VP8 ' && bytes.length>=30 && bytes[23]===0x9D && bytes[24]===1 && bytes[25]===0x2A) {
      width=view.getUint16(26,true)&0x3FFF;height=view.getUint16(28,true)&0x3FFF;
    } else if(chunk==='VP8L' && bytes.length>=25 && bytes[20]===0x2F) {
      width=1+(bytes[21]|((bytes[22]&0x3F)<<8));height=1+((bytes[22]>>6)|(bytes[23]<<2)|((bytes[24]&0x0F)<<10));
    }
    format='WebP';
  }
  if(!format || width===undefined || height===undefined)throw new Error('格式无效，请使用JPEG、PNG或静态WebP；不支持SVG、GIF。');
  return {...validateDimensions(width,height),format};
}
export function letterboxGeometry(width,height,size=INPUT_SIZE) {
  validateDimensions(width,height);
  if(!Number.isInteger(size)||size<1)throw new Error('Invalid input size');
  const scale=Math.min(size/width,size/height);
  const resizedWidth=roundEven(width*scale),resizedHeight=roundEven(height*scale);
  if(resizedWidth<1||resizedHeight<1)throw new Error('图片过于狭长，请裁剪后再试。');
  const left=roundEven((size-resizedWidth)/2-0.1),top=roundEven((size-resizedHeight)/2-0.1);
  return {width,height,size,scale,resizedWidth,resizedHeight,left,top};
}
export function rgbaToCHW(rgba,width=INPUT_SIZE,height=INPUT_SIZE) {
  const count=width*height;
  if(!Number.isInteger(width)||!Number.isInteger(height)||width<1||height<1||rgba.length!==count*4)throw new Error('输入像素尺寸不正确。');
  const tensor=new Float32Array(count*3);
  for(let i=0;i<count;i++) {
    tensor[i]=rgba[i*4]/255;tensor[count+i]=rgba[i*4+1]/255;tensor[count*2+i]=rgba[i*4+2]/255;
  }
  return tensor;
}
export function intersectionOverUnion(a,b) {
  const intersection=Math.max(0,Math.min(a[2],b[2])-Math.max(a[0],b[0]))*Math.max(0,Math.min(a[3],b[3])-Math.max(a[1],b[1]));
  const areaA=Math.max(0,a[2]-a[0])*Math.max(0,a[3]-a[1]),areaB=Math.max(0,b[2]-b[0])*Math.max(0,b[3]-b[1]);
  const union=areaA+areaB-intersection;
  return union>0?intersection/union:0;
}
export function classAwareNms(candidates,threshold=0.7,maxDet=300) {
  if(!Number.isFinite(threshold)||threshold<0||threshold>1||!Number.isInteger(maxDet)||maxDet<1)throw new Error('NMS参数无效。');
  const ordered=candidates.slice().sort((a,b)=>b.confidence-a.confidence||a.anchor-b.anchor).slice(0,30000),kept=[];
  for(const candidate of ordered) {
    if(!kept.some((prior)=>prior.classId===candidate.classId&&intersectionOverUnion(prior.box,candidate.box)>threshold)) {
      kept.push(candidate);if(kept.length>=maxDet)break;
    }
  }
  return kept;
}
export function decodeDetections(data,dims,geometry,{confidence=0.25,iou=0.7,maxDet=300}={}) {
  if(!Number.isFinite(confidence)||confidence<0||confidence>1)throw new Error('置信度参数无效。');
  if(!dims||dims.length!==3||!OUTPUT_SHAPE.every((n,i)=>dims[i]===n)||data.length!==6*34000)throw new Error('模型输出形状不符，期待[1,6,34000]。');
  if(!geometry||!Number.isFinite(geometry.scale)||geometry.scale<=0)throw new Error('坐标变换无效。');
  for(const value of data)if(!Number.isFinite(value))throw new Error('模型输出包含无效数值，停止显示结果。');
  const n=dims[2],candidates=[];
  for(let anchor=0;anchor<n;anchor++) {
    const hat=data[4*n+anchor],person=data[5*n+anchor],classId=person>hat?1:0,score=Math.max(hat,person);
    if(score<0||score>1.000001)throw new Error('模型类别分数无效。');
    if(!(score>confidence))continue;
    const cx=data[anchor],cy=data[n+anchor],w=data[2*n+anchor],h=data[3*n+anchor];
    if(w<=0||h<=0)continue;
    candidates.push({anchor,classId,confidence:score,box:[cx-w/2,cy-h/2,cx+w/2,cy+h/2]});
  }
  const clip=(v,limit)=>Math.max(0,Math.min(limit,v));
  // NMS acts in model input coordinates, before inverse mapping and clipping.
  return classAwareNms(candidates,iou,maxDet).map((candidate)=>{
    const [x1,y1,x2,y2]=candidate.box;
    const box=[clip((x1-geometry.left)/geometry.scale,geometry.width),clip((y1-geometry.top)/geometry.scale,geometry.height),clip((x2-geometry.left)/geometry.scale,geometry.width),clip((y2-geometry.top)/geometry.scale,geometry.height)];
    return {classId:candidate.classId,className:CLASS_NAMES[candidate.classId],label:CLASS_LABELS[candidate.classId],confidence:candidate.confidence,box};
  }).filter((row)=>row.box[2]>row.box[0]&&row.box[3]>row.box[1]);
}
