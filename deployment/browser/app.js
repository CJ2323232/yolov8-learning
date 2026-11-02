// SPDX-License-Identifier: AGPL-3.0-only
import {INPUT_SIZE,CLASS_LABELS,MAX_FILE_BYTES,inspectImageHeader,validateDimensions,letterboxGeometry,rgbaToCHW,decodeDetections} from './detector.mjs';

const $=(id)=>document.getElementById(id);
const ui=Object.fromEntries(['image-file','image-info','confidence','confidence-value','nms-iou','iou-value','detect','clear','status','model-state','result-tag','result-canvas','empty-canvas','summary','timing','detections','download-png','download-csv'].map((id)=>[id,$(id)]));
const base=new URL('.',import.meta.url);
let sourceCanvas=null,rows=null,sessionPromise=null,modelInfo=null,resultParameters=null;
let revision=0,busy=false;

function setStatus(message,error=false) {
  ui.status.textContent=message;
  ui.status.classList.toggle('error',error);
}
function setBusy(value) {
  busy=value;
  ui['image-file'].disabled=value;ui.confidence.disabled=value;ui['nms-iou'].disabled=value;
  ui.detect.disabled=value||!sourceCanvas;ui.detect.textContent=value?'正在检测…':'开始检测';
  ui['download-png'].disabled=value||rows===null;ui['download-csv'].disabled=value||rows===null;
}
function emptyRows(message) {
  ui.detections.replaceChildren();
  const tr=document.createElement('tr'),td=document.createElement('td');
  td.colSpan=7;td.textContent=message;tr.append(td);ui.detections.append(tr);
}
function invalidateResult() {
  rows=null;resultParameters=null;
  ui['download-png'].disabled=true;ui['download-csv'].disabled=true;
  ui.summary.textContent='检测结果将在这里显示。';ui.timing.textContent='';emptyRows('尚未检测。');
}
function drawSource() {
  const canvas=ui['result-canvas'];canvas.width=sourceCanvas.width;canvas.height=sourceCanvas.height;
  canvas.getContext('2d',{alpha:false}).drawImage(sourceCanvas,0,0);
  canvas.hidden=false;ui['empty-canvas'].hidden=true;
}
function clearImage() {
  revision++;
  if(sourceCanvas){sourceCanvas.width=1;sourceCanvas.height=1;sourceCanvas=null;}
  ui['image-file'].value='';ui['image-info'].textContent='还没有选择图片。';
  ui['result-canvas'].width=1;ui['result-canvas'].height=1;ui['result-canvas'].hidden=true;
  ui['empty-canvas'].hidden=false;ui['result-tag'].textContent='等待图片';
  invalidateResult();setBusy(busy);
  setStatus(busy?'图片已清除。当前计算结束后可重新选择。':'图片已清除，仅保留已加载的模型。');
}

async function selectImage() {
  const file=ui['image-file'].files?.[0];if(!file)return;
  const ticket=++revision;invalidateResult();
  if(sourceCanvas){sourceCanvas.width=1;sourceCanvas.height=1;sourceCanvas=null;}
  ui['result-canvas'].width=1;ui['result-canvas'].height=1;ui['result-canvas'].hidden=true;
  ui['empty-canvas'].hidden=false;ui['result-tag'].textContent='读取图片';
  setBusy(true);setStatus('正在本地读取图片…');
  let bitmap;
  try {
    if(file.size>MAX_FILE_BYTES)throw new Error('图片超过12 MiB，请压缩后再试。');
    const bytes=new Uint8Array(await file.arrayBuffer());inspectImageHeader(bytes);
    bitmap=await createImageBitmap(file,{imageOrientation:'from-image'});
    if(ticket!==revision)return;
    validateDimensions(bitmap.width,bitmap.height);letterboxGeometry(bitmap.width,bitmap.height);
    sourceCanvas=document.createElement('canvas');sourceCanvas.width=bitmap.width;sourceCanvas.height=bitmap.height;
    const context=sourceCanvas.getContext('2d',{alpha:false});
    context.fillStyle='rgb(114,114,114)';context.fillRect(0,0,bitmap.width,bitmap.height);context.drawImage(bitmap,0,0);
    drawSource();
    ui['image-info'].textContent=`${bitmap.width} × ${bitmap.height} px · ${(file.size/1024/1024).toFixed(2)} MiB · 文件名不记录`;
    ui['result-tag'].textContent='原图预览';setStatus('图片已准备好。点击“开始检测”，图片不会上传。');
  } catch(error) {
    if(ticket===revision) {
      setStatus(error instanceof Error?error.message:'图片读取失败，请换一张图片。',true);
      ui['result-tag'].textContent='图片不可用';ui['image-info'].textContent='图片未加载。';ui['image-file'].value='';
      if(sourceCanvas){sourceCanvas.width=1;sourceCanvas.height=1;sourceCanvas=null;}
    }
  } finally {bitmap?.close();setBusy(false);}
}

async function loadSession() {
  if(sessionPromise)return sessionPromise;
  sessionPromise=(async()=>{
    if(!globalThis.ort||!globalThis.crypto?.subtle)throw new Error('浏览器推理组件未就绪，请用最新版Chrome、Edge、Firefox或Safari，通过HTTPS重新打开页面。');
    ort.env.wasm.numThreads=1;ort.env.wasm.proxy=false;ort.env.wasm.wasmPaths=new URL('./vendor/',base).href;ort.env.logLevel='error';
    ui['model-state'].textContent='正在下载并校验P2模型，图片保留在浏览器内…';
    const infoResponse=await fetch(new URL('model_info.json',base),{credentials:'omit',cache:'force-cache',redirect:'follow'});
    if(!infoResponse.ok)throw new Error('模型说明加载失败，请稍后重试。');
    const info=await infoResponse.json();if(!/^[a-f0-9]{64}$/.test(info.sha256??''))throw new Error('模型校验信息无效，停止加载。');
    const modelResponse=await fetch(new URL('helmet_p2.onnx',base),{credentials:'omit',cache:'force-cache',redirect:'follow'});
    if(!modelResponse.ok)throw new Error('模型文件加载失败，请检查网络后重试。');
    // HF serves large artifacts through its official CDN. No image is sent.
    for(const response of [infoResponse,modelResponse]) {
      const origin=new URL(response.url).origin;
      if(origin!==base.origin && origin!=='https://us.aws.cdn.hf.co')throw new Error('模型资源跳转到未允许的来源，已停止加载。');
    }
    const modelBytes=await modelResponse.arrayBuffer(),digest=await crypto.subtle.digest('SHA-256',modelBytes);
    const hash=Array.from(new Uint8Array(digest),(b)=>b.toString(16).padStart(2,'0')).join('');
    if(hash!==info.sha256)throw new Error('模型文件校验不一致，已拒绝运行；请刷新页面后重试。');
    const session=await ort.InferenceSession.create(modelBytes,{executionProviders:['wasm'],graphOptimizationLevel:'all'});
    if(session.inputNames.length!==1||session.inputNames[0]!=='images'||session.outputNames.length!==1){await session.release();throw new Error('模型输入输出接口不符，停止运行。');}
    modelInfo={sha256:hash};ui['model-state'].textContent=`P2模型已校验 · SHA-256 ${hash.slice(0,12)}… · 640 × 640 · FP32 / WASM`;
    return session;
  })();
  try{return await sessionPromise;}catch(error){sessionPromise=null;modelInfo=null;ui['model-state'].textContent='模型未加载成功；可以重新点击检测。';throw error;}
}
function prepareInput(source) {
  const geometry=letterboxGeometry(source.width,source.height),canvas=document.createElement('canvas');
  canvas.width=INPUT_SIZE;canvas.height=INPUT_SIZE;
  const context=canvas.getContext('2d',{willReadFrequently:true,alpha:false});
  context.fillStyle='rgb(114,114,114)';context.fillRect(0,0,INPUT_SIZE,INPUT_SIZE);
  context.imageSmoothingEnabled=true;context.imageSmoothingQuality='low';
  context.drawImage(source,0,0,source.width,source.height,geometry.left,geometry.top,geometry.resizedWidth,geometry.resizedHeight);
  const input=rgbaToCHW(context.getImageData(0,0,INPUT_SIZE,INPUT_SIZE).data);canvas.width=1;canvas.height=1;
  return {geometry,input};
}
function showRows(detections) {
  ui.detections.replaceChildren();if(!detections.length){emptyRows('当前参数下没有检出目标。');return;}
  const fragment=document.createDocumentFragment();
  detections.forEach((row,index)=>{
    const tr=document.createElement('tr');
    const values=[index+1,`${row.className} · ${row.label}`,row.confidence.toFixed(4),...row.box.map((x)=>x.toFixed(1))];
    values.forEach((value)=>{const td=document.createElement('td');td.textContent=String(value);tr.append(td);});fragment.append(tr);
  });
  ui.detections.append(fragment);
}
function render(detections) {
  drawSource();
  const canvas=ui['result-canvas'],context=canvas.getContext('2d',{alpha:false});
  const lineWidth=Math.max(2,Math.round(Math.min(canvas.width,canvas.height)/400)),font=Math.max(12,Math.round(Math.min(canvas.width,canvas.height)/45));
  context.lineWidth=lineWidth;context.font=`600 ${font}px system-ui,sans-serif`;
  detections.forEach((row,index)=>{
    const color=row.classId===0?'#057b4e':'#ba460a',[x1,y1,x2,y2]=row.box;
    context.strokeStyle=color;context.strokeRect(x1,y1,x2-x1,y2-y1);
    const label=`#${index+1} ${row.className} ${row.confidence.toFixed(2)}`,labelWidth=context.measureText(label).width+8;
    const labelX=Math.max(0,Math.min(x1,canvas.width-labelWidth)),labelY=Math.max(0,Math.min(canvas.height-font-8,y1-font-8));
    context.fillStyle=color;context.fillRect(labelX,labelY,labelWidth,font+8);context.fillStyle='#fff';context.fillText(label,labelX+4,labelY+font+1);
  });
}
async function detect() {
  if(busy||!sourceCanvas)return;
  const ticket=revision,confidence=Number(ui.confidence.value),iou=Number(ui['nms-iou'].value);
  invalidateResult();setBusy(true);setStatus('正在准备本地推理…');
  let inputTensor,outputTensor;
  try {
    const start=performance.now(),{geometry,input}=prepareInput(sourceCanvas),preprocessMs=performance.now()-start;
    const loadingStart=performance.now(),session=await loadSession(),loadMs=performance.now()-loadingStart;
    if(ticket!==revision)return;
    setStatus('模型已就绪，正在浏览器内检测…');
    await new Promise((resolve)=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
    if(ticket!==revision)return;
    inputTensor=new ort.Tensor('float32',input,[1,3,INPUT_SIZE,INPUT_SIZE]);
    const inferenceStart=performance.now(),outputs=await session.run({images:inputTensor});
    outputTensor=outputs[session.outputNames[0]];
    const inferenceMs=performance.now()-inferenceStart,postStart=performance.now();
    const detections=decodeDetections(outputTensor.data,outputTensor.dims,geometry,{confidence,iou,maxDet:300});
    if(ticket!==revision)return;
    render(detections);showRows(detections);rows=detections;
    resultParameters={confidence,iou,width:geometry.width,height:geometry.height,modelSha256:modelInfo.sha256};
    const postMs=performance.now()-postStart,counts=[0,0];detections.forEach((row)=>counts[row.classId]++);
    ui['result-tag'].textContent='P2 · 普通检测';
    ui.summary.textContent=`检出 ${detections.length} 个头部：${CLASS_LABELS[0]} ${counts[0]} 个，${CLASS_LABELS[1]} ${counts[1]} 个。`;
    ui.timing.textContent=`本次参数：conf ${confidence.toFixed(2)} / NMS IoU ${iou.toFixed(2)} / max_det 300。预处理 ${Math.round(preprocessMs)} ms，模型就绪 ${Math.round(loadMs)} ms，推理 ${Math.round(inferenceMs)} ms，后处理与绘图 ${Math.round(postMs)} ms。模型就绪时间首次包含下载和初始化；非GPU基准。`;
    setStatus('检测完成。图片与结果仅保留在当前页面；可下载或清除。');
  } catch(error) {
    if(ticket===revision){if(sourceCanvas)drawSource();invalidateResult();setStatus(error instanceof Error?error.message:'推理失败，请刷新或换一张较小的图片重试。',true);}
  } finally {inputTensor?.dispose?.();outputTensor?.dispose?.();setBusy(false);}
}
function saveBlob(blob,name) {
  const href=URL.createObjectURL(blob),a=document.createElement('a');a.href=href;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(href),1000);
}
ui['download-png'].addEventListener('click',()=>{
  if(rows===null||busy)return;ui['result-canvas'].toBlob((blob)=>{if(blob)saveBlob(blob,'helmet-detection.png');},'image/png');
});
ui['download-csv'].addEventListener('click',()=>{
  if(rows===null||busy||!resultParameters)return;
  const p=resultParameters,csv=['id,class_id,class_name,confidence,x1,y1,x2,y2,input_width,input_height,conf_threshold,nms_iou,model_sha256',...rows.map((row,index)=>[index+1,row.classId,row.className,row.confidence,...row.box,p.width,p.height,p.confidence,p.iou,p.modelSha256].join(','))].join('\r\n');
  saveBlob(new Blob(['\uFEFF',csv],{type:'text/csv;charset=utf-8'}),'helmet-detections.csv');
});
ui['image-file'].addEventListener('change',selectImage);ui.detect.addEventListener('click',detect);ui.clear.addEventListener('click',clearImage);
for(const [input,output] of [['confidence','confidence-value'],['nms-iou','iou-value']])ui[input].addEventListener('input',()=>{
  ui[output].value=Number(ui[input].value).toFixed(2);
  if(rows!==null)setStatus('参数已改变；画面和下载仍是上一次结果，请点击“开始检测”重新计算。');
});
