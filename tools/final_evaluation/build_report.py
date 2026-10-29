"""Build public numeric evidence and local HTML from frozen evaluation outputs."""
from pathlib import Path
import csv
import html
import json
import os
import re
import shutil
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'error_analysis/final_five_model_evaluation_2026-10-28'
REPORT=ROOT/'reports/final_evaluation'
REPORT.mkdir(exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(SOURCE/'matplotlib'))
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p,o):p.write_text(json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
p=read(SOURCE/'protocol.json');metrics=read(SOURCE/'metrics.json');benchmark=read(SOURCE/'benchmark.json');uncertainty=read(SOURCE/'paired_uncertainty.json');verification=read(SOURCE/'verification.json')
assert verification['status']=='passed'
records=p['models'];ids=[r['id'] for r in records]
standard,p2,cbam=metrics[ids[0]],metrics[ids[2]],metrics[ids[3]]
pc=lambda x:'—' if x is None else f'{x*100:.2f}%'
pp=lambda x:f'{x*100:+.3f}'

def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(x) for x in r)+'|' for r in rows])
def csvout(name,rows):
    with (REPORT/name).open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

for name in ['protocol.json','metrics.json','paired_uncertainty.json','verification.json','prediction_verification.json']:
    shutil.copy2(SOURCE/name,REPORT/name)
save(REPORT/'benchmark_summary.json',{k:v for k,v in benchmark.items() if k!='raw'})
save(REPORT/'benchmark_raw.json',benchmark['raw'])
history=ROOT/'05_新增两组实验对比_20261028/五组逐轮原始指标与损失.csv'
if history.exists():shutil.copy2(history,REPORT/'training_history.csv')
else:assert (REPORT/'training_history.csv').is_file()

summary=[];area=[];tiny=[];classes=[];per_image=[]
for record in records:
    mid=record['id'];m=metrics[mid];fixed=m['fixed'];speed=benchmark['results'][mid]
    summary.append({'model':record['label'],'validation_selected_epoch':record['best_epoch'],'parameters':record['checkpoint_structure']['parameters'],**fixed['overall'],
                    'coco_bbox_AP300':m['coco_bbox_ap300']['all']['AP50_95'],'coco_bbox_APsmall300':m['coco_bbox_ap300']['small']['AP50_95'],
                    'core_mean_ms':speed['core_ms']['mean'],'core_median_ms':speed['core_ms']['median'],'core_p95_ms':speed['core_ms']['p95'],
                    'rendered_mean_ms':speed['rendered_ms']['mean'],'rendered_p95_ms':speed['rendered_ms']['p95']})
    for size in ['small','medium','large']:
        for cls in ['all','0','1']:
            ap=m['coco_bbox_ap300'][size] if cls=='all' else m['coco_bbox_ap300'][size]['classes'][cls]
            area.append({'model':record['label'],'original_area_group':size,'category':cls,'AP50_95':ap['AP50_95'],'AP50':ap['AP50'],'AR50_95':ap['AR50_95'],**fixed['area'][size][cls]})
    for size,x in fixed['helmet_short'].items():tiny.append({'model':record['label'],'short_side_group_scaled640':size,**x})
    for cid,x in fixed['classes'].items():classes.append({'model':record['label'],'class_id':cid,**x})
detail=read(SOURCE/'per_image_details.json')
for record in records:
    for im,d in zip(detail['images'],detail['models'][record['id']]):
        per_image.append({'image':im['file_name'],'group_id':im['group_id'],'model':record['label'],**d['overall'],'hat_tp':d['classes']['0']['tp'],'hat_fp':d['classes']['0']['fp'],'hat_fn':d['classes']['0']['fn']})
csvout('summary.csv',summary);csvout('area_metrics.csv',area);csvout('tiny_helmet_recall.csv',tiny);csvout('class_fixed_metrics.csv',classes);csvout('per_image_counts.csv',per_image)
csvout('benchmark_raw.csv',[{k:v for k,v in r.items() if k!='class_counts'} for r in benchmark['raw']])
ground=read(SOURCE/'ground_truth_coco.json')
support={'original_area_disjoint_counts':{s:metrics[ids[0]]['fixed']['area'][s]['all']['gt'] for s in ['small','medium','large']},
         'exact_32_squared_boundary':sum(a['area']==1024 for a in ground['annotations']),
         'exact_96_squared_boundary':sum(a['area']==9216 for a in ground['annotations']),
         'coco_default_inclusive_counts':{s:sum(lo<=a['area']<=hi for a in ground['annotations']) for s,lo,hi in [('small',0,1024),('medium',1024,9216),('large',9216,1e10)]}}
save(REPORT/'size_support.json',support)

# Scientific figures use actual computed values; no illustrative performance numbers.
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
labels=['Standard','Std + CBAM@P3','P2','P2 + CBAM@P3','P2 + CBAM@P2']
colors=['#7791af','#ad8d66','#2260a2','#b45153','#8971a1']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':140})
fig,ax=plt.subplots(figsize=(11,4.8));x=np.arange(3);width=.15
for i,mid in enumerate(ids):ax.bar(x+(i-2)*width,[metrics[mid]['coco_bbox_ap300'][s]['AP50_95']*100 for s in ['small','medium','large']],width,label=labels[i],color=colors[i])
ax.set_xticks(x,['Small','Medium','Large']);ax.set_ylabel('COCO bbox AP@[.50:.95], maxDets=300 (%)');ax.set_ylim(0,85);ax.legend(ncol=3,loc='upper left');ax.grid(axis='y',alpha=.18);fig.tight_layout();fig.savefig(REPORT/'area_ap.png');fig.savefig(REPORT/'area_ap.svg');plt.close(fig)
fig,ax=plt.subplots(figsize=(10.4,4.7));x=np.arange(4)
for i,mid in enumerate(ids):ax.bar(x+(i-2)*width,[metrics[mid]['fixed']['helmet_short'][s]['recall']*100 for s in ['lt8','8to16','16to32','ge32']],width,label=labels[i],color=colors[i])
ax.set_xticks(x,['<8 px (n=139)','8–16 px (n=78)','16–32 px (n=135)','>=32 px (n=594)']);ax.set_ylabel('Helmet recall @ conf=.25, IoU=.50 (%)');ax.set_xlabel('GT short side after long image side scaled to 640');ax.set_ylim(0,110);ax.legend(ncol=3,loc='upper left');ax.grid(axis='y',alpha=.18);fig.tight_layout();fig.savefig(REPORT/'tiny_helmet_recall.png');fig.savefig(REPORT/'tiny_helmet_recall.svg');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.5))
for i,mid in enumerate(ids):
    speed=benchmark['results'][mid]['core_ms'];f1=metrics[mid]['fixed']['overall']['f1']*100
    axes[0].scatter(speed['mean'],f1,color=colors[i],s=75,label=labels[i])
    axes[0].annotate(['Std','Std+CBAM','P2','P2+P3 CBAM','P2+P2 CBAM'][i],(speed['mean'],f1),xytext=([5,5,5,5,5][i],[7,-15,5,-13,6][i]),textcoords='offset points',fontsize=8)
    axes[1].plot(range(1,6),speed['round_means'],'o-',color=colors[i],label=labels[i])
axes[0].set_xlabel('Read/decode + inference + NMS + counts (ms/image)');axes[0].set_ylabel('Micro F1 @ conf=.25 (%)');axes[0].set_xlim(19,40);axes[0].set_ylim(86.2,87.6);axes[0].grid(alpha=.2)
axes[1].set_xlabel('Rotated-order round');axes[1].set_ylabel('Mean core latency, 32 validation images (ms)');axes[1].set_xticks(range(1,6));axes[1].grid(alpha=.2);axes[1].legend(fontsize=8);fig.tight_layout();fig.savefig(REPORT/'latency_tradeoff.png');fig.savefig(REPORT/'latency_tradeoff.svg');plt.close(fig)

sf=standard['fixed']['overall'];pf=p2['fixed']['overall'];cf=cbam['fixed']['overall']
hat=p2['fixed']['classes']['0'];chat=cbam['fixed']['classes']['0']
decision={'recommended_model':'03_P2','recommended_confidence':.25,'no_runtime_registry_change':True,
          'reason':'P2 retains the best validation AP and highest micro F1 at the common .25 operating point; P3-CBAM recovers20 more targets but produces142 more unmatched predictions, including45 more helmet predictions. CPU speeds are close; do not claim a meaningful speed/stability advantage.',
          'p3_cbam_minus_p2':{'tp':cf['tp']-pf['tp'],'fp':cf['fp']-pf['fp'],'fn':cf['fn']-pf['fn'],'micro_f1_pp':(cf['f1']-pf['f1'])*100,'helmet_fp':chat['fp']-hat['fp'],'helmet_tp':chat['tp']-hat['tp'],
              'core_mean_ratio':benchmark['results'][ids[3]]['core_ms']['mean']/benchmark['results'][ids[2]]['core_ms']['mean']},
          'limits':['Retrospective recommendation on previously used data, not a new external blind decision.','Fixedconf.25 is a common diagnostic point, not evidence of best achievable calibration for every model.','No repeated training seeds or independent hardware benchmark; runtime variation is not training stability.','Recommendation retained as documentation; application code and thresholds are unchanged.']}
save(REPORT/'model_selection.json',decision)
original_path=ROOT/'06_五组测试集评估_20261028/测试集汇总.json'
original=read(original_path if original_path.exists() else REPORT/'original_gpu_test_metrics.json')
validation_table=table(['模型','最佳轮次','参数量','验证mAP50–95','原GPU测试mAP50–95'],[[r['label'],r['best_epoch'],r['checkpoint_structure']['parameters'],pc(r['validation_map50_95']),pc(t['map50_95'])] for r,t in zip(records,original)])
area_table=table(['模型','APsmall','APmedium','APlarge','Rsmall@.25','Rmedium@.25','Rlarge@.25'],[[r['label'],*[pc(metrics[r['id']]['coco_bbox_ap300'][s]['AP50_95']) for s in ['small','medium','large']],*[pc(metrics[r['id']]['fixed']['area'][s]['all']['recall']) for s in ['small','medium','large']]] for r in records])
hat_area_table=table(['模型','小安全帽AP50–95','小安全帽Recall@.25','匹配/标注'],[[r['label'],pc(metrics[r['id']]['coco_bbox_ap300']['small']['classes']['0']['AP50_95']),pc(metrics[r['id']]['fixed']['area']['small']['0']['recall']),f"{metrics[r['id']]['fixed']['area']['small']['0']['tp']}/220"] for r in records])
tiny_table=table(['模型','<8px /139','8–16px /78','16–32px /135','≥32px /594','<16px累计 /217'],[[r['label'],*[f"{metrics[r['id']]['fixed']['helmet_short'][s]['tp']} ({pc(metrics[r['id']]['fixed']['helmet_short'][s]['recall'])})" for s in ['lt8','8to16','16to32','ge32','lt16']]] for r in records])
fixed_table=table(['模型','TP','FP未匹配预测','FN漏检','Precision','Recall','F1'],[[r['label'],*[metrics[r['id']]['fixed']['overall'][k] for k in ['tp','fp','fn']],*[pc(metrics[r['id']]['fixed']['overall'][k]) for k in ['precision','recall','f1']]] for r in records])
helmet_table=table(['模型','戴帽TP','戴帽FP','戴帽FN','戴帽Precision','戴帽Recall','戴帽F1'],[[r['label'],*[metrics[r['id']]['fixed']['classes']['0'][k] for k in ['tp','fp','fn']],*[pc(metrics[r['id']]['fixed']['classes']['0'][k]) for k in ['precision','recall','f1']]] for r in records])
speed_table=table(['模型','核心均值ms','核心中位ms','核心P95ms','含绘图编码均值ms','含绘图编码P95ms','轮均值CV'],[[r['label'],*[f"{benchmark['results'][r['id']]['core_ms'][k]:.2f}" for k in ['mean','median','p95']],*[f"{benchmark['results'][r['id']]['rendered_ms'][k]:.2f}" for k in ['mean','p95']],pc(benchmark['results'][r['id']]['core_ms']['round_mean_cv'])] for r in records])
interval=uncertainty['comparisons']['04_P2_P3_CBAM_minus_03_P2']['micro_f1_delta95']
apgain=(p2['coco_bbox_ap300']['small']['AP50_95']-standard['coco_bbox_ap300']['small']['AP50_95'])*100
hatgain=(p2['coco_bbox_ap300']['small']['classes']['0']['AP50_95']-standard['coco_bbox_ap300']['small']['classes']['0']['AP50_95'])*100

lines=['# 五组结构实验收尾：尺寸、固定工作点与完整本地推理耗时','',
'2026-10-28。五组既有best.pt，无训练、无测试阈值搜索、无检查点重选。', '',
'**结论：保留P2、conf=0.25作为当前单模型推荐。** P2在本批小目标AP与固定工作点F1上优于标准版。P2＋P3 CBAM有局部召回收益和原GPU测试AP的小幅收益，但固定工作点新增未匹配预测较多，尚不值得替换P2。CPU速度接近，不把微小耗时差或抖动解释为更稳定的网络。', '',
'## 1. 完整保留五组训练与原测试结果','',validation_table,'',
'原GPU表使用Ultralytics8.4.164、RTX3060、FP16、batch4；最佳轮次先由验证集选定。以下新尺寸AP使用同一批模型的CPU FP32、batch1预测及COCOeval，指标实现和数值精度不同，不能拿两种口径的总AP直接计算提升。', '',
'## 2. 按原图面积统计AP与固定阈值召回','',
'AP使用官方pycocotools2.0.11的COCO bbox评价：IoU=.50:.05:.95、101点插值、maxDets=300，候选conf=.001。小/中/大按**原图标注框面积**分层；与32×32、96×96比较，不按整张图片尺寸，也不是缩放后的框尺寸。沿用[COCOeval默认面积与忽略规则](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)，名称为COCO式bbox AP@300，而非默认AP@100。', '',
area_table,'',
'固定Recall按所有GT先作同类一对一匹配，再按GT尺寸分组，互斥计数小2072、中1908、大895，总4875。此处用半开区间<32²、[32²,96²)、≥96²；COCO AP官方区间上下界包含端点。本数据端点计数见JSON核验，不把两种区间规则隐去。未匹配预测没有GT尺寸，因此不编造“按真实尺寸分组的FP或Precision”。COCO AP会按其官方忽略规则处理区间外GT和预测。', '',
'本批恰好落在32²/96²面积端点的标注均为0，因此两种区间计数在这里一致，见[尺寸分母核验](size_support.json)。候选conf=.001时达到300框上限的图数依模型为9/7/17/24/23，可能截断低分候选，完整保留[上限记录](prediction_verification.json)；conf=.25没有图达到上限，单图GT最多124个。', '',
f'P2相对标准版，APsmall增加 **{apgain:.3f}个百分点**。这支持“本次小目标整体受益”，不意味着所有微小目标都受益。','',
'![原图面积分组AP](area_ap.png)','',
'### 安全帽类别单独看','',hat_area_table,'',
f'P2的小安全帽AP提高 **{hatgain:.3f}个百分点**；固定0.25下小安全帽匹配126→130，增加4个。这个改进量不大，不能说已经解决远距离安全帽问题。','',
'### 输入尺度下的微小安全帽','',
'定义：标注框短边 ×640/原图最长边；这是微小目标代理，没有真实拍摄距离标签。小于16px累计组与前两组重叠，不和其他组再次相加。','',tiny_table,'',
'小于8px的139个安全帽：标准版匹配1个、P2为0个，P2＋P3 CBAM为4个。极小目标仍是明显短板。不能只挑8–16px组的改善、忽略更小组的负结果。','',
'![微小安全帽召回](tiny_helmet_recall.png)','',
'## 3. 同一conf=0.25的误检、漏检与F1','',
'五组均为640、rect=True、batch1、CPU FP32、NMS IoU=.7、max_det300。以score降序、同类别一对一IoU≥.5匹配；这里是微平均P/R/F1，**不是库在各自最佳F1处报告的P/R，也不是mAP**。FP指相对现有标签的未匹配预测，可能包含漏标，未经人工确认不一律称真实误检。','',fixed_table,'',helmet_table,'',
f'P2＋P3 CBAM相对P2：TP +20、FN −20、FP +142，F1 **{pp(cf["f1"]-pf["f1"])}个百分点**；安全帽仅多匹配2个，却增加45个未匹配安全帽预测。它可以多找回一些目标，但当前工作点的误检代价更大。','',
f'P2＋P3 CBAM−P2的F1差值，2000次配对风险组重采样95%区间为 **[{interval[0]*100:.3f}, {interval[1]*100:.3f}]个百分点**。本批508张测试图恰好对应508个检索组，数值上等价于按图重采样；未知同源关系可能仍存在。它只表示这批图、这些固定权重的样本不确定性，不是训练多随机种子显著性。','',
'## 4. 完整本地图片处理耗时','',
f"环境：{p['environment']['platform']} / {p['environment']['machine']}，torch{p['environment']['torch']}，CPU FP32，实际torch8线程、interop1、OpenCV1线程。32张按文件名均匀选取的验证图，每模型预热5次，5轮轮换模型顺序，共每模型160次、五模型800次。每轮图像顺序一致并固定随机种子。",'',speed_table,'',
'核心：读图/解码→RGB→预处理→网络→NMS→框取回与计数。含绘图编码：核心再加画框和内存PNG编码（压缩3）。均不含权重加载、网页/网络、文件输出；文件缓存已热。预热针对predict，绘图第一次使用的初始化仍可能进入第一轮。原图分辨率、目标数量和后台负载会影响分布；保留逐次时间和五轮均值。', '',
f"P2与P2＋P3 CBAM核心均值为{benchmark['results'][ids[2]]['core_ms']['mean']:.2f}/{benchmark['results'][ids[3]]['core_ms']['mean']:.2f}ms，只差约{(decision['p3_cbam_minus_p2']['core_mean_ratio']-1)*100:.1f}%。速度大致相当，不据此宣称P2显著更快或更稳定。两组含绘图编码的轮均值抖动也很小；这与训练稳定性是两回事。",'',
'这是Mac本地处理路径的计时，不是Windows CUDA实测，不是浏览器端到端响应，也不能直接换算成系统实时吞吐率。原GPU评估器的forward时间仅保留作历史参考。模型构造计时受导入及文件缓存先后影响，原始记录保留但不据此排冷启动速度名次。', '',
'![固定工作点F1与CPU耗时](latency_tradeoff.png)','',
'## 5. 最终选择与负结果','',
'- **当前推荐P2单模型，conf=0.25。** 它的验证mAP50–95最高，本轮固定工作点总体F1也最高；小目标和小安全帽AP优于标准版。',
'- 标准版更快，若设备预算更严格可作为轻量备选；不能省略P2约1.5倍核心CPU耗时的代价。',
'- P2＋P3 CBAM保留为结构实验：原GPU测试mAP50–95高0.228个百分点，但验证低0.339个百分点、固定阈值F1下降，微小安全帽仅补回少数。',
'- 标准版＋P3 CBAM与旧P2位置CBAM也完整保留；不删负结果，不把换接入位置后的实验合并成一个成功故事。',
'- 这是文档中的推荐；应用已使用原P2登记，本次没有自动替换权重、改阈值或改检测功能。', '',
'## 6. 实验限制','',
'1. 五组均seed=0、单次训练。50个epoch不是50次独立重复；配对图像bootstrap也不能代替多种子。',
'2. 两个新P3 CBAM均64通道、678参数且初始注意力参数一致；标准版位于22接15，P2版位于28接21。P2配对历史非CBAM初始化核验一致，标准版配对则不完全一致，因此不宣称严格控制全部初始变量的二因素交互证据。',
'3. 旧P2分支CBAM为32通道、260参数，与P3位置组同时改变位置和规模，不能称纯位置单因素实验。',
'4. 测试集已被多次查看和使用；此前五组总分在本轮前已知。新协议在本轮分层、固定点与计时前冻结，不称最初全盲预注册。没有在测试上搜索阈值，但模型推荐仍属于事后综合判断。',
'5. 已检索跨分区关联隔离不等于排除所有未知场景关系。PartA教室来源全部训练用，现有测试不证明新教室泛化。',
'6. 15张外部图没有完整真值；difficult按普通目标处理；没有实测目标距离，因此“微小头部”不等同于“远距离”。',
'7. 固定0.25比较的是共同工作点，不代表每个模型经过独立验证集校准后的最佳性能；当前不为扭转测试结论继续搜索参数。',
'8. CPU速度只在本机小批验证图测得，后台负载未控制；不能推断CUDA效率、实际现场响应或多种子训练稳定性。', '',
'## 7. 复现与证据','',
'先准备协议清单对应的数据和五组完整训练目录，权重SHA256见[协议](protocol.json)。这些新训练权重仍只在本地，不把历史公开权重当作本轮模型。','',
'```bash\npython -m pip install --target error_analysis/final_five_model_evaluation_2026-10-28/metrics_dependencies --no-deps pycocotools==2.0.11\npython -B tools/final_evaluation/tests.py\npython -B tools/final_evaluation/evaluate.py prepare\npython -B tools/final_evaluation/evaluate.py infer\npython -B tools/final_evaluation/evaluate.py score\npython -B tools/final_evaluation/evaluate.py benchmark\npython -B tools/final_evaluation/evaluate.py verify\npython -B tools/final_evaluation/build_report.py\n```','',
'有已冻结结果的目录不会覆盖；如复现新一轮，先使用独立项目副本/新的输出目录，不删改本轮证据。COCO工具隔离安装，不升级现有训练环境。','',
'[五组数值](metrics.json) · [汇总CSV](summary.csv) · [尺寸AP与Recall](area_metrics.csv) · [微小安全帽](tiny_helmet_recall.csv) · [逐图计数](per_image_counts.csv) · [计时摘要](benchmark_summary.json) · [800次原始计时](benchmark_raw.csv) · [置信区间](paired_uncertainty.json) · [模型选择](model_selection.json) · [核验](verification.json) · [250行训练记录](training_history.csv) · [五组原始训练配置与完成登记](training/) · [源码](../../tools/final_evaluation/evaluate.py)','',
'已核验全部7507张图片/标签、五组训练参数/CSV/权重和评价源码；10个合成测试覆盖空预测、错类、重复框、尺寸忽略、区间边界、完美预测与微平均计数。无优化器步骤。原训练、原GPU报告和模型登记保留。']
(REPORT/'README.md').write_text('\n'.join(lines)+'\n')

# Basic safe Markdown rendering is enough for the local report: headings, tables,
# paragraphs, lists, code blocks, local images, and Markdown links only.
def inline(text):
    text=html.escape(text)
    text=re.sub(r'!\[([^\]]*)\]\(([^)]+)\)',lambda m:f'<img alt="{m[1]}" src="{quote(m[2],safe="/._-")}">',text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:f'<a href="{quote(m[2],safe=":/._-#")}">{m[1]}</a>',text)
    text=re.sub(r'\*\*(.+?)\*\*',r'<strong>\1</strong>',text)
    return text
blocks=[];in_table=False;in_code=False
for line in '\n'.join(lines).splitlines():
    if line.startswith('```'):
        blocks.append('</code></pre>' if in_code else '<pre><code>');in_code=not in_code;continue
    if in_code:blocks.append(html.escape(line)+'\n');continue
    if line.startswith('|'):
        if not in_table:blocks.append('<div class="scroll"><table>');in_table=True
        if re.fullmatch(r'[|:\- ]+',line):continue
        blocks.append('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in line.strip('|').split('|'))+'</tr>');continue
    if in_table:blocks.append('</table></div>');in_table=False
    if line.startswith('#'):
        n=len(line)-len(line.lstrip('#'));blocks.append(f'<h{n}>{inline(line[n:].strip())}</h{n}>')
    elif line:blocks.append('<p>'+inline(line)+'</p>')
if in_table:blocks.append('</table></div>')
(REPORT/'index.html').write_text('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>五组实验最终评价</title><style>body{font:16px/1.75 -apple-system,"PingFang SC",sans-serif;margin:0;color:#17283d;background:#f4f6f9}main{max-width:1180px;margin:24px auto;padding:28px;background:white;border-radius:12px}h1{font-size:29px}h2{margin-top:35px;border-top:1px solid #dde4ed;padding-top:22px}a{color:#245eaa}table{width:100%;border-collapse:collapse;font-size:14px}td{border-bottom:1px solid #dde4ed;padding:10px;white-space:nowrap}tr:first-child{background:#edf3fa;font-weight:600}.scroll{overflow:auto}img{max-width:100%;height:auto}pre{overflow:auto;background:#f0f3f7;padding:16px}strong{color:#124d8a}@media(max-width:760px){main{margin:0;padding:18px}}</style></head><body><main>'+''.join(blocks)+'</main></body></html>')

# Add current links to existing navigation while preserving the older GPU table.
for page in [ROOT/'02_实验结果/00_五组实验总览.html',ROOT/'reports/completed_experiments_2026-10-28/index.html']:
    if not page.exists():continue
    s=page.read_text();url=quote(os.path.relpath(REPORT/'index.html',page.parent),safe='/._-')
    card=f'<section class="notice"><strong>三项收尾评价已完成：尺寸AP、同阈值误检漏检、完整本地耗时。</strong><br><a class="button" href="{url}">打开最终评价与模型选择</a><p>当前推荐P2，conf=0.25；原GPU表与新尺寸/固定工作点评价分开报告。</p></section>'
    s=s.replace('<section class="card"><h2>同一协议下的测试对比</h2>',card+'<section class="card"><h2>原GPU协议下的测试对比</h2>')
    s=s.replace('现有表格尚未给出尺寸分组AP。','本页是原GPU汇总；新增尺寸AP请看上方最终评价。')
    s=re.sub(r'<section class="card"><h2>接下来</h2>.*?</section>','<section class="card"><h2>当前状态</h2><p>五组训练、尺寸分层、固定工作点与CPU计时已完成，当前推荐P2。下一步是新场景评价和成果交付；不安排继续堆模块或重复训练。</p></section>',s,flags=re.S)
    page.write_text(s)
page=ROOT/'_项目导航.html';s=page.read_text() if page.exists() else '';url=quote(os.path.relpath(REPORT/'index.html',page.parent),safe='/._-')
s=s.replace('<div class="notice"><strong>当前进度：两组P3 CBAM训练、五组测试评价和回传权重核验均已完成。</strong>',f'<div class="notice"><strong>当前进度：五组训练与三项收尾评价全部完成，当前推荐P2。</strong><br><a class="button" href="{url}">查看最终尺寸、固定阈值和速度报告</a>')
if page.exists():page.write_text(s)
link=ROOT/'02_实验结果/01_最新实验结果/04_尺寸指标_固定阈值与速度'
if link.parent.exists() and not link.exists():link.symlink_to(os.path.relpath(REPORT,link.parent),target_is_directory=True)
code=ROOT/'04_代码和项目说明/五组最终评价代码'
if code.parent.exists() and not code.exists():code.symlink_to(os.path.relpath(ROOT/'tools/final_evaluation',code.parent),target_is_directory=True)
print('REPORT_READY',REPORT,flush=True)
