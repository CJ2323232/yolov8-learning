"""Update project-facing documentation from the measured five-model evidence."""
from pathlib import Path
import json
import re

ROOT=Path(__file__).resolve().parents[2]
R=ROOT/'reports/final_evaluation'
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
p=read(R/'protocol.json');metrics=read(R/'metrics.json');speed=read(R/'benchmark_summary.json')['results'];decision=read(R/'model_selection.json')
records=p['models'];ids=[r['id'] for r in records];pc=lambda x:f'{100*x:.2f}%'
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(x) for x in r)+'|' for r in rows])
original_path=ROOT/'06_五组测试集评估_20261028/测试集汇总.json'
original=read(original_path if original_path.exists() else R/'original_gpu_test_metrics.json')
model_table=table(['模型','验证mAP50–95','原GPU测试mAP50–95','参数量'],[[r['label'],pc(r['validation_map50_95']),pc(t['map50_95']),r['checkpoint_structure']['parameters']] for r,t in zip(records,original)])
fixed_table=table(['模型','FP','FN','Precision','Recall','F1','核心CPU ms/图'],[[r['label'],metrics[r['id']]['fixed']['overall']['fp'],metrics[r['id']]['fixed']['overall']['fn'],*[pc(metrics[r['id']]['fixed']['overall'][k]) for k in ['precision','recall','f1']],f"{speed[r['id']]['core_ms']['mean']:.2f}"] for r in records])
area_table=table(['模型','小目标AP50–95','小安全帽AP50–95','<16px安全帽匹配 /217'],[[r['label'],pc(metrics[r['id']]['coco_bbox_ap300']['small']['AP50_95']),pc(metrics[r['id']]['coco_bbox_ap300']['small']['classes']['0']['AP50_95']),metrics[r['id']]['fixed']['helmet_short']['lt16']['tp']] for r in records])
cn='''<div align="center">

# Helmet Detection Lab

**安全帽佩戴检测：数据治理、小目标结构对照与可解释的模型选择**

YOLOv8n · P2 · CBAM · SAHI · NMS / WBF · Gradio

[English](README_EN.md) · [快速体验](#快速体验) · [五组结果](#五组结构实验) · [最终评价](reports/final_evaluation/README.md) · [文档](docs/README.md)

</div>

![真实本地安全帽检测演示](assets/gradio_result.png)

*历史界面的真实截图，仅作演示。现有应用支持标准版、P2、NMS/WBF融合；新CBAM权重未自动替换应用默认。*

基于 **Ultralytics YOLOv8n 和 SHWD**，检测戴帽与未戴帽的**头部**。项目围绕三个问题展开：如何控制关联图片泄漏风险？P2对小目标是否有益？注意力或融合的收益是否值得误检和速度代价？

> **2026-10-28：五组50轮训练、统一测试、尺寸AP、固定工作点与完整本地图片处理计时均已完成。当前推荐P2单模型、conf=0.25。** P2＋P3 CBAM在原GPU测试AP上略高，但固定0.25时多出142个未匹配预测，F1下降1.023个百分点。全部五组、包括负结果均保留。
>
> **下载范围：** 公开下载仍为历史演示权重；本轮五组best.pt在本地完成校验，尚未发布下载。历史权重不能复现下表成绩。

## 值得看的工作

| 工作 | 可验证结果 | 证据 |
|---|---|---|
| 数据治理 | 7507张有效图；已检索跨分区候选对20466→0，491张调整分区 | [治理与限制](reports/scene_isolation/README.md) |
| P2与CBAM结构实验 | 五组50轮；两个新CBAM均接P3/64通道，保留旧P2位置组 | [五组与初始化记录](reports/final_evaluation/README.md) |
| 小目标评价 | 标准版→P2：小目标AP41.18%→43.12%，小安全帽AP37.47%→40.03% | [面积与微小头部](reports/final_evaluation/area_metrics.csv) |
| 系统取舍 | 同conf比较FP/FN/F1，5轮轮换计时；依据证据保留P2 | [最终选择](reports/final_evaluation/model_selection.json) |
| 应用与探索 | Gradio、权重校验与复用；SAHI、NMS/WBF对比保留 | [应用](docs/FUSION.md) · [历史融合](reports/fusion_evaluation/README.md) |

小目标AP为本轮COCO式bbox AP@300、原图面积<32²、IoU=.50:.95；不是所有小目标都改善。小于8px的139个安全帽中，P2匹配0个、标准版1个，极小目标仍是短板。候选对不是确认泄漏数，单种子结果也不证明稳定因果收益。

## 快速体验

无需训练，先使用已公开的历史演示权重：

```bash
git clone https://github.com/CJ2323232/yolov8-learning.git
cd yolov8-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gradio.txt
python download_weights.py
python app.py
```

Windows创建环境后使用 `.venv\\Scripts\\activate.bat` 激活，再执行同样的安装和启动命令。

上传**没有框和文字的原图**，选择“自选／历史演示模型”开始体验。标准版/P2及融合需要对应的新权重，不能用历史权重改名代替。见[模型与权重指纹](models/README.md)、[应用配置](docs/FUSION.md)、[复现范围](docs/REPRODUCIBILITY.md)。

## 五组结构实验

'''+model_table+'''

原GPU测试：同一508图、4875框，Ultralytics8.4.164、640、batch4、RTX3060、FP16、候选conf=.001；每组先用511张验证图选择best。新P3 CBAM均仅作用于Detect前的P3分支，新增678参数；旧P2分支CBAM为260参数。

训练沿用640、batch2、AdamW、lr0=.001667、lrf=.01、nbs64、seed0、50轮及原增强策略。**标准版配对的历史非预训练层初始化不完全一致，P2配对核验一致；不能称严格控制全部变量的二因素交互实验。** [详细条件](reports/final_evaluation/protocol.json)。

## 小目标到底是否受益

'''+area_table+'''

表中AP使用本轮统一CPU FP32预测与官方COCOeval，原图面积分组、maxDets300。<16px使用缩放到640后的GT短边，固定conf=.25、匹配IoU=.5，属于微小头部代理，不是实际距离。它与原图面积分组是不同定义。

![小中大目标AP](reports/final_evaluation/area_ap.png)

P2对本批小目标有改善，但极小目标仍难检；P2＋P3 CBAM在微小安全帽累计组多匹配3个，却没有提高小安全帽AP。[完整小/中/大AP、Recall、类别与分母](reports/final_evaluation/README.md)。

## 同一工作点的系统表现

所有模型统一 **conf=.25、同类一对一IoU≥.5**；这是微平均P/R/F1，不是各模型最佳F1处的P/R，也不是mAP。

'''+fixed_table+'''

FP为相对已有标签的未匹配预测，可能含漏标。P2＋P3 CBAM相对P2少漏20个、未匹配预测多142个；其中戴帽头部多匹配2个、未匹配预测多45个。它的原GPU测试mAP50–95高0.228个百分点，不等于当前系统工作点更好。

核心CPU耗时包含读图、解码、预处理、网络、NMS、框取回及计数。本机Mac、FP32、batch1、实际8线程；32张验证图×5轮、模型顺序轮换、每模型160次。P2/P2＋P3 CBAM约32.80/33.36ms，含画框和PNG编码约69.49/69.95ms，**速度接近，不声称稳定速度优势**；不含加载、网页/网络或文件输出，不代表Windows CUDA速度。

![固定阈值F1与CPU耗时](reports/final_evaluation/latency_tradeoff.png)

**最终推荐P2：** 验证mAP最高、固定工作点F1最高、小目标AP优于标准版。标准版速度更快，可作资源受限备选。两种CBAM位置实验保留作消融与错误取舍证据；不删负结果，也不要求最终模型必须包含注意力。[选择依据与原始计时](reports/final_evaluation/README.md)。

## 数据与执行流程

```mermaid
flowchart LR
    A[SHWD标注与图片] --> B[校验去重与关联隔离]
    B --> C[固定6488/511/508划分]
    C --> D[五组50轮训练]
    D --> E[验证集选best]
    E --> F[已有测试复评: AP/尺寸/固定工作点]
    F --> G[精度与CPU效率取舍]
    G --> H[P2默认推荐与Gradio演示]
```

`0: hat`是戴安全帽的头部；`1: person`是未戴帽头部，**不是整个人体**。PartA教室图片全部在训练集，现有验证/测试不证明新教室泛化。15张外部图仅定性展示。

## 复现与结论范围

- 五组均单种子、单次运行；不把50轮当50次重复，不把图像bootstrap当多种子证明。
- 测试集历史上多次使用，五组总分在本轮分层评价前已知；本轮先冻结规则再统计，仍不是新的外部盲测。
- 标准版配对初始化限制、旧/新CBAM位置与通道规模差异均保留。
- 共享工作点不代表每模型校准后的最优成绩；没有为了改变测试排名继续搜索阈值。
- CPU耗时受图像大小、目标数、首次绘图和后台负载影响；运行抖动不等于训练稳定性。

[最终实验与复现命令](reports/final_evaluation/README.md) · [五组逐轮数据](reports/final_evaluation/training_history.csv) · [历史SAHI](reports/sahi_evaluation.md) · [历史融合](reports/fusion_evaluation/README.md) · [代码流程](docs/WORKFLOW.md) · [简历与证据](docs/RESUME.md)

后续重点是公开新权重、干净环境复现、新场景评价；更强研究结论再考虑多种子。当前结构实验已收尾，无需继续堆模块。

## 来源与致谢

基础模型：[Ultralytics](https://github.com/ultralytics/ultralytics)；数据：[SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)；切片：[SAHI](https://github.com/obss/sahi)；融合：[WBF](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)；注意力：[CBAM](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)；评价：[COCO API](https://github.com/cocodataset/cocoapi)；界面：[Gradio](https://github.com/gradio-app/gradio)。

项目贡献为数据治理、实验设计、评价与工程集成，不宣称原创网络。第三方代码、模型和图片沿用各自许可与来源，见[第三方说明](THIRD_PARTY_NOTICES.md)。未验证用于实际安全管理。
'''
(ROOT/'README.md').write_text(cn)
english_names=['Standard','Standard + CBAM@P3','P2','P2 + CBAM@P3','P2 + CBAM@P2 (old)']
english_table=table(['Model','Validation mAP50–95','Original GPU test mAP50–95','FP @.25','FN @.25','Micro F1 @.25','Core CPU ms'],[[name,pc(r['validation_map50_95']),pc(t['map50_95']),metrics[r['id']]['fixed']['overall']['fp'],metrics[r['id']]['fixed']['overall']['fn'],pc(metrics[r['id']]['fixed']['overall']['f1']),f"{speed[r['id']]['core_ms']['mean']:.2f}"] for name,r,t in zip(english_names,records,original)])
en='''# Helmet Detection Lab

**Auditable helmet-wearing head detection with five structure experiments and explicit deployment tradeoffs.**

[中文](README.md) · [Full evaluation](reports/final_evaluation/README.md) · [Documentation](docs/README.md)

![Historical local Gradio demo](assets/gradio_result.png)

Built on Ultralytics YOLOv8n and SHWD. Classes are helmeted heads (`hat`) and bare heads (`person`), not full-body pedestrians. The contribution is data governance, experimental design, evaluation and engineering integration, not a novel backbone or attention mechanism.

**2026-10-28: all five 50-epoch runs, test evaluation, area-stratified AP, fixed-confidence errors and CPU timing are complete. Retain P2 at conf=.25 as the current recommendation.** Public downloads still provide only the historical demo checkpoint; the five new checkpoints are verified locally but are not yet published.

## Five-model results

'''+english_table+'''

Original GPU AP uses Ultralytics8.4.164, RTX3060, FP16, batch4 and the same508-image/4875-object test split; checkpoints were selected on511 validation images. Fixed-confidence columns are a separate CPU FP32, batch1 experiment, common conf=.25 and class-aware one-to-one IoU>=.5 matching. FP means unmatched predictions relative to existing labels and may include annotation omissions.

P2+CBAM@P3 gains0.228 percentage points in the original GPU test mAP50–95, but adds142 unmatched predictions for20 additional matches at the common operating point. Its microF1 drops1.023 points; helmet-specific unmatched predictions rise67→112 for only2 more matches. A small AP gain does not establish a better default system.

## Do small targets benefit?

Using official COCOeval bbox AP@[.50:.95], original-image area<32² and maxDets300:

- Small-object AP: Standard 41.18%, P2 43.12%.
- Small-helmet AP: Standard 37.47%, P2 40.03%.
- Helmet recall for GT short side<16px after scaling image long side to640: Standard 31/217, P2 35/217, P2+CBAM@P3 38/217.
- Below8px: Standard 1/139, P2 0/139, P2+CBAM@P3 4/139. Extremely tiny targets remain a major limitation.

![Area-stratified AP](reports/final_evaluation/area_ap.png)

These new area AP values use common CPU FP32 predictions and COCOeval; do not mix them with the original GPU/Ultralytics total AP. The short-side criterion is a tiny-target proxy, not a measured physical distance. The report retains all size/class denominators and negative outcomes.

## Efficiency and model choice

CPU core timing includes image read/decode, RGB conversion, preprocessing, forward, NMS, output transfer and class counts. SameMac, FP32, batch1, actual8 threads,32 validation images×5 rotated-order rounds,160 calls/model. P2/P2+CBAM@P3 average32.80/33.36ms; including drawing and in-memory PNG encoding69.49/69.95ms. Speeds are close; no meaningful speed or stability advantage is claimed. Model loading, browser/network and output-file writes are excluded; these are not CUDA figures.

![F1 and latency tradeoff](reports/final_evaluation/latency_tradeoff.png)

P2 has the highest validation AP and fixed-point microF1, with improved small-object AP over Standard. Standard remains the faster resource-constrained option. Both CBAM placements and the SAHI/fusion experiments remain available, including failed tradeoffs. The application/model registry was not silently changed.

## Quick start

Run the published historical demo, which cannot reproduce the new model scores:

```bash
git clone https://github.com/CJ2323232/yolov8-learning.git
cd yolov8-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gradio.txt
python download_weights.py
python app.py
```

Windows: activate with `.venv\\Scripts\\activate.bat`. Upload an original image without prediction overlays and select the historical/custom preset. The new Standard/P2 and ensemble presets need their specific checkpoints. [Model availability](models/README.md).

## Reproducibility and limitations

- Fixed split:6488 train/511 validation/508 test. Retrieved cross-split candidate pairs20466→0 after491 assignment changes; candidates are not confirmed leaks or proof that all scene relations were found.
- Training:640,batch2,AdamW,lr0=.001667,lrf=.01,nbs64,seed0,50epochs, original augmentation settings. New CBAM modules both operate on P3 with64channels/678parameters; the old P2-position module has32channels/260parameters.
- Historical non-pretrained initialization differs for the Standard/Standard+CBAM pair; P2-pair initialization checks match. Do not claim a fully controlled two-factor interaction study.
- Single seed/run. Image bootstrap measures conditional sampling uncertainty, not training reproducibility. Runtime jitter is not training stability.
- The test set has been explored previously; overall five-model scores were known before this completion analysis. No new external blind evaluation or test-threshold sweep is claimed.
- AllPartA classrooms are training-only;15 external images lack complete labels. Neither supports unseen-classroom AP claims.
- Published process evidence is not a one-command fully reproducible experiment without the original data and unpublished checkpoints.

[Full report and commands](reports/final_evaluation/README.md) · [Machine-readable metrics](reports/final_evaluation/metrics.json) · [Frozen protocol](reports/final_evaluation/protocol.json) · [Raw timing](reports/final_evaluation/benchmark_raw.csv) · [Model selection](reports/final_evaluation/model_selection.json) · [SAHI](reports/sahi_evaluation.md) · [Fusion](reports/fusion_evaluation/README.md)

Next priorities are weight delivery, clean-environment reproduction and new-scene evaluation; repeated seeds only if stronger research conclusions are needed. No additional modules are required to close this project.

## Acknowledgements

[Ultralytics](https://github.com/ultralytics/ultralytics), [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset), [CBAM](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html), [SAHI](https://github.com/obss/sahi), [WBF](https://github.com/ZFTurbo/Weighted-Boxes-Fusion), [COCO API](https://github.com/cocodataset/cocoapi), [Gradio](https://github.com/gradio-app/gradio). See[third-party notices](THIRD_PARTY_NOTICES.md). Research/learning project, not validated for operational safety management.
'''
(ROOT/'README_EN.md').write_text(en)

# Update current indexes; preserve dated historical records instead of rewriting them.
q=ROOT/'reports/README.md';text=q.read_text();text=text.replace('## 当前实验入口','## 当前实验入口\n\n**2026-10-28：五组50轮、尺寸AP、共同conf=.25与CPU完整处理计时均已完成，当前推荐P2。** [最终评价与模型选择](final_evaluation/README.md)。')
text=text.replace('| P2＋CBAM | [结构](../docs/P2_CBAM.md) · [正式训练入口](../docs/P2_CBAM_TRAINING.md) | 准备完成，尚未训练 |','| 五组P2/CBAM结构对照 | [完整结果、尺寸与固定工作点](final_evaluation/README.md) | 五组已训练评价；负结果保留，当前推荐P2 |')
q.write_text(text)
q=ROOT/'docs/README.md';text=q.read_text().replace('| 已完成哪些实验 | [报告索引](../reports/README.md) |','| 已完成哪些实验 | [五组最终评价](../reports/final_evaluation/README.md) · [报告索引](../reports/README.md) |');q.write_text(text)
q=ROOT/'docs/REPRODUCIBILITY.md';text=q.read_text().replace('| P2＋CBAM | 结构、固定参数训练/续训入口、本地测试 | 原Windows环境和隔离数据；尚无训练结果 |','| 五组P2/CBAM | 五组结果、结构、固定训练参数、尺寸与固定工作点评价代码及协议 | 原数据和五组完整训练目录；新权重尚未公开下载 |')
text=text.replace('- CBAM是否有效，要等正式训练与同条件评价；需要更强结论再做多种子对照和新场景评价。','- 五组CBAM对照已经完成。当前保留P2，CBAM的局部收益与误检代价见[最终评价](../reports/final_evaluation/README.md)；更强结论再做多种子与新场景评价。')
text+='\n## 2026-10-28新增评价\n\n使用隔离安装的pycocotools2.0.11，不修改原训练环境。CPU FP32、batch1、候选conf=.001、max_det300用于尺寸AP；共同conf=.25用于微平均错误统计。32张验证图×5轮轮换顺序测核心及含绘图编码耗时。原GPU总体AP另表保留，不混用实现/精度/批尺寸。\n\n标准版配对的历史非预训练层初始化不完全一致；两个新P3 CBAM初始注意力参数一致，P2配对核验一致。不是完整严格二因素交互证据。\n\n[评价代码与复现命令](../reports/final_evaluation/README.md)。完整运行需要五组原始目录；仅有公开历史权重不能重现新成绩。\n';q.write_text(text)
q=ROOT/'models/README.md';text=q.read_text().replace('| P2＋CBAM | 待执行的新对照 | 只有结构和训练代码，没有训练权重 |','| 两种P3 CBAM与旧P2位置CBAM | 五组结构对照已完成；当前不替换P2默认 | 本地训练权重已核验，尚未发布下载 |')
text+='\n## 五组已完成模型（2026-10-28）\n\n'+table(['模型','best.pt SHA256'],[[r['label'],'`'+r['best_sha256']+'`'] for r in records])+'\n\n[五组尺寸、固定工作点与效率评价](../reports/final_evaluation/README.md)。当前推荐P2、conf=.25；P3 CBAM在固定工作点的误检代价更大。模型可用性以本页下载状态为准，本地已存在不等于已发布。\n';q.write_text(text)
q=ROOT/'CHANGELOG.md';text=q.read_text().replace('# 更新记录','# 更新记录\n\n## 2026-10-28\n\n- 完成五组50轮结构实验与回传权重、67项参数及代码快照核验。\n- 补齐官方COCO式面积分层AP、微小安全帽召回、共同conf=.25的P/R/F1及逐图错误统计。\n- 完成Mac CPU每模型160次轮换计时，区分核心处理与绘图编码；不混同GPU或网页端到端耗时。\n- 保留负结果，依据固定工作点权衡推荐P2；披露单种子、初始化与测试集历史使用限制。\n- 更新中英文首页、报告索引、模型目录、复现范围和简历证据；新训练权重仍未公开下载。');q.write_text(text)
q=ROOT/'THIRD_PARTY_NOTICES.md';text=q.read_text().replace('| Gradio |','| COCO评价工具 | [COCO API](https://github.com/cocodataset/cocoapi)，使用pycocotools2.0.11官方bbox评价器，隔离安装；代码和许可沿用上游 |\n| Gradio |');q.write_text(text)
q=ROOT/'docs/PROJECT_STORY.md';text=q.read_text();text+='''

## 2026-10-28：用五组结构对照收尾

已完成标准版、P2、标准版+P3 CBAM、P2+P3 CBAM及旧P2位置CBAM。两个新模块采用相同P3/64通道接入，保留标准版历史初始化不完全一致的限制，不宣称严格统计交互证据。

通过官方COCO式面积AP确认：本批小目标AP从标准版41.18%到P2 43.12%，小安全帽37.47%到40.03%；但输入尺度短边<8px的139个安全帽，P2仍为0匹配。整体受益不等于极小目标问题已解决。

原GPU测试中P2+P3 CBAM的mAP50–95高0.228个百分点，但共同conf=.25时只少漏20个、未匹配预测多142个，F1低1.023个百分点。两者本机CPU核心均值32.80/33.36ms，速度接近。最终保留P2，不为展示模块而采用更差的工作点取舍。

这是从问题、对照、分层评价到最终选择的实验过程，不能缩写为“CBAM显著提升”。所有五组、失败结果、协议、原始计时与限制见[最终报告](../reports/final_evaluation/README.md)。
''';q.write_text(text)
q=ROOT/'docs/RESUME.md';old=q.read_text();prefix='''# 当前推荐的项目经历表述（2026-10-28）

**安全帽佩戴检测与小目标结构实验｜YOLOv8n、PyTorch、CBAM、Gradio**

- 基于SHWD构建7507张样本的数据审计与关联隔离流程，记录图片/标签哈希并调整491张分区，使已检索跨分区候选关系20466→0；明确未知场景关系仍可能存在。
- 完成标准版、P2及不同CBAM接入的五组50轮结构对照，补齐官方COCO式尺寸AP与微小安全帽召回；本次小目标AP由41.18%提高至43.12%，小安全帽AP由37.47%提高至40.03%，同时保留极小目标仍大量漏检的负结果。
- 实现共同工作点误检/漏检评价与每模型160次CPU轮换计时；发现P2+P3 CBAM少漏20个但多142个未匹配预测，依据P2的87.41%总体F1和综合取舍保留单模型默认，并提供Gradio及可追溯报告。

以上是单次已有测试集复评，不写成原创模块、稳定显著提升或工业实时部署。面积AP是COCO式bbox AP@300、原图尺寸、IoU=.50:.95；F1为conf=.25、匹配IoU=.5。CPU核心32.80ms不含绘图编码、模型加载或网页；含绘图编码69.49ms。新权重尚未公开下载。

面试应能解释：小目标与极小目标的定义为何不同；为什么AP略高但固定工作点更差；为什么选择P2而不是强行采用CBAM；标准版初始化与测试集历史使用如何限制结论。[完整证据](../reports/final_evaluation/README.md)。

---

下面保留各阶段历史表述，当前简历优先采用上面的五组收尾版本。

''';q.write_text(prefix+old)
print('README_AND_PROJECT_DOCS_UPDATED')
