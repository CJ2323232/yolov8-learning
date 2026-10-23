# 安全帽佩戴检测 · YOLOv8 + Gradio

> **2026-10-23 更新：标准版与 P2 均完成50轮，产物已核验。** 同一511张验证集，最佳轮次 mAP50–95：标准版 **60.244%**，P2 **61.301%（+1.057个百分点）**。单次运行且标准版跨设备续训，小目标收益与速度仍待统一评估。[结果、曲线和完整条件](reports/p2_training_comparison/README.md) · [P2说明](docs/P2.md)。

基于 **Ultralytics YOLOv8n 与 SHWD** 的计算机视觉实践项目，覆盖数据处理、模型训练、结果评价、数据质量审计与本地交互演示。目标是检测**戴安全帽与未戴安全帽的头部**，并通过实验记录说明模型的表现与边界。

**Python · PyTorch · Ultralytics · OpenCV · Gradio**

[English](README_EN.md) · [快速体验](#快速体验) · [实验结果](#实验结果与适用范围) · [技术复盘](docs/PROJECT_STORY.md) · [模型下载](https://github.com/CJ2323232/yolov8-learning/releases/tag/v1.0.0)

![Gradio 本地演示：原图、检测框与分类统计](assets/gradio_result.png)

*支持图片上传、置信度调整、分类计数和结果下载。截图展示普通检测界面；输入图片是否参与过训练未核实，不作为独立测试证据。*

> **当前状态：** 可下载已有权重直接运行。旧数据划分发现重复泄漏，已完成审计修复与原模型的测试子集复评；成绩适用范围见下文。新划分两版训练产物已核验，见下方P2结果；历史模型成绩单独标注。

## 项目完成了什么

| 工作 | 实现内容 | 可核查资料 |
| --- | --- | --- |
| 数据处理 | VOC XML 转 YOLO；校验类别、图片尺寸和框范围，保留 difficult/truncated 元数据 | [转换代码](voc_to_yolo.py) · [标注协议](docs/WORKFLOW.md) |
| 训练与推理 | 完成历史 50 轮训练实验，保存训练记录与最佳权重，集成单图预测 | [训练记录](reports/training/) · [模型说明](models/README.md) |
| 数据质量审计 | 内容去重、冲突标注隔离；记录图片与标签哈希，支持按场景组划分 | [审计结果](reports/clean_dataset_audit.json) · [划分代码](split_dataset.py) |
| 实验可追溯 | 记录模型、数据清单、有效参数及环境；评价入口核验对应关系 | [流程代码](workflow.py) · [修复验证](reports/fix_validation.json) |
| 评价与取舍 | 比较 416 / 640 / 960 输入尺寸；筛查原测试集后复评，公开每类指标 | [尺寸实验](reports/size_comparison.md) · [复评报告](reports/legacy_test_reassessment.md) |
| 交互与错误分析 | Gradio 展示；分析无人佩戴安全帽误检、小目标及遮挡场景 | [界面代码](app.py) · [失败案例](reports/failure_cases.md) |
| 切片推理评估 | 集成SAHI；比较640/960/1280切片，定位并修复单块额外合并，完成529张复验并公开收益与代价 | [完整实验](reports/sahi_evaluation.md) · [代码](helmet_inference.py) |
| 场景风险隔离 | 保留人工判断，隔离已检索关联；新划分 6488 / 511 / 508 | [隔离证据与清单](reports/scene_isolation/README.md) |
| P2 结构对照 | 官方 nano 标准/P2 配置，共有层同权重初始化，前向/反向与训练器交接检查 | [设置与验证](reports/p2_progress.md) · [初始化代码](helmet_models.py) |
| 跨设备续训 | 检查点、优化器、EMA、数据身份核对；Mac→Windows 输出路径重定位 | [迁移记录与源码](docs/WINDOWS_MIGRATION.md) |

项目工作集中在**数据、实验和应用集成**；检测网络及训练框架来自 Ultralytics，未提出新的检测架构。

## 数据治理产生了什么效果

首轮内容去重后，部分疑似同源图片仍跨训练、验证或测试集。进一步按检索关联和人工核查构建保守风险组，把关联图片安排在同一分区。

| 数据质量指标 | 首轮去重版本 | 分组隔离版本 |
| --- | ---: | ---: |
| 已检索关联候选的跨分区对数 | 20,466 | **0** |
| 人工确认同组图片的跨分区对数 | 10 | **0** |
| PartA教室来源位于验证／测试的图片 | 206／200 | **0／0** |
| 保留的有效图片 | 7,507 | **7,507** |
| 训练／验证／测试图片 | 6005／750／752 | **6488／511／508** |

共调整491张图片的分区，图片与标签清单中的内容哈希保持一致。20,466是涉及2,126张图片的**候选配对数**，同一图片可参与多对，不是重复图片数量，也不是全部确认泄漏。人工已核查20对：10同组、9不同组、1不确定。

这项工作的直接结果是消除**已检索关联的跨分区风险**，建立更可信的评价条件；尚未证明所有未知同源关系均已发现。PartA全部放入训练集也使val/test无法衡量新教室泛化。数据和训练设置同时改变，旧、新模型的分数差不能单独归因于去重或分组。

[前后指标与可运行核验](reports/scene_isolation/README.md) · [旧、新标准版参数对比](reports/baseline_evolution.md) · [P2训练结果](reports/p2_training_comparison/README.md)


## 快速体验

**不需要训练，也不需要下载完整数据集。** 下载仓库 ZIP 并解压，或在 macOS / Linux 终端运行：

```bash
git clone https://github.com/CJ2323232/yolov8-learning.git
cd yolov8-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gradio.txt
python download_weights.py
python app.py
```

Windows CMD 使用 `python -m venv .venv` 创建环境，使用 `.venv\Scripts\activate.bat` 激活；后续安装、权重准备和启动命令相同。

打开终端显示的本地地址（通常为 `http://127.0.0.1:7860`），上传图片并点击 **Submit**。保持终端运行，按 `Control+C` 停止。设备自动选择 MPS / CUDA / CPU；当前应用仅在本机提供单张图片检测。

网页提供“普通检测 / SAHI 切片检测”选项，默认普通检测；普通模式可选择 640 / 960 输入尺寸。SAHI使用同一模型，按640×640、20%重叠切片，多块时补充整图预测并合并；宽、高均不超过640时直接沿用普通检测。已完成切片对比与修复复验：召回有收益，但综合F1仍低于普通检测。[接入说明](docs/SAHI.md) · [完整实验](reports/sahi_evaluation.md)。

权重保存在 `models/helmet_yolov8n.pt`，通过 SHA256 校验。网页优先选择已登记的新训练产物；没有新产物时使用并标明历史演示模型。[详细安装与运行说明](docs/WORKFLOW.md)。

已验证本地环境：Python **3.14.7** / PyTorch **2.14.0** / Ultralytics **8.4.164** / Gradio **6.30.0**。历史训练使用 Mac MPS；复评使用 CPU。Mac 环境快照不代表 Windows 全依赖兼容性保证。

## 实验结果与适用范围

### 新划分标准版与 P2（2026-10-23）

| 最佳轮次验证指标 | 标准版 | P2 |
| --- | ---: | ---: |
| Precision | 89.160% | 90.248% |
| Recall | 86.152% | 86.829% |
| mAP50 | 91.584% | 92.995% |
| mAP50–95 | 60.244% | 61.301% |

两版均50轮，表中为各自best.pt对应的第40/48轮，使用同一511张验证集。共有层预训练初始化与主要参数一致；标准版前13轮MPS、随后CUDA续训，P2记录为CUDA训练。当前只能报告本次运行的改善，不能声称严格结构因果、稳定收益或小目标专门提升。P/R不是固定conf0.25的成绩。

[完整结果、曲线、原始CSV和局限](reports/p2_training_comparison/README.md)。新权重已在本地核验，本仓库可下载演示权重仍为历史模型。


### 原模型的测试子集复评

沿用历史 50 轮实验的同一份 `best.pt`，只从**原测试集 759 张**图片中筛查，排除 285 张确认重复、测试内部冗余或疑似关联图片，在保留的 **474 张、4,816 个标注框**上重新评价，没有训练或微调。

| 评价对象 | 图片数 | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原测试集筛查子集 | 474 | 0.9079 | 0.8544 | 0.9173 | 0.6063 |

**适用范围：** 这是事后筛查子集结果，不能证明全部场景独立，也不代表模型获得提升。285 张排除图并非全部确诊泄漏。评价使用 CPU、imgsz=640、batch=4；P/R 是最大平均 F1 置信度点的类别平均，不能当成网页 conf=0.25 的成绩。

[方法、每类结果与局限](reports/legacy_test_reassessment.md) · [逐图筛查决策](reports/legacy_test_review/screening_decisions.csv) · [完整数值](reports/legacy_test_review/screened_test_metrics.json)

已补充普通检测在固定 conf=0.25 下的[小目标与错误诊断](reports/small_object_analysis.md)，统计不同大小目标的漏检并提供人工核查步骤；这不是新的独立测试或 SAHI 提升结论。

### SAHI 切片对比与单块修复

在推理前冻结的 **529 张原验证集筛查图片**上，固定同一历史模型与 conf=0.25，对比切片尺寸并修复单块图片额外合并的问题；修正版已对529张全部重新运行。

| 方案 | 总体 Recall | 最小档 Recall | 未匹配预测 FP | 微平均 F1 |
| --- | --- | --- | --- | --- |
| 普通检测 | 93.13% | 81.22% | 747 | 90.33% |
| 旧版 SAHI 640 | 92.63% | 83.11% | 975 | 88.34% |
| 修正版 SAHI 640 | 93.47% | 83.84% | 1055 | 88.20% |

修正版相比旧版少漏48个目标，但多80个未匹配预测；相比普通检测净少漏19个，但多308个未匹配预测。**继续采用普通检测为默认，SAHI不作为已验证的综合性能提升。** 最小档包含两种头部，FP也包含重复框和定位差异等。该子集用于探索与修复验证，不是新独立测试。

[完整工作、五种配置与局限](reports/sahi_evaluation.md) · [公开数值与复现入口](reports/sahi_evaluation/) · [接入代码](helmet_inference.py)

### 输入尺寸与推理耗时

同一模型、同一组 758 张历史验证图片，仅改变输入尺寸：

| 输入尺寸 | mAP50–95 | CPU 模型推理耗时（ms/图） |
| ---: | ---: | ---: |
| 416 | 0.5484 | 7.64 |
| 640 | 0.6145 | 15.31 |
| 960 | 0.6307 | 33.77 |

![输入尺寸、验证精度与 CPU 模型推理耗时](assets/size_comparison.svg)

这次运行中，960 相比 640 的 mAP50–95 高 **1.62 个百分点**，模型推理耗时约为 **2.2 倍**。结果来自存在重复泄漏的历史划分；每档仅运行一次，耗时不含预处理、后处理和网页交互，因此用于探索取舍，不能作为独立泛化或稳定性能基准。没有据此证明小目标 AP 提升。

[完整条件与原始记录](reports/size_comparison.md) · [全部历史实验](reports/README.md)

## 一个值得分析的错误

![无人佩戴的安全帽被误判为戴帽头部](assets/failure_3.jpg)

模型把地上无人佩戴的安全帽识别为 `hat`，置信度约 **0.79**。这说明“识别到帽子外观”不能等同于“正确识别佩戴状态”。增加此类负样本或显式利用头部与帽子的关系是待验证方向，尚未实现为性能改进。

另外两类观察是手持安全帽负例与远处遮挡目标。[案例、来源和预测参数](reports/failure_cases.md) · [15 张外部图片定性对照](reports/external_gallery.md)。外部图片没有人工真值框，不报告其 mAP 或漏检率。

## 数据与复现

数据来自 [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)，本仓库不打包原始数据。

| 类别 | 标注含义 |
| --- | --- |
| `0: hat` | 戴安全帽的头部 |
| `1: person` | 未戴安全帽的头部，不是整个人体 |

数据流：**VOC 图片与 XML → 标签转换与校验 → 去重、冲突隔离及划分 → 训练与产物登记 → 评价 / 预测 → Gradio 展示**。

首轮审计隔离了 **37 组、74 张**同图不同标注的版本，得到 6,005 / 750 / 752 张划分。随后结合 20 对人工判断与 20,466 对检索候选完成保守分组隔离，保留全部 **7,507 张**接受图片，491 张变更分区，当前训练 / 验证 / 测试为 **6,488 / 511 / 508**。

已检索候选不再跨分区，但不代表穷尽所有同源关系。1,999 张 PartA 教室来源图片全部放入训练集，验证/测试不覆盖新教室泛化。历史模型不能用于这份重新划分数据的独立基准。[完整范围、哈希与清单](reports/scene_isolation/README.md)。

历史实验、结构接入验证和新训练结果分开记录。**标准版与P2的50轮产物已核验，训练验证结果已公开**。P2本次mAP50–95高1.057个百分点；下一步进行固定阈值、目标尺寸分层与耗时的统一评价。[训练流程](docs/WORKFLOW.md) · [待评价方案](reports/p2_progress.md)。历史 1 / 5 / 50 轮记录不是三组独立从零训练的控制实验。

| 代码入口 | 职责 |
| --- | --- |
| [voc_to_yolo.py](voc_to_yolo.py) / [split_dataset.py](split_dataset.py) | 标注转换、数据检查、去重及划分 |
| [workflow.py](workflow.py) / [run.py](run.py) | 完整性检查、训练产物管理与执行入口 |
| [train_helmet.py](train_helmet.py) / [test_helmet.py](test_helmet.py) | 训练与评价 |
| [review_legacy_test.py](review_legacy_test.py) | 原测试集筛查与原模型复评 |
| [benchmark_sizes.py](benchmark_sizes.py) | 同权重的输入尺寸比较 |
| [app.py](app.py) / [predict_external.py](predict_external.py) | 网页演示与外部图片预测 |

[数据修复记录](docs/FIXES_2026-10-13.md) · [模型说明](models/README.md) · [常见问题](docs/FAQ.md) · [项目技术复盘](docs/PROJECT_STORY.md)

## 后续方向

- 基于已有标注整理误检、漏检案例，分析目标大小与遮挡的影响。
- 重复测量模型推理与端到端耗时，在验证集选择置信度阈值。
- 使用两版现有best.pt统一验证固定阈值表现、小目标召回与耗时；方案冻结后评价测试集。

当前没有公网在线服务、视频检测或批量上传；未证明可用于实际安全管理决策。

## 来源与许可

模型与 API 来自 [Ultralytics](https://github.com/ultralytics/ultralytics)，使用遵循其 [AGPL-3.0 / Enterprise 条款](https://www.ultralytics.com/license)；界面使用 [Gradio](https://github.com/gradio-app/gradio)。

数据来自 [njvisionpower/SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)。其仓库标有 MIT，但部分图片来自网络及 SCUT-HEAD，本仓库不再分发原始数据。外部示例逐图记录来源，并遵循 [Pexels License](https://www.pexels.com/license/)。检测输出不代表图片人物的真实安全状态或背书。
