# YOLOv8 安全帽检测与 Gradio 演示

[English](README_EN.md) · [快速开始](#快速开始) · [模型下载](models/README.md) · [实验报告](reports/size_comparison.md) · [失败案例](reports/failure_cases.md)

基于 **Ultralytics YOLOv8n + SHWD** 的学习与复现项目：完成数据处理、模型训练、保留测试集评价，并用 Gradio 提供上传图片、调节置信度、显示检测框和数量的本地界面。支持 Mac、Windows；CPU 也可演示。

![Gradio 实际检测：原图、检测框和数量](assets/gradio_result.png)

*用户提供的真实操作截图：conf=0.25，界面输出 8 个 hat。截图用于展示交互流程，未核实该输入是否参与训练，不作为独立测试证据。当前是本地应用，尚未部署公网在线 Demo。*

| 已完成 | 结果 |
| --- | --- |
| 数据处理与训练 | SHWD 两类；1 / 5 / 50 轮历史训练记录 |
| 保留测试集评价 | 759 张；mAP50 **91.19%**，mAP50-95 **57.09%** |
| 输入尺寸对比 | 同一权重、758 张验证图；416 / 640 / 960，报告精度与 CPU 推理耗时 |
| 应用演示 | 单张图片上传、conf 滑块、检测统计和结果图片下载 |
| 复现资料 | 公开已训练权重、校验值、脚本、配置与原始数值 |

## 快速开始

下载仓库 ZIP 并解压，或克隆：

```bash
git clone https://github.com/CJ2323232/yolov8-learning.git
cd yolov8-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gradio.txt
python download_weights.py
python app.py
```

Windows CMD 把创建环境改为 `python -m venv .venv`，激活改为 `.venv\Scripts\activate.bat`；其余 `python` 命令相同。

打开终端显示的地址（通常 `http://127.0.0.1:7860`），上传图片，保持 conf=0.25，点击 Submit。结果区显示画框图和检测数量，可下载结果图片。终端需保持运行；Control+C 停止。

**无需重新训练即可演示。** `download_weights.py` 优先使用仓库自带模型，校验后安装到默认路径；已有不同模型时会停止，不覆盖你的训练结果。默认选择 MPS / CUDA / CPU。

以后在项目目录只需 `./.venv/bin/python app.py`；Mac 也可双击 `启动安全帽检测.command`，ZIP 下载后若不能执行，先运行 `chmod +x 启动安全帽检测.command`。

## 模型与数据

[下载已训练权重](https://raw.githubusercontent.com/CJ2323232/yolov8-learning/main/models/helmet_yolov8n.pt)（约 5.94 MiB） · [模型说明与 SHA256](models/README.md)

模型来自本项目 50 轮实验的 best.pt，与测试报告中的权重一致。best.pt 是该训练过程验证表现最好的权重，不一定是最后一轮。原始数据不打包：需要训练或评价时，从 [SHWD 原仓库](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)取得数据。

| 类别 | 含义 |
| --- | --- |
| hat / 0 | 戴安全帽的头部目标 |
| person / 1 | 未戴安全帽的头部目标，不是整个人体 |

本地划分：训练 6,064 张 / 验证 758 张 / 测试 759 张。Pexels 外部 15 张照片无人工框，只作定性观察。

## 实验结果

**保留测试集，2026-10-10：** 同一公开 best.pt，CPU，imgsz 640，batch 4。

| Precision | Recall | mAP50 | mAP50-95 |
| ---: | ---: | ---: | ---: |
| 0.9100 | 0.8508 | 0.9119 | 0.5709 |

[完整测试成绩与模型身份](reports/test_metrics.json) · [测试清单](reports/test_manifest.csv) · [训练历史](reports/training/)

**输入尺寸对比，2026-10-11：** 同一模型与 758 张验证图，不重新训练。

| imgsz | Precision | Recall | mAP50 | mAP50-95 | 模型推理 ms/图 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 416 | 0.9013 | 0.7963 | 0.8611 | 0.5484 | 7.64 |
| 640 | 0.9446 | 0.8788 | 0.9343 | 0.6145 | 15.31 |
| 960 | 0.9404 | 0.9010 | 0.9526 | 0.6307 | 33.77 |

![验证精度与 CPU 推理耗时](assets/size_comparison.svg)

960 相比 640 的 mAP50-95 高约 **1.62 个百分点**，本次 CPU 模型推理耗时约 **2.2 倍**。计时是库内模型推理，不是网页端到端延迟；每档只运行一次，不能当作稳定硬件性能结论。

[实验条件、局限与重跑命令](reports/size_comparison.md) · [完整 JSON](reports/size_comparison.json) · [CSV](reports/size_comparison.csv)

历史 1 / 5 / 50 轮记录是最后一轮的验证成绩；50 轮配置记录了恢复训练，不能当作三组独立从零训练的严格对比。测试与验证成绩分开展示。

## 失败与边界

![地上的安全帽被判为戴帽目标](assets/failure_3.jpg)

模型可能把无人佩戴的安全帽识别为 hat，也会受到小目标、遮挡、模糊和场景变化影响。不能用画框数量代替准确性，更不能直接用作安全管理判定。

[查看失败与边界案例分析](reports/failure_cases.md) · [15 张历史外部照片对照及来源](reports/external_gallery.md)

## 代码与详细复现

| 入口 | 用途 |
| --- | --- |
| `app.py` | Gradio 单图演示，显示画框和数量 |
| `download_weights.py` | 获取或安装模型并校验，不覆盖不同的已有文件 |
| `run.py` | 通用设备检查、预测、训练、评价入口 |
| `voc_to_yolo.py` / `split_dataset.py` | 标签转换、固定随机种子划分数据 |
| `train_helmet.py` / `test_helmet.py` | 安全帽训练与测试，保存结果和运行记录 |
| `benchmark_sizes.py` | 在同一验证集上比较输入尺寸 |
| `predict_external.py` | 外部照片预测，保存命令与环境 |

[完整安装、数据准备与代码解读](docs/WORKFLOW.md) · [常见问题](docs/FAQ.md) · [实验登记](experiments.csv) · [可复用的项目介绍](docs/PROJECT_STORY.md)

已验证环境：Python 3.14.7 / torch 2.14.0 / Ultralytics 8.4.164 / Gradio 6.30.0。历史训练使用 MPS，本次评价与对比使用 CPU；跨环境结果和耗时可能变化。

后续计划：不同模型大小的训练对比、更多随机种子、外部照片人工标注、重复速度测量。目前未实现视频检测、批量上传或公网在线服务。

## 来源与许可

本项目调用 [Ultralytics](https://github.com/ultralytics/ultralytics) 提供的模型与 API，不是 YOLOv8 原创算法；模型使用遵循其 [AGPL-3.0 / Enterprise 条款](https://www.ultralytics.com/license)。界面使用 [Gradio](https://github.com/gradio-app/gradio)。

数据来自 [njvisionpower/SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)；其仓库标有 MIT，但部分图片来自网络及 SCUT-HEAD，原始数据不在本仓库再分发。外部示例遵循 [Pexels License](https://www.pexels.com/license/)，逐图来源见展示页面。画框不代表人物真实安全状态，也不暗示背书。
