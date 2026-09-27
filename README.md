# yolov8-learning

YOLOv8 目标检测入门项目：从第一张检测图，到小规模训练和对比实验。

支持在 Mac、Windows 或 Linux 上运行。自动优先使用 NVIDIA CUDA，其次使用 Mac 的 MPS，最后使用 CPU。

> 当前状态：入门代码与实验计划已准备；尚未完成模型预测或训练验证。实验表中的数值留空，所有实验均为计划状态。

## 1. 下载项目

GitHub 页面点击 **Code → Download ZIP**，解压后在项目文件夹打开终端。也可以使用 Git 克隆本仓库。

建议安装 Python 3.11 或 3.12。若 Mac 的 `python3` 提示缺少开发工具，可先从 [Python 官网](https://www.python.org/downloads/) 安装 Python，再重新打开终端。

## 2. 配置环境

### Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Windows（命令提示符 CMD）

```bat
py -3 -m venv .venv
.venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

NVIDIA 显卡需要兼容的驱动和 CUDA 版 PyTorch。若下方检查显示 CUDA 不可用，按 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/) 在当前虚拟环境安装适合的版本。

## 3. 检查设备

```bash
python run.py check
```

`selected_device` 为 `mps` 表示使用苹果 GPU，`0` 表示使用 NVIDIA GPU，`cpu` 表示使用 CPU。MPS 是否可用还取决于系统及 PyTorch 版本。

## 4. 第一张检测图

```bash
python run.py predict
```

首次运行需要联网下载官方预训练权重和示例图片。结果保存到终端显示的 `runs/predict` 或自动递增目录。

检测自己的图片：

```bash
python run.py predict --source "your-image.jpg"
```

如果 MPS 运行遇到不支持的操作，先用 CPU 验证流程：

```bash
python run.py predict --device cpu
```

## 5. 完成一次小规模训练

```bash
python run.py train --name smoke
```

默认使用 `yolov8n.pt`、COCO8、3 轮、图片尺寸 640、batch 2。首次使用 COCO8 会下载数据；数据实际位置由 Ultralytics 的数据目录设置决定。

**COCO8 只有 8 张图片，只用于检查流程，不代表真实场景的模型效果。**

结果通常包含 `weights/best.pt`（最佳模型）、`weights/last.pt`（最后一轮模型）、`results.csv`、`args.yaml` 和训练图表。脚本另存 `environment.json` 记录运行环境。

用刚训练的模型预测（如果目录递增，请替换为实际目录）：

```bash
python run.py predict --model runs/smoke/weights/best.pt
```

## 6. 准备自己的数据

复制 `data.example.yaml` 为 `data.yaml`，修改数据路径和类别。文件夹结构：

```text
dataset/
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

图片 `example.jpg` 对应标签 `example.txt`，每个目标占一行：

```text
class_id x_center y_center width height
```

类别从 0 开始，坐标和宽高按图片尺寸归一化到 0～1。训练集用来学习，验证集用来调参，测试集留到最终评价。同一视频的相邻帧或重复图片应放在同一组，避免数据泄漏。

## 7. 对比实验

准备好自定义数据后运行以下三组；耗时取决于数据和设备：

```bash
python run.py train --data data.yaml --model yolov8n.pt --epochs 30 --imgsz 640 --name A
python run.py train --data data.yaml --model yolov8n.pt --epochs 30 --imgsz 416 --name B
python run.py train --data data.yaml --model yolov8s.pt --epochs 30 --imgsz 640 --name C
```

保持数据划分、训练轮数、batch 和种子一致。A 是基准，B 只改变输入尺寸，C 只改变模型大小。在 `experiments.csv` 记录真实结果；速度只能在相同硬件、输入及计时口径下比较。算力允许时用多个种子重复实验。

验证训练后的模型：

```bash
python run.py val --model runs/A/weights/best.pt --data data.yaml --name A-val
```

完成方案选择后，再评价保留的测试集：

```bash
python run.py val --model runs/A/weights/best.pt --data data.yaml --split test --name A-test
```

`val` 会保存 `metrics.json`。主要指标是 mAP50-95、Precision 和 Recall，同时应观察漏检、误检图片。

## 8. 保存与分享

- 仓库保留代码、配置、少量效果图和真实实验结论。
- `.gitignore` 已排除虚拟环境、数据集、模型权重和完整运行目录；网页手动上传时仍需自行避开这些文件。
- 安装并跑通后，运行 `python -m pip freeze > requirements-lock.txt` 记录当前机器的精确依赖。当前 `requirements.txt` 是安装范围，不是经过训练验证的锁定环境。
- 大数据和权重单独保存，在 README 提供获取方式。不要上传账号令牌或私人图片。

## 代码来源与复现说明

- `run.py`、数据模板和实验表是为这个入门项目新写的调用脚本与模板，参考了 Ultralytics 官方文档中的 Python API 用法；没有复制 YOLOv8 模型内部实现。
- YOLOv8 模型、预训练权重、训练与预测能力来自 [Ultralytics 官方项目](https://github.com/ultralytics/ultralytics)。本仓库不是 YOLOv8 原论文或模型的原创实现；Ultralytics 表示 YOLOv8 没有单独发布正式论文。
- `coco8.yaml` 与 COCO8 示例数据由 Ultralytics 提供，仅用于流程检查。使用其他人的图片、标注或代码时，需要记录原始链接、作者、版本和相应许可。
- 截至当前，本仓库只上传了代码与实验计划，尚未跑出预测图片、训练权重或对比实验数值。写复现报告时应标明真实运行环境、数据来源、具体命令、实际结果，以及与你参考项目的差异。
- Ultralytics 说明其 YOLOv8 软件和模型按 AGPL-3.0 或 Enterprise 条款提供。若进一步发布基于它的应用或用于商业场景，先核对[官方许可说明](https://www.ultralytics.com/license)。公开仓库不等于获得任意使用、再分发第三方代码或数据的许可。

## 参考

- [YOLOv8 官方说明](https://docs.ultralytics.com/models/yolov8/)
- [安装指南](https://docs.ultralytics.com/quickstart/)
- [训练与 Mac MPS](https://docs.ultralytics.com/modes/train/)
- [YOLO 检测数据格式](https://docs.ultralytics.com/datasets/detect/)
- [COCO8](https://docs.ultralytics.com/datasets/detect/coco8/)

本项目调用 Ultralytics 软件与权重。数据集遵循各自许可。
