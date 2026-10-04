# YOLOv8 安全帽目标检测复现

这是一个基于 [Ultralytics YOLOv8](https://docs.ultralytics.com/models/yolov8/) 的学习项目。已在 Mac 上跑通示例图片预测、COCO8 流程检查，以及本地 `hat` / `person` 数据的 1、5、50 轮训练。仓库展示代码、配置方法和真实训练记录；数据图片、完整模型权重没有上传。

## 已得到的结果

本地 `helmet_dataset` 含训练图片 6,064 张、验证图片 758 张、预留测试图片 759 张。三次安全帽训练使用 YOLOv8n、640 像素、batch 4、随机种子 0 和 Apple MPS。下面都是**各次训练结束时在验证集上的指标**，不是预留测试集的成绩。1、5、50 轮是训练时长的探索，不代表已完成原先计划的模型大小或输入尺寸对比。

| 本地运行名 | 轮数 | Precision | Recall | mAP50 | mAP50-95 | 训练记录耗时 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `helmet_try` | 1 | 0.7946 | 0.6938 | 0.7751 | 0.4413 | 221 秒 |
| `helmet_test` | 5 | 0.9082 | 0.8274 | 0.8950 | 0.5620 | 1,092 秒 |
| `helmet_50` | 50 | 0.9444 | 0.8790 | 0.9344 | 0.6145 | 2,913 秒 |

`helmet_test` 只是本地运行名，这一行仍是**验证集**结果。表格从本地训练记录的最后一轮提取；完整数值见 [运行摘要](reports/helmet_runs.csv) 和 [50 轮逐轮记录](reports/helmet_50_history.csv)。这组观察显示，训练轮数增加时验证指标上升；还不能据此证明模型对独立新场景的效果。

![50 轮训练的验证集 mAP50-95 曲线](reports/helmet_50_map.svg)

示例公交车图片的预测也已经在本地完成。那次预测使用官方预训练模型，是流程检查，不是本项目安全帽模型的效果展示。COCO8 的 3 轮训练同样只用于流程检查；它只有 8 张图片，成绩不与上表比较。

## 仓库里有什么

| 文件 | 用途 |
| --- | --- |
| `run.py` | 检查设备、预测、训练和验证的通用入口 |
| `train_helmet.py` | 本地安全帽项目使用的 50 轮训练设置 |
| `voc_to_yolo.py` | 将 VOC XML 检测框转为 YOLO TXT 标签 |
| `split_dataset.py` | 检查标签后，按固定随机种子划分训练/验证/测试集 |
| `requirements.txt` | 安装 Ultralytics 的依赖说明 |
| `data.example.yaml` | 两类安全帽数据的配置示例 |
| `experiments.csv` | 已完成和计划中的实验登记 |
| `reports/` | 从本地训练日志提取的数值和曲线 |

数据处理脚本依据本机 `/Users/cj/yolov8/` 中的版本整理为相对路径，便于从仓库目录运行；不会自动下载数据。`split_dataset.py` 发现目标目录已存在时会停止，以免覆盖已有划分。

## 怎样复现

建议使用独立的 Python 环境。安装依赖：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py check
```

Windows CMD 的激活命令是 `.venv\Scripts\activate.bat`。安装 CUDA 版 PyTorch 时按 [PyTorch 官方说明](https://pytorch.org/get-started/locally/)选择匹配版本。

先跑通官方示例图片预测：

```bash
python run.py predict
```

本地数据来自 [SHWD（Safety Helmet Wearing Dataset）](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)。原仓库提供下载入口，介绍了 VOC2028 目录结构与 `hat`、`person` 两个类别；其中 `hat` 表示戴安全帽的目标，`person` 是未戴安全帽的头部目标。取得有使用权限的数据后，把它放成下面的目录结构：

```text
VOC2028/
  Annotations/  # VOC XML
  JPEGImages/   # 对应图片
```

然后在仓库根目录执行：

```bash
python voc_to_yolo.py
python split_dataset.py
python train_helmet.py
```

数据转换生成 `VOC2028/labels/`；划分脚本生成 `helmet_dataset/` 和其中的 `data.yaml`。`train_helmet.py` 使用 YOLOv8n 预训练权重、50 轮、640 像素、batch 4；结果写到 `runs/helmet_50/` 或自动递增目录。也可以用通用入口运行较短实验：

```bash
python run.py train --data helmet_dataset/data.yaml --epochs 5 --batch 4 --name helmet_test
```

想严格复现上表，需要使用相同数据、相同划分和匹配的软件环境；仅使用同名数据文件夹不能保证得到相同结果。当前已保存的 `requirements.txt` 是版本范围，流程检查时记录的环境为 Python 3.14.7、PyTorch 2.14.0、Ultralytics 8.4.164，使用 MPS；安全帽训练日志也记录了 `device: mps`。这些信息仍不足以保证跨机器结果完全一致。

## 尚待完成

- 在预留的 759 张测试图片上单独评价最终模型；目前没有测试集成绩。
- 核对 SHWD 原始图片和引用数据的再分发许可；当前仓库没有上传原始图片。
- 在同一划分和硬件上完成原计划的模型大小、输入尺寸对比；计划见 `experiments.csv`。
- 分析不同场景下的误检和漏检，并在确认图片可公开后添加可视化示例。

## 来源与许可

本仓库的脚本调用 Ultralytics 提供的 YOLOv8 模型、权重及训练/预测 API，并非 YOLOv8 算法的原创实现。[官方项目](https://github.com/ultralytics/ultralytics)和[许可说明](https://www.ultralytics.com/license)提供相应来源与使用条款。数据来自 [njvisionpower 的 SHWD 仓库](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)；其页面标有 MIT 许可证，但原仓库说明部分图片来自网络及 SCUT-HEAD，具体图片的再分发权限需要分别核对。本仓库只公开处理脚本与数值摘要。
