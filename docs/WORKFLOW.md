# 安装、数据处理、训练与代码解读

> 2026-10-13 修正：旧划分已确认重复泄漏。公开 v1.0.0 权重与旧分数只用于历史演示，不能作为新划分基准。原始数据、原始框和历史训练结果均保留。

## 1. 运行历史演示

在仓库目录创建并激活虚拟环境后：

```bash
python -m pip install -r requirements-gradio.txt
python download_weights.py
python app.py
```

Mac 创建环境：`python3 -m venv .venv`，激活：`source .venv/bin/activate`。
Windows CMD 创建环境：`python -m venv .venv`，激活：`.venv\Scripts\activate.bat`。

打开终端给出的本地地址，上传图片、调节置信度、点击 Submit。结果区可保存画框图；数量不是准确率。终端保持运行，Control+C 停止。Mac 可双击启动脚本；只监听 127.0.0.1，不是公网服务。

历史模型保存在 `models/helmet_yolov8n.pt`，下载脚本先校验 SHA256，再原子安装，不占用新训练目录。网页优先读取 `runs/latest_helmet_training.json` 登记的训练产物，否则明确标记为历史演示。自定义权重可通过 `YOLO_MODEL_PATH` 指定；网页要求 0=hat、1=person 的头部检测模型。

核心直接依赖已固定版本；`requirements-lock-macos-py314.txt` 是这台 Mac 的已安装完整包版本快照，只用于相同平台和 Python 版本，不保证可原样用于 Windows。

## 2. 数据准备与标注协议

从 [SHWD 原仓库](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) 获取数据，将 JPEGImages 和 Annotations 放在仓库 VOC2028 目录。原图不上传到本仓库。

```bash
python voc_to_yolo.py
python split_dataset.py
```

转换程序校验类别、尺寸和框范围，并保存 difficult/truncated 元数据。当前协议是 **difficult 作为普通目标参与训练与评价**，不是 VOC difficult-ignore 协议。已知非目标 dog 被记录并跳过，其他未知类别会停止转换。坐标保留现有连续 xyxy 公式，未擅自作端点偏移。

划分生成新的 `helmet_dataset_clean/`，不覆盖已有输出：
- 按图片内容哈希去重；同图不同框的所有版本隔离至 quarantine，不自动判断哪个框正确。
- 已知 JPEG 末尾问题只在副本补齐结束标记，不重新编码像素；之后记录实际图片哈希。
- 使用 seed=42 固定分配；图片、标签、内容哈希与组编号写入 split_manifest.csv。
- dataset_audit.json 保存冲突清单、划分统计、JPEG 处理和评价协议；annotation_metadata.csv 保存原始 VOC 标记。

本机已生成 6,005 / 750 / 752 张的新划分，隔离 74 张。它只消除了完全相同的跨集合图片；近重复和场景独立性仍需人工核查。

有同一视频/场景信息时，可传入完整 CSV（image,group_id）：

```bash
python split_dataset.py --groups-file scene_groups.csv --output helmet_dataset_grouped
```

同一组不会跨集合。CSV 必须覆盖全部源图；组大小不同会使划分比例近似，而非精确 8:1:1。

## 3. 新训练与产物登记

```bash
python train_helmet.py
```

默认使用 helmet_dataset_clean/data.yaml，从官方 yolov8n.pt 开始新训练：50 轮、640、batch=4、seed=0、AdamW、lr0=0.001667、momentum/beta1=0.9、warmup_bias_lr=0、线性学习率调度。参数通过 run.py 显式传递；有效参数另存 effective_args.json，不把请求参数冒充全部生效参数。

训练前后都会核对完整数据清单。成功生成 best.pt 后：
1. 在实际输出目录保存 training_artifact.json。
2. 记录模型 SHA256、数据清单 SHA256、有效参数和环境。
3. 原子更新 runs/latest_helmet_training.json。
4. 测试、尺寸比较和网页读取该记录，避免目录递增后仍使用旧模型。

```bash
python train_helmet.py --epochs 30 --device cpu --name helmet_clean_seed1 --seed 1
```

每个种子单独命名。不要把恢复训练、不同增强或不同初始权重的运行当成只改变轮数的对比。

恢复中断的新训练，必须明确给出 last.pt：

```bash
python train_helmet.py --model runs/实际目录/weights/last.pt --resume
```

恢复只接受带有匹配数据身份记录的 checkpoint。历史未登记模型不能静默恢复成新基准；已完成训练的 last.pt 可能已被库移除优化器状态，也不适合作为继续恢复的起点。

## 4. 评价与推理

```bash
python test_helmet.py
python benchmark_sizes.py --device cpu
python helmet_external_test/download_images.py
python predict_external.py --conf 0.25 --batch 4
python app.py
```

尚无新训练记录时，测试和尺寸比较会停止，不退回到历史公开模型。评价默认使用登记时的数据划分，并核对模型与数据身份。显式自定义模型/data 也必须通过数据审计；已知历史公开权重不能在重新划分数据上当作独立基准。

仅诊断旧结果时必须明确标记：

```bash
python test_helmet.py --model models/helmet_yolov8n.pt --data helmet_dataset/data.yaml --allow-legacy-data
```

这种输出会记录 legacy diagnostic 状态。不能从它得出清理后模型的泛化结论。

评价保存 metrics.json、command.json、environment.json、effective_args.json 和 input_identity.json。AP 使用低候选置信度 0.001；NMS IoU=0.7 与计算 AP 时的匹配 IoU 不同。P/R 是库在最大平均 F1 置信度点返回的类别平均值，**不是 conf=0.25 的部署成绩**。

尺寸比较仅使用验证集，固定权重；单次库内模型耗时不代表网页端到端延迟。外部照片无人工框时只作定性展示。

## 5. 代码如何工作

| 文件 | 职责 |
| --- | --- |
| voc_to_yolo.py | VOC→YOLO；校验并保留 difficult/truncated 协议与元数据 |
| split_dataset.py | 图片内容去重、冲突隔离、JPEG 副本修复、分组划分、全量哈希清单 |
| workflow.py | 数据完整性检查、类别映射检查、模型校验、训练产物登记与选择 |
| run.py | 训练/评价/预测入口；显式训练参数、真实预测 batch、身份及有效参数记录 |
| train_helmet.py | 新基准默认参数；导入模块不会启动训练 |
| test_helmet.py | 用已登记权重和数据评价，避免误测旧模型 |
| benchmark_sizes.py | 同模型、同验证划分的尺寸比较及完整身份记录 |
| app.py | 复用模型，画框、RGB 转换、计数，显示来源与历史模型局限 |
| predict_external.py | 登记模型或明确标记的历史演示模型的外部定性预测 |
| download_weights.py | 固定版本下载、校验、原子安装且不覆盖不同文件 |
| tests/test_workflow.py | 泄漏、冲突、数据变化、错误映射、错误权重和 batch 等回归检查 |

运行回归检查：`python -m unittest discover -s tests -t . -v`。检查不下载模型、不启动网页、不进行完整训练。详细修复和验证见 [FIXES_2026-10-13.md](FIXES_2026-10-13.md)。

## 来源

模型/API：[Ultralytics](https://github.com/ultralytics/ultralytics)；使用遵循其 [AGPL-3.0 / Enterprise 条款](https://www.ultralytics.com/license)。
数据：[SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)。外部照片：[Pexels](https://www.pexels.com/license/)，逐图来源保留在清单中。
