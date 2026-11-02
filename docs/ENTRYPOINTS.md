# 安装、推理与数据路径

本页记录第二阶段增加的可移植入口。历史训练脚本、结构、权重和冻结的最终评价代码保持原样。新入口只负责参数、路径、输入检查和输出封装，网络和 NMS 仍由原版本 Ultralytics 执行。

## 环境

先按 [环境说明](ENVIRONMENTS.md) 选择应用环境或历史 Windows 训练环境。不要把 Python 3.10 的训练记录和使用 NumPy 2.5.3 的应用依赖混为同一环境。以下命令从仓库根目录执行，`python` 指已激活的相应虚拟环境。

## 图片推理

使用已有、可信的本地 `.pt`。不需要数据集、真实标注或训练登记文件。缺失权重直接报错，不搜索作者目录，也不自动下载替代模型。

历史演示权重若已随仓库获取，先将一张有权使用、没有预测框和文字的原图放入 `my_images/example.jpg`，再在**本地研究演示**中检查：

```bash
python cli.py predict --weights models/helmet_yolov8n.pt --source my_images/example.jpg --imgsz 640 --conf 0.25 --iou 0.7 --device cpu --output outputs/legacy_example
```

文件夹示例（输入应为没有预测框的原图；请自行准备有权使用的图片）：

```bash
python cli.py predict --weights models/helmet_yolov8n.pt --source my_images --conf 0.25 --iou 0.7 --output outputs/folder_example
```

`--weights` 可换为五组中的任一真实 checkpoint；[交付清单](../models/release_manifest.json) 列出了预期 SHA256，但这些新权重尚未公开下载。可加 `--sha256 EXPECTED_DIGEST` 校验。不要把历史权重改名当作新模型。数据和权重许可限制见 [许可证审计](license_audit.md)。

- `--source`：本地图片或文件夹（递归）；支持 jpg/jpeg/png/webp/bmp/tif/tiff。
- `--weights`、`--source`、`--output`：相对路径均按当前工作目录解释，也可显式传入绝对路径。
- `--conf` 默认 .25，`--iou` 默认 .7；后者是 NMS 阈值，与评价时判断匹配的 IoU 不同。
- `--imgsz` 默认 640；`--device` 默认 auto，可指定 cpu、mps、0。Windows CUDA 可显式 `--device 0`。
- `--dry-run` 只检查文件和参数，不加载模型，不写输出。
- 输出目录必须尚不存在，避免覆盖。每次命令只加载一个选定模型一次。
- 输出 `images/` 为真实预测画框图，`detections.json` 逐图记录类别、置信度和原图像素坐标 xyxy；`run.json` 记录权重哈希、参数和完成状态。文件夹层级保留，避免同名文件相互覆盖。
- `elapsed_seconds_including_image_output` 是这次命令的耗时，不是标准化基准，不可直接当论文 FPS。

推理不读取 labels、验证集标注或测试集真值。错误输入会明确报错；处理中途失败时保留 `running` 状态和已产生的图片，不能把该目录算作完整运行。

## 数据检查与评估

新入口采用唯一规则：YAML 中相对 `path` 按 **YAML 所在目录** 解释；`--data-root` 则按**当前目录**解释，并优先于 YAML。生成的运行期 YAML 使用计算出的绝对根路径，项目审核和 Ultralytics 加载同一根目录，不修改原 YAML。

```bash
python cli.py check-data --data data.example.yaml --data-root datasets/helmet_scene_isolated_v1
python cli.py val --weights run/helmet_p2_isolated_640_windows/weights/best.pt --data data.example.yaml --data-root datasets/helmet_scene_isolated_v1 --split val --imgsz 640 --batch 2 --device cpu --output outputs/p2_validation --dry-run
```

第二条移除 `--dry-run` 才会实际评估；本阶段不重跑完整验证或测试。新用户应先按 [隔离划分说明](../reports/scene_isolation/README.md) 准备数据和校验清单；空目录或只有 YAML 不能通过检查。

`val` 委托原 `run.py`，保持候选 conf=.001、NMS IoU=.7、max_det=300、rect=True。AP 评价不使用推理默认 conf=.25。默认仍要求完整数据审计清单；`--allow-legacy-data` 只用于明确标记的历史诊断，不能将未审计数据变成可信独立基线。历史演示模型用于新划分的限制仍由原 runner 拦截。

实际评估后保留 `runtime_data.yaml`、`data_resolution.json`、`portable_command.json` 和原 runner 的记录。原 runner 的命令可能包含已删除的临时 YAML 路径；应按 `portable_command.json` 重现命令，或使用保留下来的 `runtime_data.yaml`。这类输出属于新运行，不会改写历史五组结果。

## 原入口对应关系

| 用途 | 当前入口 | 与原入口的关系 |
|---|---|---|
| 图片或文件夹推理 | `python cli.py predict ...` | 新增明确的本地权重、IoU、输出参数；原 `run.py predict` 保留 |
| 审计数据路径 | `python cli.py check-data ...` | 规范化路径后调用原 `workflow.verify_dataset` |
| 常规验证/测试 | `python cli.py val ...` | 规范化后委托原 `run.py val`；默认 val |
| 标准/P2 原训练 | `python train_helmet.py --variant standard` / `p2` | 原始参数与初始化不变；本阶段未启动 |
| 旧 P2 分支 CBAM 原训练 | `python train_p2_cbam.py` | 原始训练实现保留 |
| 标准版 P3-CBAM 原训练 | `python tools/p3_cbam_standard/train.py` | 原始独立实验代码保留 |
| P2 版 P3-CBAM 原训练 | `python tools/p3_cbam_p2/train.py` | 原始独立实验代码保留 |
| 历史五组统一评价 | `tools/final_evaluation/evaluate.py` | 冻结协议代码，含作者本机路径，不宣称可直接跨平台运行 |
| 交互演示 | `python app.py` | 现有 Gradio 支持五组已核验本地权重、IoU 调整及同图双模型对照；缺失权重的方案禁用，详见 [Demo 说明](GRADIO_DEMO.md) |

原训练入口并不自动使用新路径适配。需要在另一电脑复现时，先明确数据根目录并生成一份机器本地配置，避免直接把相对 YAML 交给旧 runner：

```bash
python cli.py check-data --data data.example.yaml --data-root datasets/helmet_scene_isolated_v1 --write-config outputs/data_resolved.yaml
python train_helmet.py --variant p2 --data outputs/data_resolved.yaml --dry-run
```

这里只做训练前检查，不启动训练。原 `train_helmet.py` 还要求准备项目根目录的官方 `yolov8n.pt` 初始化权重，不能用训练后的 best.pt 替代。P3 两个实验仍按各自已冻结的迁移说明配置，不擅自更改其参数或初始化。

## 结构与自定义模块

- 标准：P3/P4/P5 检测；P2：额外的 P2 浅层高分辨率分支。
- 标准 + P3-CBAM：特征层 15 → CBAM 22 → Detect 23，64 通道。
- P2 + P3-CBAM：特征层 21 → CBAM 28 → Detect 29，64 通道。
- 旧 P2 + P2-CBAM：特征层 18 → CBAM 28 → Detect 29，32 通道。

三种注意力结构复用 `helmet_cbam.P2CBAM`，类名保留用于兼容现有权重；它并不表示所有调用都作用于 P2。`cli.py` 加载模型前注册该类，不修改安装的 Ultralytics 源码。详见 [模块来源](../models/configs/p2_cbam_sources.json) 与 [最终协议](../reports/final_evaluation/protocol.json)。
