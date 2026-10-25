# P2＋CBAM 正式训练与 Windows 迁移

2026-10-25：训练、检查、续训和验证集评价入口已补齐；**尚未启动正式训练，尚无CBAM准确率结果**。

## GitHub使用者入口

下面的Finder更新包是作者本地迁移产物，尚未作为Release发布。GitHub使用者可取得本仓库的完整代码，在具备**同一隔离数据清单、官方预训练权重和原Windows环境**的项目中使用 `tools/windows_cbam/00_check.bat`；不要将本地更新包路径误认为公开下载链接。公开材料范围见[复现说明](REPRODUCIBILITY.md)。

## 作者本机迁移步骤

1. 在 Mac 的 `yolov8 → 02_实验结果` 找到 `01_P2_CBAM_Windows更新包.zip`，传到游戏本。这个小包只有代码和配置，不包含数据集、权重或 Mac 虚拟环境。
2. 解压后，将 `P2_CBAM_Windows_Update` 文件夹放进**上次训练P2的项目根目录**，与旧 `train_helmet.py`、`datasets`、`runs` 同级。不要在另一个空文件夹里创建新项目。
3. 双击更新文件夹中的 `00_apply_update.bat`。它先核对更新包和原数据清单，再备份旧代码并更新文件；不训练、不改原图/标签/权重、不安装或升级Python依赖。
4. 回到项目，打开 `tools → windows_cbam`，双击 `00_check.bat`。只有看到“核查通过，没有开始训练”才继续。
5. 双击 `01_train.bat`，开始新的 **P2＋CBAM 50轮训练**。建议游戏本接电，使用与上次相同的电源和性能模式。

**不要再运行旧的 standard 或 p2 命令来训练 CBAM。** 也不用自动把标准版、P2重新跑一遍。当前先完成新增的P2＋CBAM实验。

## 固定参数：与上次 P2 相同

依据 `runs2/helmet_p2_isolated_640_windows/args.yaml` 提取并逐项核对，保存在 `models/configs/p2_cbam_training_protocol.json` 与独立参考表 `p2_reference_parameters.json`。

| 项目 | 固定设置 |
|---|---|
| 数据 | helmet_scene_isolated_v1；原7507张的清单、图片、标注、组隔离证据逐项核查 |
| 训练/验证/测试划分 | 6488 / 511 / 508，不重新分配 |
| 输入 / batch / workers | 640 / 2 / 0 |
| 轮数 / 随机种子 | 50 / 0 |
| 优化器 / 初始学习率 | AdamW / 0.001667 |
| 最终学习率比例 / 权重衰减 | 0.01 / 0.0005 |
| momentum / nbs | 0.9 / 64 |
| 预热 | 3轮，warmup_momentum=0.8，warmup_bias_lr=0 |
| Mosaic | 1.0，最后10轮关闭 |
| scale / translate / mixup | 0.5 / 0.1 / 0 |
| 水平翻转 / 垂直翻转 | 0.5 / 0 |
| HSV | h=0.015、s=0.7、v=0.4 |
| AMP / deterministic / cos_lr | true / true / false |
| patience | 100，与原记录相同；本实验最多50轮 |

其余损失权重、增强、训练和验证设置同样由协议显式传入，并在首次参数更新前核对实际生效值。命令行不接受改epochs、batch、seed、lr0、imgsz或data等参数。显存不足时停止，不允许自动减小batch后继续。

初始化与原P2保持同一策略：官方YOLOv8n的共有层0～15加载预训练参数，其余原P2层使用相同种子的初始化。新增CBAM后，原P2的所有非CBAM初始状态逐张量对齐，仅增加CBAM参数。**没有从原P2的50轮best.pt继续微调**；那会形成另一种实验条件。

## 环境与公平性的边界

正式训练要求沿用原记录：Windows、Python **3.10.10**、torch **2.14.0+cu126**、Ultralytics **8.4.164**，GPU设备0。使用同一台RTX3060游戏本。版本不符会停止，不会偷偷换CPU或升级环境。

脚本依次寻找 `YOLO_PYTHON` 环境变量、项目 `.venv`、项目 `venv`、最后PATH里的python。如果自动选到了不同环境，请在**上次训练P2的同一个终端环境**运行：

```bat
python -B train_p2_cbam.py --check
python -B train_p2_cbam.py
```

也可以将 `YOLO_PYTHON` 设置为原训练环境的python.exe完整路径，再启动脚本。不要为了通过检查重新安装最新版PyTorch。

本次记录GPU型号、显存、CUDA版本和已安装包版本。历史记录没有完整依赖锁与GPU驱动信息，因此不能声称所有运行环境细节已被追溯证明相同。固定参数和初始化不等于多次训练统计证据；单次小幅提升仍可能受训练随机性影响。

`04_control_p2_optional.bat` 是**可选**的匹配P2控制组入口，使用同一新入口与相同协议，仅去掉CBAM，结果保存到单独目录。需要更严格控制环境时再运行；它不会被检查或CBAM训练脚本自动启动。

## 中断后续训

- 当前轮结束保存后再按一次Ctrl+C，等待程序退出。
- `03_check_resume.bat`：先核查last.pt、完整优化器/EMA/scaler、轮次、results.csv、代码和数据身份，不训练。
- `02_resume.bat`：从已保存的完整轮次继续，总预算仍为50轮。
- 完成50轮后被训练库精简过的last.pt不能续训；best.pt也不能当作完整续训检查点。
- 续训检查失败会停止，绝不会自动从第1轮重开。
- 中断会丢失本轮尚未保存的进度，且训练库未完整保存随机数与数据加载器状态，不能承诺与不中断训练逐位相同。

默认续训文件：
`runs/helmet_p2_cbam_isolated_640_windows/weights/last.pt`

指定其他已登记实验目录时：

```bat
python -B train_p2_cbam.py --resume "runs\自己的实验名\weights\last.pt" --check
python -B train_p2_cbam.py --resume "runs\自己的实验名\weights\last.pt"
```

新训练遇到已存在目录会停止，避免覆盖。如果之前在保存任何轮次前就失败，先解决错误，再使用新的文件夹名，例如 `--name helmet_p2_cbam_retry`。自定义名字只改变存储位置，不能改变参数。续训该名字时请传对应last.pt路径。

## 训练结束后交哪些结果

保留完整 `runs/helmet_p2_cbam_isolated_640_windows` 文件夹，包括：

- `results.csv`、训练曲线、验证图和 `weights/best.pt`。
- `experiment_identity.json`、`initialization.json`：协议、数据、代码指纹和初始化来源。
- `effective_args.json`、`environment.json`、`launch_events.json`：实际参数、环境、每次启动和续训记录。
- `completed.json`、`training_artifact.json`：完成记录和最佳权重哈希。

训练只用train更新参数、用val选择最佳权重，不自动跑test，也不改变当前Gradio应用的正式模型。记录保存在独立的 `runs/latest_p2_cbam_experiment.json`。

如需训练后重新验证：

```bat
python -B tools\windows_cbam\evaluate_validation.py
```

该入口使用已登记的best.pt和相同验证集，AP的候选框阈值0.001、NMS IoU 0.7、max_det300；不会把固定置信度下的召回误称为mAP。现有508张test已经被多次查看，不再称为全新盲测集。

## 这次代码做了什么，验证了什么

- `train_p2_cbam.py`：固定协议、路径迁移、数据校验、初始化、训练前断言、严格续训、独立结果登记。
- `train_helmet.py`：兼容新入口 `--variant p2_cbam`；原standard/p2路径保持原行为。
- `helmet_cbam.py`：新模块注册和受限检查点加载支持；CBAM算法和接入位置不变。
- `tools/windows_cbam`：检查、训练、续训、可选控制组、验证与备份安装脚本。

本机完成结构、参数、续训拒绝逻辑、更新备份和旧模型回归测试，并实际校验全量数据、执行640合成输入前向。**没有启动正式训练，没有在这台Mac上验证Windows批处理执行、CUDA训练显存占用或完整50轮收敛。** 游戏本上的00_check是迁移后必要的实际环境检查，检查通过也不等于训练显存一定足够。
