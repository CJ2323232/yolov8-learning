# 安装环境：应用与历史训练分开

原训练环境和当前Mac应用环境使用同一代框架，但不是同一份安装清单。不要通过升级框架来消除差异，也不要直接将Mac的`.venv`复制到Windows。

| 用途 | Python / 设备 | 依赖入口 | 已有证据 |
|---|---|---|---|
| 当前应用、图片推理、Gradio | Python≥3.12；本机实际为3.14.7 / macOS arm64 | `requirements.txt`；Demo用`requirements-gradio.txt` | 既有环境记录；2026-11-02 独立新环境安装与 CPU 应用检查通过，见[验收](release_checklist.md) |
| 五组历史训练对照 | Python3.10.10 / Windows x64 / RTX3060 / CUDA12.6 | `requirements-training-windows.txt` | 原训练`environment.json`中的实际版本；本阶段未重新训练或在Windows重装 |
| 同平台环境审计 | 仅相同macOS/架构/Python | `requirements-lock-macos-py314.txt` | 已安装环境快照，不是跨平台锁文件 |

## 1. 应用或Demo

先确认`python3 --version`为3.12或更高，在仓库根目录运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gradio.txt
python -m pip check
python download_weights.py
python app.py
```

只使用命令行推理时安装`requirements.txt`，不必安装Gradio和SAHI。不要把虚拟环境安装到已有原训练环境上；环境名称已存在时先核对，不覆盖。

Windows应用可以使用Python3.12以上的新环境，例如CMD中：

```bat
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements-gradio.txt
.venv\Scripts\python.exe -m pip check
```

2026-11-02 已在 macOS arm64 / Python3.14.7 的全新独立环境完成安装、轻量测试、CPU 推理与 Gradio HTTP 检查，见[验收记录](release_checklist.md)。**未在干净 Windows 应用环境安装，也未验证所有 Python/系统组合**。支持Python下限不等于已验证每个Python/操作系统组合。NumPy2.5.3本地包元数据要求Python≥3.12，因此Python3.10不能使用此应用清单。

## 2. 保留历史Windows训练环境

历史对照固定以下实际版本：

| 包 | 训练环境 | 当前应用环境 |
|---|---|---|
| Ultralytics | 8.4.164 | 8.4.164 |
| PyTorch | 2.14.0+cu126 | 2.14.0（本机CPU/MPS构建） |
| torchvision | 0.29.0+cu126 | 0.29.0 |
| NumPy | 2.2.6 | 2.5.3 |

先使用游戏本上原来的环境；不要仅为运行推理或查看结果重装它。在仓库根目录，用原环境执行以下只读版本检查，不会训练：

```bat
python -c "import sys,torch,torchvision,numpy,ultralytics; print(sys.version); print(torch.__version__,torchvision.__version__,numpy.__version__,ultralytics.__version__); print('CUDA',torch.cuda.is_available(),torch.version.cuda)"
python -m pip check
```

对应记录应为Python3.10.10、torch2.14.0+cu126、torchvision0.29.0+cu126、NumPy2.2.6、Ultralytics8.4.164；CUDA可用且运行时为12.6。预检查仍由原训练入口核对，不通过修改epoch、batch、种子或版本绕过失败。

如果确实需要另建同版本Windows环境，先在 Python3.10.10 环境从PyTorch官方来源安装**完全匹配**的CUDA12.6构建，再安装其余依赖：

```bat
python -m pip install --index-url https://download.pytorch.org/whl/cu126 torch==2.14.0+cu126 torchvision==0.29.0+cu126
python -m pip install -r requirements-training-windows.txt
python -m pip check
```

清单保留`+cu126`，因此缺少精确CUDA构建时会报错，不会把CPU或`cu130`误记为原训练环境。2026-10-29 已只读检查官方 [torch 索引](https://download.pytorch.org/whl/cu126/torch/) 和 [torchvision 索引](https://download.pytorch.org/whl/cu126/torchvision/)，均列出对应的 `cp310-cp310-win_amd64` 构建。**没有下载轮子或在干净 Windows 环境安装**，索引中存在文件不等于目标机器安装验证通过。若以后官方源不再提供匹配构建，应保留原环境及其环境导出；不要寻找来源不明的轮子或静默换版后声称复现。

直接依赖来自[标准版P3-CBAM实际环境](../reports/final_evaluation/training/02_standard_P3_CBAM/environment.json)与[P2版P3-CBAM实际环境](../reports/final_evaluation/training/04_P2_P3_CBAM/environment.json)。这些JSON保留全部原始记录，不是由本次依赖文件倒推或改写。

## 3. 历史迁移包是什么

`tools/windows_migration/`中的安装器是早期迁移包源码快照，原安装器选择Python3.14/3.12并安装`cu130`；它不是五组已完成对照的Python3.10/`cu126`重装入口。该清单的NumPy2.5.3只适用于其Python≥3.12范围。本次保留安装器及依赖数值，只补充用途提示。

新P3实验的`tools/p3_cbam_standard/`、`tools/p3_cbam_p2/`使用已有环境，运行前核对原版本。它们与早期迁移安装器用途不同。

## 4. 本阶段验证边界

- 对照已保存的训练JSON和本机包元数据，检查版本及Python约束。
- 使用轻量清单测试检查两类环境没有混用NumPy或CUDA构建，框架版本未升级。
- 没有安装、升级或卸载现有`.venv`里的任何依赖；没有下载官方权重或运行训练。
- 依赖清单为直接版本约束，不是所有平台的完整环境锁；Windows驱动、CUDA可用性、精确轮子可获取性与干净环境安装仍需在目标机器验证。
