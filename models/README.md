# 模型目录

五组原始 PT 权重已准备成本地审核包，尚未发布；这与下文已公开的浏览器 P2 ONNX 分别管理。见[机器可读清单](release_manifest.json)、[模型卡](MODEL_CARD.md)、[许可证审计](../docs/license_audit.md)。每份约6.2MB，合计31,209,003字节；清单使用相对路径并列出五组真实哈希。数据/预训练权利和 checkpoint 内部历史路径仍需审核，不能把“已校验”当成“可公开商用”。

当前将SHWD相关模型用于学术研究实验记录，沿用上游材料范围，不因此授予未核实的再分发权利；根代码的AGPL条款保持原样。五组数据、训练/初始化、best选择、全部评价口径和局限集中在[统一实验报告](../docs/experiment_report.md)。

已有可信本地权重可通过 `python cli.py predict --weights ...` 使用，详见[可移植入口](../docs/ENTRYPOINTS.md)。`python tools/prepare_model_delivery.py --check` 仅核验本地原件；不会上传文件，也不伪造下载地址。

| 模型 | 当前用途 | 公开状态 |
|---|---|---|
| 历史YOLOv8n PT | 本地 Gradio 快速体验与历史记录 | v1.0.0可下载；不可用于复现新成绩 |
| 浏览器 P2 ONNX | 已托管的单模型普通图片检测 | 独立脱敏导出已公开；不是原始 PT，不包含全部实验功能 |
| 新标准版 / P2 | 隔离数据上的50轮实验、融合应用 | 权重本地已核验，尚未发布下载 |
| 两种P3 CBAM与旧P2位置CBAM | 五组结构对照已完成；当前不替换P2默认 | 本地训练权重已核验，尚未发布下载 |

新标准版SHA256：`15dd7f9b51930a6a4a03cbf0b9d86092fbaa1938c70c2167035bb767d5bc9b8b`。

新P2 SHA256：`c6322674f5ddda41b2354551d283d8fb386a3be1e56a1d50666b29d2a2f70e23`。

本地应用的五组权重身份和相对路径见 [release_manifest.json](release_manifest.json)，历史融合设置见 [fusion_config.json](fusion_config.json)。应用优先读取五组清单，并允许显式环境变量覆盖路径；见[应用说明](../docs/FUSION.md)。路径存在和哈希核验通过才提供对应选项，历史演示模型不能改名替代新权重。

## 公开浏览器 P2 ONNX

[打开在线演示](https://cj2323232-helmet-detection-browser.static.hf.space) · [完整托管文件](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/tree/main) · [ONNX 下载](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/resolve/main/helmet_p2.onnx) · [模型卡](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/blob/main/MODEL_CARD.md)

- 独立导出文件：`helmet_p2.onnx`，12,417,480 字节；SHA256：`7122cb4675126e230b7b0322bca2ef5eceb34382c3fe9ac914d0fb8a5ac92d91`。
- 由上列 P2 原始 PT 导出、清理 ONNX 元信息后另存；原始 PT 未修改、未公开。身份与验证范围见 [browser_demo_manifest.json](browser_demo_manifest.json)。
- 在线版仅支持 P2、640 输入普通图片检测，不能直接作为本地 PyTorch/Gradio 的 `.pt` 权重使用；原五组训练与评价不因此完整可复现。
- 已有有限同张量转换与公开网页功能核查，没有新增整套 ONNX AP 评测或通用商用许可承诺。

## 历史演示权重

> 2026-10-13 审计更新：本页模型和数值来自已发现重复泄漏的旧划分，只作历史/定性记录，不代表新划分性能。[修复说明](../docs/FIXES_2026-10-13.md)。

`helmet_yolov8n.pt` 是本项目 YOLOv8n 在 SHWD 本地划分上的 50 轮实验最佳权重，与 `reports/test_metrics.json` 中的模型一致。未重新训练，保留原始文件校验值。

- [v1.0.0 发布页（权重、校验文件、源码）](https://github.com/CJ2323232/yolov8-learning/releases/tag/v1.0.0)
- [下载 v1.0.0 权重](https://github.com/CJ2323232/yolov8-learning/releases/download/v1.0.0/helmet_yolov8n.pt)
- [下载 SHA256SUMS](https://github.com/CJ2323232/yolov8-learning/releases/download/v1.0.0/SHA256SUMS)
- 大小：6,230,058 字节（约 5.94 MiB）
- 类别：0 = hat（戴安全帽的头部），1 = person（未戴安全帽的头部）
- SHA256：`55be821e9eefdd0a69caf4abd3fdb2a22277059e21ec155222102b32cef2e650`
- 测试集：759 张，mAP50 0.911927，mAP50-95 0.570874；不是工业安全认证。

在项目目录运行 `python download_weights.py`，在 models 目录校验或安装历史演示模型；不会覆盖不同的已有权重。若目录中已有这份正确文件，会直接核验；若缺失，下载脚本会使用上列真实历史 Release。首次联网下载后才能离线使用，不能仅凭代码仓库存在就假定所有权重已随包提供。

模型基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)，使用遵循其 [AGPL-3.0 / Enterprise 条款](https://www.ultralytics.com/license)。数据来源为 [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)。不附带原始训练图片，不承诺新场景的准确性。

原训练流程通过 `runs/latest_helmet_training.json` 登记模型；新版本地 Gradio 使用五组清单与历史融合配置，按需加载可用且通过核验的模型。P2 可用时优先选择 P2；只有历史 PT 时使用“自选／历史演示模型”。CBAM 可供显式选择，不自动取代推荐 P2。


## 模型结构配置

configs/保存standard/P2的两类nano网络蓝图和来源记录，不是训练好的权重。它们通过helmet_models.py从同一官方yolov8n.pt迁移共有层0..15，新增/分歧部分保持初始化。详见 [P2说明](../docs/P2.md)。两版50轮检查点已在本地核验，SHA256与验证结果见[结果汇总](../reports/p2_training_comparison/summary.json)。五组原始 PT 尚未发布为下载资产；历史 PT 未替换为 P2。浏览器使用的是独立公开的 P2 ONNX，二者格式和使用入口不同。

## 五组已完成模型（2026-10-28）

|模型|best.pt SHA256|
|---|---|
|标准版|`15dd7f9b51930a6a4a03cbf0b9d86092fbaa1938c70c2167035bb767d5bc9b8b`|
|标准版+P3-CBAM|`8dec06cfff37c47b2ab933dc2dbd7a1f90ffec23fcdad25443bcb151e4117ab0`|
|原始P2|`c6322674f5ddda41b2354551d283d8fb386a3be1e56a1d50666b29d2a2f70e23`|
|P2+P3-CBAM|`553d67ca469c2699f2ba272eb7c7423bf8c53d79e5aa05d7a3daab690a85f591`|
|P2+P2-CBAM|`4349cd89a8bfd8b26b4fbe6f45cd588e2cd7f34f4c78835ee98f424bf2491ecb`|

[五组尺寸、固定工作点与效率评价](../reports/final_evaluation/README.md)。当前推荐P2、conf=.25；P3 CBAM在固定工作点的误检代价更大。模型可用性以本页下载状态为准，本地已存在不等于已发布。
