# 模型目录

| 模型 | 当前用途 | 公开状态 |
|---|---|---|
| 历史YOLOv8n | 快速体验与历史记录 | v1.0.0可下载；不可用于复现新成绩 |
| 新标准版 / P2 | 隔离数据上的50轮实验、融合应用 | 权重本地已核验，尚未发布下载 |
| 两种P3 CBAM与旧P2位置CBAM | 五组结构对照已完成；当前不替换P2默认 | 本地训练权重已核验，尚未发布下载 |

新标准版SHA256：`15dd7f9b51930a6a4a03cbf0b9d86092fbaa1938c70c2167035bb767d5bc9b8b`。

新P2 SHA256：`c6322674f5ddda41b2354551d283d8fb386a3be1e56a1d50666b29d2a2f70e23`。

加载路径和阈值见[fusion_config.json](fusion_config.json)，迁移与环境变量见[应用说明](../docs/FUSION.md)。历史演示模型不能改名替代这两份权重。

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

在项目目录运行 `python download_weights.py`，在 models 目录校验或安装历史演示模型；不会覆盖不同的已有权重。克隆仓库或下载完整 ZIP 已含此文件，可以离线完成这一步。

模型基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)，使用遵循其 [AGPL-3.0 / Enterprise 条款](https://www.ultralytics.com/license)。数据来源为 [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)。不附带原始训练图片，不承诺新场景的准确性。

原训练流程通过 `runs/latest_helmet_training.json` 登记模型；新版网页根据[fusion_config.json](fusion_config.json)分别加载标准/P2，在有核验P2时优先使用P2。历史模型明确单列为演示模式。CBAM使用独立实验登记，不自动替换网页模型。


## 模型结构配置

configs/保存standard/P2的两类nano网络蓝图和来源记录，不是训练好的权重。它们通过helmet_models.py从同一官方yolov8n.pt迁移共有层0..15，新增/分歧部分保持初始化。详见 [P2说明](../docs/P2.md)。两版50轮检查点已在本地核验，SHA256与验证结果见[结果汇总](../reports/p2_training_comparison/summary.json)。新权重尚未发布为下载资产；这里的公开演示权重未替换为P2。

## 五组已完成模型（2026-10-28）

|模型|best.pt SHA256|
|---|---|
|标准版|`15dd7f9b51930a6a4a03cbf0b9d86092fbaa1938c70c2167035bb767d5bc9b8b`|
|标准版+P3-CBAM|`8dec06cfff37c47b2ab933dc2dbd7a1f90ffec23fcdad25443bcb151e4117ab0`|
|原始P2|`c6322674f5ddda41b2354551d283d8fb386a3be1e56a1d50666b29d2a2f70e23`|
|P2+P3-CBAM|`553d67ca469c2699f2ba272eb7c7423bf8c53d79e5aa05d7a3daab690a85f591`|
|P2+P2-CBAM|`4349cd89a8bfd8b26b4fbe6f45cd588e2cd7f34f4c78835ee98f424bf2491ecb`|

[五组尺寸、固定工作点与效率评价](../reports/final_evaluation/README.md)。当前推荐P2、conf=.25；P3 CBAM在固定工作点的误检代价更大。模型可用性以本页下载状态为准，本地已存在不等于已发布。
