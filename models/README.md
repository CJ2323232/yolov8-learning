# 已训练的模型

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

新训练模型由 `runs/latest_helmet_training.json` 登记，测试和网页优先读取该记录，不会静默把历史权重当成新训练产物。
