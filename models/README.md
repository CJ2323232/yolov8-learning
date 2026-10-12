# 已训练的模型

`helmet_yolov8n.pt` 是本项目 YOLOv8n 在 SHWD 本地划分上的 50 轮实验最佳权重，与 `reports/test_metrics.json` 中的模型一致。未重新训练，保留原始文件校验值。

- [v1.0.0 发布页（权重、校验文件、源码）](https://github.com/CJ2323232/yolov8-learning/releases/tag/v1.0.0)
- [下载 v1.0.0 权重](https://github.com/CJ2323232/yolov8-learning/releases/download/v1.0.0/helmet_yolov8n.pt)
- [下载 SHA256SUMS](https://github.com/CJ2323232/yolov8-learning/releases/download/v1.0.0/SHA256SUMS)
- 大小：6,230,058 字节（约 5.94 MiB）
- 类别：0 = hat（戴安全帽的头部），1 = person（未戴安全帽的头部）
- SHA256：`55be821e9eefdd0a69caf4abd3fdb2a22277059e21ec155222102b32cef2e650`
- 测试集：759 张，mAP50 0.911927，mAP50-95 0.570874；不是工业安全认证。

在项目目录运行 `python download_weights.py`，把此模型复制到默认推理路径并校验；不会覆盖不同的已有权重。克隆仓库或下载完整 ZIP 已含此文件，可以离线完成这一步。

模型基于 [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)，使用遵循其 [AGPL-3.0 / Enterprise 条款](https://www.ultralytics.com/license)。数据来源为 [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)。不附带原始训练图片，不承诺新场景的准确性。
