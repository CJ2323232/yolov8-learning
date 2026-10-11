# 输入尺寸与精度、耗时的对比

2026-10-11，在**同一已训练 best.pt、同一 758 张验证图片**上比较 416 / 640 / 960；不重新训练，不使用 759 张测试集调参。每档尺寸各评价一次，固定顺序 416 → 640 → 960。

| imgsz | Precision | Recall | mAP50 | mAP50-95 | 模型推理 ms/图 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 416 | 0.9013 | 0.7963 | 0.8611 | 0.5484 | 7.64 |
| 640 | 0.9446 | 0.8788 | 0.9343 | 0.6145 | 15.31 |
| 960 | 0.9404 | 0.9010 | 0.9526 | 0.6307 | 33.77 |

![精度与耗时比较](../assets/size_comparison.svg)

本次观察：640 相比 416 的 mAP50-95 高 6.61 个百分点；960 相比 640 高 1.62 个百分点，模型推理耗时约为 2.21 倍。当前演示继续使用 640；这不是已证明适合所有场景的最优尺寸。

## 固定条件

- 权重 SHA256：`55be821e9eefdd0a69caf4abd3fdb2a22277059e21ec155222102b32cef2e650`。
- 后端 CPU，Apple Silicon Mac / arm64；batch 4，workers 0，seed 0。
- conf=0.001、NMS IoU=0.7、max_det=300、rect=True、augment=False。
- Python 3.14.7，torch 2.14.0，Ultralytics 8.4.164。
- 指标由 Ultralytics validator 计算，**验证的 conf=0.001 与演示的 conf=0.25 用途不同**。

耗时是库报告的模型推理毫秒/图，不包含预处理、后处理、图片上传和网页渲染，不代表完整网页延迟或 MPS 性能。仅一次运行，顺序、系统负载、温度等可能影响计时；应增加重复运行后再得出稳定速度结论。本实验没有单独计算小目标 AP，不能仅凭整体 mAP 上升断言小目标改善。

[原始完整数值](size_comparison.json) · [CSV](size_comparison.csv) · [验证集图片与标签校验清单](validation_manifest.csv)

## 如何重跑

在仓库根目录，准备权重与同一数据划分后运行：

```bash
python benchmark_sizes.py --model runs/helmet_50/weights/best.pt --data helmet_dataset/data.yaml --sizes 416 640 960 --device cpu --output reports/my_size_comparison
```

输出目录不能已存在，避免覆盖旧结果。只改变 `imgsz`；模型大小和训练输入尺寸对比需要另行训练，尚未完成。
