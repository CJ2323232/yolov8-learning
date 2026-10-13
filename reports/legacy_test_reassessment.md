# 沿用原模型的旧测试集筛查与复评

> 未重新训练。权重与原 best.pt 的 SHA256 完全一致。这是原测试集的事后筛查子集评价，不是新采集外部测试，也不证明所有场景独立。

## 方法与筛查结果

原训练集 6,064 张、原验证集 758 张作为参考；只从原测试集 759 张中筛选，不把 train/val 图片重新分入 test。规则在评价前固定，未根据检测结果或 mAP 调整样本。

检查包括文件 SHA256、解码像素哈希、感知哈希、官方预训练 YOLOv8n 检索向量，以及 ORB/RANSAC 局部特征关联。分批读取图片，原文件保持不动。

| 项目 | 图片数 |
| --- | ---: |
| 原测试集 | 759 |
| 排除：文件字节完全相同 | 5 |
| 排除：文件不同但解码像素完全相同 | 1 |
| 排除：感知近似候选，采取保守策略 | 219 |
| 排除：局部几何关联候选，采取保守策略 | 60 |
| 最终保留 | 474 |

285 张排除图片不是全部已确诊泄漏，也包含测试内部冗余和未确认候选。代表关系中，251 张关联 train、26 张关联 val、8 张关联其他 test。原 test 的 209 张 PartA 图均被筛除，保留集合为其他来源 262 张与 PartB 212 张。

新例子：`part2_000029.jpg`（原 test）与 `part2_000622.jpg`（原 val）文件不同，解码像素完全相同。

视觉复核覆盖全部 60 个几何候选、全部 28 个非 PartA 感知候选，另查看 24 个感知候选间隔抽样。可以看到同教室不同时间帧，以及裁剪、色调和水印变体。几何候选中 6 对主要由图库公共水印等产生匹配，不能确认同一照片或场景，仍按预先声明的保守策略排除。没有宣称所有保留图片通过穷尽式独立性审核。

## 实际评价

CPU / imgsz=640 / batch=4 / conf=0.001 / NMS IoU=0.7 / max_det=300 / rect=True。difficult 作为普通目标。保留集合含 hat 816 个、person 4,000 个，共 4,816 个框。

| 集合 | 图片 | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 原旧测试集（已发现泄漏） | 759 | 0.9100 | 0.8508 | 0.9119 | 0.5709 |
| 原测试集筛查后的子集 | 474 | 0.9079 | 0.8544 | 0.9173 | 0.6063 |

| 类别 | Precision | Recall | AP50 | AP50–95 |
| --- | ---: | ---: | ---: | ---: |
| hat | 0.8923 | 0.8122 | 0.8827 | 0.6527 |
| person | 0.9235 | 0.8965 | 0.9519 | 0.5599 |

**两行使用同一模型，但测试组成、目标数量与难度不同。分数差异不能解释为模型提升，也不能估算去除泄漏的因果效果。** P/R 为最大平均 F1 置信度点的类别平均，不是网页 conf=0.25 的成绩。

## 身份与局限

- 模型 SHA256：`55be821e9eefdd0a69caf4abd3fdb2a22277059e21ec155222102b32cef2e650`。
- 子集清单 SHA256：`602a7b56bd97ca7729e17329514bae7115a29a63c76e6102e4babf905de300cb`。
- 决策 SHA256：`271f278d7008e7934709d935d2b46e4292a6b8d36ef2ce61b3f8cc3567c60d3a`。
- XML 缺少可用场景/视频标识，自动检索只查有限候选，不能排除所有遗漏的近重复、同源或同场景关系。
- 历史训练未保存不可变的数据清单；本次参考现有原 train/val，不能倒推证明训练时每个文件都没有变化。
- 原 test 已评价过，本次是事后敏感性分析。严格最终泛化评价仍宜使用新收集、独立标注的数据。
- 原始图片、标签、旧划分、模型未改写；JPEG 结束标记只在评价副本补齐，没有训练或微调。

## 文件与复现

- [规则与统计](legacy_test_review/screening_report.json)
- [逐图保留/排除决策](legacy_test_review/screening_decisions.csv)
- [train/val 参考哈希](legacy_test_review/original_reference_manifest.csv)
- [保留图、源图和标签哈希](legacy_test_review/retained_test_manifest.csv)
- [完整评价结果](legacy_test_review/screened_test_metrics.json)
- [几何候选视觉复核记录](legacy_test_review/visual_review_geometric.csv)

在项目根目录运行，输出目录必须尚不存在：

```bash
python review_legacy_test.py prepare --legacy-root /你的/yolov8目录 --output reviewed_legacy_test
python review_legacy_test.py evaluate --legacy-root /你的/yolov8目录 --output reviewed_legacy_test
```

需要原划分、官方 yolov8n.pt（检索用）与原 runs/helmet_50/weights/best.pt。脚本不调用 train()。原数据图片和本地候选对照图不上传 GitHub。
