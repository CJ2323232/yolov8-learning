# 报告与适用范围

## 当前实验入口

| 阶段 | 报告 | 状态 |
|---|---|---|
| 数据治理 | [分组隔离](scene_isolation/README.md) · [基线变化](baseline_evolution.md) | 已完成已检索风险隔离，未知关系仍可能存在 |
| 标准 / P2 | [50轮训练对照](p2_training_comparison/README.md) | 单次运行，设备迁移差异已披露 |
| SAHI | [切片消融与修复](sahi_evaluation.md) | 已完成，未作为默认方案 |
| 融合 | [验证选阈值、冻结测试与速度](fusion_evaluation/README.md) | 已完成，无新训练 |
| P2＋CBAM | [结构](../docs/P2_CBAM.md) · [正式训练入口](../docs/P2_CBAM_TRAINING.md) | 准备完成，尚未训练 |

## 历史记录及范围

2026-10-13 已确认旧划分存在重复图片泄漏。test_metrics、size_comparison、training、helmet_runs、helmet_50_history 等旧文件中的分数均保留作历史记录，不能当成清理后数据的独立基准。外部图片没有人工真值，只能用于定性观察。

最新工作：[P2与当前进度](p2_progress.md) · [场景风险隔离](scene_isolation/README.md) · [跨设备迁移](../docs/WINDOWS_MIGRATION.md)。

首轮去重资料（不是当前场景隔离版本）：
- clean_dataset_audit.json：冲突隔离、JPEG 副本处理和划分统计。
- clean_dataset_manifest.csv：新划分完整图片/标签哈希。
- fix_validation.json：修复回归与小规模流程验证；不是新的正式性能结果。

以上为首轮修复时的记录。2026-10-23：新划分标准/P2两版50轮产物已核验，[验证成绩、曲线与日志](p2_training_comparison/README.md)已公开；后续固定阈值融合评价、分层召回与CPU计时已完成，见[融合评价](fusion_evaluation/README.md)。参见 [修复说明](../docs/FIXES_2026-10-13.md)。

补充：[沿用旧模型的原测试子集复评](legacy_test_reassessment.md)，未重训；结果限定为事后筛查子集，不替代新外部独立评价。

新增：[普通模式的小目标与错误诊断](small_object_analysis.md)，固定 conf=0.25、IoU=0.5；该报告只讨论普通模式；后续SAHI对比见下方。

新增：[SAHI接入、切片消融与单块修复复验](sahi_evaluation.md)：单图诊断、529张640/960/1280对比、单块额外合并定位及修正版实跑。综合F1未超过普通检测，继续默认普通；公开逐图数字、清单与复现入口。
