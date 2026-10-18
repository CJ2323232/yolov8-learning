# 报告与适用范围

2026-10-13 已确认旧划分存在重复图片泄漏。test_metrics、size_comparison、training、helmet_runs、helmet_50_history 等旧文件中的分数均保留作历史记录，不能当成清理后数据的独立基准。外部图片没有人工真值，只能用于定性观察。

当前新资料：
- clean_dataset_audit.json：冲突隔离、JPEG 副本处理和划分统计。
- clean_dataset_manifest.csv：新划分完整图片/标签哈希。
- fix_validation.json：修复回归与小规模流程验证；不是新的正式性能结果。

完整数据的新训练与评价尚未执行。参见 [修复说明](../docs/FIXES_2026-10-13.md)。

补充：[沿用旧模型的原测试子集复评](legacy_test_reassessment.md)，未重训；结果限定为事后筛查子集，不替代新外部独立评价。

新增：[普通模式的小目标与错误诊断](small_object_analysis.md)，固定 conf=0.25、IoU=0.5；该报告只讨论普通模式；后续SAHI对比见下方。

新增：[SAHI接入、切片消融与单块修复复验](sahi_evaluation.md)：单图诊断、529张640/960/1280对比、单块额外合并定位及修正版实跑。综合F1未超过普通检测，继续默认普通；公开逐图数字、清单与复现入口。
