# SAHI 实验公开资料

[完整报告](../sahi_evaluation.md) · [项目主页](../../README.md)

- `case04_*`：单张图片的配对诊断与错误预测来源结论，非独立测试。
- `slice_*`：2026-10-17普通与640/960/1280切片对比。
- `single_tile_postprocess_diagnosis.json`：额外合并的离线重放诊断。
- `code_fix_validation.json`：单块分支代码修复当时的回归检查。
- `fix_*`：2026-10-18修正版529张实跑，与历史两条基线对照。
- `selected_manifest.csv`、`screening_decisions.csv`：固定名单、哈希与筛查决策。
- `reproduce.py`、`repro/`：发布整理时新增的复现入口、冻结旧实现及原实验评价函数；命令和环境要求见完整报告。
- `artifact_manifest.json`：原始本地文件与公开副本的SHA256映射；个人绝对路径已改为相对路径，数值未改。

原图、原标签文件与本地图片对照页不随仓库分发。完整原始预测日志保留本地，公开逐图计数、汇总和日志哈希；需要具体框时可用匹配清单的数据复现。原实验核验JSON中的SHA指当时原始文件，不直接等于经过路径整理的公开副本SHA，请结合artifact_manifest.json查看。
