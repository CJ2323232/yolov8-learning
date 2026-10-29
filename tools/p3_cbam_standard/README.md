# 标准YOLOv8n＋P3-CBAM：统一P3位置的结构对照

本轮新实验与另一包配对：standard/P2 × 无注意力/固定P3注意力。两组都使用原helmet_cbam.P2CBAM(64,16,7)，P3 stride8、640输入为64×80×80，仅影响Detect的P3输入；其他检测分支保持父网络原路径。原P2位置的P2＋CBAM作为历史探索保留，不当成本轮P3组。

## 结构与初始化

父特征层15 → 新CBAM层22 → Detect层23，Detect输入[22, 18, 21]。额外678参数。层号因父网络不同而不同，但这里匹配了尺度、通道、公式、超参数、检测前的位置和注意力初始张量。

父网络从同一官方yolov8n.pt加载共有0～15层，其余使用原seed0初始化策略；所有非CBAM状态与本机同环境重建的父网络逐张量一致，不用已训练50轮的best.pt微调。CBAM额外在独立随机数作用域用seed0初始化，两新组注意力初始状态完全相同，避免不同网络长度消耗随机数的影响。

固定原数据6488/511/508、imgsz640、batch2、workers0、AdamW、lr0=0.001667、lrf0.01、nbs64、seed0、50轮；共67项与旧对照核对。不可用命令行改epochs/batch/lr/seed。

## 安装与运行

1.把解压后的整包文件夹放入上次训练的原项目，与train_helmet.py和datasets同级。
2.双击包根目录00_apply_update.bat，只增加tools/p3_cbam_standard。
3.在原项目进入tools/p3_cbam_standard，先双击00_check.bat。
4.检查通过后双击01_train_50epochs.bat才正式训练。两新组按顺序运行，不同时启动。

默认Python命令只检查，正式训练必须有--train：

```bat
python -B tools\p3_cbam_standard\train.py --check
python -B tools\p3_cbam_standard\train.py --train
python -B tools\p3_cbam_standard\train.py --resume --check
python -B tools\p3_cbam_standard\train.py --resume --train
```

沿用原Windows环境Python3.10.10、torch2.14.0+cu126、Ultralytics8.4.164。启动器搜索项目和上级的.venv-yolo/.venv/venv；也可通过YOLO_PYTHON明确指定原python.exe。不要升级依赖来绕过检查。安装包不含数据、权重或Mac环境。

## 独立产物

正式目录：runs/helmet_standard_p3_cbam_isolated_640_windows/。保存配置/代码快照、初始化和环境、每次启动、实际参数、CSV、图和weights/best.pt/last.pt。只写独立实验登记，不替换演示模型。

检查日志：project_work/p3_cbam_standard_training/logs/。游戏本上先03_check_resume.bat核查last.pt再02_resume.bat续训；总预算50轮，不从第1轮静默重开，不覆盖已有目录。显存不足时停止，不自动减batch。

完成后把上述整个实验目录传回，再统一评估四组。04_evaluate_val_after_training.bat仅提供val复查，不自动重复测试集。

## 检查与结论范围

本机为Mac CPU：640和矩形前向、合成2×128×128检测损失反向、通道/空间注意力与骨干梯度、Identity替换、完整父状态和官方来源、训练器重建、保存重载、参数与安装保护均检查。没有optimizer.step，也没有启动正式50轮。需要游戏本实际CUDA预检查；本机合成梯度不保证640/batch2训练显存。

原始父模型的预训练共有参数指纹必须一致；随机新增层的历史摘要也记录。不同torch构建/平台时不声称历史随机张量逐位相同；Windows预检会再次记录，若同torch构建下seed0摘要不符则停止。两组新CBAM本身使用独立seed0并核对一致。

本轮按用户约定不把续训差异作为主要比较因素。主表是单种子2×2结构对照，不是多种子统计交互显著性结论；最终模型按统一评价选择，不预设组合更好。
