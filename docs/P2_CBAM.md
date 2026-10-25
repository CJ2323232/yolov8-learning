# P2＋CBAM：结构接入第一步

2026-10-24。状态：结构接通并完成合成输入核查，**没有训练，没有准确率结论，没有替换当前检测网页的正式模型**。

## 在 Finder 中查看

打开 `yolov8 → 02_实验结果 → 00_查看P2加CBAM结构核查.command`。

报告保存在 `reports/p2_cbam_structure/index.html`，机器记录是同目录 `structure_check.json`。代码统一保存在本项目目录。

## 结构和代码解读

- `helmet_cbam.py` 的 `P2CBAM`：先做通道注意力，再做空间注意力。平均池化和最大池化共用一个 MLP；用1×1卷积实现，与对池化向量做全连接等价。压缩比16，P2实际32通道，隐藏层2通道。空间卷积7×7。
- `models/configs/helmet_yolov8n_p2_cbam.yaml`：在原 P2 配置的末端增加第28层注意力，读取第18层。第29层 Detect 读取 `[28,21,24,27]`。这样P3/P4/P5分支及下采样连接不变；只增强直接送往检测头的P2特征。这个配置只支持nano宽度，不能直接改成s/m/l/x。
- `register_cbam()`：运行时告诉 Ultralytics 如何找到新类。不改虚拟环境安装包，不覆盖它自带的 CBAM 类。
- `build_cbam_model()`：先按现有策略构建原P2初始化，核对类型、拓扑、键和形状，再将原检测头28映射到新检测头29。逐张量验证非CBAM状态相同，新增CBAM状态未被覆盖。保留训练器重建时的初始化。
- `tools/check_p2_cbam.py`：检查640输入的特征尺寸和有限输出，生成中文页面和JSON记录。可加 `--self-test` 做完整测试。
- `tests/test_p2_cbam.py`：验证公式、分支隔离、初始化、恒等消融、梯度、重建、保存重载与随机状态。

这是单个 P2＋CBAM 网络，与两个已训练模型预测框融合不同。没有将SAHI、NMS/WBF融合或其他新模块同时加入本次结构。

## 来源和实现差异

机制来自 [CBAM，ECCV 2018](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)，参考[作者实现](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)。这里独立实现论文的通道→空间机制，并在来源文件中记录差异：

- 使用平均＋最大池化的共享MLP。当前安装的Ultralytics内置ChannelAttention只有平均池化和单层卷积，不等同于本实现。
- 空间注意力使用无偏置7×7卷积，**没有作者参考代码中的空间BatchNorm**。因此称为本项目CBAM实现，不能声称逐行复现作者代码。
- 没有额外残差或恒等门控，随机初始化会直接改变P2特征幅值。因此尚未训练的结果不适合与训练好的P2比较精度。
- 接入方法是已有注意力模块的应用与消融设计，不宣称原创CBAM。

## 初始化与后续训练

默认沿用已审计的官方预训练策略：共有层0～15加载官方YOLOv8n参数；其他原P2层与相同种子的P2初始化一致；CBAM新增初始化。没有加载旧50轮的P2最佳权重继续训练。默认种子0，来源和配置哈希见 `models/configs/p2_cbam_sources.json`。

这里的“与原P2一致”指相同种子下的**初始参数**，不指你已经训练完成的最佳权重。加载日志里的“442/442”是已构建模型内部复用，不能解释为442个张量均来自官方COCO预训练。官方迁移的210个状态张量和新增层状态分别记在JSON里。

2026-10-25更新：正式训练入口已补齐，支持 `train_helmet.py --variant p2_cbam`。参数固定为原P2协议，迁移后先检查再训练。见 [正式训练与Windows迁移](P2_CBAM_TRAINING.md)。

后续建议：先保持640输入、相同数据隔离版本、训练预算、优化器和增强策略，仅比较P2与P2＋CBAM。已有P2可用于初步探索参照；严谨对照需核对环境和初始化，必要时在同一游戏本做匹配控制；再有依据地增加种子重复。不能预先承诺召回提升或误检减少。

结构检查不消耗正式训练轮次：完整测试只用合成图像做一次损失反向传播，不调用optimizer.step，不读取真实训练/验证/测试图片。临时检查点自动删除，既有模型文件不改。

## 重新运行核查（不是训练）

```bash
cd /Users/cj/yolov8
.venv/bin/python -B tools/check_p2_cbam.py --self-test
```

用于后续开发的Python调用：

```python
from helmet_cbam import build_cbam_model
model, audit = build_cbam_model(seed=0)
# 这里只构建并初始化，不调用 model.train()。
```

自定义结构需先导入注册模块再由项目入口使用；不能保证任意原生 `yolo` CLI 能独立识别新类。当前已通过本机结构和训练入口检查；Windows CUDA运行仍需迁移后用00_check.bat验证。
