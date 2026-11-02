# 开源许可证审计与授权范围

审计日期：2026-10-29。范围：当前项目源码、五组模型配置及来源记录、直接运行依赖的本机包元数据、SHWD/SCUT-HEAD、官方初始化权重、五组和历史微调权重、公开展示图片。读取本地材料并核对下文上游来源；未删除、移动或改写原实验材料。

**结论：为项目有权授权的程序代码补齐 AGPL-3.0-only；数据、微调权重和部分照片的完整授权链仍未核实。代码许可补齐不等于全部材料通过发布或商业用途审查。** 当前并非一次完整的法务尽调、代码查重或全依赖供应链审计。

## 本项目采用的使用与分发规则（2026-10-29）

维护者决定沿用各组成部分的上游条件，不另造一份覆盖所有材料的“研究用途许可证”：

| 材料 | 本项目采用的规则 |
|---|---|
| 有权授权的项目程序代码 | AGPL-3.0-only；保留适用的源码提供、修改和再分发义务。**不在 AGPL 上追加“仅研究/禁止商用”条款** |
| Ultralytics 派生配置与依赖 | 保留上游 AGPL 声明；不移除版权，不声称取得 Enterprise 授权 |
| WBF 等独立第三方组件 | 保留其 MIT 等原许可和通知，不统一重新授权 |
| SHWD/SCUT-HEAD 研究材料 | 保留 SHWD 的原许可声明及来源；涉及 SCUT-HEAD 的材料按其“学术共同体研究用途”的范围处理，不擅自扩大为通用商用授权。需从原始来源按其条件取得数据，本仓库不打包原始数据 |
| 基于上述数据的五组及历史微调权重 | 当前仅作为学术研究实验记录使用和说明；**这是一项维护者用途与发布策略，不是已获得全部权利的再许可声明**。对外再分发条件仍待核实，不新增发布、不提供商业可用承诺；上游已经适用的 AGPL 义务保持有效 |
| Pexels 演示照片 | 按 Pexels 原条款使用，保留具体来源及人物/商标等边界；不擅自给这些照片加上项目自有版权或统一研究用途许可 |
| SHWD 照片、嵌图截图、社交预览 | 不纳入根代码的额外授权，不因加框、截图或项目公开而扩大权利；许可不明时不新增打包分发 |

这些规则落实了上游限制，**不能补出原本缺失的照片授权，也不代表微调权重的公开分发条件已经核实**。使用者应同时满足程序许可和实际使用材料的条件。第三阶段的[统一实验报告](experiment_report.md)只整理既有证据，不产生新的数据/权重许可。

## 根许可证的范围

根 [LICENSE](../LICENSE) 是未经改写的 GNU Affero General Public License v3 全文。项目新增并有权授权的 Python、脚本、测试和自有说明文本，除已有独立声明外，采用 **AGPL-3.0-only**；本审计说明也承担项目授权声明。Ultralytics 派生配置保留上游版权和原许可条款，不改变上游自己的版本选择权。

下列材料有独立边界：

- `tools/third_party/weighted_boxes_fusion.py` 保持原 MIT 许可及版权，见 [LICENSE-WBF](../tools/third_party/LICENSE-WBF)。整合到程序时仍需满足适用的整体 AGPL 条件。
- 安装的第三方依赖按各自许可证，不因根许可证而被重新授权。
- 数据集、标注、预训练和微调 checkpoint、照片、嵌有照片的截图及其衍生图，**不因本项目的授权声明获得额外权利**。上游已经适用的模型 AGPL 条件也不因此取消；另需核对数据与素材权利。
- 图表中的事实数值不被包装成项目独占的算法或数据所有权。第三方商标和人物权益不在代码许可中。

这不是在 AGPL 正文上添加“禁止商用”等额外限制，而是明确本项目无权替第三方授权。AGPL 本身不等同于“禁止收费”；须履行其适用的源码与再分发义务，修改版本提供网络服务还应关注第 13 条。Ultralytics 的 [许可说明](https://www.ultralytics.com/license) 另列出其对代码、架构、预训练与微调模型的许可立场，以及 Enterprise 方案。不能把它概括为“任何商业使用一律违法”，也不能据此承诺本项目可直接闭源商用。

许可证全文从本机已安装的 Ultralytics 8.4.164 发行包 `licenses/LICENSE` 原样复制，与 [Ultralytics 上游 LICENSE](https://github.com/ultralytics/ultralytics/blob/main/LICENSE) 所示 AGPLv3 文本类型一致；复制前后字节、长度和 SHA256 记录在本地阶段检查记录中。未添加未经证实的作者姓名或版权转让声明。

## 自有实现、修改与引用的区分

| 范围 | 实际来源及核验 | 授权处理 |
|---|---|---|
| 标准/P2 YAML | 从 Ultralytics 配置修改，原头部含 AGPL；[sources.json](../models/configs/sources.json) 记录 8.4.164、原配置与本地配置哈希 | 保留上游声明，不能统一改成 MIT/Apache |
| 三组 CBAM YAML | 由上述标准/P2 图结构修改；P3 两组配置位于独立 `tools/p3_cbam_*/` 目录 | 同上；项目贡献是结构接入与对照实现，不是发明 YOLO/P2 |
| `helmet_cbam.P2CBAM` | 项目实现共享 avg/max 通道 MLP 和空间门；1×1 Conv、空间无 BN，与作者参考表达有差异，见 [来源记录](../models/configs/p2_cbam_sources.json) | 纳入项目代码许可，明确机制来自 [CBAM 论文](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)；不声称注意力机制原创 |
| CBAM 作者参考 | [作者代码](https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py)，[MIT 许可](https://github.com/Jongchan/attention-module/blob/master/LICENSE)，2019 Jongchan Park | 未发现整文件复制；论文引用本身不是复制代码，若未来复制片段需保留原许可 |
| WBF | 完整复制、重命名自 ZFTurbo 的 `ensemble_boxes_wbf.py`，固定 commit `796d4b1b6125ebef795cf14363dae00da78d5a1c` | 保留 MIT 与 2020 ZFTurbo (Roman Solovyev) 版权；[版本与哈希](../tools/third_party/WBF_SOURCE.json) |
| 训练/初始化/推理/数据治理/报告脚本 | 项目定制实现，现有材料未显示为其他完整上游文件复制；它们调用 PyTorch/Ultralytics 等框架 | 有权授权的新增部分使用根 AGPL；不能据此保证全部历史片段都不存在未记录来源 |
| Ultralytics 框架源码 | 从 pip 安装于虚拟环境，非本项目重新实现，也没有把整个包作为项目源码提交 | 维持发行包 AGPL，部署或打包时保留其完整对应源码与许可义务 |

WBF 实际文件 SHA256 已与固定版本记录交叉核对：

- 代码 `a9663b421bc3adf4c42d74ccb572d1e2e149ee2a1c5aab2a6e88eec46c6b13b9`。
- MIT 文本 `3e1ed0965a0cb43ad76a1b86def4ba45da853bb4a88206efb93dc4dbbbc69c8d`。

## 依赖许可

以下按本机安装的 `*.dist-info/METADATA`、附带 LICENSE/NOTICE 及 COCO 代码头核查，不代表所有间接依赖已经穷尽；历史 Windows 安装包还需独立分发审计。仅安装依赖不同于把其发行包或二进制一起再分发。

| 依赖/实际版本 | 发行包的许可记录 |
|---|---|
| ultralytics 8.4.164 / ultralytics-thop 2.2.1 | AGPL-3.0；分类元数据含 v3 or later，保留上游自身声明 |
| torch 2.14.0 | Apache-2.0、Apache-2.0 WITH LLVM-exception、BSD-2-Clause、BSD-3-Clause、BSL-1.0、MIT 等组合声明 |
| torchvision 0.29.0 | BSD，另附发行包 LICENSE |
| NumPy 2.5.3 | BSD-3-Clause、0BSD、MIT、Zlib、CC0-1.0 组合声明 |
| Pillow 12.3.0 | MIT-CMU |
| opencv-python 5.0.0.93 | Apache 2.0，并附第三方许可清单 |
| PyYAML 6.0.3 | MIT |
| Gradio 6.30.0 | Apache-2.0 |
| SAHI 0.12.8 | MIT |
| pycocotools 2.0.11 | METADATA 写 FreeBSD，代码与 [上游文本](https://github.com/ppwwyyxx/cocoapi/blob/master/license.txt) 对应 Simplified / BSD-2-Clause |
| matplotlib 3.11.2 | Matplotlib 自有许可及历史组成许可，分类为 PSF 类 |
| pandas 3.0.6 | BSD-3-Clause |

COCO API 的代码许可不授予 COCO 照片、更不授予 SHWD 照片的使用权。未来如果分发 Docker、打包环境、桌面安装器或二进制，需要进一步收集实际分发内容的许可证和 NOTICE；本阶段不宣称这类交付已过审。

## SHWD、SCUT-HEAD 与权重

[SHWD LICENSE](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/blob/master/LICENSE) 确实为 MIT，版权为 2019 njvisionpower。但其 [README](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset#introduction) 明确说明：戴帽正样本来自 Google/Baidu，部分负样本来自 SCUT-HEAD。搜索平台不是所有原摄影作品的权利人，根 MIT 声明不能证明这些作品全部得到可再许可授权。

[SCUT-HEAD 官方 Remark](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release#remark) 明确限定学术研究用途。项目的 PartA/PartB 样本因此不能仅靠 SHWD 的根 MIT 声明认定可自由商用。上游对衍生模型发布和商业使用的具体适用范围仍需核实，不能从“含研究数据”直接推出权重一定禁止发布，也不能反过来推断权重完全没有限制。

| 对象 | 已知来源 | 尚未核实 |
|---|---|---|
| `yolov8n.pt` 初始化 | Ultralytics 官方模型；[记录的 SHA256](../models/configs/sources.json) 为 `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36` | 供应方模型许可之外的数据权利不能由本项目保证 |
| 五组 best.pt | 项目训练日志、初始化记录与 SHWD 隔离划分可追溯；五组哈希见 [清单](../models/release_manifest.json) | 数据权利及微调模型对外再分发的完整依据；原 checkpoint 内还留有历史训练路径 |
| 历史 `models/helmet_yolov8n.pt` | 已有 v1.0.0，对应旧划分 | 公开存在不等于本次重新确认全部权利；同样需要数据和上游模型许可核查 |

本阶段只做本地交付准备，**未发布新的权重 Release，也未宣称旧 Release 已完成许可整改**。不修改原 checkpoint 去清除元信息，以免破坏实验身份。若以后需提供脱敏版，必须另存新文件、新哈希，并验证预测一致性。

## 演示图片的来源

| 文件 | 核实来源 | 边界 |
|---|---|---|
| `assets/external/01.jpg`…`15.jpg` | [逐图来源](../helmet_external_test/photo_sources.csv)、[展示页](../reports/external_gallery.md)，来自 Pexels | 是缩放/预测对照图，不是原始输入或项目原创照片 |
| `assets/failure_1.jpg` / `2` / `3` | Pexels 13795569 / 8487720 / 30592246，见 [失败案例](../reports/failure_cases.md) | 预测框不消除底图权利 |
| `assets/gradio_interface.png` | 项目 Gradio 空界面截图，已目视核实 | 与包含第三方照片的截图分开归类，UI 依赖仍有原许可 |
| `assets/gradio_result.png` | 其中施工照片已定位到本地 SHWD `VOC2028/JPEGImages/000106.jpg`；缩略图检索后目视核实 | 用户操作截图不代表拥有嵌入照片的版权；独立摄影许可未查明 |
| `assets/social-preview.jpg` | 上述结果截图转换，见 [展示说明](REPOSITORY_PRESENTATION.md) | 继承相同底图权利问题 |
| 根 `bus.jpg` | 与安装包自带 `ultralytics/assets/bus.jpg` 字节相同，SHA256 `c02019c4979c191eb739ddd944445ef408dad5679acab6fd520ef9d434bfbc63` | 可追溯供应方示例，但没有独立摄影授权链；不作为新自有素材提交 |
| 批次图、错误核查图 | 通常可回溯 SHWD，未逐张查摄影权利 | 保持本地历史记录，不新增打包发布 |

[Pexels 许可](https://www.pexels.com/license/) 允许修改与展示，并有不得暗示代言、不得使可识别人物处于冒犯性语境、不得作为图库再分发等限制。保留原图片页来源，不能把这些图片再标为 AGPL 自有素材，也不宣称所有肖像或商标使用场景都已获得许可。

## 为什么不选 MIT、Apache-2.0 或 GPL-3.0 统一覆盖

| 方案 | 对当前仓库的适用性 |
|---|---|
| [MIT](https://opensource.org/license/mit) | 可用于有权独立授权的自有组件，但不能替换 Ultralytics 派生配置的 AGPL 条件。当前整个集成项目不宜统称 MIT |
| [Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0) | 同样不能重授上游 AGPL 材料；其专利条款也不会解决照片/数据的权利缺口 |
| GPL-3.0 | 不能通过换根文件来去掉 AGPL 的网络交互源码义务；AGPL 第 13 条存在特定 GPLv3 组合规则，不等于将已有 AGPL-only 部分重新授权为 GPL-only |
| AGPL-3.0 | 与当前公开程序、Ultralytics 派生结构和集成方式相符。根采用 v3-only 授权自有部分，上游有其他版本选择的声明仍原样保留；WBF 仍保留 MIT |

## 还需要确认和解决什么

1. **项目代码权利**：维护者确认新增集成代码是否受学校、雇主、合作贡献者或其他未记录片段约束。有遗漏时补明来源和授权；此次技术审计不能代替权利转让证明。
2. **数据/权重发布依据**：如要扩大用途或发布新权重，需要向 SHWD/SCUT-HEAD 等权利人核实相关许可边界。维护者点“同意”不能替代第三方授权。
3. **演示素材**：建议后续用自己拍摄且可公开的施工照片，或权利清晰的允许演示照片重新产生真实预测截图。当前未删除旧图，也未替换成伪造输出。
4. **部署目标**：如果将来做闭源产品或商用服务，分别核对 Ultralytics 授权、训练数据和人物/场景权益；本阶段不作商业合规承诺。
5. **分发方式**：新权重许可和历史路径未清理前不发布；如打包环境，还需完整间接依赖和二进制许可清单。是否处理已有公开图片/历史 Release，由维护者在具体整改方案后决定，本阶段不擅自删除远端材料。

以上缺口已经明示，不是“全部文件许可通过”。本阶段完成的是可授权代码的许可落地和来源边界记录。


## 浏览器研究演示的单独发布记录（2026-11-01）

维护者明确授权创建公开浏览器演示，并要求不泄露个人信息。本次仅发布与原始checkpoint分离的P2 ONNX、静态前端及对应源码与许可；三份配置经过逐项内容核验。原五组PT、训练照片、标签、私有日志及个人路径不随本次发布，原始文件保持不变。新身份与网址见[浏览器演示清单](../models/browser_demo_manifest.json)。此前“不新增发布”的保守策略，对这一经明确授权的独立研究演示作了范围限定的更新；不能据此宣称全部模型或原始素材均获完整再分发授权。

SHWD根MIT及其公开模型行为、SCUT-HEAD的数据库学术研究用途声明中，没有找到明确禁止本项目训练后模型或ONNX研究演示的条款；这不等于SCUT明确授予了衍生权重的不限用途再分发权。页面与模型卡披露该不确定性，不打包原始数据，不承诺商用许可。AGPL代码不额外附加非商业条款；对应源码、配置、导出方法及第三方许可均在Space提供。

ONNX的说明/训练元信息已清除，所有protobuf文本字段经过个人路径、邮箱和Token模式检查，禁止外部tensor引用。明确元信息核验不是对模型参数不含任何训练信息的形式化证明。平台会显示既有公开项目账号并处理普通访问请求；应用不上传访客照片。
