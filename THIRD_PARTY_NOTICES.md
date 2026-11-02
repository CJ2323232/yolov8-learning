# 第三方来源与范围

本仓库是基于已有方法的学习与实验工程，数据、基础网络和注意力机制不属于本项目原创。

项目有权授权的程序代码已采用 [AGPL-3.0-only](LICENSE)。完整适用范围、依赖清单和未解决的图片/权重权利问题见 [许可证审计](docs/license_audit.md)。根许可证不替代第三方独立条款。

维护者已采用[按组件区分的使用与分发规则](docs/license_audit.md#本项目采用的使用与分发规则2026-10-29)：SCUT-HEAD相关研究材料保留其学术用途范围；Pexels图片依原条款；代码AGPL不额外附加非商业限制；五组原始 PT 的发布审查尚未完成。独立公开的浏览器 P2 ONNX 见[身份清单](models/browser_demo_manifest.json)及其[模型卡](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/blob/main/MODEL_CARD.md)，仍不承诺完整衍生模型或商业用途权利。

| 组成 | 来源与记录 |
|---|---|
| YOLOv8及P2结构 | [Ultralytics](https://github.com/ultralytics/ultralytics)，来源版本记录在[配置目录](models/configs/)；原项目包含AGPL-3.0及商业授权选项 |
| SHWD数据 | [Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)，训练图片不随本项目打包；示例和历史预测图见相应报告来源 |
| SAHI | [obss/sahi](https://github.com/obss/sahi)，通过依赖调用，沿用上游许可 |
| WBF | [作者仓库](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)；本地保留[MIT许可](tools/third_party/LICENSE-WBF)和[版本/文件哈希](tools/third_party/WBF_SOURCE.json) |
| CBAM | [论文](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)与[作者实现](https://github.com/Jongchan/attention-module)；本项目按论文机制实现，差异和来源见[记录](models/configs/p2_cbam_sources.json) |
| COCO评价工具 | [COCO API](https://github.com/cocodataset/cocoapi)，使用pycocotools2.0.11官方bbox评价器，隔离安装；代码和许可沿用上游 |
| Gradio | [gradio-app/gradio](https://github.com/gradio-app/gradio)，通过依赖调用 |

外部图片只作定性展示，不当作自有标注测试集，来源见[外部图片说明](helmet_external_test/photo_sources.csv)。不因公开可访问就宣称拥有图片再许可权。

SHWD 根仓库虽有 MIT 声明，其部分图片来自研究用途受限的 SCUT-HEAD，另有 Google/Baidu 搜集照片。`assets/gradio_result.png` 中的底图已核实为 SHWD `000106.jpg`，`social-preview.jpg` 为该截图转换；它们不应被当作项目自有摄影作品。详情见审计报告，当前不新增发布这些原始图片、原始数据或五组原始 PT；浏览器 ONNX 为独立导出产物，公开范围与局限如上所述。

各组件适用其原始条款。本页是来源索引，不额外授予第三方材料的使用权，也不将整个仓库笼统标记为MIT许可。
