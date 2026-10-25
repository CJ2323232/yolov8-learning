# 第三方来源与范围

本仓库是基于已有方法的学习与实验工程，数据、基础网络和注意力机制不属于本项目原创。

| 组成 | 来源与记录 |
|---|---|
| YOLOv8及P2结构 | [Ultralytics](https://github.com/ultralytics/ultralytics)，来源版本记录在[配置目录](models/configs/)；原项目包含AGPL-3.0及商业授权选项 |
| SHWD数据 | [Safety-Helmet-Wearing-Dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset)，训练图片不随本项目打包；示例和历史预测图见相应报告来源 |
| SAHI | [obss/sahi](https://github.com/obss/sahi)，通过依赖调用，沿用上游许可 |
| WBF | [作者仓库](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)；本地保留[MIT许可](tools/third_party/LICENSE-WBF)和[版本/文件哈希](tools/third_party/WBF_SOURCE.json) |
| CBAM | [论文](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html)与[作者实现](https://github.com/Jongchan/attention-module)；本项目按论文机制实现，差异和来源见[记录](models/configs/p2_cbam_sources.json) |
| Gradio | [gradio-app/gradio](https://github.com/gradio-app/gradio)，通过依赖调用 |

外部图片只作定性展示，不当作自有标注测试集，来源见[外部图片说明](helmet_external_test/photo_sources.csv)。不因公开可访问就宣称拥有图片再许可权。

各组件适用其原始条款。本页是来源索引，不额外授予第三方材料的使用权，也不将整个仓库笼统标记为MIT许可。
