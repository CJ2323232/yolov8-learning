# 公开版本验收与复现边界

2026-11-02。验收对象为公开仓库基线加本次审核的代码、文档、许可证与测试。发布后核对提交中的文件字节与已测快照一致。没有重新训练或重算五组 AP，历史训练配置、日志和原始权重保持不变。

## 已实际通过

- 在同一台 macOS arm64 / Python 3.14.7 上，从公开 GitHub 仓库取得独立源码快照，创建不继承系统包的全新虚拟环境，从官方 PyPI 安装 `requirements-gradio.txt`；没有复制旧 `.venv` 或新五组私有 PT。
- `python -m pip check`：无依赖冲突。检查了框架、Gradio 等导入来自新虚拟环境，应用来自独立源码快照。
- 下列轻量测试 **57 项通过**，不需要 GPU、数据集、五组训练权重或训练操作。
- 浏览器检测协议 **11 项通过**，覆盖 letterbox、RGB/NCHW、原图坐标恢复、输入异常和分类 NMS；不等于所有浏览器兼容性通过。
- `python cli.py --help` 正常；使用公开仓库自带且 SHA256 核验通过的历史 `v1.0.0` PT，在 CPU 上完成一张外部图片的真实预测，生成画框图和 JSON。
- Gradio 在独立环境真实启动，本机 HTTP 返回 200，`/detect` 接口完成同图预测，检测图、逐框表、CSV 和 JSON 均生成。640 / conf=0.25 / NMS IoU=0.7；只有历史模型时如实显示历史选项，正式五组及双模型比较不可用。
- 核对 134 份受保护历史文件未改变；本次新增公开文件不含数据集、照片、原始 PT、ONNX 或运行库二进制。

汇总见 [机器可读检查结果](../reports/open_source_release_checks.json)。完整安装日志、运行输出和测试图片仅本地保存，不作为新性能数据公开。

## 可以重复的轻量检查

从仓库根目录、已激活的新应用环境运行：

```bash
python -m pip check
python cli.py --help
python -m unittest tests.test_portable_cli tests.test_environment_requirements tests.test_helmet_inference tests.test_fusion_app tests.test_gradio_demo tests.test_model_delivery tests.test_space_deployment tests.test_p2_cbam.CBAMMechanismTests
```

如已安装 Node.js，可运行：

```bash
node tests/browser_detector.test.mjs
```

不建议无条件运行全部结构集成测试：部分检查需要另外合法取得的官方 `yolov8n.pt` 或历史 PT。轻量测试使用合成夹具验证软件契约，不证明检测精度。

真实推理与本地网页：

```bash
python download_weights.py
python cli.py predict --weights models/helmet_yolov8n.pt --source my_images/example.jpg --conf 0.25 --iou 0.7 --device cpu --output outputs/my_first_prediction
python app.py
```

输入图片需自行准备并有权使用；输出目录必须尚不存在。历史 PT 已随当前公开仓库提供，下载器先校验；缺失时才从真实历史 Release 获取。它不能重现新五组成绩。完整安装方法见 [环境说明](ENVIRONMENTS.md)，应用行为见 [Gradio 说明](GRADIO_DEMO.md)。

若本机代理导致 `127.0.0.1` 的启动检查返回 502，先让本机地址绕过代理。例如 macOS/Linux：

```bash
NO_PROXY=127.0.0.1,localhost no_proxy=127.0.0.1,localhost python app.py
```

这是本机网络设置，不需要关闭证书验证或改模型。

## 公开交付检查

- [x] 中英文首页统一指向免费 P2 浏览器演示，区分完整本地 Gradio。
- [x] 五组数字来自原始评估记录，保留单种子、初始化差异和探索性测试集限制。
- [x] 根 AGPL-3.0-only 与第三方独立许可/材料权利说明分开；没有商用许可承诺。
- [x] 原五组 PT 未公开、历史 PT 可用、独立脱敏 P2 ONNX 已在 Hugging Face 公开，三者用途明确。
- [x] 浏览器目录标记为源码模板；完整模型与 JS/WASM 运行包从真实托管页面取得。
- [x] 只按明确文件清单同步；不删除历史公开材料，不新增照片/数据/原始权重。

## 尚未验证或未解决

- 没有在干净 Windows/CUDA、Linux/Docker 或全部受支持 Python 版本安装；同 Mac 新环境检查不能替代这些检查。
- 没有重新训练、跨种子验证、新独立数据集 AP 或浏览器 ONNX 整套 AP。
- 原五组 PT 的公开获取仍未解决；原始权重内历史路径和衍生材料权利需要单独审核。当前研究演示不表示全部材料可自由再分发或商用。
- 此轮只做 Gradio HTTP 与接口验收，没有重新逐项点击其全部网页控件；既有 UI 检查的日期和范围保留在 Demo 文档。
- CI 尚未配置；以上为本地实际执行结果，不是 GitHub Actions 状态。
- 已存在的历史提交、报告和截图可能保留历史个人路径或第三方图片。此次不改写 Git 历史，不把已有来源风险说成已经全部清除。

## 维护者手动设置

建议检查 GitHub About 中的网站是否指向在线演示，按实际功能维护 Topics/置顶项目。分支保护、合并规则和后续 Release 由维护者决定；本次不发布新的 Release 或原始五组权重。
