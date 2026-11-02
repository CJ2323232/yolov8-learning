# 在线检测与本地 Gradio

## 免费浏览器在线版（2026-11-01 已部署并验证检测）

[打开在线安全帽检测](https://cj2323232-helmet-detection-browser.static.hf.space) · [托管项目与对应源码](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/tree/main)

- 访客不需要登录或安装 Python；维护者电脑关机后仍可访问托管文件。平台持续可用性取决于其服务与规则，不保证永久不限量。
- 在线版仅为 P2 单模型普通图片检测，固定640、默认 conf=0.25 / NMS IoU=0.70 / max_det=300；完整 Gradio、多模型、SAHI、融合仍保留本地。
- 图片、文件名及 EXIF 不由应用上传；网页会下载模型与运行库，HF及其官方CDN仍会接收普通网络请求。
- 仅公开独立导出的 ONNX、必要源码、配置与许可文件；未公开原始PT、训练数据、日志、个人绝对路径、邮箱、密码或Token。公开项目账号名仍由平台显示。
- ONNX清理元信息后另存，原P2权重不变。数据与衍生模型权利范围仍有局限，不作通用商用许可承诺。
- 已完成三个同张量模型转换检查、11项JS核心回归、原框架/JS后处理对照，以及本地服务关闭后的公开网页真实检测。没有重训或新增AP评测。
- 公开页面的PNG与CSV真实下载已验证，PNG不带EXIF；未测试所有手机、Safari和Firefox。首次模型加载耗时依网络和设备而异。

源码：[deployment/browser](../deployment/browser/)；模型身份：[browser_demo_manifest.json](../models/browser_demo_manifest.json)；本地验收记录位于 `project_work/browser_demo_2026-11-01/`。Finder可以双击 `02_实验结果/00_打开在线安全帽检测.command`。

## 本地完整 Gradio与历史部署准备

以下保留既有完整 Gradio 与临时分享的使用方法。它们与上方已部署的静态浏览器演示是不同入口，历史Docker准备包没有在本次发布。

部署复用现有 `app.py` 和五组已训练模型，不重新训练。原Finder启动方式仍在本机运行；线上入口单独位于 `deployment/hf/start.py`。

### 本地完整应用与临时分享

本机使用仍双击顶层 `01_启动安全帽检测.command`。

如果需要让别人临时访问，双击 `02_实验结果/01_生成临时公开链接.command`，或从项目根目录运行：

```bash
.venv/bin/python -B deployment/hf/start.py --host 127.0.0.1 --port 7872 --share
```

Windows在已安装应用依赖的环境中使用 `python deployment/hf/start.py --host 127.0.0.1 --port 7872 --share`。
命令行显示 `Running on public URL` 后，把对应 `https://…gradio.live` 链接发给别人；每次重新启动，链接可能改变。

- 不需要新账号、订阅或重新训练。
- 模型实际在提供演示的这台电脑运行，保持网络与终端运行；按Control+C结束分享。
- Gradio启动信息说明链接最多约一周、尽力提供，不能保证持续可用。电脑休眠、退出程序或网络断开都可能使它提前失效。
- 分享链接公开可访问；模型/数据目录被文件接口阻止下载，上传上限12MB；不要上传敏感图片。
- `--share`是显式开启临时隧道；默认不带它时不创建公网链接。临时分享不是云端常驻托管，不应把短期链接写成永久简历入口。
- 若只看到本机地址或分享失败，本机功能仍可使用；公网可用性以真实打开、上传和推理检查为准。

后面的Hugging Face准备步骤作为可选历史方案保留，目前不需要继续注册其他平台。

## 可选历史方案：云端托管与账号

完整 Gradio/Docker 的历史云端部署配置已准备，但该方案尚未上传或完成云端验收。上方免费 P2 静态浏览器演示已另行部署；它不等于完整 Gradio/Docker 上线。
Hugging Face需要维护者自己的账号；注册、邮箱验证和平台协议由本人完成，密码和Token不要发到聊天中。
当前普通Gradio/Docker Space创建资格可能涉及付费计划，创建前以账号实际页面和[官方规则](https://huggingface.co/docs/hub/en/spaces-overview)为准，不能将CPU Basic无小时费用理解为账号完全免费。

权重对外再分发条件、checkpoint历史路径隐私仍在[许可证审计](license_audit.md)中待核实。先用私有Space验收；公开上线与权重分发需解决这些已知事项。准备包不代表已取得额外授权。

## 生成最小部署目录

在项目根目录运行，输出必须是 `project_work` 下不存在的新目录：

```bash
python tools/prepare_space.py --output project_work/my_space_package
```

默认只打包明确列出的应用文件与许可，不含数据集、原图、训练日志、个人登记、密钥或模型。只有代码的包不能推理，云入口会明确拒绝无权重启动。

本地准备包含原始权重的私有验收包：

```bash
python tools/prepare_space.py --output project_work/my_private_space_package --with-weights
```

复制前后核对清单中的SHA256，保留模型相对位置，不编辑权重或其内嵌历史元信息。`DEPLOYMENT_MANIFEST.json`记录打包范围；脚本不会上传、创建Space、commit或push。

## 运行与平台配置

包根目录的README含 `sdk: docker` 和 `app_port: 7860`。Docker使用Python3.12、现有固定版本、官方CPU PyTorch源和非root用户；不改成新模型版本。

已有应用环境中可先检查：

```bash
python deployment/hf/start.py --check
python deployment/hf/start.py --host 127.0.0.1 --port 7881
```

第二条命令默认只监听本机；Docker通过环境变量HOST绑定0.0.0.0，并监听7860。默认不创建Gradio临时公网隧道；只有显式加上 `--share` 才临时分享。不会自动下载替代模型权重。

若有Docker，进入生成的包目录再执行：

```bash
docker build -t helmet-demo .
docker run --rm -p 7860:7860 helmet-demo
```

没有实际完成构建时，不能宣称Docker/Linux验收通过。本地Mac测试也不能代替托管平台的构建与运行日志。

## 完整 Gradio/Docker 方案的预期行为（尚未云端验收）

- 五组模型按需加载，P2默认，普通640、conf=.25；原有融合和SAHI参数不变。
- 最大上传12MB，原有4000万像素保护和单请求串行队列保持有效。
- 线上明确提示图片在托管服务器处理，不再显示“所有处理在本机”。
- Gradio托管缓存每10分钟检查，清理超过1小时的缓存。原始逐框导出文件可保留到实例销毁；清空界面不立即擦除服务器文件。
- 模型目录和部署目录不通过Gradio文件接口提供下载；这不隐藏公共Space仓库里的文件，不能代替许可或隐私审查。
- 不包含默认示例照片，不从训练集/测试集抽图作为普通推理输入，不输出无标注图片的准确率。

上线后必须检查网页可访问、真实预测和下载有效、模型身份正确，再把真实URL加入GitHub。暂不填写占位链接冒充已上线。
