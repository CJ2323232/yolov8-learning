# 常见问题

## 找不到 best.pt

在项目根目录运行 `python download_weights.py`。仓库自带模型位于 `models/helmet_yolov8n.pt`，脚本会校验并复制到默认路径。你自己的模型可以通过 `YOLO_MODEL_PATH` 指定，例：`YOLO_MODEL_PATH=runs/my_run/weights/best.pt python app.py`（Mac/Linux）。

## 要不要每次重新训练或编辑代码

不需要。安装依赖、准备权重完成后，以后只运行 `python app.py`，或 Mac 双击启动文件。

## 浏览器打不开 / 关闭终端后不能用了

先查看终端是否有报错，使用终端实际显示的地址；7860 被占用时可能换端口。应用需持续运行，Control+C 或关机后就停止。本项目没有部署公网服务，别人的电脑访问自己的 127.0.0.1 不会访问到你的 Mac。

## 安装失败

确认激活 `.venv`，用 `python -m pip install -r requirements-gradio.txt`。`requirements.txt` 是文本安装清单。Mac、Windows 的虚拟环境激活方式不同，见主页。不要关闭 TLS 证书校验；若下载权重遇到证书问题，可从主页链接手动下载，并通过脚本校验安装。

## 可以用 CPU 吗

可以。默认自动选 MPS / CUDA / CPU。公开的输入尺寸对比使用 CPU，耗时不是 GPU/MPS 或网页端到端速度。

## hat/person 表示什么

hat 是戴安全帽的头部，person 是未戴安全帽的头部，不是整个人体。独立的安全帽物体不是目标任务中的戴帽头部。

## mAP50 91.19% 是 91.19% 图片都正确吗

不是。它是检测任务在 IoU=0.50 条件下的平均精度，不是图片正确率。展示用 conf=0.25，正式评价通常用低阈值生成完整 PR 曲线。

## 为什么不直接改进 conf 来提高测试成绩

阈值会影响误检与漏检。用验证集选设置，最终测试集用于独立评价；15 张无标注外部照片不能计算 mAP。
