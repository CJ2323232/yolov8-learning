# SAHI 切片检测接入

在项目虚拟环境内安装网页依赖，再启动现有 app.py：

```bash
python -m pip install -r requirements-gradio.txt
python app.py
```

网页在上传图片和置信度选项下新增“检测方式”。选择“普通检测”或“SAHI 切片检测”后点击 Submit；输出仍为结果图片和分类数量。建议对同一张大图使用相同置信度逐次比较。数量变化本身不代表准确性变化。

## 实现与参数

- SAHI 固定为 0.12.8，写入 requirements-gradio.txt；沿用已有 YOLO 权重和模型来源校验，不进行训练。
- [helmet_inference.py](../helmet_inference.py) 复用 app.py 加载的模型；SAHI 包装器在首次使用时创建并缓存，不重复加载第二份权重。
- 普通模式输入尺寸 640，NMS IoU=0.7，max_det=300。SAHI 每块也使用输入尺寸 640，保留模型默认的块内检测设置；合并后结果数量可能超过普通模式的单图上限。
- 切片尺寸 640×640，横纵重叠 20%。原图较小时可能只有一块。
- perform_standard_pred=True：多块时增加一次整图检测；合并使用 SAHI 的按类别 GREEDYNMM、IOS=0.5，将可能由切片边缘产生的部分重叠框合并。IOS 是交集面积与较小框面积之比，区别于 IoU；合并阈值与块内模型 NMS 是两个不同步骤。
- 图片按 EXIF 方向转换为 RGB；SAHI 合并后的框使用原图坐标，最终统一绘图和计数。
- 两种模式共享模型，使用锁串行执行，避免并发请求混用置信度或输出。
- 超过 256 个切片的请求会提示缩小图片或使用普通模式。
- 默认沿用自动设备选择，可通过 YOLO_DEVICE 指定 cpu、mps 或 cuda:0。界面耗时包含本次检测、切片、合并、绘图及首次初始化，不含上传和浏览器传输，不作为稳定性能基准。

## 当前范围

本次只接入并检查运行流程，没有在完整标注集上证明小目标召回率或 mAP 提高。原模型及其数据评价限制继续适用。切片可能增加误检、边界截断、重复框和耗时；图片本身没有的细节无法由切片恢复。

安装依赖后，原有普通模式保持为默认。模型代码和已训练权重不需要改成新网络。

来源：[SAHI 官方实现](https://github.com/obss/sahi) · [Ultralytics 接入说明](https://docs.ultralytics.com/guides/sahi-tiled-inference/) · [SAHI 论文](https://arxiv.org/abs/2202.06934)。

## 接入验证

23 项回归检查通过；真实图片的普通与切片模式、单块小图、阈值切换、Gradio 三输入 HTTP 接口均已检查。实际检查设备为 CPU。[机器可读记录](../reports/sahi_integration_validation.json)。模型与输入图片内容保持一致；这些检查不代表精度或稳定速度实验。
