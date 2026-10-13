import os
from pathlib import Path

import cv2
import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO
from workflow import select_model, sha256

# app.py 和 runs 文件夹放在同一项目目录。
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH, MODEL_ORIGIN = select_model(os.environ.get("YOLO_MODEL_PATH"))
MODEL_SHA256 = sha256(MODEL_PATH)

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "0"
else:
    DEVICE = "cpu"

print(f"加载模型：{MODEL_PATH}")
print(f"模型来源：{MODEL_ORIGIN}；SHA256：{MODEL_SHA256}")
print(f"推理设备：{DEVICE}")
model = YOLO(str(MODEL_PATH))
if model.task != 'detect' or model.names != {0: 'hat', 1: 'person'}:
    raise ValueError('网页只支持 0=hat、1=person 的两类头部检测模型')


def detect(image: Image.Image | None, conf: float):
    if image is None:
        return None, "请先上传一张图片。"

    result = model.predict(
        source=image,
        conf=conf,
        imgsz=640,
        device=DEVICE,
        verbose=False,
    )[0]

    # YOLO 的 plot() 返回 BGR；Gradio 展示 PIL RGB 图片。
    annotated = Image.fromarray(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))

    counts = {}
    for cls_id in result.boxes.cls.tolist():
        name = model.names[int(cls_id)]
        counts[name] = counts.get(name, 0) + 1

    total = len(result.boxes)
    summary = f"共检测到 {total} 个目标（置信度阈值：{conf:.2f}）。"
    if counts:
        summary += "\n" + "\n".join(f"{name}: {count} 个" for name, count in counts.items())
    else:
        summary += "\n没有检测到超过当前阈值的目标，可适当降低 conf 试试。"
    return annotated, summary


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="上传施工现场图片"),
        gr.Slider(minimum=0.05, maximum=0.95, value=0.25, step=0.05, label="置信度阈值 conf"),
    ],
    outputs=[
        gr.Image(label="检测结果（可以保存图片）"),
        gr.Textbox(label="检测统计"),
    ],
    title="YOLOv8 安全帽检测 Demo",
    description=("上传图片后点击 Submit。hat=戴帽头部，person=未戴帽头部，结果需人工核对。"
                 f" 当前模型：{MODEL_PATH.name}，SHA256={MODEL_SHA256[:12]}，来源={MODEL_ORIGIN}。"
                 + ("历史演示模型使用过存在重复泄漏的旧划分，不代表清理后数据的性能。"
                    if MODEL_ORIGIN == 'historical_demo_not_clean_baseline' else "")),
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1", share=False)
