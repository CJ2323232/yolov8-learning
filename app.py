import os
from pathlib import Path

import cv2
import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO

# app.py 和 runs 文件夹放在同一项目目录。
BASE_DIR = Path(__file__).resolve().parent
# 默认使用 50 轮实验的 best.pt；也可通过 YOLO_MODEL_PATH 指定其他权重。
MODEL_PATH = Path(os.environ.get("YOLO_MODEL_PATH", "runs/helmet_50/weights/best.pt")).expanduser()
if not MODEL_PATH.is_absolute():
    MODEL_PATH = BASE_DIR / MODEL_PATH

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"找不到训练好的模型：{MODEL_PATH}\n"
        "请先训练模型或放入已有权重；也可设置 YOLO_MODEL_PATH 指定 best.pt。"
    )

if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "0"
else:
    DEVICE = "cpu"

print(f"加载模型：{MODEL_PATH}")
print(f"推理设备：{DEVICE}")
model = YOLO(str(MODEL_PATH))


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
    description="上传图片后点击 Submit 进行检测。hat 表示戴安全帽的头部目标，person 表示未戴安全帽的头部目标。检测结果需要人工核对。",
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1", share=False)
