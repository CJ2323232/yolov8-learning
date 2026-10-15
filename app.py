import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
from ultralytics import YOLO
from workflow import select_model, sha256
from helmet_inference import HelmetPredictor, NORMAL_MODE, SLICED_MODE

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
DEVICE = os.environ.get("YOLO_DEVICE", DEVICE)

print(f"加载模型：{MODEL_PATH}")
print(f"模型来源：{MODEL_ORIGIN}；SHA256：{MODEL_SHA256}")
print(f"推理设备：{DEVICE}")
model = YOLO(str(MODEL_PATH))
if model.task != 'detect' or model.names != {0: 'hat', 1: 'person'}:
    raise ValueError('网页只支持 0=hat、1=person 的两类头部检测模型')
predictor = HelmetPredictor(model, DEVICE)


def detect(image: Image.Image | None, conf: float, mode: str = NORMAL_MODE):
    if image is None:
        return None, "请先上传一张图片。"

    try:
        annotated, counts, details = predictor.predict(image, conf, mode)
    except (ImportError, ValueError) as exc:
        raise gr.Error(str(exc)) from exc

    total = sum(counts.values())
    summary = f"共检测到 {total} 个目标（置信度阈值：{conf:.2f}）。"
    if counts:
        summary += "\n" + "\n".join(f"{name}: {count} 个" for name, count in counts.items())
    else:
        summary += "\n没有检测到超过当前阈值的目标，可适当降低 conf 试试。"
    summary += f"\n检测方式：{mode}；本次处理耗时：{details['seconds']:.2f} 秒。"
    if mode == SLICED_MODE:
        summary += f"\n切片数：{details['tiles']}（640×640，重叠 20%）。"
        if details['tiles'] == 1:
            summary += "\n图片较小，本次没有分成多个切片。"
        else:
            summary += "\n结果合并了切片与整图预测，并按类别去重。"
    return annotated, summary


demo = gr.Interface(
    fn=detect,
    inputs=[
        gr.Image(type="pil", label="上传施工现场图片"),
        gr.Slider(minimum=0.05, maximum=0.95, value=0.25, step=0.05, label="置信度阈值 conf"),
        gr.Radio(choices=[NORMAL_MODE, SLICED_MODE], value=NORMAL_MODE,
                 label="检测方式", info="大图中的远处目标可尝试切片检测，处理通常更慢。"),
    ],
    outputs=[
        gr.Image(label="检测结果（可以保存图片）"),
        gr.Textbox(label="检测统计"),
    ],
    title="YOLOv8 安全帽检测 Demo",
    description=("上传图片后点击 Submit。hat=戴帽头部，person=未戴帽头部，结果需人工核对。"
                 "可选择普通检测或 SAHI 切片检测。"
                 f" 当前模型：{MODEL_PATH.name}，SHA256={MODEL_SHA256[:12]}，来源={MODEL_ORIGIN}。"
                 + ("历史演示模型使用过存在重复泄漏的旧划分，不代表清理后数据的性能。"
                    if MODEL_ORIGIN == 'historical_demo_not_clean_baseline' else "")),
    analytics_enabled=False,
)

if __name__ == "__main__":
    demo.launch(inbrowser=True, server_name="127.0.0.1", share=False)
