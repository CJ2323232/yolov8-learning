"""Local helmet/head detection UI with reproducible single/ensemble presets."""
import os
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from workflow import select_model
from helmet_inference import NORMAL_MODE, SLICED_MODE
from helmet_detection import HelmetDetectionService, STANDARD, P2, NMS, WBF, LEGACY

BASE_DIR = Path(__file__).resolve().parent
DEVICE = os.environ.get('YOLO_DEVICE') or ('mps' if torch.backends.mps.is_available()
             else '0' if torch.cuda.is_available() else 'cpu')
try:
    MODEL_PATH, MODEL_ORIGIN = select_model(os.environ.get('YOLO_MODEL_PATH'))
except FileNotFoundError:
    if os.environ.get('YOLO_MODEL_PATH'):
        raise
    MODEL_PATH, MODEL_ORIGIN = None, 'missing'

service = HelmetDetectionService(BASE_DIR, DEVICE, MODEL_PATH)
CHOICES = [STANDARD, P2, NMS, WBF] + ([LEGACY] if MODEL_PATH else [])
DEFAULT = (LEGACY if os.environ.get('YOLO_MODEL_PATH') else P2 if service.available('p2')
           else STANDARD if service.available('standard') else LEGACY if MODEL_PATH else P2)


def preset_note(preset):
    if preset == NMS:
        return '运行标准版和P2，再按类别去除重叠框。推荐阈值0.30，固定640整图；当前实验更侧重减少漏检，耗时增加。'
    if preset == WBF:
        return '运行标准版和P2，再对重叠框的位置与分数做融合。推荐阈值0.17，固定640整图；当前测试总体F1较高。'
    if preset == LEGACY:
        return ('当前为自选模型。其结果不能套用标准版/P2实验指标。' if MODEL_ORIGIN != 'historical_demo_not_clean_baseline'
                else '当前为历史演示权重，使用过存在重复泄漏的旧划分；不能套用新模型实验指标。')
    return ('推荐阈值'+f'{service.threshold(preset):.2f}。单模型通常更快；切片或960为其他设置，不能直接套用640整图实验结果。')


def change_preset(preset):
    fusion = preset in [NMS, WBF]
    return (gr.update(value=service.threshold(preset)),
            gr.update(value=NORMAL_MODE, interactive=not fusion),
            gr.update(value=640, interactive=not fusion), preset_note(preset))


def detect(image: Image.Image | None, conf: float, mode=NORMAL_MODE,
           ordinary_imgsz=640, preset=DEFAULT):
    if image is None:
        return None, '请先上传一张图片。'
    try:
        annotated, counts, details = service.predict(image, preset, conf, mode, ordinary_imgsz)
    except (ImportError, ValueError, FileNotFoundError) as exc:
        raise gr.Error(str(exc)) from exc
    summary = (f'检测方案：{preset}\n共检测到 {sum(counts.values())} 个目标；最终置信度阈值 {conf:.2f}。'
               + ('\n'+'\n'.join(f'{k}: {v} 个' for k,v in counts.items()) if counts else '\n没有超过当前阈值的目标。'))
    summary += (f'\n检测方式：{mode}；输入分辨率上限：{details["imgsz"]}。'
                f'\n本次处理与绘图耗时：{details["seconds"]:.2f} 秒（首次可能包含加载模型，不等于稳定推理速度）。')
    if mode == SLICED_MODE:
        summary += f'\n切片数：{details["tiles"]}；单块时沿用普通检测。'
    if preset != LEGACY and not details['calibrated_settings']:
        summary += '\n当前已调整阈值、尺寸或切片方式，不是报告中的固定实验设置。'
    if preset == LEGACY:
        summary += '\n'+preset_note(preset)
    return annotated, summary


with gr.Blocks(analytics_enabled=False, title='YOLOv8 安全帽检测') as demo:
    gr.Markdown('# YOLOv8 安全帽检测\n上传图片，选择方案后点击「开始检测」。hat=戴安全帽头部，person=未戴帽头部；结果需人工核对。')
    with gr.Row():
        with gr.Column():
            image = gr.Image(type='pil', label='上传原始图片')
            preset = gr.Radio(CHOICES, value=DEFAULT, label='检测方案')
            note = gr.Markdown(preset_note(DEFAULT))
            conf = gr.Slider(.05, .95, value=service.threshold(DEFAULT), step=.01, label='最终置信度阈值')
            mode = gr.Radio([NORMAL_MODE, SLICED_MODE], value=NORMAL_MODE, label='单模型检测方式')
            size = gr.Dropdown([640, 960], value=640, label='单模型输入分辨率')
            with gr.Row():
                submit = gr.Button('开始检测', variant='primary')
                clear = gr.ClearButton(value='清空图片')
        with gr.Column():
            output = gr.Image(label='检测结果（可保存图片）')
            summary = gr.Textbox(label='检测统计', lines=9)
            gr.Markdown('融合会运行两个模型，通常更慢；固定阈值来自验证集。具体收益及类别差异见项目内「02_实验结果 → 融合方案正式验证」。')
    preset.change(change_preset, preset, [conf, mode, size, note])
    submit.click(detect, [image, conf, mode, size, preset], [output, summary], concurrency_limit=1)
    clear.add([image, output, summary])
demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == '__main__':
    print(f'推理设备：{DEVICE}；默认方案：{DEFAULT}；模型按需加载并复用。')
    demo.launch(inbrowser=True, server_name='127.0.0.1', share=False)
