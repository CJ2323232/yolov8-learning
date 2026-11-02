"""Local research demo: verified models, transparent outputs and same-image comparison."""
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import tempfile

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault('GRADIO_TEMP_DIR', str(BASE_DIR/'project_work/gradio_cache'))
os.environ.setdefault('GRADIO_ANALYTICS_ENABLED', 'False')

import gradio as gr
import torch
from PIL import Image

from workflow import select_model, RELEASE_SHA256, sha256
from helmet_inference import NORMAL_MODE, SLICED_MODE, SLICE_SIZE, OVERLAP
from helmet_detection import (HelmetDetectionService, STANDARD, P2, NMS, WBF, LEGACY,
    STANDARD_P3_CBAM, P2_P3_CBAM, P2_P2_CBAM, SINGLE_MODELS)

LOGGER = logging.getLogger(__name__)
CLASS_NAMES = {'hat': '戴安全帽头部', 'person': '未戴帽头部'}
TABLE_HEADERS = ['编号', '类别', '置信度', 'x1', 'y1', 'x2', 'y2']
DEVICE = os.environ.get('YOLO_DEVICE') or ('mps' if torch.backends.mps.is_available()
             else '0' if torch.cuda.is_available() else 'cpu')
ONLINE_DEMO = os.environ.get('YOLO_ONLINE_DEMO') == '1'


def legacy_selection(selector=select_model):
    explicit = os.environ.get('YOLO_MODEL_PATH')
    try:
        path, origin = selector(explicit)
        return path, origin, ''
    except (OSError, ValueError, KeyError, TypeError) as exc:
        LOGGER.warning('Optional model registration unavailable: %s', exc)
        warning = ('指定的自选模型不可用，请检查YOLO_MODEL_PATH；下方只列出其他可用模型。'
                   if explicit else '旧训练登记不可用，已跳过；不影响清单中可用的正式实验模型。')
        historical = BASE_DIR/'models/helmet_yolov8n.pt'
        if not explicit and historical.is_file() and sha256(historical) == RELEASE_SHA256:
            return historical, 'historical_demo_not_clean_baseline', warning
        return None, 'missing', warning


MODEL_PATH, MODEL_ORIGIN, STARTUP_WARNING = legacy_selection()
service = HelmetDetectionService(BASE_DIR, DEVICE, MODEL_PATH)
CHOICES = service.available_presets()
DEFAULT = (LEGACY if os.environ.get('YOLO_MODEL_PATH') and LEGACY in CHOICES else P2 if P2 in CHOICES
           else STANDARD if STANDARD in CHOICES else CHOICES[0] if CHOICES else None)
COMPARISON_CHOICES = [choice for choice in CHOICES if choice in SINGLE_MODELS]
COMPARE_A = STANDARD if STANDARD in COMPARISON_CHOICES else next(iter(COMPARISON_CHOICES), None)
COMPARE_B = P2 if P2 in COMPARISON_CHOICES and P2 != COMPARE_A else next(
    (choice for choice in COMPARISON_CHOICES if choice != COMPARE_A), None)


def preset_note(preset):
    if preset is None:
        return '没有可用权重，请查看模型状态。不会自动下载或用历史权重冒充新模型。'
    if preset == NMS:
        return '历史融合验证阈值0.30；固定640整图、模型NMS IoU=0.7、融合IoU=0.55。局部召回收益有误检和耗时代价。'
    if preset == WBF:
        return '历史融合验证阈值0.17；WBF会重标度分数，不能直接等同于单模型阈值。历史同批实验的结果不能套用新设置。'
    if preset == LEGACY:
        return ('当前为自选模型，其结果不能套用五组实验指标。' if MODEL_ORIGIN != 'historical_demo_not_clean_baseline'
                else '历史演示权重见过旧划分数据，不能用于复现新的五组成绩。')
    if preset in [STANDARD_P3_CBAM, P2_P3_CBAM, P2_P2_CBAM]:
        return 'CBAM结构实验模型；共同比较默认阈值0.25，尚未单独校准阈值，不默认优于P2。'
    return (f'历史验证选择的阈值为{service.threshold(preset):.2f}。'
            '普通640为默认；手动改变参数后属于自定义设置，不能直接套用报告成绩。')


def mode_note(mode, preset):
    if preset in [NMS, WBF]:
        return '双模型融合固定640整图与模型NMS IoU=0.7；不叠加切片。融合IoU固定0.55。'
    if mode == SLICED_MODE:
        return (f'SAHI固定切片{SLICE_SIZE}×{SLICE_SIZE}，横纵重叠{OVERLAP:.0%}，同时运行整图。'
                '模型NMS IoU=0.7；跨片合并使用IOS=0.5，两者含义不同。单块时回退普通检测。')
    return '普通检测支持640/960及NMS IoU调节。NMS IoU控制重叠预测的去除，不是评价时的匹配IoU。'


def change_mode(mode, preset):
    fixed = mode == SLICED_MODE or preset in [NMS, WBF]
    return (gr.update(value=640, interactive=False) if fixed else gr.update(interactive=True),
            mode_note(mode, preset),
            gr.update(value=.7, interactive=False) if fixed else gr.update(interactive=True))


def change_preset(preset):
    fusion = preset in [NMS, WBF]
    return (gr.update(value=service.threshold(preset)), gr.update(value=NORMAL_MODE, interactive=not fusion),
            gr.update(value=640, interactive=not fusion), preset_note(preset), mode_note(NORMAL_MODE, preset),
            gr.update(value=.7, interactive=not fusion))


def table_rows(details):
    return [[row['id'], CLASS_NAMES.get(row['class_name'], row['class_name']), round(row['confidence'], 4),
             *[round(value, 2) for value in row['xyxy']]] for row in details['detections']]


def export_payload(details):
    return {'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'image': {'width': details['image_width'], 'height': details['image_height']},
            'run': {key: value for key, value in details.items() if key not in
                    ['detections', 'image_width', 'image_height']},
            'detections': details['detections']}


def export_directory():
    root = BASE_DIR/'project_work/gradio_exports'
    root.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix='request_', dir=root))


def write_exports(details):
    directory = export_directory()
    csv_path, json_path = directory/'detections.csv', directory/'detections.json'
    with csv_path.open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['id', 'class_id', 'class_name', 'confidence', 'x1', 'y1', 'x2', 'y2'])
        for row in details['detections']:
            writer.writerow([row['id'], row['class_id'], row['class_name'], row['confidence'], *row['xyxy']])
    json_path.write_text(json.dumps(export_payload(details), ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    return str(csv_path), str(json_path)


def error_message(exc):
    text = str(exc).lower()
    if isinstance(exc, RuntimeError) and any(word in text for word in ['out of memory', 'not enough memory']):
        return '设备内存或显存不足。请使用普通640单模型检测，关闭其他占用程序，或用YOLO_DEVICE=cpu启动。'
    if isinstance(exc, (ValueError, FileNotFoundError, ImportError)):
        return str(exc)
    return '图片、权重或设备处理失败。请确认图片可打开、权重完整、设备可用；详细原因已写到启动终端。'


def run_prediction(image, preset, conf, mode, size, iou, labels, scores, ids):
    if image is None:
        raise ValueError('请先上传没有检测框和文字的原图。')
    return service.predict(image, preset, conf, mode, size, nms_iou=iou,
                           show_labels=labels, show_conf=scores, show_box_ids=ids)


def describe(counts, details):
    text = (f'实际方案：{details["preset"]}\n共预测{sum(counts.values())}个头部；'
            + '，'.join(f'{CLASS_NAMES.get(name, name)}：{count}' for name, count in counts.items())
            + f'\n方式：{details["mode"]}；尺寸：{details["imgsz"]}；设备：{details["device"]}'
            + f'\nconf={details["confidence"]:.2f}；模型NMS IoU={details["nms_iou"]:.2f}'
            + f'\n处理与绘图：{details["seconds"]:.3f}秒（可能含首次加载，不是稳定速度或网页完整延迟）。')
    if not counts:
        text += '\n没有超过当前阈值的目标。'
    if details.get('sahi'):
        text += f'\n切片数：{details["tiles"]}；跨片IOS合并阈值0.5。'
    if details['preset'] == LEGACY:
        text += '\n'+preset_note(LEGACY)
    elif not details['calibrated_settings']:
        text += '\n此配置未独立校准或已改变参数；不直接套用固定实验成绩。'
    text += '\n单张未标注图片不计算准确率、误检、漏检、召回率或F1；置信度不是准确率。'
    return text


def detect(image, conf, mode=NORMAL_MODE, ordinary_imgsz=640, preset=DEFAULT,
           nms_iou=.7, show_labels=True, show_conf=True, show_box_ids=False):
    try:
        annotated, counts, details = run_prediction(image, preset, conf, mode, ordinary_imgsz, nms_iou,
                                                    show_labels, show_conf, show_box_ids)
        csv_path, json_path = write_exports(details)
        return annotated, describe(counts, details), table_rows(details), export_payload(details)['run'], csv_path, json_path
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        LOGGER.exception('Single-image demo request failed')
        raise gr.Error(error_message(exc)) from exc


def compare(image, model_a, model_b, conf=.25, nms_iou=.7,
            show_labels=True, show_conf=True, show_box_ids=False):
    try:
        if model_a not in COMPARISON_CHOICES or model_b not in COMPARISON_CHOICES:
            raise ValueError('同图对照需要两份可用的五组实验权重；不使用历史演示权重代替。')
        if model_a == model_b:
            raise ValueError('请为A和B选择不同模型。')
        first = run_prediction(image, model_a, conf, NORMAL_MODE, 640, nms_iou, show_labels, show_conf, show_box_ids)
        second = run_prediction(image, model_b, conf, NORMAL_MODE, 640, nms_iou, show_labels, show_conf, show_box_ids)
        rows = [[name, sum(counts.values()), counts.get('hat', 0), counts.get('person', 0), round(details['seconds'], 3)]
                for name, (_, counts, details) in zip([model_a, model_b], [first, second])]
        payload = {'comparison': 'same original image, ordinary640, shared confidence and NMS IoU',
                   'note': 'Predictions only; no ground truth or accuracy claims. Timings may include first loading.',
                   'A': export_payload(first[2]), 'B': export_payload(second[2])}
        path = export_directory()/'comparison.json'
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
        return first[0], second[0], rows, table_rows(first[2]), table_rows(second[2]), str(path)
    except (ImportError, ValueError, RuntimeError, OSError) as exc:
        LOGGER.exception('Same-image comparison request failed')
        raise gr.Error(error_message(exc)) from exc


def empty_single():
    return None, '', [], {}, None, None


def empty_comparison():
    return None, None, [], [], [], None


def lock_controls():
    return [gr.update(interactive=False) for _ in RUN_CONTROLS]


def unlock_controls(preset, mode):
    fusion = preset in [NMS, WBF]
    adjustable = not fusion and mode == NORMAL_MODE
    flags = [True, bool(CHOICES), True, not fusion, adjustable, adjustable,
             bool(CHOICES), True, True, bool(COMPARISON_CHOICES), bool(COMPARISON_CHOICES),
             True, True, len(COMPARISON_CHOICES) >= 2, True, True, True, True]
    return [gr.update(interactive=flag) for flag in flags]


with gr.Blocks(analytics_enabled=False, title='YOLOv8 安全帽检测') as demo:
    default_name = 'P2' if DEFAULT == P2 else DEFAULT or '暂无可用模型'
    gr.Markdown(f'# YOLOv8 安全帽头部检测\n默认{default_name}、普通640、conf={service.threshold(DEFAULT):.2f}。'
                '上传没有预测框和文字的原图；结果需人工核对。')
    if STARTUP_WARNING:
        gr.Markdown('⚠ '+STARTUP_WARNING)
    with gr.Accordion('模型文件与可用状态', open=False):
        gr.Dataframe(headers=['模型', '状态', '项目内文件'], value=service.model_status(), interactive=False)
        gr.Markdown('只列出本地可用模型供选择，不自动下载。实验权重的使用范围与分发条件见项目许可证说明。')
    with gr.Accordion('画框显示选项（不改变预测）', open=False):
        with gr.Row():
            labels = gr.Checkbox(True, label='显示类别')
            scores = gr.Checkbox(True, label='显示置信度')
            ids = gr.Checkbox(False, label='显示框编号（对应明细表）')
    with gr.Tab('单图检测'):
        with gr.Row():
            with gr.Column():
                image = gr.Image(type='pil', label='上传原图', sources=['upload'], format='png', height=420)
                preset = gr.Dropdown(CHOICES, value=DEFAULT, label='模型 / 方案')
                note = gr.Markdown(preset_note(DEFAULT))
                conf = gr.Slider(.05, .95, value=service.threshold(DEFAULT), step=.01, label='置信度阈值')
                with gr.Accordion('高级检测设置（普通 / SAHI / NMS）', open=False):
                    mode = gr.Radio([NORMAL_MODE, SLICED_MODE], value=NORMAL_MODE, label='单模型检测方式',
                                    interactive=DEFAULT not in [NMS, WBF])
                    size = gr.Dropdown([640, 960], value=640, label='普通检测输入尺寸',
                                       interactive=DEFAULT not in [NMS, WBF])
                    iou = gr.Slider(.05, .95, value=.7, step=.01, label='模型NMS IoU',
                                    interactive=DEFAULT not in [NMS, WBF])
                    mode_info = gr.Markdown(mode_note(NORMAL_MODE, DEFAULT))
                with gr.Row():
                    submit = gr.Button('开始检测', variant='primary', interactive=bool(CHOICES))
                    clear = gr.ClearButton(value='清空图片与结果')
            with gr.Column():
                output = gr.Image(label='真实预测结果（可下载）', format='png', height=420)
                summary = gr.Textbox(label='本次实际运行信息', lines=10)
        detail_table = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False, label='逐框明细 · 坐标为原图像素')
        with gr.Accordion('权重身份与完整运行参数', open=False):
            parameters = gr.JSON(label='实际运行参数与权重SHA256')
        with gr.Row():
            csv_download = gr.File(label='下载逐框CSV', interactive=False)
            json_download = gr.File(label='下载JSON（含参数）', interactive=False)
        single_outputs = [output, summary, detail_table, parameters, csv_download, json_download]
        preset.change(change_preset, preset, [conf, mode, size, note, mode_info, iou])
        mode.change(change_mode, [mode, preset], [size, mode_info, iou])
        image.change(empty_single, outputs=single_outputs, queue=False, api_name=False)
        clear.add([image, *single_outputs])
    with gr.Tab('同图双模型对照'):
        gr.Markdown('同一原图、普通640、共同conf和NMS IoU。默认conf=0.25用于比较，不改写历史融合阈值。'
                    '框数增加不等于召回率提高；无标注时只比较预测。耗时可能包含首次加载，不能当速度基准。')
        if len(COMPARISON_CHOICES) < 2:
            gr.Markdown('需要至少两份可用的正式实验权重；当前仅可使用单图演示。')
        compare_image = gr.Image(type='pil', label='对照原图', sources=['upload'], format='png', height=360)
        with gr.Row():
            model_a = gr.Dropdown(COMPARISON_CHOICES, value=COMPARE_A, label='模型A')
            model_b = gr.Dropdown(COMPARISON_CHOICES, value=COMPARE_B, label='模型B')
        with gr.Row():
            compare_conf = gr.Slider(.05, .95, value=.25, step=.01, label='共同置信度')
            compare_iou = gr.Slider(.05, .95, value=.7, step=.01, label='共同NMS IoU')
        with gr.Row():
            compare_submit = gr.Button('运行同图对照', variant='primary', interactive=len(COMPARISON_CHOICES) >= 2)
            compare_clear = gr.ClearButton(value='清空对照')
        with gr.Row():
            output_a = gr.Image(label='模型A真实预测', format='png', height=420)
            output_b = gr.Image(label='模型B真实预测', format='png', height=420)
        compare_summary = gr.Dataframe(headers=['模型', '总预测数', '戴帽头部', '未戴帽头部', '处理绘图秒'],
                                       value=[], interactive=False, label='预测数量与单次耗时（非准确率）')
        with gr.Row():
            table_a = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False, label='A逐框明细')
            table_b = gr.Dataframe(headers=TABLE_HEADERS, value=[], interactive=False, label='B逐框明细')
        comparison_download = gr.File(label='下载两组预测和参数JSON', interactive=False)
        comparison_outputs = [output_a, output_b, compare_summary, table_a, table_b, comparison_download]
        compare_image.change(empty_comparison, outputs=comparison_outputs, queue=False, api_name=False)
        compare_clear.add([compare_image, *comparison_outputs])
    # Lock both uploaders and clears until the chained request finishes, including
    # on failure, so an old result cannot be applied to a newly uploaded image.
    RUN_CONTROLS = [image, preset, conf, mode, size, iou, submit, clear, compare_image,
                    model_a, model_b, compare_conf, compare_iou, compare_submit, compare_clear,
                    labels, scores, ids]
    single_event = submit.click(lock_controls, outputs=RUN_CONTROLS, queue=False, api_name=False).then(
        detect, [image, conf, mode, size, preset, iou, labels, scores, ids], single_outputs,
        concurrency_id='helmet_inference', concurrency_limit=1, api_name='detect')
    single_event.success(unlock_controls, [preset, mode], RUN_CONTROLS, queue=False, api_name=False)
    single_event.failure(unlock_controls, [preset, mode], RUN_CONTROLS, queue=False, api_name=False)
    pair_event = compare_submit.click(lock_controls, outputs=RUN_CONTROLS, queue=False, api_name=False).then(
        compare, [compare_image, model_a, model_b, compare_conf, compare_iou, labels, scores, ids],
        comparison_outputs, concurrency_id='helmet_inference', concurrency_limit=1, api_name='compare')
    pair_event.success(unlock_controls, [preset, mode], RUN_CONTROLS, queue=False, api_name=False)
    pair_event.failure(unlock_controls, [preset, mode], RUN_CONTROLS, queue=False, api_name=False)
    gr.Markdown('hat=戴安全帽头部；person=未戴帽头部，并非全身行人。代码采用AGPL，研究数据/图片/权重另有上游条件。'
                + ('在线研究演示：图片在提供演示的设备上处理并产生临时文件，请勿上传敏感图片；清空界面不会立即删除这些文件。'
                   if ONLINE_DEMO else
                   '所有处理在本机进行；导出保存在项目project_work/gradio_exports，清空界面不会删除这些文件。'))

demo.queue(default_concurrency_limit=1, max_size=8)

if __name__ == '__main__':
    print(f'推理设备：{DEVICE}；默认方案：{DEFAULT}；模型按需加载并复用。')
    demo.launch(inbrowser=True, server_name='127.0.0.1', share=False)
