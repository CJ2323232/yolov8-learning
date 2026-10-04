
from pathlib import Path
from ultralytics import YOLO
import torch

# 1. 设置数据集路径
BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "helmet_dataset" / "data.yaml"

# 2. 自动选择计算设备
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "0"
else:
    device = "cpu"

print(f"当前训练设备：{device}")

# 3. 加载官方预训练模型
model = YOLO("yolov8n.pt")

# 4. 开始训练
model.train(
    data=str(DATA),
    epochs=50,
    imgsz=640,
    batch=4,
    device=device,
    workers=0,
    project=str(BASE_DIR / "runs"),
    name="helmet_50"
)

print("训练完成！")

