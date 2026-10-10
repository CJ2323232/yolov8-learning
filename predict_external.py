"""对 15 张外部照片预测，并保存参数、设备信息和画框结果。"""
import sys
from pathlib import Path
from run import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    main(["predict", "--model", str(root / "runs/helmet_50/weights/best.pt"),
          "--source", str(root / "helmet_external_test/external_test"),
          "--project", str(root / "runs"), "--name", "external_15_new",
          *sys.argv[1:]])
