"""评价已训练的安全帽 best.pt；Mac、Windows、CPU 均可自动选设备。"""
import sys
from pathlib import Path
from run import main

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    main(["val", "--model", str(root / "runs/helmet_50/weights/best.pt"),
          "--data", str(root / "helmet_dataset/data.yaml"), "--split", "test",
          "--imgsz", "640", "--batch", "4", "--project", str(root / "runs"),
          "--name", "helmet_final_test", *sys.argv[1:]])
