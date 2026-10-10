"""YOLOv8 入门：检查设备、预测、训练和验证。"""
import argparse
import json
import platform
from pathlib import Path


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=["check", "predict", "train", "val"])
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--source", default="https://ultralytics.com/images/bus.jpg")
    p.add_argument("--data", default="coco8.yaml")
    p.add_argument("--device", default="auto", help="auto、cpu、mps 或 CUDA 设备编号 0")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="实验名称；已有目录会自动递增")
    p.add_argument("--project", default="runs")
    p.add_argument("--conf", type=float, default=None, help="预测置信度阈值；不填则使用库默认值")
    p.add_argument("--split", choices=["val", "test"], default="val")
    return p


def select_device(requested, torch):
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if min(args.epochs, args.imgsz, args.batch) < 1:
        p.error("epochs、imgsz 和 batch 必须是正整数")
    if args.conf is not None and not 0 <= args.conf <= 1:
        p.error("conf 必须在 0 到 1 之间")
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("请先激活项目环境并运行：python -m pip install -r requirements.txt") from exc

    device = select_device(args.device, torch)
    environment = {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "cuda_available": torch.cuda.is_available(),
        "mps_available": torch.backends.mps.is_available(),
        "selected_device": device,
    }
    print(json.dumps(environment, ensure_ascii=False, indent=2))
    if args.mode == "check":
        return

    model = YOLO(args.model)
    options = dict(device=device, imgsz=args.imgsz, project=args.project, name=args.name or args.mode)
    if args.mode == "predict":
        source = int(args.source) if args.source.isdecimal() else args.source
        for _ in model.predict(source=source, save=True, stream=True,
                               **({"conf": args.conf} if args.conf is not None else {}), **options):
            pass
        output = Path(model.predictor.save_dir)
    elif args.mode == "train":
        model.train(data=args.data, epochs=args.epochs, batch=args.batch,
                    seed=args.seed, workers=0, **options)
        output = Path(model.trainer.save_dir)
    else:
        validation_output = {}
        model.add_callback("on_val_end", lambda validator: validation_output.update(
            save_dir=str(validator.save_dir)))
        metrics = model.val(data=args.data, batch=args.batch, workers=0,
                            split=args.split, **options)
        output = Path(validation_output["save_dir"])
        summary = {key: float(value) for key, value in metrics.results_dict.items()}
        summary.update(model=args.model, data=args.data, split=args.split,
                       device=device, imgsz=args.imgsz, batch=args.batch)
        (output / "metrics.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "command.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "environment.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"完成，结果目录：{output.resolve()}")


if __name__ == "__main__":
    main()
