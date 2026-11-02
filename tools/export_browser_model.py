#!/usr/bin/env python3
"""Export a separately named, privacy-reviewed browser artifact. AGPL-3.0-only.

Original checkpoints are never overwritten. Install export-only dependencies in
a separate environment; this script does not install, download, train or upload.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_fields(message, prefix=""):
    """Enumerate protobuf text, excluding tensor raw bytes and numeric arrays."""
    for field, value in message.ListFields():
        values = value if field.is_repeated else [value]
        for i, item in enumerate(values):
            key = f"{prefix}.{field.name}[{i}]"
            if field.type == field.TYPE_MESSAGE:
                yield from text_fields(item, key)
            elif field.type == field.TYPE_STRING:
                yield key, item


def clear_descriptions(message):
    for field, value in list(message.ListFields()):
        if field.name in {"doc_string", "metadata_props"}:
            message.ClearField(field.name)
        elif field.type == field.TYPE_MESSAGE:
            for item in value if field.is_repeated else [value]:
                clear_descriptions(item)


def sanitize_onnx(model):
    clear_descriptions(model)
    model.producer_name = "Ultralytics / PyTorch"
    model.producer_version = "8.4.164 / 2.14.0"
    model.graph.name = "helmet_p2"
    for i, node in enumerate(model.graph.node):
        node.name = f"node_{i}"
    forbidden = re.compile(
        r"/Users/|/home/|/private/|/var/folders/|[A-Za-z]:[\\/]|"
        r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|hf_[A-Za-z0-9]{16,}|gh[pousr]_[A-Za-z0-9]+",
        re.I,
    )
    bad = [key for key, value in text_fields(model) if forbidden.search(value)]
    if bad:
        raise ValueError(f"Private-looking ONNX text remains in fields: {bad}")
    for tensor in model.graph.initializer:
        if tensor.external_data or tensor.data_location != 0:
            raise ValueError("External tensor references are forbidden for this artifact")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory, must not exist")
    parser.add_argument("--image", type=Path, action="append", default=[], help="Local verification image; never copied")
    args = parser.parse_args(argv)
    if not args.weights.is_file():
        parser.error("The specified trusted checkpoint does not exist")
    original_hash = sha256(args.weights)
    if original_hash != args.expected_sha256:
        parser.error("Checkpoint SHA256 does not match the reviewed source")
    if args.output.exists():
        parser.error("Output must be a NEW directory; refusing to replace existing artifacts")
    for image in args.image:
        if not image.is_file():
            parser.error("A specified verification image does not exist")

    import cv2
    import numpy as np
    import onnx
    import onnxruntime as ort
    import torch
    import ultralytics
    from PIL import Image, ImageOps
    from ultralytics import YOLO
    from ultralytics.data.augment import LetterBox

    if ultralytics.__version__ != "8.4.164":
        raise RuntimeError("Use the reviewed Ultralytics 8.4.164 export environment")
    torch.set_num_threads(2)
    args.output.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="export_private_", dir=args.output.parent) as temp:
        copied = Path(temp) / "helmet_p2.pt"
        shutil.copyfile(args.weights, copied)
        yolo = YOLO(str(copied), task="detect")
        exported = Path(yolo.export(format="onnx", imgsz=640, batch=1, device="cpu",
                                   dynamic=False, simplify=False, opset=17, nms=None))
        graph = onnx.load(exported, load_external_data=False)
        sanitize_onnx(graph)
        onnx.checker.check_model(graph)
        output = args.output / "helmet_p2.onnx"
        onnx.save_model(graph, output, save_as_external_data=False)
        clean = onnx.load(output, load_external_data=False)
        sanitize_onnx(clean)  # repeat the privacy scan on the serialized artifact
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        session = ort.InferenceSession(str(output), options, providers=["CPUExecutionProvider"])
        if session.get_inputs()[0].shape != [1, 3, 640, 640]:
            raise ValueError("Unexpected input shape")
        if session.get_outputs()[0].shape != [1, 6, 34000]:
            raise ValueError("Unexpected output shape")

        # Same tensor comparison isolates model conversion from resize algorithms.
        reference = YOLO(str(copied), task="detect").model.float().cpu().eval()
        reference.fuse(verbose=False)
        cases = [("zero", np.zeros((1, 3, 640, 640), dtype=np.float32)),
                 ("seeded_noise", np.random.default_rng(0).random((1, 3, 640, 640), dtype=np.float32))]
        for i, path in enumerate(args.image):
            with Image.open(path) as im:
                rgb = np.asarray(ImageOps.exif_transpose(im).convert("RGB"))
            image = LetterBox((640, 640), auto=False)(image=rgb)
            cases.append((f"local_image_{i + 1}", np.ascontiguousarray(image.transpose(2, 0, 1)[None], dtype=np.float32) / 255))
        checks = []
        with torch.inference_mode():
            for name, tensor in cases:
                pred = reference(torch.from_numpy(tensor))
                pred = pred[0] if isinstance(pred, (tuple, list)) else pred
                expected = pred.numpy()
                actual = session.run(None, {"images": tensor})[0]
                delta = np.abs(expected - actual)
                np.testing.assert_allclose(actual, expected, rtol=2e-3, atol=2e-3)
                checks.append({"case": name, "shape": list(actual.shape),
                               "max_absolute_error": float(delta.max()),
                               "mean_absolute_error": float(delta.mean()),
                               "allclose_rtol": 0.002, "allclose_atol": 0.002, "passed": True})
                # Private scratch for browser postprocessing verification, not a public asset.
                np.save(args.output / f"{name}_output.npy", actual)
                np.save(args.output / f"{name}_input.npy", tensor)
        if sha256(args.weights) != original_hash:
            raise RuntimeError("Original checkpoint changed unexpectedly")
        info = {"schema_version": 1, "model": "YOLOv8n-P2", "sha256": sha256(output),
                "bytes": output.stat().st_size, "source_checkpoint_sha256": original_hash,
                "input_shape": [1, 3, 640, 640], "output_shape": [1, 6, 34000],
                "classes": {"0": "hat: helmeted head", "1": "person: bare head (not full body)"},
                "confidence": 0.25, "nms_iou": 0.7, "max_detections": 300,
                "format": "ONNX FP32 opset 17; raw predictions, class-aware NMS in browser",
                "code_license": "AGPL-3.0-only", "model_rights_note": "See MODEL_CARD.md; no commercial-rights assurance",
                "export": {"ultralytics": ultralytics.__version__, "torch": torch.__version__,
                           "onnx": onnx.__version__, "onnxruntime_validation": ort.__version__,
                           "nms": None, "simplify": False, "dynamic": False},
                "privacy": {"training_metadata_removed": True, "private_text_scan_passed": True,
                            "external_tensors": False, "original_checkpoint_unchanged": True}}
        (args.output / "model_info.json").write_text(json.dumps(info, indent=2) + "\n")
        (args.output / "validation.json").write_text(json.dumps({"same_tensor_checks": checks}, indent=2) + "\n")
        print(json.dumps({"model_bytes": info["bytes"], "sha256": info["sha256"],
                          "cases_passed": len(checks), "original_unchanged": True}))


if __name__ == "__main__":
    main()
