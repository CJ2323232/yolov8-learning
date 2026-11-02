---
title: Helmet Detection Browser Demo
emoji: 🪖
colorFrom: blue
colorTo: green
sdk: static
app_file: index.html
license: agpl-3.0
pinned: false
---

# Helmet Detection Browser Demo

This GitHub directory is the **source template** for the YOLOv8n-P2 browser demo. It is not a complete runnable release: the GitHub source delivery does not add the ONNX model, `model_info.json`, the vendored JavaScript/WASM binaries, or a duplicated `source/` tree. References to those files in the application require the complete hosted package.

**[Use the verified hosted demo](https://cj2323232-helmet-detection-browser.static.hf.space)** · **[Get the complete runnable files](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/tree/main)**

It uses an existing trained model; no new training or accuracy result is introduced by this source template.

The page and model are static files. Detection runs on the visitor's device through ONNX Runtime Web rather than on the project owner's computer. The linked public Static Space has been deployed; its visitors do not need a Hugging Face account, Python, or access to the owner's computer.

This is a small single-model image demo. It does not reproduce the full local Gradio application, its five-model comparison, SAHI, or model fusion.

## Use the demo

1. Open the app and allow the model and browser runtime to load.
2. Choose an image that you have permission to use.
3. Set the confidence threshold and NMS IoU threshold, then run detection.
4. Inspect the annotated image and detection details. Empty results mean that no prediction survived the selected thresholds.

Defaults: input **640 × 640**, confidence **0.25**, class-aware NMS IoU **0.70**, maximum **300** detections. These are inference settings, not a claim of optimal settings for every scene.

The SHWD class names refer to **head boxes**:

| ID | Model label | Meaning |
|---|---|---|
| 0 | `hat` | Helmeted head |
| 1 | `person` | Bare head; not a full-person detector |

The first visit downloads the **12,417,480-byte ONNX model** plus runtime files. Startup and inference time depend on the visitor's device, browser, and connection. No browser speed benchmark is claimed here.

## Run the complete hosted package locally

Download the full files from the [Hugging Face Files tab](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/tree/main), retaining its `vendor/` directory, `helmet_p2.onnx`, `model_info.json` and license/notices. No account credentials or tokens belong in browser code. Serving this GitHub source template alone will not load a model.

From the downloaded directory containing `index.html`:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000). Python serves the files only; inference still runs in the browser. Serve over HTTP locally or HTTPS on a host, rather than opening the HTML file directly.

If loading fails, inspect the browser console and Network panel. Confirm that `helmet_p2.onnx`, `model_info.json`, and the packaged JavaScript/WASM runtime files are served successfully. Do not place access tokens in browser code.

## Model, validation, and source

- [Model card](MODEL_CARD.md): provenance, tensor format, export checks, and limitations.
- Hosted [model information](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/blob/main/model_info.json) and [ONNX artifact](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/resolve/main/helmet_p2.onnx); these are not included in this GitHub source template.
- [Corresponding project source](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/tree/main/source), including the [export script](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/blob/main/source/export_browser_model.py), [training entry](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/blob/main/source/train_helmet.py), and [P2 configuration](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/blob/main/source/models/configs/helmet_yolov8n_p2.yaml). This GitHub repository keeps its own [export tool](../../tools/export_browser_model.py), [training entry](../../train_helmet.py) and [configuration](../../models/configs/helmet_yolov8n_p2.yaml).
- [Full research project](https://github.com/CJ2323232/yolov8-learning): five-model experiments and local Gradio workflow.

Recorded conversion checks cover three identical-input tensor comparisons between PyTorch and native ONNX Runtime. The largest absolute output difference was **0.001861572265625**. These checks do not establish browser UI correctness, full-dataset accuracy, or unchanged predictions for all images.

Browser image decoding and Canvas resizing can differ from the Python/OpenCV preprocessing used for the tensor checks. This demo therefore makes no new AP, Recall, or Gradio-parity claim. See the model card for the precise scope of validation.

## Privacy

Selected images and detections are processed locally by this application. The app does not upload them or include analytics. Model and runtime downloads still make ordinary requests to the hosting service, whose own logging and privacy practices are outside the app's control. See [PRIVACY.md](PRIVACY.md).

## Research use and licensing

This service is presented as an academic research and learning demonstration. The model was trained on [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset), which includes SCUT-HEAD-derived samples and web-collected images. [SCUT-HEAD](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release#remark) states an academic-research-use condition for its database. The complete image-rights chain and explicit scope of derived-model redistribution are not fully established; this project does not promise commercial-use rights.

Project code is provided under [AGPL-3.0-only](../../LICENSE), within the documented scope. Ultralytics-derived material retains its applicable upstream conditions. The research purpose of this service is not an additional restriction added to AGPL code. Dataset and image rights remain separate. See [third-party notices](THIRD_PARTY_NOTICES.md) and the model card.

The demo is provided without warranty. It is an experimental head detector, not evidence that an unattended workplace-safety system is suitable for deployment.
