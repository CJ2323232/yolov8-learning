# YOLOv8n-P2 browser artifact

## Scope and provenance

This package exports an existing YOLOv8n-P2 checkpoint for single-image browser inference. It does not retrain, replace, or revise the original experiment. There is no CBAM module in this selected P2 model.

The source model was trained on SHWD using Ultralytics YOLOv8. Its detection head uses P2/P3/P4/P5 features with strides **4/8/16/32**. The source inventory records **2,926,824 parameters** and validation-based selection of epoch **48** from a **50-epoch** run. Historical research results belong to the [full project](https://github.com/CJ2323232/yolov8-learning); they are not new measurements of this browser artifact.

Class IDs preserve SHWD's original terminology:

| ID | Label | Interpretation |
|---|---|---|
| 0 | `hat` | Helmeted head |
| 1 | `person` | Bare head, not a full-body person |

## Artifact identity

| Item | Value |
|---|---|
| Public artifact | [helmet_p2.onnx](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/resolve/main/helmet_p2.onnx) |
| File size | 12,417,480 bytes |
| ONNX SHA-256 | `7122cb4675126e230b7b0322bca2ef5eceb34382c3fe9ac914d0fb8a5ac92d91` |
| Original checkpoint SHA-256 | `c6322674f5ddda41b2354551d283d8fb386a3be1e56a1d50666b29d2a2f70e23` |
| Format | FP32 ONNX, opset 17 |
| Input | `images`: float32 RGB, shape `[1, 3, 640, 640]` |
| Output | `output0`: shape `[1, 6, 34000]`, decoded raw predictions |
| Default confidence / NMS IoU | 0.25 / 0.70 |
| NMS | Class-aware; performed by browser code, maximum 300 detections |

The original checkpoint is not bundled. The public ONNX is a separately named artifact with its own checksum. Export preparation preserved the original checkpoint bytes.

The 34,000 output positions are candidate locations across four feature scales, not a detected-object count. The six channels contain decoded `center_x, center_y, width, height` in input-pixel coordinates followed by two class probabilities. There is **no additional objectness channel**; class scores have already passed through sigmoid.

## Preprocessing and postprocessing

1. Decode an image to RGB.
2. Preserve aspect ratio while resizing into a centered 640 × 640 letterbox with padding value 114.
3. Arrange RGB channels as NCHW float32 and divide by 255.
4. Run the ONNX graph.
5. Select each candidate's highest-scoring class, filter by confidence, convert boxes to corner coordinates, and apply class-aware NMS.
6. Remove letterbox offsets, restore the original-image scale, and clip boxes to image bounds.

The fixed-shape export uses a square input. Comparisons against Python should use the same input tensor or fixed-square preprocessing, not silently compare with a rectangular `rect=True` pipeline. Browser Canvas interpolation can differ from OpenCV interpolation, and browser image decoding may also differ.

## Export settings and recorded checks

The export used Ultralytics **8.4.164**, PyTorch **2.14.0**, ONNX **1.23.2**, and native ONNX Runtime **1.31.1** for validation:

- `imgsz=640`, `batch=1`, `device=cpu`.
- `dynamic=False`, `simplify=False`, `opset=17`.
- `nms=None`: retain raw predictions under this Ultralytics version's export semantics.
- FP32 graph; no quantization experiment.

Three same-tensor comparisons against the PyTorch model were recorded:

| Input case | Maximum absolute output error |
|---|---:|
| Zero tensor | 0.001220703125 |
| Seeded random tensor, seed 0 | 0.000579833984375 |
| One locally prepared image tensor | 0.001861572265625 |

Each comparison passed NumPy `assert_allclose` with `rtol=0.002` and `atol=0.002`. The same input tensor was supplied to both runtimes, so these checks isolate model conversion from image decoding and resizing.

This is **limited conversion validation**, not a full accuracy evaluation. It does not establish browser end-to-end equivalence, constant latency, all-image equivalence, or statistical significance. No new AP, Recall, F1, or inference-speed result is claimed. Hosted UI and browser-runtime checks must be documented separately when performed.

The [source directory](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/tree/main/source) includes the configuration, training-related code, provenance records, and [export script](https://huggingface.co/spaces/CJ2323232/helmet-detection-browser/blob/main/source/export_browser_model.py). Training data, original checkpoints, private verification tensors, and private images are not part of this static demo.

## Privacy treatment

Export preparation removed ONNX descriptions and metadata properties, replaced graph/node names with generic names, and scanned serialized text fields for personal-path, email, and token patterns. External tensor references were disallowed. The published model uses the checksum above.

These are checks on the exported artifact's explicit text fields; they are not a claim of formal privacy certification or a proof that learned parameters can never reveal information about training data. See [PRIVACY.md](PRIVACY.md) for the application's data flow.

## Intended use and limitations

The service is an academic research and learning demonstration of helmeted-head and bare-head detection. Small, occluded, blurred, distant, or unfamiliar heads can be missed or misclassified. Confidence is not a calibrated safety probability. An empty result does not establish that a scene contains no heads or no safety issue.

The demo contains one P2 model. The full project's Gradio interface, model selection, SAHI, ensemble methods, and evaluation tools are separate functionality; parity with them is not claimed.

## Licensing and data-rights disclosure

- Project code is [AGPL-3.0-only](../../LICENSE) within its documented scope; Ultralytics-derived architecture and model retain applicable upstream AGPL conditions. The page provides access to corresponding project source rather than presenting an exported model as a replacement for source.
- [Ultralytics licensing information](https://www.ultralytics.com/license) describes its open-source licensing basis for YOLO code and trained models.
- [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset) has a [root MIT license](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset/blob/master/LICENSE), while its README identifies web-collected photographs and SCUT-HEAD-derived samples.
- [SCUT-HEAD's stated database purpose](https://github.com/HCIILAB/SCUT-HEAD-Dataset-Release#remark) is academic-community research use. No explicit upstream provision governing this project's exported weights was identified in that statement. This absence is not a claim of affirmative, unrestricted model-redistribution permission.
- The full image-rights chain and derived-model redistribution scope remain incompletely established. This project does not grant rights it does not hold, does not distribute the training images here, and does not assure commercial-use rights.
- Research use describes this service's context; it does not add a noncommercial restriction to AGPL-covered code. Separate rights in training data and images are not relicensed by the repository's AGPL declaration.
- ONNX Runtime Web and other distributed components retain their own licenses and notices. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

A checksum identifies an artifact; it does not establish ownership, permission, accuracy, or safety suitability.
