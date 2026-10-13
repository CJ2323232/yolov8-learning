# Helmet-Wearing Head Detection · YOLOv8 + Gradio

A computer vision practice project using **Ultralytics YOLOv8n and SHWD**, covering data processing, training, evaluation, data quality auditing and a local interactive demo. The task detects helmet-wearing and unhelmeted **heads**.

**Python · PyTorch · Ultralytics · OpenCV · Gradio**

[中文](README.md) · [Quick start](#quick-start) · [Results](#results-and-scope) · [Technical retrospective (Chinese)](docs/PROJECT_STORY.md) · [Model release](https://github.com/CJ2323232/yolov8-learning/releases/tag/v1.0.0)

![Local Gradio demo: input image, detections and class counts](assets/gradio_result.png)

*Upload an image, adjust confidence, inspect counts and download the result. The screenshot demonstrates interaction; its training-set membership is unverified, so it is not independent test evidence.*

> **Status:** Existing weights run without retraining. Duplicate leakage was found in the historical split; auditing, workflow fixes and a screened-test reassessment are complete. Evaluation limits are stated below. Full training on the new split has not been completed.

## Implemented work

| Area | Implementation | Evidence |
| --- | --- | --- |
| Data processing | VOC-to-YOLO conversion; class/dimension/box validation; difficult/truncated metadata | [Converter](voc_to_yolo.py) · [Protocol](docs/WORKFLOW.md) |
| Training and inference | Historical 50-epoch experiment, best checkpoint and single-image predictions | [Training records](reports/training/) · [Model card](models/README.md) |
| Data quality | Content deduplication, conflicting annotation quarantine, image/label hashes, group-aware splitting | [Audit](reports/clean_dataset_audit.json) · [Splitter](split_dataset.py) |
| Traceability | Model/data identities, effective parameters, environment records and evaluation checks | [Workflow](workflow.py) · [Fix validation](reports/fix_validation.json) |
| Evaluation | 416/640/960 input-size experiment; unchanged-model reassessment with per-class metrics | [Size experiment](reports/size_comparison.md) · [Reassessment](reports/legacy_test_reassessment.md) |
| Application and analysis | Gradio UI and qualitative analysis of false positives and challenging scenes | [App](app.py) · [Cases](reports/failure_cases.md) |

The work focuses on data handling, experiments and integration. The detector architecture and training framework are provided by Ultralytics; no new detector architecture is proposed.

## Quick start

**No training or full dataset download is required.** Download the repository ZIP, or use these macOS / Linux commands:

```bash
git clone https://github.com/CJ2323232/yolov8-learning.git
cd yolov8-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gradio.txt
python download_weights.py
python app.py
```

On Windows CMD, create the environment with `python -m venv .venv` and activate it with `.venv\Scripts\activate.bat`; use the same subsequent commands. Open the printed local URL, usually `http://127.0.0.1:7860`, upload an image and click **Submit**. Keep the terminal running; stop with `Control+C`.

The app selects MPS / CUDA / CPU automatically. Weights are stored at `models/helmet_yolov8n.pt` and verified with SHA256. Registered new training artifacts take precedence; otherwise the app explicitly labels the historical demo model. It serves single-image inference locally without public hosting. [Detailed workflow (Chinese)](docs/WORKFLOW.md).

Verified local environment: Python **3.14.7**, PyTorch **2.14.0**, Ultralytics **8.4.164**, Gradio **6.30.0**. Historical training used Mac MPS; reassessment used CPU. The Mac environment snapshot does not guarantee full dependency compatibility on Windows.

## Results and scope

### Screened-test reassessment with unchanged weights

The same best checkpoint from the historical 50-epoch experiment was evaluated without training or fine-tuning. Screening retained **474 images with 4,816 ground-truth boxes** from the **original 759-image test set**. The other 285 images include confirmed duplicates, within-test redundancy and possible related images; they are not all confirmed leakage.

| Evaluation set | Images | Precision | Recall | mAP50 | mAP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Screened original-test subset | 474 | 0.9079 | 0.8544 | 0.9173 | 0.6063 |

**Scope:** This is post-hoc subset sensitivity analysis, not proof of complete scene independence or improved model performance. Evaluation used CPU, imgsz=640 and batch=4. P/R are class means at the maximum mean-F1 confidence point, not results at the UI threshold of 0.25.

[Method and limitations](reports/legacy_test_reassessment.md) · [Per-image decisions](reports/legacy_test_review/screening_decisions.csv) · [Raw metrics](reports/legacy_test_review/screened_test_metrics.json)

### Input size versus inference time

Same checkpoint and 758 historical validation images; only input size changes:

| Input size | mAP50–95 | CPU model inference ms/image |
| ---: | ---: | ---: |
| 416 | 0.5484 | 7.64 |
| 640 | 0.6145 | 15.31 |
| 960 | 0.6307 | 33.77 |

![Input size, validation precision and CPU inference time](assets/size_comparison.svg)

In this run, 960 produced **1.62 percentage points** higher mAP50–95 than 640 with about **2.2×** the model inference time. These are exploratory measurements on the historical split with duplicate leakage. Each size was measured once; timing excludes preprocessing, postprocessing and UI work. They are not independent generalization or stable hardware benchmarks; small-object AP was not separately measured.

[Conditions and original records](reports/size_comparison.md) · [Historical experiments](reports/README.md)

## A concrete failure

![A helmet on the ground is incorrectly classified as a helmet-wearing head](assets/failure_3.jpg)

A helmet on the ground was predicted as `hat` at confidence around **0.79**, despite having no wearer. This illustrates the difference between helmet appearance and the task of helmet-wearing heads. Negative examples and explicit head/helmet relations are hypotheses to test, not implemented accuracy improvements.

[Cases, sources and settings](reports/failure_cases.md) · [15 external-image examples](reports/external_gallery.md). The external images have no ground-truth boxes; no external mAP or miss rate is claimed.

## Data and reproducibility

Source: [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset). Original dataset images are not bundled. `0: hat` means a helmet-wearing head; `1: person` means an unhelmeted head, not a full-body person.

Flow: **VOC images/XML → conversion and validation → deduplication, quarantine and split → training and artifact registration → evaluation/inference → Gradio**.

Auditing quarantined **74 conflicting image variants across 37 groups** and generated a **6,005 / 750 / 752** train/validation/test split. Cross-split exact-image duplication was removed; near-duplicate and scene grouping review remains necessary. The historical checkpoint cannot serve as an independent baseline on this reshuffled split.

Historical experiments, corrected code and small-scale workflow checks are recorded separately. **Full new-split training has not been completed.** Review scene grouping before starting a new baseline, then follow the [training workflow](docs/WORKFLOW.md). Historical 1/5/50-epoch records are not three independent from-scratch controlled experiments.

| Entry points | Purpose |
| --- | --- |
| [voc_to_yolo.py](voc_to_yolo.py), [split_dataset.py](split_dataset.py) | Conversion, validation, deduplication and splitting |
| [workflow.py](workflow.py), [run.py](run.py) | Integrity checks, artifact management and execution |
| [train_helmet.py](train_helmet.py), [test_helmet.py](test_helmet.py) | Training and evaluation |
| [review_legacy_test.py](review_legacy_test.py) | Original-test screening and unchanged-model reassessment |
| [benchmark_sizes.py](benchmark_sizes.py) | Input-size experiments |
| [app.py](app.py), [predict_external.py](predict_external.py) | Local UI and external-image predictions |

[Fix log](docs/FIXES_2026-10-13.md) · [Model card](models/README.md) · [FAQ](docs/FAQ.md)

## Next steps

- Analyze false positives and misses using existing annotations, including size and occlusion effects.
- Repeat model and end-to-end timing; choose operating confidence on validation data.
- Complete scene grouping and external annotations before new baseline/model-size experiments.

Video detection, batch uploads and public hosting are not implemented. Suitability for real safety-management decisions has not been established.

## Credits and terms

Models and APIs: [Ultralytics](https://github.com/ultralytics/ultralytics), subject to its [AGPL-3.0 / Enterprise terms](https://www.ultralytics.com/license). UI: [Gradio](https://github.com/gradio-app/gradio).

Dataset: [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset). Its repository is MIT-licensed, but images include web and SCUT-HEAD sources; original data is not redistributed here. External examples have per-image credits and follow the [Pexels License](https://www.pexels.com/license/). Predictions are not claims about the subjects' actual safety status or endorsement.
