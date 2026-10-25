<div align="center">

# Helmet Detection Lab

**Helmet-wearing head detection with auditable data splits and small-object experiments**

YOLOv8 · P2 · SAHI · NMS / WBF · Gradio

[中文](README.md) · [Quick start](#quick-start) · [Results](#results) · [Documentation](docs/README.md) · [Report an issue](https://github.com/CJ2323232/yolov8-learning/issues/new/choose)

</div>

![Local helmet detection demo](assets/gradio_result.png)

*Actual screenshot of the earlier UI, not independent evaluation evidence. Current code adds standard/P2 and NMS/WBF presets.*

A computer-vision project built on **Ultralytics YOLOv8n and SHWD**, exploring how to reduce missed small heads while measuring false alarms, latency and data leakage risks.

**Status — 2026-10-25:** standard/P2 training and fusion evaluation are complete. P2+CBAM architecture and frozen-parameter training/resume code are ready, **but CBAM has not been trained or evaluated for accuracy**. Only the historical demo weights are publicly downloadable; the new standard/P2 checkpoints are pending publication.

## What this project demonstrates

| Work | Evidence | Details |
| --- | --- | --- |
| Data governance | 7,507 retained images; retrieved cross-split candidate pairs 20,466→0; 491 assignments changed | [Audit](reports/scene_isolation/README.md) |
| P2 comparison | Best validation mAP50–95: 60.244% standard, 61.301% P2 | [Curves and conditions](reports/p2_training_comparison/README.md) |
| Ensemble tradeoffs | NMS: 23 fewer misses than P2, approximately 1.71× CPU latency | [Frozen evaluation](reports/fusion_evaluation/README.md) |
| Deployment and ablation preparation | Local Gradio presets; P2-only CBAM branch; original training settings locked | [App](docs/FUSION.md) · [Training](docs/P2_CBAM_TRAINING.md) |

Candidate pairs are not confirmed leaks. These are project-specific observations, not SOTA claims or evidence of consistent gains across seeds. The contribution is data analysis, experimental design and engineering integration; the underlying architectures and attention mechanism are prior work.

## Quick start

Run the **historical demo** without training or downloading the dataset:

```bash
git clone https://github.com/CJ2323232/yolov8-learning.git
cd yolov8-learning
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-gradio.txt
python download_weights.py
python app.py
```

On Windows, create the environment with `python -m venv .venv` and activate it with `.venv\Scripts\activate.bat`. Then use the same installation and launch commands.

Open the printed local URL, upload an unannotated image and click **开始检测**. With only the published historical checkpoint, select the historical/custom model preset. New standard/P2 and ensemble presets require the specific, currently unpublished checkpoints listed in [the model catalog](models/README.md); the old demo checkpoint cannot reproduce their results.

Local app environment: Python3.14.7, torch2.14.0, Ultralytics8.4.164, Gradio6.30.0. CBAM training reuses the previous Windows CUDA environment rather than upgrading dependencies. [Reproducibility and availability](docs/REPRODUCIBILITY.md).

## Results

**Training validation:** both models trained for 50 epochs and were evaluated on the same 511-image validation split.

| Metric at each best epoch | Standard, epoch40 | P2, epoch48 |
| --- | ---: | ---: |
| mAP50 | 91.584% | 92.995% |
| mAP50–95 | 60.244% | 61.301% |

The standard model resumed on CUDA after 13 MPS epochs; P2 trained on CUDA. This is a single-run comparison, not a strict architecture-only or multi-seed result. [Original records](reports/p2_training_comparison/README.md).

**Fusion:** thresholds selected on 511 validation images, frozen before evaluation on 508 test images with 4,875 annotations.

| Method | Threshold | Precision | Recall | F1 | CPU ms/image |
| --- | ---: | ---: | ---: | ---: | ---: |
| Standard | 0.29 | 87.82% | 86.19% | 87.00% | 21.24 |
| P2 | 0.25 | 87.23% | 87.59% | 87.41% | 33.55 |
| NMS ensemble | 0.30 | 87.51% | 88.06% | 87.78% | 57.44 |
| WBF ensemble | 0.17 | 87.95% | 87.86% | 87.90% | 53.91 |

These are fixed-threshold micro metrics with class-aware one-to-one IoU≥0.5 matching, **not mAP**. NMS was the primary fusion comparison fixed before testing. Its F1 improvement interval crosses zero; helmet-class precision decreases. The corpus has been examined previously and is not a new external blind test. CPU timing uses 8 threads, 24 images ×2 rounds after warmup, includes decode/predict/fusion, excludes weight loading, plotting and UI; it is not GPU performance. [Full evidence and limitations](reports/fusion_evaluation/README.md).

## Data, failure cases and scope

SHWD classes are `hat` = helmeted head and `person` = bare head, not full-body person detection. The current split is 6,488 train / 511 validation / 508 test. Retrieved cross-split relations have been isolated, but unknown relations may remain. All PartA classroom images are training-only, so validation/test do not measure unseen-classroom generalization.

![Failure case: an unworn helmet detected as a helmeted head](assets/failure_3.jpg)

Unworn helmets, occlusion, blur and tiny heads remain challenging. SAHI has recall/false-alarm tradeoffs and was not retained as the default. The 15 external images lack ground-truth boxes and support qualitative review only. [Failure analysis](reports/failure_cases.md) · [SAHI experiment](reports/sahi_evaluation.md).

## Roadmap

- [x] Data audit, grouped isolation and traceable manifests.
- [x] Standard/P2 training, SAHI comparison, calibrated NMS/WBF fusion and CPU benchmark.
- [x] Gradio presets, verified weights and lazy model reuse.
- [x] P2+CBAM integration and fixed-parameter training/resume preparation; 36 related local tests passed.
- [ ] Publish the new standard/P2 checkpoints with verified downloads.
- [ ] Train P2+CBAM with the original settings and report outcomes, including failures.
- [ ] New-scene evaluation, repeated seeds and CUDA benchmarking.

[Documentation](docs/README.md) · [Changelog](CHANGELOG.md) · [Contributing](CONTRIBUTING.md)

## Acknowledgements

Built with [Ultralytics](https://github.com/ultralytics/ultralytics), [SHWD](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset), [SAHI](https://github.com/obss/sahi), [WBF](https://github.com/ZFTurbo/Weighted-Boxes-Fusion), [CBAM](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html) and [Gradio](https://github.com/gradio-app/gradio). Raw training data is not bundled. See [third-party notices](THIRD_PARTY_NOTICES.md) for attribution and license references. Research/learning project; not validated for operational safety decisions.
