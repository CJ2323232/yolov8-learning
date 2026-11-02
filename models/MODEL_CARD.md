# Five-model checkpoint inventory / 五组模型说明

Status: **locally staged, not published**. No download URL is currently available for these five checkpoints. This card records exact original files; it does not authorize redistribution. The historical `v1.0.0` release is a different model delivery and cannot reproduce the five-group comparison.

## Task and provenance

These models predict **head boxes**, using SHWD classes `0: hat` (helmeted head) and `1: person` (bare head). The second class does not mean a full person. Intended use is research, learning, and manual error analysis of helmet detection. Do not treat confidence scores as calibrated safety probabilities or use these experiments as evidence of suitability for unattended safety enforcement.

Training uses the [SHWD dataset](https://github.com/njvisionpower/Safety-Helmet-Wearing-Dataset), [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics), and a local implementation of [CBAM](https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html). See [third-party notices](../THIRD_PARTY_NOTICES.md) and the [license audit](../docs/license_audit.md). SHWD includes SCUT-HEAD-derived samples and images collected from Google/Baidu. Upstream academic-use terms and image rights require review before any weight redistribution or commercial-use statement. The repository code license (`AGPL-3.0-only`, within the documented scope) does **not** resolve checkpoint, pretraining, or dataset rights.

## Exact checkpoints and recorded results

The machine-readable [inventory](release_manifest.json) contains source-relative paths, exact byte sizes, SHA-256 checksums, configuration checksums, and the matching historical evaluation evidence. The following values come from the original GPU test records, cross-checked against `reports/completed_experiments_2026-10-28/verification.json`; these are not new evaluations.

| Model | Attention input → module → Detect | Parameters | Best epoch | Test mAP50 | Test mAP50–95 |
|---|---|---:|---:|---:|---:|
| YOLOv8n | None | 3,011,238 | 40 | 86.742% | 57.575% |
| YOLOv8n + P3-CBAM | P3: 15 → 22 → 23, 64 channels | 3,011,916 | 46 | 87.064% | 57.562% |
| YOLOv8n-P2 | None | 2,926,824 | 48 | 86.780% | 58.183% |
| YOLOv8n-P2 + P3-CBAM | P3: 21 → 28 → 29, 64 channels | 2,927,502 | 39 | 87.689% | 58.410% |
| YOLOv8n-P2 + P2-CBAM | P2: 18 → 28 → 29, 32 channels | 2,927,084 | 44 | 87.019% | 57.532% |

The unrounded ratios are in the inventory. Original GPU AP used Ultralytics 8.4.164, RTX 3060, FP16, batch 4, and input size 640. Later CPU fixed-confidence measurements and COCO area AP are different protocols; they must not be silently substituted for this table.

## Training and limits

- Dataset split: 6,488 train / 511 validation / 508 test images. Test annotations: 4,875 heads (946 hat / 3,929 person). Dataset manifest SHA-256: `5781ecdffa173010f539189b15565220f5e2ac597b5f8ea72f5cf7b1e8bf63e4`.
- All five runs used 50 epochs, imgsz 640, batch 2, AdamW, lr0 0.001667, lrf 0.01, nbs 64, seed 0, and the recorded original augmentation policy. Best checkpoints were selected using validation results, not the table's test scores. Source logs and configuration records remain preserved.
- Training environment recorded Python 3.10.10, PyTorch 2.14.0+cu126 and Ultralytics 8.4.164. This is historical provenance, not a claim that this environment has been freshly installed or reproduced here. Use the project environment documentation for supported installation instructions.
- All variants share the recorded official YOLOv8n pretraining transfer over common layers 0–15; newly added/non-transferred layers were initialized separately. Official pretraining SHA-256: `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`.
- Single seed only: small differences are not statistically established. The standard pair has historical differences in some non-pretrained initial tensors; the P2 pair has matching historical initialization checks. The P2-CBAM versus P3-CBAM comparison also changes feature scale and channel count. Do not call the entire five-model comparison a strict all-variable-controlled two-factor experiment.
- The test split has already informed project exploration. It is not an untouched final external test. PartA classroom images are in training; do not claim unseen-classroom generalization. The separate 15-image showcase has no full ground-truth labels and cannot establish AP.
- The current working recommendation is the P2 single model at confidence 0.25, based on the recorded fixed-working-point tradeoff. P2+P3-CBAM has +0.2276 percentage points test mAP50–95 relative to P2, but recovered 20 true positives while adding 142 false positives at the common working point, lowering F1 by 1.023 percentage points. Highest AP does not imply best performance in every scene. See [model selection evidence](../reports/final_evaluation/model_selection.json).

## Compatibility and integrity

CBAM checkpoints require the existing `helmet_cbam.P2CBAM` implementation and project registration/loading helpers; keep the full project source available rather than loading them from an arbitrary directory with a bare upstream command. Do not rename or rewrite classes inside an already trained checkpoint. PyTorch checkpoints can contain serialized Python objects: only load files whose provenance you trust. The preparation script below performs byte-level checks and ZIP/pickle instruction inspection **without unpickling or executing a model**.

The original five `.pt` files retain historical personal absolute paths inside serialized training metadata (read-only inspection found 6 / 5 / 5 / 5 / 6 strings respectively). These do not appear in the public inventory. Original weights have not been edited or sanitized. Before public delivery, review metadata privacy and decide whether to create separately named, separately hashed distribution artifacts with validation; never overwrite these originals.

## Local delivery preparation

Run from the repository root, without a GPU or model dependencies:

```bash
python tools/prepare_model_delivery.py --check
python tools/prepare_model_delivery.py --prepare --output project_work/model_delivery_review
```

The default is read-only check mode. `--prepare` requires an explicit **new** output directory, refuses an existing directory and original `run` locations, then copies exact checkpoint bytes into distinct descriptive filenames and verifies each copy. It adds this card, the inventory, verification details, `SHA256SUMS`, and a `NOT_PUBLISHED.txt` notice. It does not train, load a model, download data, create a release, commit, push, or upload anything. Check mode requires the locally preserved original checkpoints; a fresh GitHub clone cannot pass it until the matching files are obtained through a legitimate, documented channel.

Publication blockers remain: dataset/pretraining-derived weight rights, metadata privacy, and explicit user publication authorization. `release_url`, `release_tag`, and each `download_url` stay `null`; a prepared folder is not a published release.
