---
title: YOLOv8 Safety Helmet Detection
emoji: 🪖
colorFrom: blue
colorTo: orange
sdk: docker
app_port: 7860
license: agpl-3.0
---

# YOLOv8 safety helmet research demo

Upload an original image, choose a verified model, and inspect real predictions.
The default model is YOLOv8n-P2, ordinary inference at 640 and confidence 0.25.
`hat` means a head wearing a safety helmet; `person` means a head without one,
not a full-body pedestrian. The five models and same-image comparison reuse
the project implementation. Predictions are not safety certification.

Images are processed on this hosted server and create temporary files. Do not
upload sensitive images. Clearing the page does not immediately erase server
files. First model loading can be slower; displayed one-shot timings are not
a benchmark. Unlabelled images do not establish precision, recall, or AP.

[Project and experiments](https://github.com/CJ2323232/yolov8-learning)

Code is supplied under AGPL-3.0-only, with independent notices in
`THIRD_PARTY_NOTICES.md`. Dataset, images and checkpoint rights are separate;
see `models/MODEL_CARD.md` and `docs/license_audit.md`. No dataset or example
photos are bundled. This preparation does not establish checkpoint publication
rights or commercial permission. Keep staging private while publication
conditions are unresolved.
