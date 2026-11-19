---
title: Chest X-ray Opacity + Grad-CAM (education only)
emoji: 🩻
colorFrom: gray
colorTo: blue
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: DenseNet-121 on RSNA with calibrated probability + Grad-CAM
---

**Education and research only. Not a medical device, not for clinical use.**

Demo for [aiml-02-medical-imaging](https://github.com/adhi2801/aiml-02-medical-imaging). It runs the same decoding,
preprocessing, temperature and per-view thresholds as the repo's reading-room worker. All numbers and limitations are in
the repo's README and model card. Example images are held-out RSNA test images (NIH ChestX-ray8/14, see
`examples/ATTRIBUTION.md`).
