# Model Card: Chest X-ray Lung Opacity Classifier

> **Educational prototype. Not a medical device, not cleared by any regulator, not for clinical use.**

## Model details
- **Architecture:** DenseNet-121 (ImageNet-pretrained, torchvision) with a single-logit head. Input is a 384×384 grayscale image replicated to 3 channels.
- **Training:** AdamW, one-cycle LR, AMP, class-balanced BCE (`pos_weight = neg/pos`), early stopping on validation AUROC.
- **Post-processing:** temperature scaling (fit on validation) and a Youden-optimal threshold (fit on validation).
- **Explanation:** Grad-CAM on the final dense block (12×12 at 384 px input, upsampled bilinearly).
- **Version:** `densenet121-<sha256[:8] of weights>`, shown on every result and in every audit entry.

## Intended use
- Teaching and demonstrating an AI-in-the-loop radiology workflow: triage suggestion, explanation, human sign-off, audit.
- **Out of scope:** any diagnostic or treatment decision, pediatric images, CT/MRI, lateral views, non-chest images.

## Data
- **RSNA Pneumonia Detection Challenge (2018):** 26,684 frontal chest radiographs from the NIH ChestX-ray14 collection,
  re-annotated by radiologists (RSNA, STR). Label = "Lung Opacity" (positive) vs. "Normal" or "No Lung Opacity / Not Normal" (negative).
- **Split:** 70/15/15, stratified on label × view position. One image per patient, so there is no patient leakage.

## Metrics
Filled from `reports/metrics.json` and `reports/subgroups.csv` after `dvc repro`. Report each with its 95% CI:
AUROC, AUPRC, sensitivity, specificity, PPV, NPV, ECE (before/after calibration), Brier score,
Grad-CAM pointing game, and Grad-CAM IoU.

## Known limitations and bias risks
1. **View-position shortcut.** AP (portable, bedside) films come from sicker inpatients and have much higher opacity
   prevalence than PA films. A model can partly learn "AP ⇒ opacity" from markers, tubes and positioning instead of lung
   findings. *Mitigation:* stratified split, per-view metrics, a UI warning on AP films, and Grad-CAM review.
2. **Single source.** All images come from one US hospital system (NIH Clinical Center). Expect a performance drop on other
   scanners, populations and protocols. No external validation has been done.
3. **Label noise.** Labels come from radiologist reads of the image alone, without clinical correlation. "Lung opacity" is
   broader than pneumonia (it includes edema, atelectasis, hemorrhage and more).
4. **Demographics.** Only sex and age are available; race/ethnicity is not recorded, so fairness across those groups cannot be measured.
5. **Explanation limits.** Grad-CAM is coarse and class-discriminative, not causal. A heatmap on the lung does not prove
   the model used clinically valid features. That is why localization is scored against radiologist boxes.
6. **Calibration drift.** The temperature is fit on in-distribution validation data and is not guaranteed to hold after a domain shift.
7. **Binary output.** The model does not localize, grade severity or distinguish causes of opacity.

## Ethical considerations
- The human reviewer is mandatory: results start as `pending_review`, and rejection requires a written reason.
- Every inference and review is logged with model version, actor and timestamp in a tamper-evident hash chain.
- The demo uses only public, de-identified data. Do not upload real patient data to this system.

## Segmentation model (MSD Spleen)
MONAI 3D U-Net (5 levels, residual units) trained on 96³ patches at 1.5×1.5×2.0 mm spacing with DiceCE loss.
Evaluated with sliding-window inference using Dice and HD95. Limitations: only 41 labeled volumes in total (about 33 train / 8 validation) from one
institution, a single organ, portal-venous-phase CT only.
