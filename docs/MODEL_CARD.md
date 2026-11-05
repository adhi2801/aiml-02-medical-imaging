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
Held-out test split: 4,003 images, 22.5 % positive. DenseNet-121 at 384 px, best validation AUROC 0.891 at epoch 8 of 11
(early stopping), trained on a Kaggle T4. Source: `reports/metrics.json`, `reports/subgroups.csv`.

| Metric | Value |
|---|---|
| AUROC (95 % bootstrap CI) | 0.882 (0.870 – 0.894) |
| AUPRC | 0.699 (chance = 0.225) |
| Sensitivity / specificity at the validation-selected threshold (0.317) | 0.839 / 0.759 |
| PPV / NPV | 0.503 / 0.942 |
| ECE before → after temperature scaling (T = 1.29); Brier after | 0.073 → 0.071; 0.119 |
| Grad-CAM pointing game / mean IoU vs radiologist boxes | 0.544 / 0.237 |
| Metadata-only baseline: view / view + sex + age (no pixels) | 0.707 / 0.725 |

| Subgroup | n | Prevalence | AUROC (95 % CI) | Sens / Spec |
|---|---|---|---|---|
| Female | 1,672 | 0.21 | 0.892 (0.875 – 0.908) | 0.84 / 0.77 |
| Male | 2,331 | 0.24 | 0.875 (0.859 – 0.890) | 0.84 / 0.75 |
| Age < 40 | 1,321 | 0.27 | 0.909 (0.893 – 0.924) | 0.87 / 0.78 |
| Age 40–59 | 1,702 | 0.21 | 0.889 (0.870 – 0.905) | 0.85 / 0.77 |
| Age 60+ | 980 | 0.20 | **0.816 (0.787 – 0.848)** | 0.75 / 0.73 |
| AP view | 1,826 | 0.38 | 0.838 (0.820 – 0.855) | 0.91 / 0.54 |
| PA view | 2,177 | 0.09 | 0.862 (0.837 – 0.884) | **0.58** / 0.88 |

**What the numbers say:**
- **The model reads the image, not just the acquisition.** Metadata alone reaches 0.725. The image model reaches 0.882 overall,
  and 0.84–0.86 *within* AP films and *within* PA films, where the view shortcut carries no information.
- **Patients aged 60+ are served worse.** Their AUROC CI (0.787 – 0.848) does not overlap the under-40 group's. Plausible causes
  are more co-morbid findings (effusion, cardiomegaly, chronic changes) that look like opacity. This needs targeted error analysis
  before any use on older patients.
- **One threshold does not fit both views.** With the shared threshold, sensitivity is 0.91 on AP films but only 0.58 on PA films,
  where prevalence is 9 %. A per-view threshold, or explicitly reporting this trade-off to the reader, would be the next step.
- **Calibration was already good.** Temperature scaling barely moved ECE (0.073 → 0.071).
- **Grad-CAM is only partly on target.** The heatmap peak falls inside a radiologist box in 54 % of positive images. That is
  consistent with limitation 5 below, and the reason the UI shows heatmaps as a review aid, not as a finding.

## Known limitations and bias risks
1. **View-position shortcut.** AP (portable, bedside) films come from sicker inpatients and have much higher opacity
   prevalence than PA films. A model can partly learn "AP ⇒ opacity" from markers, tubes and positioning instead of lung
   findings. This has been demonstrated on chest X-ray pneumonia classifiers in the literature. *Mitigation:* stratified split,
   per-view metrics, a UI warning on AP films, Grad-CAM review, and a **metadata-only baseline**
   (`baseline_view_only_auroc` and `baseline_metadata_auroc` in `reports/metrics.json`). That baseline is a logistic regression on view,
   sex and age with no pixels. The image model has to clearly beat it, and also hold up *within* AP and *within* PA films
   (`reports/subgroups.csv`), before we can say it reads the lungs rather than the acquisition.
2. **Single source.** All images come from one US hospital system (NIH Clinical Center). Expect a performance drop on other
   scanners, populations and protocols. No external testing (CLAIM 2024 terminology) has been done.
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
Evaluated with sliding-window inference using Dice and HD95. Post-processing keeps the largest connected component,
since the spleen is a single organ. Raw and post-processed scores are both reported (`reports/seg_postproc.json`,
`reports/seg_per_case.csv`), because a few distant false-positive blobs inflate HD95 by tens of millimetres.

**Results (8 validation CTs, final epoch, no checkpoint selection):** after 600 epochs on a Kaggle T4, mean Dice
**0.885** and HD95 **6.7 mm** after post-processing (raw 0.881 / 25.7 mm). The median is 0.944 and 6 of 8 CTs score at
least 0.94, but spleen_26 fails at 0.53 and spleen_49 reaches 0.86 (`reports/seg_per_case.csv`).

**Why the final epoch and not the best one:** the best of the 120 validation checkpoints (epoch 224) scores 0.939, with
spleen_26 at 0.93 (`reports/seg_postproc_best.json`). That checkpoint was chosen *on these same 8 CTs*, so its score is
biased upwards. The gap between 0.939 and 0.885, and one case swinging from 0.93 to 0.53 between epochs, show that a single
8-case split cannot support a precise number either way. **5-fold cross-validation** (`medimg.seg.cv_seg`) therefore scores
every one of the 41 labelled CTs exactly once, with a model that never trained on it, at a fixed schedule with
weight averaging (SWA) and no selection. Its mean ± SD will replace this paragraph's numbers.

Post-processing matters most for HD95: on spleen_53 it removes a distant false-positive blob and takes HD95 from 154 mm to
2.0 mm. A 200-epoch run on a 4 GB laptop GTX 1650 is kept for reference in `reports/seg_laptop_200ep/` (best-epoch
selection, so optimistic in the same way).

**Context:** MONAI's reference bundle for this task reports a mean Dice of about 0.96 on its own split. The splits differ,
so the numbers are not directly comparable, and with 8 validation volumes one case moves the mean by about 0.01.

Limitations: only 41 labeled volumes in total (33 train / 8 validation) from one institution, a single organ,
portal-venous-phase CT only. The validation set is small, so per-case results are listed rather than only the mean.

## Regulatory context (for orientation only)
This prototype is not a regulated device. The design follows ideas from the FDA / Health Canada / MHRA
*Good Machine Learning Practice* principles (2021) and their guiding principles for *Predetermined Change Control Plans*
(2023):
- **Separate training and test data, with a reproducible split:** fixed seed, stratified, versioned with DVC.
- **Focus on human–AI team performance:** mandatory radiologist sign-off, agreement rate tracked in `/api/stats`.
- **Clear information for users:** this model card, the disclaimer on every screen and API response, and Grad-CAM shown with its limits.
- **Monitoring and controlled change:** the model version is stamped on every result and audit entry. A real product would
  specify in advance which retraining changes are allowed and how each one is re-tested (a PCCP). Here, `dvc repro`
  re-runs the identical evaluation for any new model.
