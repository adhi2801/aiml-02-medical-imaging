"""DVC stage `eval_cls`: the evidence behind every number in the README and the model card.

  1. Temperature scaling + operating threshold, both fit on VAL only (test stays untouched).
  2. TEST: AUROC/AUPRC with bootstrap 95% CI, sens/spec/PPV/NPV, ECE and Brier before/after calibration.
  3. Subgroup audit by sex, age band and view position (AP vs PA) -> reports/subgroups.csv.
  4. Grad-CAM localization vs. radiologist boxes (pointing game + IoU) on test positives.
  5. Figures: ROC, PR, reliability diagram, Grad-CAM gallery (TP / FP / FN).
"""
from __future__ import annotations

import json

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_curve
from torch.utils.data import DataLoader
from tqdm import tqdm

from medimg import ROOT, load_params
from medimg.data.dataset import RSNADataset
from medimg.eval.stats import (bootstrap_auc, confusion_stats, ece, fit_temperature, reliability_curve,
                               shortcut_baseline, sigmoid, youden_threshold)
from medimg.explain.gradcam import boxes_to_mask, cam_iou, gradcam, overlay, pointing_hit
from medimg.models.classifier import load_checkpoint
from medimg.train_cls import predict

FIG = ROOT / "reports" / "figures"


def age_band(a) -> str:
    if pd.isna(a):
        return "unknown"
    return "<40" if a < 40 else "40-59" if a < 60 else "60+"


def subgroup_table(df: pd.DataFrame, thr: float, iters: int) -> pd.DataFrame:
    rows = []
    for col in ("sex", "age_band", "view"):
        for val, g in df.groupby(col):
            y, p = g.label.values, g.prob.values
            row = {"attribute": col, "group": val, "n": len(g), "prevalence": round(y.mean(), 3)}
            if 0 < y.sum() < len(y):
                auc, lo, hi = bootstrap_auc(y, p, iters)
                row |= {"auroc": round(auc, 4), "auroc_ci_low": round(lo, 4), "auroc_ci_high": round(hi, 4)}
            cs = confusion_stats(y, p, thr)
            row |= {"sensitivity": cs["sensitivity"], "specificity": cs["specificity"]}
            rows.append(row)
    return pd.DataFrame(rows)


def plot_curves(y, p_raw, p_cal, bins):
    fpr, tpr, _ = roc_curve(y, p_cal)
    prec, rec, _ = precision_recall_curve(y, p_cal)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].plot(fpr, tpr), ax[0].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[0].set(title="ROC (test)", xlabel="1 - specificity", ylabel="sensitivity")
    ax[1].plot(rec, prec), ax[1].axhline(y.mean(), color="k", ls="--", lw=0.8)
    ax[1].set(title="Precision-recall (test)", xlabel="recall", ylabel="precision")
    for name, p in (("uncalibrated", p_raw), ("temperature-scaled", p_cal)):
        c, a, _ = reliability_curve(y, p, bins)
        ax[2].plot(c, a, "o-", label=name)
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.8)
    ax[2].set(title="Reliability diagram (test)", xlabel="predicted probability", ylabel="observed frequency")
    ax[2].legend()
    fig.tight_layout()
    fig.savefig(FIG / "test_curves.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = load_params()
    ep = p["eval"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    FIG.mkdir(parents=True, exist_ok=True)

    model, ckpt = load_checkpoint(ROOT / "models" / "classifier.pt", device)
    size = ckpt["img_size"]
    data_dir = ROOT / p["data"]["processed_dir"]
    meta = pd.read_csv(data_dir / "metadata.csv")

    def run(split):
        df = meta[meta.split == split].reset_index(drop=True)
        dl = DataLoader(RSNADataset(df, data_dir / "images", size, False), batch_size=32, num_workers=4)
        logits, y = predict(model, dl, device, amp)
        return df, logits, y.astype(int)

    val_df, val_logits, val_y = run("val")
    test_df, test_logits, test_y = run("test")

    T = fit_temperature(val_logits, val_y)
    thr = youden_threshold(val_y, sigmoid(val_logits / T))
    p_raw, p_cal = sigmoid(test_logits), sigmoid(test_logits / T)

    auc, lo, hi = bootstrap_auc(test_y, p_cal, ep["bootstrap_iters"], p["seed"])
    metrics = {
        "test_n": int(len(test_y)), "test_prevalence": round(float(test_y.mean()), 4),
        "auroc": round(auc, 4), "auroc_ci95": [round(lo, 4), round(hi, 4)],
        "auprc": round(float(average_precision_score(test_y, p_cal)), 4),
        "temperature": round(T, 4), "threshold": round(thr, 4),
        "ece_before": round(ece(test_y, p_raw, ep["ece_bins"]), 4),
        "ece_after": round(ece(test_y, p_cal, ep["ece_bins"]), 4),
        "brier_after": round(float(brier_score_loss(test_y, p_cal)), 4),
        **confusion_stats(test_y, p_cal, thr),
    }

    test_df = test_df.assign(prob=p_cal, age_band=test_df.age.map(age_band))
    subgroups = subgroup_table(test_df, thr, ep["bootstrap_iters"])
    subgroups.to_csv(ROOT / "reports" / "subgroups.csv", index=False)
    by_view = subgroups[subgroups.attribute == "view"].set_index("group")
    if {"AP", "PA"} <= set(by_view.index):
        metrics["auroc_gap_ap_minus_pa"] = round(float(by_view.loc["AP", "auroc"] - by_view.loc["PA", "auroc"]), 4)

    # ---- Shortcut check: how far does metadata alone get, with no pixels? ----
    train_df = meta[meta.split == "train"]
    for name, feats in (("view_only", ["view"]), ("metadata", ["view", "sex", "age"])):
        b_auc, b_lo, b_hi = bootstrap_auc(test_y, shortcut_baseline(train_df, test_df, feats), ep["bootstrap_iters"], p["seed"])
        metrics[f"baseline_{name}_auroc"] = round(b_auc, 4)
        metrics[f"baseline_{name}_auroc_ci95"] = [round(b_lo, 4), round(b_hi, 4)]

    # ---- Grad-CAM localization on test positives ----
    pos = test_df[test_df.label == 1].reset_index(drop=True)
    hits, ious, gallery = [], [], {"TP": [], "FN": [], "FP": []}
    ds = RSNADataset(pos, data_dir / "images", size, False)
    for start in tqdm(range(0, len(pos), 16), desc="Grad-CAM"):
        idx = list(range(start, min(start + 16, len(pos))))
        x = torch.stack([ds[i][0] for i in idx]).to(device)
        _, cams = gradcam(model, x)
        for i, cam in zip(idx, cams.cpu().numpy()):
            row = pos.iloc[i]
            gt = boxes_to_mask(json.loads(row.boxes), (row.orig_h, row.orig_w), cam.shape)
            hits.append(pointing_hit(cam, gt))
            ious.append(cam_iou(cam, gt, ep["cam_threshold"]))
            kind = "TP" if row.prob >= thr else "FN"
            if len(gallery[kind]) < 4:
                gallery[kind].append((row, cam))
    metrics["gradcam_pointing_game"] = round(float(np.mean(hits)), 4)
    metrics["gradcam_mean_iou"] = round(float(np.mean(ious)), 4)

    fps = test_df[(test_df.label == 0) & (test_df.prob >= thr)].nlargest(4, "prob")
    if len(fps):
        fds = RSNADataset(fps, data_dir / "images", size, False)
        _, cams = gradcam(model, torch.stack([fds[i][0] for i in range(len(fps))]).to(device))
        gallery["FP"] = list(zip((r for _, r in fps.iterrows()), cams.cpu().numpy()))
    save_gallery(gallery, data_dir / "images")

    plot_curves(test_y, p_raw, p_cal, ep["ece_bins"])
    (ROOT / "reports" / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (ROOT / "models" / "calibration.json").write_text(json.dumps({"temperature": T, "threshold": thr}, indent=2))
    print(json.dumps(metrics, indent=2))
    print(subgroups.to_string(index=False))


def save_gallery(gallery: dict, image_dir) -> None:
    kinds = [k for k in ("TP", "FN", "FP") if gallery[k]]
    if not kinds:
        return
    fig, axes = plt.subplots(len(kinds), 4, figsize=(16, 4.2 * len(kinds)), squeeze=False)
    for r, kind in enumerate(kinds):
        for c in range(4):
            ax = axes[r, c]
            ax.axis("off")
            if c >= len(gallery[kind]):
                continue
            row, cam = gallery[kind][c]
            img = cv2.imread(str(image_dir / f"{row.patient_id}.png"), cv2.IMREAD_GRAYSCALE)
            ax.imshow(overlay(img, cam))
            s = img.shape[0] / row.orig_h
            for x, y, w, h in json.loads(row.boxes):
                ax.add_patch(plt.Rectangle((x * s, y * s), w * s, h * s, fill=False, ec="lime", lw=2))
            ax.set_title(f"{kind}  p={row.prob:.2f}  {row.view}", fontsize=10)
    fig.suptitle("Grad-CAM vs radiologist boxes (green)", fontsize=14)
    fig.tight_layout()
    fig.savefig(FIG / "gradcam_gallery.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
