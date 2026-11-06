"""Figures for the spleen 5-fold cross-validation (reads the outputs of medimg.seg.cv_seg).

    python -m medimg.seg.figures_cv

reports/seg_figures/seg_cv_per_case.png  every CT's Dice by fold (the SWA weights cv_seg scores), mean and 95 % CI
reports/seg_figures/seg_cv_failures.png  the three lowest-Dice CTs, axial + coronal, predicted by the fold model that
                                         never trained on them, so the failure modes can be discussed, not only the mean
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.data import Dataset
from monai.inferers import sliding_window_inference

from medimg import ROOT, load_params
from medimg.seg.cv_seg import all_cases
from medimg.seg.figures_seg import FIG, keep_largest
from medimg.seg.train_seg import build_model, transforms


def per_case_plot(df: pd.DataFrame, summary: dict) -> None:
    swa = df[df.model == "swa"]
    s = summary["swa"]
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    lo, hi = s["dice_ci95"]
    ax.axhspan(lo, hi, color="tab:blue", alpha=0.12, label=f"95 % CI of the mean ({lo:.3f}–{hi:.3f})")
    ax.axhline(s["dice_mean"], color="tab:blue", lw=1.2, label=f"mean {s['dice_mean']:.3f}")
    ax.axhline(s["dice_median"], color="tab:blue", lw=1, ls=":", label=f"median {s['dice_median']:.3f}")
    for k, g in swa.groupby("fold"):
        x = k + rng.uniform(-0.15, 0.15, len(g))
        ax.scatter(x, g.dice, s=22, color="black", zorder=3)
        for xi, (_, r) in zip(x, g.iterrows()):
            if r.dice < 0.9:
                ax.annotate(r.case, (xi, r.dice), xytext=(6, -3), textcoords="offset points", fontsize=7)
    ax.set(xticks=range(summary["folds"]), xticklabels=[f"fold {k}" for k in range(summary["folds"])],
           ylabel="Dice (largest component)", ylim=(0.3, 1.0),
           title=f"Spleen 5-fold CV: each of {summary['n_cases']} CTs scored once by a model that never saw it")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "seg_cv_per_case.png", dpi=150)
    plt.close(fig)


def failures_plot(df: pd.DataFrame, n_worst: int = 3) -> None:
    p = load_params()
    sp = p["segmentation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    by_name = {c["image"].replace("\\", "/").split("/")[-1].split(".")[0]: c for c in all_cases(sp, p["seed"])}
    worst = df[df.model == "swa"].nsmallest(n_worst, "dice")

    # RAS arrays -> radiology display (anterior up / superior up, patient's left on the image right)
    def axial(a: np.ndarray, z: int) -> np.ndarray:
        return np.flip(a[..., z].T, (0, 1))

    def coronal(a: np.ndarray, y: int) -> np.ndarray:
        return np.flip(a[:, y, :].T, (0, 1))

    fig, axes = plt.subplots(2, n_worst, figsize=(5 * n_worst, 9.5))
    model = build_model().to(device).eval()
    for col, (_, r) in enumerate(worst.iterrows()):
        state = torch.load(ROOT / "models" / f"seg_cv_fold{int(r.fold)}.pt", map_location=device, weights_only=False)
        model.load_state_dict(state["state_dict"])
        item = Dataset([by_name[r.case]], transforms(sp, False))[0]
        with torch.no_grad(), torch.autocast(device.type, enabled=sp["amp"] and device.type == "cuda"):
            x = item["image"][None].to(device).float()
            raw = sliding_window_inference(x, sp["roi_size"], 4, model, overlap=0.5).argmax(1)[0].cpu().numpy()
        img, gt, pr = item["image"][0].numpy(), item["label"][0].numpy(), keep_largest(raw)
        z, y = int(gt.sum(axis=(0, 1)).argmax()), int(gt.sum(axis=(0, 2)).argmax())
        for row, (cut, i) in enumerate(((axial, z), (coronal, y))):
            ax = axes[row, col]
            ax.imshow(cut(img, i), cmap="gray", aspect="auto" if cut is coronal else None)
            ax.contour(cut(gt, i), levels=[0.5], colors="lime", linewidths=1.5)
            if cut(pr, i).any():
                ax.contour(cut(pr, i), levels=[0.5], colors="red", linewidths=1.2, linestyles="--")
            ax.axis("off")
        vol = float(np.prod(sp["pixdim"])) / 1000  # mL per voxel
        axes[0, col].set_title(f"{r.case} (fold {int(r.fold)}): Dice {r.dice:.3f} · HD95 {r.hd95_mm:.1f} mm\n"
                               f"spleen {gt.sum() * vol:.0f} mL, predicted {pr.sum() * vol:.0f} mL")
        print(json.dumps({"case": r.case, "dice": r.dice, "gt_ml": round(gt.sum() * vol), "pred_ml": round(pr.sum() * vol)}))
    axes[0, 0].text(-0.05, 0.5, "axial", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right")
    axes[1, 0].text(-0.05, 0.5, "coronal", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right")
    fig.suptitle("Lowest-Dice CTs in cross-validation: radiologist (green) vs. held-out fold model (red dashed)", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(FIG / "seg_cv_failures.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ROOT / "reports" / "seg_cv_per_case.csv")
    summary = json.loads((ROOT / "reports" / "seg_cv.json").read_text())
    per_case_plot(df, summary)
    failures_plot(df)


if __name__ == "__main__":
    main()
