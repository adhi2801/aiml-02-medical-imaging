"""Figures for the spleen model: learning curve and predicted vs. radiologist outlines on validation CTs.

    python -m medimg.seg.figures_seg       (after train_seg; reads models/seg_unet3d.pt and reports/seg_history.csv)

Per-case Dice/HD95 go to reports/seg_per_case.csv so the worst case can be discussed, not just the mean.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from monai.apps import DecathlonDataset
from monai.data import PersistentDataset
from monai.data.utils import json_hashing
from monai.inferers import sliding_window_inference
from monai.metrics import compute_dice, compute_hausdorff_distance
from monai.networks.utils import one_hot

from medimg import ROOT, load_params
from medimg.seg.train_seg import build_model, transforms

FIG = ROOT / "reports" / "seg_figures"


def learning_curve(history: pd.DataFrame) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.epoch, history.train_loss)
    ax[0].set(xlabel="epoch", ylabel="Dice + CE loss", title="Training loss")
    val = history.dropna(subset=["val_dice"])
    ax[1].plot(val.epoch, val.val_dice, marker="o", ms=3)
    best = val.loc[val.val_dice.idxmax()]
    ax[1].axvline(best.epoch, color="grey", ls="--", lw=1)
    ax[1].set(xlabel="epoch", ylabel="Dice", ylim=(0, 1), title=f"Validation Dice (best {best.val_dice:.3f} @ {int(best.epoch)})")
    fig.tight_layout()
    fig.savefig(FIG / "seg_learning_curve.png", dpi=150)
    plt.close(fig)


def main() -> None:
    p = load_params()
    sp = p["segmentation"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    FIG.mkdir(parents=True, exist_ok=True)
    learning_curve(pd.read_csv(ROOT / "reports" / "seg_history.csv"))

    model = build_model().to(device).eval()
    model.load_state_dict(torch.load(ROOT / "models" / "seg_unet3d.pt", map_location=device)["state_dict"])
    split = DecathlonDataset(str(ROOT / sp["root_dir"]), sp["task"], "validation", download=False,
                             val_frac=0.2, seed=p["seed"], cache_rate=0.0).data
    ds = PersistentDataset(split, transforms(sp, False), cache_dir=ROOT / "data" / "processed" / "msd_cache" / "val",
                           hash_transform=json_hashing)

    rows, panels = [], []
    with torch.no_grad():
        for item, case in zip(ds, split):
            x = item["image"][None].to(device).float()
            y = item["label"][None].to(device).long()
            with torch.autocast(device.type, enabled=sp["amp"] and device.type == "cuda"):
                logits = sliding_window_inference(x, sp["roi_size"], 4, model, overlap=0.5)
            pred = logits.argmax(1, keepdim=True)
            p1, y1 = one_hot(pred, 2), one_hot(y, 2)
            dice = compute_dice(p1, y1, include_background=False).item()
            hd95 = compute_hausdorff_distance(p1, y1, include_background=False, percentile=95,
                                              spacing=sp["pixdim"]).item()
            rows.append({"case": Path(case["image"]).name.split(".")[0], "dice": round(dice, 4), "hd95_mm": round(hd95, 2)})
            img, gt, pr = x[0, 0].cpu().numpy(), y[0, 0].cpu().numpy(), pred[0, 0].cpu().numpy()
            z = int(gt.sum(axis=(0, 1)).argmax())  # axial slice with the most spleen
            panels.append((rows[-1], img[..., z], gt[..., z], pr[..., z]))

    per_case = pd.DataFrame(rows).sort_values("dice")
    per_case.to_csv(ROOT / "reports" / "seg_per_case.csv", index=False)
    print(per_case.to_string(index=False))

    # Best, median and worst validation case, so the figure is honest about failures too.
    order = sorted(panels, key=lambda t: t[0]["dice"])
    picks = [("worst", order[0]), ("median", order[len(order) // 2]), ("best", order[-1])]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
    # RAS arrays -> radiology display: anterior at the top, patient's left (the spleen side) on the image right.
    def view(a: np.ndarray) -> np.ndarray:
        return np.flip(a.T, (0, 1))

    for ax, (tag, (row, img, gt, pr)) in zip(axes, picks):
        ax.imshow(view(img), cmap="gray")
        ax.contour(view(gt), levels=[0.5], colors="lime", linewidths=1.5)
        ax.contour(view(pr), levels=[0.5], colors="red", linewidths=1.2, linestyles="--")
        ax.set_title(f"{tag}: {row['case']}\nDice {row['dice']:.3f} · HD95 {row['hd95_mm']:.1f} mm")
        ax.axis("off")
    fig.suptitle("Spleen on validation CT: radiologist (green) vs. model (red dashed)")
    fig.tight_layout()
    fig.savefig(FIG / "seg_examples.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
