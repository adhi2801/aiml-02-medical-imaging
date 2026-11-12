"""External validation: the released classifier, unchanged, on CheXpert (Stanford Hospital, a different country,
scanner fleet, patient mix and labelling process than RSNA/NIH).

Nothing is refit here. The temperature and the operating threshold come from models/calibration.json, which was fit on
the RSNA validation split. A model that only works on the hospital it was trained on is the most common way medical
imaging AI fails in deployment, so this is the number that says whether the RSNA results transfer.

Two test sets, reported separately because their labels differ in quality:

  valid  CheXpert's validation studies, labelled by a consensus of three board-certified radiologists (the strong
         reference standard, but small).
  train  a sample of one frontal image per patient from CheXpert's training set, labelled by the CheXpert NLP labeller
         from the radiology report. Larger, so the CI is tighter, but the labels are noisy; uncertain mentions (-1) are
         dropped and unmentioned findings count as negative, the usual convention.

Target: "Lung Opacity", the RSNA target ("Lung Opacity" vs "Normal / No Lung Opacity"). Pneumonia and Consolidation are
reported as secondary targets. Only frontal views (AP, PA) are scored, like the RSNA training data.

    python -m medimg.external_cls --root /path/to/CheXpert-v1.0-small [--train-sample 5000]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_curve
from torch.utils.data import DataLoader, Dataset

from medimg import ROOT, load_params
from medimg.data.dataset import preprocess
from medimg.eval.stats import bootstrap_auc, confusion_stats, ece, sigmoid
from medimg.models.classifier import load_checkpoint
from medimg.train_cls import predict

TARGETS = ("Lung Opacity", "Pneumonia", "Consolidation")
PRIMARY = "Lung Opacity"


def load_split(root: Path, split: str, sample: int | None, seed: int) -> pd.DataFrame:
    """Frontal images of one CheXpert split, with paths resolved against `root`'s parent (CheXpert CSV paths start
    with the dataset folder name) and the view (AP / PA) and patient id parsed out."""
    df = pd.read_csv(root / f"{split}.csv")
    df = df[df["Frontal/Lateral"] == "Frontal"].copy()
    df["view"] = df["AP/PA"].where(df["AP/PA"].isin(["AP", "PA"]), "other")
    df["patient"] = df.Path.str.extract(r"(patient\d+)", expand=False)
    if sample:  # one image per patient, so bootstrap resamples are independent
        df = df.sample(frac=1, random_state=seed).drop_duplicates("patient")
        df = df.sample(min(sample, len(df)), random_state=seed)
    df = df.reset_index(drop=True)
    # CSV paths start with the dataset folder ("CheXpert-v1.0-small/valid/..."); mirrors differ on whether that folder
    # exists on disk, so resolve against whichever layout this copy has.
    first = Path(df.Path.iloc[0])
    base = root.parent if (root.parent / first).exists() else None
    df["file"] = [str(base / p) if base else str(root / Path(p).relative_to(Path(p).parts[0])) for p in df.Path]
    return df


def binary_labels(col: pd.Series) -> pd.Series:
    """CheXpert label column -> 1 / 0 / NaN (uncertain). Blank (not mentioned) counts as negative."""
    return col.fillna(0).replace({-1: np.nan}).astype(float)


class FileDataset(Dataset):
    def __init__(self, files: list[str], size: int):
        self.files, self.size = files, size

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, i: int):
        img = cv2.imread(self.files[i], cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(self.files[i])
        return preprocess(img, self.size), torch.tensor(0.0), i


def score_target(y: np.ndarray, p: np.ndarray, thr: float, iters: int, seed: int, bins: int) -> dict:
    auc, lo, hi = bootstrap_auc(y, p, iters, seed)
    return {"n": int(len(y)), "prevalence": round(float(y.mean()), 4),
            "auroc": round(auc, 4), "auroc_ci95": [round(lo, 4), round(hi, 4)],
            "auprc": round(float(average_precision_score(y, p)), 4),
            "ece": round(ece(y, p, bins), 4), **confusion_stats(y, p, thr)}


def evaluate(df: pd.DataFrame, probs: np.ndarray, thr: float, iters: int, seed: int, bins: int) -> dict:
    out = {}
    for target in TARGETS:
        y = binary_labels(df[target])
        keep = y.notna().values
        if keep.sum() == 0 or y[keep].nunique() < 2:
            continue
        res = score_target(y[keep].values.astype(int), probs[keep], thr, iters, seed, bins)
        if target == PRIMARY:
            res["by_view"] = {}
            for view in ("AP", "PA"):
                m = keep & (df.view == view).values
                yv = y[m].values.astype(int)
                if m.sum() and 0 < yv.sum() < len(yv):
                    res["by_view"][view] = score_target(yv, probs[m], thr, iters, seed, bins)
        out[target] = res
    return out


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="External validation on CheXpert")
    ap.add_argument("--root", type=Path, required=True, help="the CheXpert-v1.0-small folder (has valid.csv)")
    ap.add_argument("--train-sample", type=int, default=5000, help="patients sampled from train.csv (0 = skip)")
    ap.add_argument("--out", type=Path, default=ROOT / "reports" / "external_chexpert.json")
    args = ap.parse_args(argv)

    p = load_params()
    ep, seed = p["eval"], p["seed"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_checkpoint(ROOT / "models" / "classifier.pt", device)
    calib = json.loads((ROOT / "models" / "calibration.json").read_text())
    T, thr = calib["temperature"], calib["threshold"]

    report = {"model": "models/classifier.pt (RSNA-trained, unchanged)",
              "temperature": round(T, 4), "threshold": round(thr, 4),
              "note": "temperature and threshold fit on the RSNA validation split; nothing is refit on CheXpert"}
    curves = {}
    splits = [("valid_radiologist", "valid", None)]
    if args.train_sample:
        splits.append(("train_nlp_labels", "train", args.train_sample))
    for name, split, sample in splits:
        df = load_split(args.root, split, sample, seed)
        dl = DataLoader(FileDataset(df.file.tolist(), ckpt["img_size"]), batch_size=32, num_workers=4)
        logits, _ = predict(model, dl, device, device.type == "cuda")
        probs = sigmoid(logits / T)
        report[name] = evaluate(df, probs, thr, ep["bootstrap_iters"], seed, ep["ece_bins"])
        report[name]["images"] = int(len(df))
        y = binary_labels(df[PRIMARY])
        keep = y.notna().values
        curves[name] = (y[keep].values.astype(int), probs[keep])
        print(name, json.dumps(report[name][PRIMARY], indent=2))

    rsna = json.loads((ROOT / "reports" / "metrics.json").read_text())
    report["rsna_internal_test"] = {k: rsna[k] for k in ("test_n", "auroc", "auroc_ci95", "sensitivity", "specificity")}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, (y, s) in curves.items():
        fpr, tpr, _ = roc_curve(y, s)
        r = report[name][PRIMARY]
        ax.plot(fpr, tpr, label=f"CheXpert {name.replace('_', ' ')}: AUROC {r['auroc']:.3f} (n={r['n']})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set(title=f"External validation, '{PRIMARY}'\nRSNA-trained model, no refitting "
                 f"(RSNA internal test AUROC {rsna['auroc']:.3f})",
           xlabel="1 - specificity", ylabel="sensitivity")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "external_chexpert_roc.png", dpi=150)
    plt.close(fig)
    return report


if __name__ == "__main__":
    main()
