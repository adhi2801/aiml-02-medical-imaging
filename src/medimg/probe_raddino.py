"""Foundation-model baseline: a linear probe on RAD-DINO (Microsoft, Nature Machine Intelligence 2025), a ViT-B/14
self-supervised on 882k chest X-rays, against our DenseNet-121 trained end to end.

The encoder is frozen; only a logistic regression is fit on its CLS embeddings, on the same RSNA train split. Its
regularisation strength is chosen on the val split by AUROC, and the test split is scored once. If a frozen
foundation model with a linear head matches or beats the fine-tuned CNN, that is the honest headline.

    python -m medimg.probe_raddino            # writes reports/raddino_probe.json and reports/predictions_raddino.csv
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from medimg import ROOT, load_params
from medimg.eval.stats import bootstrap_auc

MODEL = "microsoft/rad-dino"
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0)


@torch.no_grad()
def embed(paths: list, processor, model, device, batch: int = 64) -> np.ndarray:
    out = []
    for i in range(0, len(paths), batch):
        imgs = [Image.open(p).convert("RGB") for p in paths[i:i + batch]]
        x = processor(images=imgs, return_tensors="pt").to(device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            h = model(**x).pooler_output
        out.append(h.float().cpu().numpy())
        if i % (batch * 50) == 0:
            print(f"embedded {i + len(imgs)}/{len(paths)}", flush=True)
    return np.concatenate(out)


def choose_c(xtr, ytr, xva, yva) -> tuple[float, dict]:
    """Regularisation chosen on val only."""
    scores = {}
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000).fit(xtr, ytr)
        scores[c] = roc_auc_score(yva, clf.predict_proba(xva)[:, 1])
    return max(scores, key=scores.get), {str(k): round(v, 4) for k, v in scores.items()}


def main() -> None:
    from transformers import AutoImageProcessor, AutoModel

    p = load_params()
    data_dir = ROOT / p["data"]["processed_dir"]
    meta = pd.read_csv(data_dir / "metadata.csv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoImageProcessor.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()

    feats, ys, dfs = {}, {}, {}
    for split in ("train", "val", "test"):
        df = meta[meta.split == split].reset_index(drop=True)
        feats[split] = embed([data_dir / "images" / f"{pid}.png" for pid in df.patient_id], processor, model, device)
        ys[split], dfs[split] = df.label.values.astype(int), df

    scaler = StandardScaler().fit(feats["train"])
    x = {k: scaler.transform(v) for k, v in feats.items()}
    c, val_aucs = choose_c(x["train"], ys["train"], x["val"], ys["val"])
    clf = LogisticRegression(C=c, max_iter=3000).fit(x["train"], ys["train"])
    prob = clf.predict_proba(x["test"])[:, 1]
    auc, lo, hi = bootstrap_auc(ys["test"], prob, p["eval"]["bootstrap_iters"], p["seed"])

    report = {
        "encoder": MODEL, "frozen": True, "embedding_dim": int(feats["train"].shape[1]),
        "train_n": int(len(ys["train"])), "test_n": int(len(ys["test"])),
        "C_chosen_on_val": c, "val_auroc_by_C": val_aucs,
        "test_auroc": round(auc, 4), "test_auroc_ci95": [round(lo, 4), round(hi, 4)],
    }
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "raddino_probe.json").write_text(json.dumps(report, indent=2))
    dfs["test"].assign(prob_raddino=np.round(prob, 5))[["patient_id", "label", "prob_raddino"]].to_csv(
        ROOT / "reports" / "predictions_raddino.csv", index=False)
    print(json.dumps(report, indent=2))


def paired_comparison(a: np.ndarray, b: np.ndarray, y: np.ndarray, iters: int = 2000, seed: int = 0) -> dict:
    """AUROC(b) - AUROC(a) on the same test images, with a paired bootstrap 95 % CI (resampling images, so both models
    see the same resample: the correlation between them is kept, which a comparison of two separate CIs ignores)."""
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(iters):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            diffs.append(roc_auc_score(y[i], b[i]) - roc_auc_score(y[i], a[i]))
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"auroc_a": round(roc_auc_score(y, a), 4), "auroc_b": round(roc_auc_score(y, b), 4),
            "diff_b_minus_a": round(roc_auc_score(y, b) - roc_auc_score(y, a), 4),
            "diff_ci95": [round(lo, 4), round(hi, 4)], "significant": bool(lo > 0 or hi < 0)}


if __name__ == "__main__":
    main()
