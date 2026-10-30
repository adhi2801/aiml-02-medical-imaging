"""Clinical-grade evaluation helpers: confidence intervals, calibration, operating points."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, roc_curve


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def bootstrap_auc(y: np.ndarray, s: np.ndarray, iters: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """AUROC with a percentile bootstrap 95% CI. A point estimate without a CI hides small-sample noise."""
    rng = np.random.default_rng(seed)
    n, aucs = len(y), []
    for _ in range(iters):
        idx = rng.integers(0, n, n)
        if y[idx].min() != y[idx].max():
            aucs.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(aucs, [2.5, 97.5])
    return float(roc_auc_score(y, s)), float(lo), float(hi)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar that fixes over-confidence, fit on val only."""
    z, t = torch.tensor(logits, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    log_T = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z / log_T.exp(), t)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_T.exp())


def ece(y: np.ndarray, p: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_curve(y: np.ndarray, p: np.ndarray, bins: int = 15) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    conf, acc, cnt = [], [], []
    for b in range(bins):
        m = idx == b
        if m.any():
            conf.append(p[m].mean()), acc.append(y[m].mean()), cnt.append(m.sum())
    return np.array(conf), np.array(acc), np.array(cnt)


def youden_threshold(y: np.ndarray, p: np.ndarray) -> float:
    fpr, tpr, thr = roc_curve(y, p)
    return float(thr[np.argmax(tpr - fpr)])


def confusion_stats(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    pred = p >= thr
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())

    def div(a, b):
        return round(a / b, 4) if b else None

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "sensitivity": div(tp, tp + fn), "specificity": div(tn, tn + fp),
            "ppv": div(tp, tp + fp), "npv": div(tn, tn + fn)}
