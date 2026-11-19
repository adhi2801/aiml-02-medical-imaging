"""Knowing when not to answer: selective prediction and conformal prediction sets for the binary classifier.

Selective prediction (Geifman & El-Yaniv, NeurIPS 2017): the model defers the cases it is least sure of to a
radiologist. Confidence is the distance of the case's logit from the operating threshold's logit, so "unsure" means
"close to the decision boundary actually used", not "close to 0.5". The deferral cutoff is fit on validation for a
target coverage and then applied unchanged to test.

Split conformal prediction (Vovk et al. 2005; Angelopoulos & Bates 2021): with the nonconformity score
1 - p(true label) and its finite-sample quantile on validation, the set {labels whose score is under the quantile}
contains the true label with probability >= 1 - alpha on exchangeable data. A set holding both labels is the model
saying "cannot tell". Mondrian (group-conditional) conformal fits one quantile per group (sex, age band, view), which
guarantees the coverage within each group instead of only on average (Vovk 2012; Romano et al. 2020).
"""
from __future__ import annotations

import numpy as np


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))


def margin(p: np.ndarray, thr: float) -> np.ndarray:
    """Confidence of each decision: distance from the operating threshold, in logit space."""
    return np.abs(logit(p) - logit(np.array(thr)))


def risk_coverage(y: np.ndarray, p: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Error rate among the k most confident cases, for every k. Returns (coverage, risk, AURC); AURC is the mean
    risk over all coverages (lower is better; a confidence that ranks errors last gives the minimum)."""
    order = np.argsort(-margin(p, thr), kind="stable")
    wrong = ((p[order] >= thr) != (y[order] == 1)).astype(float)
    k = np.arange(1, len(y) + 1)
    risk = np.cumsum(wrong) / k
    return k / len(y), risk, float(risk.mean())


def cutoff_for_coverage(p_val: np.ndarray, thr: float, coverage: float) -> float:
    """Margin cutoff, fit on validation, that keeps `coverage` of the cases (the rest are deferred)."""
    return float(np.quantile(margin(p_val, thr), 1 - coverage))


def selective_stats(y: np.ndarray, p: np.ndarray, thr: float, cutoff: float) -> dict:
    keep = margin(p, thr) >= cutoff
    pred = p >= thr
    yk, pk = y[keep] == 1, pred[keep]
    pos, neg = yk.sum(), (~yk).sum()
    return {
        "coverage": round(float(keep.mean()), 4),
        "error_accepted": round(float((pk != yk).mean()), 4) if keep.any() else None,
        "error_all": round(float((pred != (y == 1)).mean()), 4),
        "sensitivity_accepted": round(float((pk & yk).sum() / pos), 4) if pos else None,
        "specificity_accepted": round(float((~pk & ~yk).sum() / neg), 4) if neg else None,
        "deferred": int((~keep).sum()),
        "positives_deferred": int((~keep & (y == 1)).sum()),
    }


def conformal_qhat(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample-corrected (1 - alpha) quantile of calibration scores; inf when there are too few to guarantee it
    (the set then holds every label, the honest answer)."""
    n = len(scores)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    return float("inf") if n == 0 or level > 1 else float(np.quantile(scores, level, method="higher"))


def scores(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Nonconformity: 1 - the probability the model gave the true label."""
    return np.where(y == 1, 1 - p, p)


def prediction_sets(p: np.ndarray, qhat: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
    """(contains 0, contains 1) per case: a label is in the set when its nonconformity is within the quantile."""
    return p <= qhat, (1 - p) <= qhat


def set_report(y: np.ndarray, p: np.ndarray, qhat: np.ndarray | float) -> dict:
    has0, has1 = prediction_sets(p, qhat)
    covered = np.where(y == 1, has1, has0)
    k, n = int(covered.sum()), len(y)
    lo, hi = wilson(k, n)
    return {
        "n": n,
        "coverage": round(k / n, 4) if n else None,
        "coverage_ci95": [round(lo, 4), round(hi, 4)],
        "both_labels": round(float((has0 & has1).mean()), 4) if n else None,
        "empty": round(float((~has0 & ~has1).mean()), 4) if n else None,
    }


def mondrian_qhat(y: np.ndarray, p: np.ndarray, groups: np.ndarray, alpha: float) -> dict:
    """One conformal quantile per group, from that group's validation cases only."""
    s = scores(y, p)
    return {g: conformal_qhat(s[groups == g], alpha) for g in np.unique(groups)}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score 95 % interval for a proportion (stays inside [0, 1] and behaves at small n)."""
    if n == 0:
        return float("nan"), float("nan")
    phat = k / n
    den = 1 + z * z / n
    centre = (phat + z * z / (2 * n)) / den
    half = z * np.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n)) / den
    return float(centre - half), float(centre + half)


def underdiagnosis(y: np.ndarray, p: np.ndarray, thr: float, groups: np.ndarray, iters: int = 1000,
                   seed: int = 0) -> dict:
    """False-negative rate (missed disease) per group with a Wilson 95 % CI, and each group's gap to the best-served
    group with a bootstrap 95 % CI (Seyyed-Kalantari et al., Nature Medicine 2021: underdiagnosis falls hardest on
    under-served groups). A gap whose CI excludes 0 is a measured disparity, not noise."""
    miss = (p < thr) & (y == 1)
    out = {}
    for g in np.unique(groups):
        pos = (groups == g) & (y == 1)
        k, n = int(miss[pos].sum()), int(pos.sum())
        lo, hi = wilson(k, n)
        out[str(g)] = {"positives": n, "fnr": round(k / n, 4) if n else None,
                       "fnr_ci95": [round(lo, 4), round(hi, 4)] if n else None}
    rated = {g: v for g, v in out.items() if v["fnr"] is not None}
    if len(rated) < 2:
        return {"groups": out}
    best = min(rated, key=lambda g: rated[g]["fnr"])
    rng = np.random.default_rng(seed)
    idx_pos = {g: np.where((groups.astype(str) == g) & (y == 1))[0] for g in rated}
    gaps = {}
    for g in rated:
        if g == best:
            continue
        draws = []
        for _ in range(iters):
            a = rng.choice(idx_pos[g], len(idx_pos[g]))
            b = rng.choice(idx_pos[best], len(idx_pos[best]))
            draws.append(miss[a].mean() - miss[b].mean())
        lo, hi = np.percentile(draws, [2.5, 97.5])
        gaps[g] = {"fnr_gap": round(rated[g]["fnr"] - rated[best]["fnr"], 4), "ci95": [round(lo, 4), round(hi, 4)],
                   "significant": bool(lo > 0)}
    return {"groups": out, "best_served": best, "gaps_vs_best": gaps}
