"""Selective prediction and conformal sets on the real predictions; writes reports/uncertainty.json and a figure.

    python -m medimg.uncertainty_report          # needs reports/predictions_rsna.csv (from medimg.evaluate_cls)

Everything is fit on the RSNA validation split and applied unchanged to the RSNA test split and, when
reports/predictions_chexpert.csv exists (from medimg.external_cls), to CheXpert. Conformal coverage is guaranteed only
for data exchangeable with the calibration set; CheXpert is a different hospital, so its coverage is a measurement of
how far that guarantee survives a distribution shift, not a guarantee.
"""
from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from medimg import ROOT
from medimg.eval import uncertainty as u

ALPHA = 0.10  # 90 % prediction sets
COVERAGES = (0.9, 0.8, 0.7)  # share of cases the model answers; the rest go to a radiologist
GROUPS = ("sex", "age_band", "view")


def age_band(a) -> str:
    if pd.isna(a):
        return "unknown"
    return "<40" if a < 40 else "40-59" if a < 60 else "60+"


def with_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Same group labels for RSNA ("M"/"F") and CheXpert ("Male"/"Female")."""
    sex = df["sex"].astype(str).str[0].str.upper().where(df["sex"].notna(), "unknown")
    return df.assign(sex=sex, age_band=df["age"].map(age_band))


def analyse(name: str, y: np.ndarray, p: np.ndarray, g: pd.DataFrame, thr: float, cutoffs: dict, qhat: float,
            mondrian: dict) -> dict:
    cov, risk, aurc = u.risk_coverage(y, p, thr)
    out = {
        "n": int(len(y)),
        "aurc": round(aurc, 4),
        "selective": {f"{c:.0%}": u.selective_stats(y, p, thr, cut) for c, cut in cutoffs.items()},
        "conformal_marginal": u.set_report(y, p, qhat),
        "conformal_by_group": {},
    }
    for col in GROUPS:
        for val in sorted(g[col].unique()):
            m = (g[col] == val).values
            if m.sum() < 30:
                continue
            q_group = mondrian.get(col, {}).get(val, float("inf"))
            out["conformal_by_group"][f"{col}={val}"] = {
                "marginal_quantile": u.set_report(y[m], p[m], qhat),
                "group_quantile": u.set_report(y[m], p[m], q_group),
            }
    out["_curve"] = (cov, risk)
    print(name, json.dumps({k: v for k, v in out.items() if k not in ("_curve", "conformal_by_group")}, indent=1))
    return out


def main() -> None:
    preds = with_groups(pd.read_csv(ROOT / "reports" / "predictions_rsna.csv"))
    thr = json.loads((ROOT / "models" / "calibration.json").read_text())["threshold"]
    val, test = preds[preds.split == "val"], preds[preds.split == "test"]
    yv, pv = val.label.values, val.prob.values

    cutoffs = {c: u.cutoff_for_coverage(pv, thr, c) for c in COVERAGES}
    qhat = u.conformal_qhat(u.scores(yv, pv), ALPHA)
    mondrian = {col: u.mondrian_qhat(yv, pv, val[col].values, ALPHA) for col in GROUPS}

    report = {
        "alpha": ALPHA,
        "threshold": round(thr, 4),
        "fit_on": f"RSNA validation (n={len(val)})",
        "conformal_quantile": round(qhat, 4),
        "deferral_margin_cutoffs": {f"{c:.0%}": round(v, 4) for c, v in cutoffs.items()},
        "sets": {},
    }
    sets = {"rsna_test": (test.label.values, test.prob.values, test)}
    cx_path = ROOT / "reports" / "predictions_chexpert.csv"
    if cx_path.exists():
        cx = with_groups(pd.read_csv(cx_path))
        for name in cx["set"].unique():
            d = cx[(cx["set"] == name) & cx["Lung Opacity"].notna()]
            sets[f"chexpert_{name}"] = (d["Lung Opacity"].values.astype(int), d.prob.values, d)
    curves = {}
    for name, (y, p, g) in sets.items():
        res = analyse(name, y, p, g, thr, cutoffs, qhat, mondrian)
        curves[name] = res.pop("_curve")
        report["sets"][name] = res

    (ROOT / "reports" / "uncertainty.json").write_text(json.dumps(report, indent=2))
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for name, (cov, risk) in curves.items():
        ax.plot(cov, risk, label=f"{name.replace('_', ' ')} (AURC {report['sets'][name]['aurc']:.3f})")
    for c in COVERAGES:
        ax.axvline(c, color="grey", lw=0.6, ls=":")
    ax.set(xlabel="coverage (share of cases the model answers)", ylabel="error rate on answered cases",
           title="Risk-coverage: deferring the least confident cases", xlim=(0.05, 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    (ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(ROOT / "reports" / "figures" / "risk_coverage.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
