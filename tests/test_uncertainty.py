"""Selective prediction and conformal sets: their guarantees hold on synthetic data where the truth is known."""
import numpy as np

from medimg.eval.uncertainty import (conformal_qhat, cutoff_for_coverage, margin, mondrian_qhat, risk_coverage,
                                     scores, selective_stats, set_report, wilson)


def calibrated(n: int, seed: int, shift: float = 0.0):
    """Labels drawn from the model's own probabilities, so p is perfectly calibrated."""
    rng = np.random.default_rng(seed)
    p = 1 / (1 + np.exp(-(rng.normal(shift, 2.0, n))))
    return (rng.random(n) < p).astype(int), p


def test_margin_is_zero_at_the_threshold_and_symmetric_in_logit_space():
    assert margin(np.array([0.3]), 0.3)[0] < 1e-9
    a, b = margin(np.array([0.3 * 1.5 / (1 + 0.3 * 0.5), 0.3 / (1.5 - 0.3 * 0.5)]), 0.3)  # logit(0.3) +- log 1.5
    assert abs(a - b) < 1e-9


def test_confident_cases_are_more_accurate_than_random_order():
    y, p = calibrated(5000, 0)
    _, risk, aurc = risk_coverage(y, p, 0.5)
    assert risk[int(0.5 * len(y))] < risk[-1]  # accepting the confident half lowers the error
    rng = np.random.default_rng(1)
    shuffled = rng.permutation(len(y))
    # a random ranking has (about) constant risk, so its AURC is the full-coverage error
    assert aurc < risk[-1] - 0.03
    assert np.isclose(risk_coverage(y[shuffled], p[shuffled], 0.5)[1][-1], risk[-1])


def test_cutoff_fit_on_val_gives_the_target_coverage_on_new_data():
    yv, pv = calibrated(4000, 2)
    yt, pt = calibrated(4000, 3)
    cut = cutoff_for_coverage(pv, 0.4, 0.8)
    s = selective_stats(yt, pt, 0.4, cut)
    assert abs(s["coverage"] - 0.8) < 0.03
    assert s["error_accepted"] < s["error_all"]
    assert s["deferred"] == int(round((1 - s["coverage"]) * len(yt)))


def test_split_conformal_covers_at_least_one_minus_alpha_on_exchangeable_data():
    alpha, hits = 0.1, []
    for seed in range(20):
        yc, pc = calibrated(500, seed)
        yt, pt = calibrated(2000, 100 + seed)
        hits.append(set_report(yt, pt, conformal_qhat(scores(yc, pc), alpha))["coverage"])
    assert np.mean(hits) >= 1 - alpha - 0.01  # the guarantee is on average over calibration draws


def test_too_few_calibration_cases_give_the_full_set():
    assert conformal_qhat(np.array([0.1, 0.2]), alpha=0.1) == float("inf")
    rep = set_report(np.array([0, 1]), np.array([0.2, 0.9]), float("inf"))
    assert rep["coverage"] == 1.0 and rep["both_labels"] == 1.0


def test_mondrian_restores_coverage_in_a_group_the_marginal_quantile_undercovers():
    # group B's probabilities are overconfident (labels flipped 30 % of the time), group A's are calibrated
    rng = np.random.default_rng(5)
    def make(n):
        y, p = calibrated(n, int(rng.integers(1e9)))
        g = np.where(rng.random(n) < 0.3, "B", "A")
        flip = (g == "B") & (rng.random(n) < 0.3)
        return np.where(flip, 1 - y, y), p, g
    yc, pc, gc = make(6000)
    yt, pt, gt = make(6000)
    alpha = 0.1
    marginal = conformal_qhat(scores(yc, pc), alpha)
    per_group = mondrian_qhat(yc, pc, gc, alpha)
    b = gt == "B"
    assert set_report(yt[b], pt[b], marginal)["coverage"] < 1 - alpha - 0.02
    assert set_report(yt[b], pt[b], per_group["B"])["coverage"] >= 1 - alpha - 0.02


def test_wilson_interval_contains_the_estimate_and_stays_in_unit_range():
    lo, hi = wilson(0, 10)
    assert lo == 0.0 or lo > -1e-12
    lo, hi = wilson(90, 100)
    assert lo < 0.9 < hi and 0 <= lo and hi <= 1


def test_report_end_to_end_on_synthetic_predictions(tmp_path, monkeypatch):
    import json

    import pandas as pd

    from medimg import uncertainty_report as ur

    rng = np.random.default_rng(0)
    rows = []
    for split, n in (("val", 800), ("test", 800)):
        y, p = calibrated(n, 7 if split == "val" else 8)
        rows.append(pd.DataFrame({"split": split, "patient_id": [f"{split}{i}" for i in range(n)],
                                  "view": rng.choice(["AP", "PA"], n), "sex": rng.choice(["M", "F"], n),
                                  "age": rng.integers(20, 90, n), "label": y, "logit": 0.0, "prob": p}))
    (tmp_path / "reports").mkdir()
    (tmp_path / "models").mkdir()
    pd.concat(rows).to_csv(tmp_path / "reports" / "predictions_rsna.csv", index=False)
    (tmp_path / "models" / "calibration.json").write_text(json.dumps({"threshold": 0.5}))
    monkeypatch.setattr(ur, "ROOT", tmp_path)
    ur.main()
    rep = json.loads((tmp_path / "reports" / "uncertainty.json").read_text())
    t = rep["sets"]["rsna_test"]
    assert set(t["selective"]) == {"90%", "80%", "70%"}
    assert t["conformal_marginal"]["coverage"] >= 0.85
    assert "sex=M" in t["conformal_by_group"] and "view=AP" in t["conformal_by_group"]
    assert (tmp_path / "reports" / "figures" / "risk_coverage.png").exists()
