"""Per-view thresholds, external validation and CPU spleen inference: CPU only, no dataset needed."""
import json
import sys
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch

from medimg.eval.stats import confusion_stats
from medimg.evaluate_cls import threshold_for_sensitivity, view_thresholds
from medimg.external_cls import binary_labels, evaluate, load_split
from medimg.seg import predict
from medimg.seg.train_seg import build_model

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "inference"))


def test_threshold_for_sensitivity_reaches_target_and_no_lower():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 2000)
    p = np.clip(rng.normal(0.35 + 0.3 * y, 0.15), 0, 1)
    for target in (0.5, 0.8, 0.95):
        t = threshold_for_sensitivity(y, p, target)
        assert confusion_stats(y, p, t)["sensitivity"] >= target
        # the highest such threshold: nudging it up drops below the target
        assert confusion_stats(y, p, t + 1e-6)["sensitivity"] < target


def test_view_thresholds_equalise_sensitivity_across_views():
    """A model that scores PA positives lower (the RSNA failure) gets a lower PA threshold, and the PA sensitivity
    on held-out data moves up to the shared target."""
    rng = np.random.default_rng(1)

    def make(n):
        view = rng.choice(["AP", "PA"], n)
        y = rng.integers(0, 2, n)
        shift = np.where(view == "PA", -0.15, 0.0)
        p = np.clip(rng.normal(0.3 + 0.35 * y + shift * y, 0.12), 0, 1)
        return pd.DataFrame({"view": view, "label": y}), p

    (val, pv), (test, pt) = make(4000), make(4000)
    shared = 0.5
    out = view_thresholds(val, pv, test, pt, shared)
    ap, pa = out["views"]["AP"], out["views"]["PA"]
    assert pa["threshold"] < ap["threshold"]
    assert pa["per_view_threshold"]["sensitivity"] > pa["shared_threshold"]["sensitivity"]
    target = out["target_sensitivity_from_val"]
    assert abs(pa["per_view_threshold"]["sensitivity"] - target) < 0.05
    assert abs(ap["per_view_threshold"]["sensitivity"] - target) < 0.05


def test_view_thresholds_fall_back_when_a_view_has_no_val_positives():
    """Small splits (the e2e smoke test, a new site) can have a view with no positives in VAL: keep the shared
    threshold for that view and say so, instead of crashing the evaluation stage."""
    val = pd.DataFrame({"view": ["AP", "AP", "PA", "PA"], "label": [1, 0, 0, 0]})
    test = pd.DataFrame({"view": ["AP", "PA", "PA"], "label": [1, 0, 0]})
    out = view_thresholds(val, np.array([0.9, 0.2, 0.3, 0.1]), test, np.array([0.8, 0.4, 0.2]), shared=0.5)
    assert out["views"]["AP"]["fitted_on_val"] and out["views"]["AP"]["threshold"] == 0.9
    assert not out["views"]["PA"]["fitted_on_val"] and out["views"]["PA"]["threshold"] == 0.5
    assert out["overall_test_per_view_policy"] == {"sensitivity": 1.0, "specificity": 1.0}
    assert threshold_for_sensitivity(np.array([0, 0]), np.array([0.1, 0.2]), 0.8) is None


def test_engine_uses_view_threshold_with_shared_fallback():
    from app.engine import Engine

    e = Engine.__new__(Engine)
    e.threshold, e.view_thresholds = 0.3, {"AP": 0.4, "PA": 0.2}
    assert e.threshold_for("PA") == 0.2
    assert e.threshold_for("AP") == 0.4
    assert e.threshold_for("LL") == 0.3
    assert e.threshold_for(None) == 0.3


def test_chexpert_labels_uncertain_dropped_blank_negative():
    s = binary_labels(pd.Series([1.0, 0.0, -1.0, np.nan]))
    assert s.iloc[0] == 1 and s.iloc[1] == 0 and np.isnan(s.iloc[2]) and s.iloc[3] == 0


@pytest.fixture
def fake_chexpert(tmp_path):
    root = tmp_path / "CheXpert-v1.0-small"
    rows = []
    rng = np.random.default_rng(0)
    for i in range(40):
        for view, fl in (("AP" if i % 2 else "PA", "Frontal"), ("", "Lateral")):
            rel = f"CheXpert-v1.0-small/valid/patient{i:05d}/study1/view{1 if fl == 'Frontal' else 2}_{fl.lower()}.jpg"
            (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp_path / rel), rng.integers(0, 255, (64, 64), dtype=np.uint8))
            rows.append({"Path": rel, "Frontal/Lateral": fl, "AP/PA": view or np.nan,
                         "Lung Opacity": float(i % 3 == 0), "Pneumonia": [1.0, 0.0, -1.0, np.nan][i % 4],
                         "Consolidation": np.nan})
    pd.DataFrame(rows).to_csv(root / "valid.csv", index=False)
    return root


def test_load_split_keeps_frontal_and_resolves_paths(fake_chexpert):
    df = load_split(fake_chexpert, "valid", None, 0)
    assert len(df) == 40 and set(df.view) == {"AP", "PA"}
    assert all(Path(f).exists() for f in df.file)
    sampled = load_split(fake_chexpert, "valid", 10, 0)
    assert len(sampled) == 10 and sampled.patient.is_unique


def test_evaluate_reports_primary_by_view_and_skips_single_class(fake_chexpert):
    df = load_split(fake_chexpert, "valid", None, 0)
    probs = df["Lung Opacity"].values * 0.5 + np.linspace(0, 0.4, len(df))
    rep = evaluate(df, probs, 0.5, iters=50, seed=0, bins=10)
    assert set(rep) == {"Lung Opacity", "Pneumonia"}  # Consolidation is all blank -> one class -> skipped
    assert rep["Pneumonia"]["n"] == 30  # the 10 uncertain labels are dropped
    assert set(rep["Lung Opacity"]["by_view"]) == {"AP", "PA"}


def test_fold_agreement():
    a = np.zeros((10, 10, 10), np.uint8)
    a[2:6, 2:6, 2:6] = 1
    assert predict.fold_agreement([a]) is None
    assert predict.fold_agreement([a, a, a]) == 1.0
    b = np.zeros_like(a)
    assert predict.fold_agreement([a, b]) == 0.0


def test_predict_end_to_end_mask_on_input_grid(tmp_path):
    """Two random-weight models on a small synthetic CT: the mask must come back on the CT's own grid and affine."""
    ct = np.full((70, 60, 30), -1000.0, np.float32)
    ct[15:55, 10:50, 5:25] = 60.0  # soft tissue block
    affine = np.diag([0.8, 0.8, 2.5, 1.0])
    nib.save(nib.Nifti1Image(ct, affine), tmp_path / "ct.nii.gz")
    models = []
    for k in range(2):
        torch.manual_seed(k)
        torch.save({"state_dict": build_model().state_dict()}, tmp_path / f"fold{k}.pt")
        models.append(str(tmp_path / f"fold{k}.pt"))
    out = tmp_path / "out"
    summary = predict.main([str(tmp_path / "ct.nii.gz"), "--models", *models, "--out", str(out), "--device", "cpu"])
    mask = nib.load(out / "ct_spleen.nii.gz")
    assert mask.shape == ct.shape
    assert np.allclose(mask.affine, affine, atol=1e-4)
    assert set(np.unique(mask.get_fdata())) <= {0.0, 1.0}
    assert (out / "ct_spleen.png").exists()
    assert json.loads((out / "ct_spleen.json").read_text())["models"] == ["fold0.pt", "fold1.pt"]
    assert summary["fold_agreement_dice"] is not None
