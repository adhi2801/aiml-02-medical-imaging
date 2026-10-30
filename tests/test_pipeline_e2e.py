"""End-to-end smoke test: the real DVC stages on a tiny synthetic RSNA-shaped dataset, then the live
inference engine on a DICOM. Catches integration bugs before spending GPU hours on Kaggle.

Synthetic "opacities" are bright blobs placed inside the labeled boxes, so the pipeline has a learnable
signal, but the point is that every stage runs and produces its artifacts, not the accuracy.
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from tests.test_core import make_dicom

REPO = Path(__file__).resolve().parents[1]
N = 48


def build_fake_rsna(root: Path) -> None:
    raw = root / "data/raw/rsna"
    (raw / "stage_2_train_images").mkdir(parents=True)
    rng = np.random.default_rng(0)
    labels, classes = [], []
    for i in range(N):
        pid = f"fake{i:03d}"
        img = rng.normal(1200, 150, (256, 256)).clip(0, 4095)
        positive = i % 2 == 0
        if positive:
            x, y = rng.integers(30, 150, 2)
            img[y:y + 60, x:x + 60] += 1500
            labels.append([pid, x * 4.0, y * 4.0, 240.0, 240.0, 1])  # boxes in 1024-px coordinates
        else:
            labels.append([pid, None, None, None, None, 0])
        classes.append([pid, "Lung Opacity" if positive else "Normal"])
        make_dicom(raw / "stage_2_train_images" / f"{pid}.dcm", img.astype(np.uint16),
                   sex="MF"[i % 2], age=str(30 + i), view=["PA", "AP"][(i // 2) % 2])
    pd.DataFrame(labels, columns=["patientId", "x", "y", "width", "height", "Target"]).to_csv(
        raw / "stage_2_train_labels.csv", index=False)
    pd.DataFrame(classes, columns=["patientId", "class"]).to_csv(raw / "stage_2_detailed_class_info.csv", index=False)

    params = yaml.safe_load((REPO / "params.yaml").read_text())
    params["data"]["cache_size"] = 128
    params["classifier"] |= {"arch": "efficientnet_b0", "img_size": 128, "batch_size": 8, "epochs": 2,
                             "num_workers": 0, "early_stop_patience": 5}
    params["eval"]["bootstrap_iters"] = 50
    (root / "params.yaml").write_text(yaml.safe_dump(params))


def stage(module: str, root: Path) -> None:
    env = {**os.environ, "MEDIMG_ROOT": str(root), "WANDB_API_KEY": ""}
    r = subprocess.run([sys.executable, "-m", module], cwd=REPO, env=env, capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=900)
    assert r.returncode == 0, f"{module} failed:\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}"


@pytest.mark.slow
def test_full_pipeline(tmp_path):
    build_fake_rsna(tmp_path)
    stage("medimg.data.prepare_rsna", tmp_path)
    meta = pd.read_csv(tmp_path / "data/processed/rsna/metadata.csv")
    assert len(meta) == N and set(meta.split) == {"train", "val", "test"}
    assert meta.loc[meta.label == 1, "boxes"].map(lambda b: len(json.loads(b))).min() == 1

    stage("medimg.train_cls", tmp_path)
    assert (tmp_path / "models/classifier.pt").exists()

    stage("medimg.evaluate_cls", tmp_path)
    metrics = json.loads((tmp_path / "reports/metrics.json").read_text())
    for key in ("auroc", "auroc_ci95", "ece_before", "ece_after", "threshold", "temperature",
                "gradcam_pointing_game", "gradcam_mean_iou", "sensitivity", "specificity"):
        assert key in metrics, key
    assert (tmp_path / "reports/figures/test_curves.png").exists()
    assert (tmp_path / "reports/figures/gradcam_gallery.png").exists()
    assert (tmp_path / "reports/subgroups.csv").exists()

    # The live service engine on a DICOM, including the Secondary Capture it writes back to Orthanc.
    sys.path.insert(0, str(REPO / "services/inference"))
    import pydicom
    from app.engine import Engine

    engine = Engine(tmp_path / "models/classifier.pt", tmp_path / "models/calibration.json")
    dcm = (tmp_path / "data/raw/rsna/stage_2_train_images/fake000.dcm").read_bytes()
    out = engine.analyze(dcm, "inst-0", tmp_path / "artifacts")
    rec = out["record"]
    assert 0 <= rec["probability"] <= 1 and rec["ai_label"] in ("opacity", "no_opacity")
    assert (tmp_path / "artifacts/inst-0_image.png").exists() and (tmp_path / "artifacts/inst-0_cam.png").exists()
    sc = pydicom.dcmread(__import__("io").BytesIO(out["secondary_capture"]))
    assert sc.Modality == "OT" and engine.is_ai_output(sc)
    assert engine.analyze(out["secondary_capture"], "inst-sc", tmp_path / "artifacts") is None  # no feedback loop
