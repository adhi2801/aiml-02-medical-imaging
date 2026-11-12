"""Build the demo pack: held-out TEST chest X-rays (never used for training, validation or threshold fitting), balanced
between opacity and no opacity, as the original DICOM files, with a manifest of the radiologists' ground truth.

    python scripts/export_demo_pack.py --n 24 --out demo_pack

Runs where the RSNA data is (the Kaggle eval-only run calls it). The images are the RSNA challenge's DICOMs, which are
NIH Clinical Center chest X-rays (ChestX-ray8/14) re-released as DICOM; see demo_pack/ATTRIBUTION.md.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

ATTRIBUTION = """# Demo pack: attribution

{n} chest X-rays from the held-out **test** split of this project (no image here was used for training, validation
or fitting the threshold), balanced between radiologist-confirmed lung opacity and no opacity. `manifest.csv` has the
ground truth (label, radiologist boxes in pixel coordinates of the 1024 x 1024 image) and the DICOM metadata used in
the subgroup analysis.

**Source.** RSNA Pneumonia Detection Challenge (Radiological Society of North America, 2018), whose images are the
NIH Clinical Center's ChestX-ray8/14 collection: Wang X, Peng Y, Lu L, Lu Z, Bagheri M, Summers RM. *ChestX-ray8:
Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common
Thorax Diseases.* IEEE CVPR 2017. The labels and boxes are from the RSNA / Society of Thoracic Radiology annotation
effort: Shih G et al., *Augmenting the NIH Chest Radiograph Dataset with Expert Annotations of Possible Pneumonia.*
Radiology: AI 2019;1(1):e180041.

The images are de-identified by their publishers. Included for education and research only; not for clinical use.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--out", type=Path, default=ROOT / "demo_pack")
    ap.add_argument("--dicoms", type=Path, default=ROOT / "data/raw/rsna/stage_2_train_images")
    args = ap.parse_args()

    meta = pd.read_csv(ROOT / "data/processed/rsna/metadata.csv")
    test = meta[meta.split == "test"]
    half = args.n // 2
    # Same picks as scripts/upload_to_orthanc.py --from-test-split (random_state=0)
    picks = pd.concat([test[test.label == 1].sample(half, random_state=0),
                       test[test.label == 0].sample(args.n - half, random_state=0)])
    out = args.out / "dicom"
    out.mkdir(parents=True, exist_ok=True)
    for pid in picks.patient_id:
        shutil.copy2(args.dicoms / f"{pid}.dcm", out / f"{pid}.dcm")
    cols = ["patient_id", "label", "class_name", "view", "sex", "age", "boxes"]
    picks[cols].to_csv(args.out / "manifest.csv", index=False)
    (args.out / "ATTRIBUTION.md").write_text(ATTRIBUTION.format(n=len(picks)))
    print(f"{len(picks)} test DICOMs -> {args.out}")


if __name__ == "__main__":
    main()
