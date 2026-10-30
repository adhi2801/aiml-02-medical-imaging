"""DVC stage `prepare`: RSNA DICOMs -> cached PNGs + one metadata table with splits.

Output metadata.csv columns:
  patient_id, label (1 = lung opacity), class_name, sex, age, view, split, boxes (JSON, original 1024px coords)

RSNA has exactly one image per patientId, so a split by image is also a split by patient,
which rules out patient leakage between train/val/test.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from medimg import ROOT, load_params
from medimg.data.dicom import read_dicom


def _process_one(args: tuple[Path, Path, int]) -> dict:
    dcm_path, out_dir, size = args
    img, meta, ds = read_dicom(str(dcm_path))
    h, w = img.shape
    cv2.imwrite(str(out_dir / f"{dcm_path.stem}.png"), cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
    return {"patient_id": dcm_path.stem, "sex": meta.sex, "age": meta.age, "view": meta.view, "orig_h": h, "orig_w": w}


def main() -> None:
    p = load_params()
    raw = ROOT / p["data"]["raw_dir"]
    out = ROOT / p["data"]["processed_dir"]
    img_out = out / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    labels = pd.read_csv(raw / "stage_2_train_labels.csv")
    classes = pd.read_csv(raw / "stage_2_detailed_class_info.csv").drop_duplicates("patientId")

    boxes = (
        labels[labels.Target == 1]
        .groupby("patientId")[["x", "y", "width", "height"]]
        .apply(lambda g: json.dumps(g.values.round(1).tolist()))
        .rename("boxes")
    )
    per_patient = (
        labels.groupby("patientId").Target.max().rename("label").to_frame()
        .join(classes.set_index("patientId")["class"].rename("class_name"))
        .join(boxes)
        .reset_index()
        .rename(columns={"patientId": "patient_id"})
    )
    per_patient["boxes"] = per_patient["boxes"].fillna("[]")

    dcm_dir = raw / "stage_2_train_images"
    jobs = [(dcm_dir / f"{pid}.dcm", img_out, p["data"]["cache_size"]) for pid in per_patient.patient_id]
    with ProcessPoolExecutor() as ex:
        meta = list(tqdm(ex.map(_process_one, jobs, chunksize=64), total=len(jobs), desc="DICOM -> PNG"))
    df = per_patient.merge(pd.DataFrame(meta), on="patient_id")

    # Stratify on label x view so AP/PA prevalence (a known shortcut in this dataset) is equal across splits.
    strata = df.label.astype(str) + "_" + df.view
    seed = p["seed"]
    test_frac, val_frac = p["data"]["test_frac"], p["data"]["val_frac"]
    trainval, test = train_test_split(df, test_size=test_frac, stratify=strata, random_state=seed)
    train, val = train_test_split(
        trainval, test_size=val_frac / (1 - test_frac), stratify=strata.loc[trainval.index], random_state=seed
    )
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"

    df.to_csv(out / "metadata.csv", index=False)
    summary = df.groupby(["split", "view"]).label.agg(["count", "mean"]).round(3)
    print(summary)
    (ROOT / "reports").mkdir(exist_ok=True)
    (ROOT / "reports" / "data_summary.json").write_text(json.dumps(
        {"n_images": len(df), "prevalence": round(df.label.mean(), 4),
         "by_split": df.groupby("split").label.agg(["count", "mean"]).round(4).to_dict(orient="index")},
        indent=2))


if __name__ == "__main__":
    main()
