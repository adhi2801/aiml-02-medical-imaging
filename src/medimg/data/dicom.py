"""DICOM -> pixel array, done the way a radiology viewer does it.

Shared by the training pipeline and the inference service so that a scan arriving
from Orthanc is preprocessed exactly like the training data (no train/serve skew).
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pydicom
from pydicom.pixels import apply_modality_lut, apply_voi_lut


@dataclass
class DicomMeta:
    patient_id: str
    sex: str            # "M" | "F" | "U"
    age: int | None
    view: str           # "PA" | "AP" | "U"
    modality: str
    study_uid: str
    series_uid: str
    sop_uid: str


def _parse_age(value) -> int | None:
    # PatientAge is an Age String like "058Y", but RSNA stores plain integers like "58".
    if value in (None, ""):
        return None
    s = str(value).strip().upper()
    try:
        if s.endswith("Y"):
            return int(s[:-1])
        if s[-1:] in ("M", "W", "D"):
            return 0
        return int(s)
    except ValueError:
        return None


def read_meta(ds: pydicom.Dataset) -> DicomMeta:
    sex = str(getattr(ds, "PatientSex", "") or "U").upper()[:1]
    view = str(getattr(ds, "ViewPosition", "") or "U").upper()
    return DicomMeta(
        patient_id=str(getattr(ds, "PatientID", "")),
        sex=sex if sex in ("M", "F") else "U",
        age=_parse_age(getattr(ds, "PatientAge", None)),
        view=view if view in ("PA", "AP") else "U",
        modality=str(getattr(ds, "Modality", "")),
        study_uid=str(getattr(ds, "StudyInstanceUID", "")),
        series_uid=str(getattr(ds, "SeriesInstanceUID", "")),
        sop_uid=str(getattr(ds, "SOPInstanceUID", "")),
    )


def to_uint8(ds: pydicom.Dataset) -> np.ndarray:
    """Return a 2D uint8 image with modality LUT, VOI LUT/windowing and MONOCHROME1 handled."""
    arr = ds.pixel_array
    if arr.ndim == 3 and getattr(ds, "SamplesPerPixel", 1) == 3:
        arr = arr.mean(axis=-1)
    elif arr.ndim == 3:  # multi-frame: take the first frame
        arr = arr[0]
    arr = apply_modality_lut(arr, ds)
    arr = apply_voi_lut(arr, ds)
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, (0.5, 99.5))
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        arr = 1.0 - arr
    return (arr * 255).astype(np.uint8)


def read_dicom(source: str | bytes) -> tuple[np.ndarray, DicomMeta, pydicom.Dataset]:
    ds = pydicom.dcmread(io.BytesIO(source) if isinstance(source, bytes) else source)
    return to_uint8(ds), read_meta(ds), ds
