"""Model wrapper used by the worker: DICOM bytes -> calibrated probability, Grad-CAM, artifacts."""
from __future__ import annotations

import hashlib
import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pydicom
import torch
from pydicom.dataset import FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from medimg.data.dataset import preprocess
from medimg.data.dicom import read_dicom
from medimg.explain.gradcam import cam_to_rgba, gradcam, overlay
from medimg.models.classifier import load_checkpoint

AI_MANUFACTURER = "medimg-ai (EDUCATIONAL - NOT FOR CLINICAL USE)"
DISPLAY_SIZE = 1024


class Engine:
    def __init__(self, model_path: Path, calibration_path: Path, device: str = "cpu"):
        self.device = torch.device(device)
        self.model, ckpt = load_checkpoint(model_path, self.device)
        self.img_size = ckpt["img_size"]
        calib = json.loads(calibration_path.read_text())
        self.temperature, self.threshold = calib["temperature"], calib["threshold"]
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()[:8]
        self.version = f"{ckpt['arch']}-{digest}"
        torch.set_num_threads(max(1, torch.get_num_threads()))

    @staticmethod
    def is_ai_output(ds: pydicom.Dataset) -> bool:
        """Never re-analyze our own Secondary Captures (would loop forever)."""
        return str(getattr(ds, "Manufacturer", "")).startswith("medimg-ai")

    def analyze(self, dicom_bytes: bytes, instance_id: str, out_dir: Path) -> dict | None:
        img, meta, ds = read_dicom(dicom_bytes)
        if self.is_ai_output(ds):
            return None
        t0 = time.perf_counter()
        x = preprocess(img, self.img_size)[None].to(self.device)
        logit, cam = gradcam(self.model, x)
        prob = float(torch.sigmoid(logit / self.temperature)[0])
        latency_ms = (time.perf_counter() - t0) * 1000
        cam = cam[0].cpu().numpy()

        # Artifacts for the web viewer: the scan and a transparent heatmap at the same resolution.
        out_dir.mkdir(parents=True, exist_ok=True)
        disp = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(out_dir / f"{instance_id}_image.png"), disp)
        rgba = cam_to_rgba(cam, (DISPLAY_SIZE, DISPLAY_SIZE))
        cv2.imwrite(str(out_dir / f"{instance_id}_cam.png"), cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))

        return {
            "record": {
                "instance_id": instance_id, "study_uid": meta.study_uid, "sop_uid": meta.sop_uid,
                "patient_id": meta.patient_id, "sex": meta.sex, "age": meta.age, "view": meta.view,
                "probability": round(prob, 4), "threshold": round(self.threshold, 4),
                "ai_label": "opacity" if prob >= self.threshold else "no_opacity",
                "model_version": self.version, "latency_ms": round(latency_ms, 1),
            },
            "secondary_capture": self.secondary_capture(ds, overlay(disp, cam), prob),
        }

    def secondary_capture(self, src: pydicom.Dataset, rgb: np.ndarray, prob: float) -> bytes:
        """Grad-CAM burned into a DICOM Secondary Capture in the same study, so any PACS viewer
        (including Orthanc's own) shows the AI result next to the original series."""
        label = f"AI p(opacity)={prob:.2f} | EDUCATIONAL USE ONLY - NOT FOR CLINICAL USE"
        rgb = rgb.copy()
        cv2.rectangle(rgb, (0, 0), (rgb.shape[1], 34), (0, 0, 0), -1)
        cv2.putText(rgb, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        ds = pydicom.Dataset()
        ds.file_meta = meta
        for tag in ("PatientID", "PatientName", "PatientSex", "PatientAge", "StudyInstanceUID", "StudyDate",
                    "StudyTime", "AccessionNumber", "StudyID", "ReferringPhysicianName"):
            if tag in src:
                setattr(ds, tag, getattr(src, tag))
        ds.SOPClassUID, ds.SOPInstanceUID = SecondaryCaptureImageStorage, meta.MediaStorageSOPInstanceUID
        ds.SeriesInstanceUID = generate_uid()
        ds.Modality, ds.ConversionType = "OT", "WSD"
        ds.SeriesNumber, ds.InstanceNumber = 9001, 1
        ds.SeriesDescription = "AI Grad-CAM overlay (not for clinical use)"
        ds.Manufacturer = AI_MANUFACTURER
        ds.ImageComments = label
        ds.Rows, ds.Columns = rgb.shape[:2]
        ds.SamplesPerPixel, ds.PhotometricInterpretation, ds.PlanarConfiguration = 3, "RGB", 0
        ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 8, 8, 7, 0
        ds.PixelData = np.ascontiguousarray(rgb, dtype=np.uint8).tobytes()
        buf = io.BytesIO()
        ds.save_as(buf, enforce_file_format=True)
        return buf.getvalue()
