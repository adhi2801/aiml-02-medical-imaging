"""Hugging Face Space: chest X-ray lung-opacity classifier with Grad-CAM. EDUCATION ONLY, NOT FOR CLINICAL USE.

Runs the same code path as the reading-room worker (services/inference/app/engine.py): DICOM decoding from
medimg.data.dicom, preprocessing from medimg.data.dataset, the temperature and per-view thresholds fitted on RSNA
validation (models/calibration.json). Nothing here is fitted on the test set.

Weights come from the HF model repo in MODEL_REPO (default below) or, when run from a clone, from ../models.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch

from medimg.data.dataset import preprocess
from medimg.data.dicom import read_dicom
from medimg.explain.gradcam import gradcam, overlay
from medimg.models.classifier import load_checkpoint

HERE = Path(__file__).resolve().parent
MODEL_REPO = os.environ.get("MODEL_REPO", "Adhi2801/aiml02-cxr-densenet121")
DISCLAIMER = ("**Education and research only. Not a medical device, not for clinical use.** The output is a suggestion "
              "that a qualified radiologist must review.")


def _asset(name: str) -> Path:
    local = HERE.parent / "models" / name
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(MODEL_REPO, name))


def _load():
    model, ckpt = load_checkpoint(_asset("classifier.pt"), "cpu")
    calib = json.loads(_asset("calibration.json").read_text())
    return model, ckpt["img_size"], calib


torch.set_num_threads(max(1, os.cpu_count() or 1))
MODEL, IMG_SIZE, CALIB = _load()


def _read(path: str) -> tuple[np.ndarray, str]:
    """Returns (uint8 grayscale, view). PNG/JPG uploads have no DICOM header, so the view is unknown."""
    if path.lower().endswith((".dcm", ".dicom")):
        img, meta, _ = read_dicom(path)
        return img, meta.view
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise gr.Error("Could not read the file. Upload a DICOM (.dcm), PNG or JPG chest X-ray.")
    return img, "U"


def analyze(path: str | None, view_choice: str):
    if not path:
        raise gr.Error("Upload a chest X-ray first.")
    img, view = _read(path)
    if view_choice != "from DICOM header":
        view = view_choice
    x = preprocess(img, IMG_SIZE)[None]
    logit, cam = gradcam(MODEL, x)
    prob = float(torch.sigmoid(logit / CALIB["temperature"])[0])
    thr = float(CALIB.get("view_thresholds", {}).get(view, CALIB["threshold"]))
    label = "lung opacity suggested" if prob >= thr else "no opacity suggested"

    disp = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)
    heat = overlay(disp, cam[0].numpy())
    margin = abs(float(logit[0]) / CALIB["temperature"])
    lines = [
        f"### {label}",
        f"Calibrated probability **{prob:.2f}** against threshold **{thr:.2f}** "
        f"({'per-view, ' + view if view in CALIB.get('view_thresholds', {}) else 'shared'} threshold, fitted on validation).",
        f"Confidence margin |logit/T| = {margin:.2f}.",
    ]
    if view == "U":
        lines.append("View unknown (no DICOM header): the shared threshold is used. Known weakness: PA-view sensitivity "
                     "is lower at the shared threshold.")
    lines += ["", DISCLAIMER]
    return heat, {"opacity": prob, "no opacity": 1 - prob}, "\n".join(lines)  # overlay() already returns RGB


EXAMPLES = sorted(str(p) for p in (HERE / "examples").glob("*") if p.suffix.lower() in (".dcm", ".png", ".jpg"))

with gr.Blocks(title="Chest X-ray opacity + Grad-CAM (education only)") as demo:
    gr.Markdown("# Chest X-ray lung opacity, with Grad-CAM\n"
                "DenseNet-121 fine-tuned on RSNA Pneumonia Detection. Held-out test AUROC 0.882 (95 % CI 0.870-0.894); "
                "Grad-CAM hottest pixel lands inside a radiologist box on 54 % of positives. "
                "[Code, evaluation and model card](https://github.com/adhi2801/aiml-02-medical-imaging).\n\n" + DISCLAIMER)
    with gr.Row():
        with gr.Column():
            inp = gr.File(label="Chest X-ray (DICOM, PNG or JPG)", file_types=[".dcm", ".dicom", ".png", ".jpg", ".jpeg"],
                          type="filepath")
            view = gr.Radio(["from DICOM header", "PA", "AP", "U"], value="from DICOM header", label="View position")
            btn = gr.Button("Analyze", variant="primary")
        with gr.Column():
            out_img = gr.Image(label="Grad-CAM overlay (where the model looked, not proof it is right)")
            out_lbl = gr.Label(label="Calibrated probability")
            out_txt = gr.Markdown()
    btn.click(analyze, [inp, view], [out_img, out_lbl, out_txt], concurrency_limit=2)
    if EXAMPLES:
        gr.Examples(EXAMPLES, inputs=[inp], label="Held-out RSNA test images (never used in training or threshold fitting)")

if __name__ == "__main__":
    demo.queue(max_size=16).launch()
