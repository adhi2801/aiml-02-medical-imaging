# AI-Assisted Medical Imaging & Diagnosis System

> ⚠️ **For education and research only. Not a medical device. Not for clinical use.**
> Every AI output in this system is a suggestion that a qualified radiologist must review.

An end-to-end platform for AI-assisted radiology: chest X-rays (DICOM) arrive at an **Orthanc PACS**, a
worker runs a **DenseNet-121** classifier with **Grad-CAM** explanations, and a **React reading-room viewer**
lets a radiologist approve or reject each finding. Every decision goes into a **hash-chained, append-only audit log**.
A separate **MONAI 3D U-Net** segments the spleen in abdominal CT.

| | |
|---|---|
| Classification | RSNA Pneumonia Detection (26,684 chest X-rays, native DICOM, radiologist boxes) |
| Segmentation | Medical Segmentation Decathlon Task09 Spleen (CT) |
| Stack | PyTorch · MONAI · Grad-CAM · Orthanc · FastAPI · React/TypeScript · DVC · Docker · W&B |

## Results

All numbers are produced by `dvc repro` and read from `reports/` (never typed by hand).

| Metric (held-out test set) | Value |
|---|---|
| AUROC (95% bootstrap CI) | _from `reports/metrics.json`_ |
| Sensitivity / Specificity @ val-selected threshold | _…_ |
| ECE before → after temperature scaling | _…_ |
| Grad-CAM pointing game (hit inside radiologist box) | _…_ |
| Spleen Dice / HD95 (mm) | _from `reports/seg_metrics.json`_ |

Subgroup performance (sex, age band, AP vs PA view) is in `reports/subgroups.csv` and discussed in the
[model card](docs/MODEL_CARD.md).

## What makes this more than a demo

- **Explanations are measured, not just shown.** RSNA includes radiologist bounding boxes, so Grad-CAM is scored with the
  *pointing game* and *IoU*. A heatmap that looks plausible is not evidence that it is correct.
- **Statistically honest evaluation.** Bootstrap 95% CIs, calibration (ECE + reliability diagram), and the operating
  threshold and temperature are fit on validation only, so the test set is touched exactly once.
- **Bias audit.** Performance is broken down by sex, age band and view position. AP (portable) films are sicker patients,
  which gives the model a known shortcut. We measure it and warn about it in the UI.
- **Real DICOM flow.** Studies enter through Orthanc (REST or DICOM C-STORE). The AI result is written back into the
  same study as a DICOM Secondary Capture, so it also appears in any PACS viewer.
- **Tamper-evident audit trail.** Each entry stores `sha256(prev_hash + entry)`. SQLite triggers block UPDATE/DELETE, and
  `/api/audit/verify` pinpoints the first altered row. Rejecting an AI finding requires a written reason.
- **No train/serve skew.** Training and the live service share the same DICOM decoding (`medimg.data.dicom`) and
  preprocessing code.
- **Reproducible.** DVC pipeline `prepare → train_cls → eval_cls` and `train_seg`, all driven by `params.yaml`.

## Architecture

```mermaid
flowchart LR
  subgraph Sources
    M[Modality / script<br/>C-STORE or REST]
  end
  subgraph Docker["docker compose"]
    O[(Orthanc PACS<br/>:8042 / :4242)]
    W[Inference worker<br/>DenseNet-121 + Grad-CAM<br/>temperature-calibrated]
    A[FastAPI<br/>:8000 /docs]
    D[(SQLite<br/>results + hash-chained audit)]
    V[React viewer<br/>nginx :8080]
  end
  R((Radiologist))
  M -->|DICOM| O
  O -->|/changes feed| W
  W -->|GET instance| O
  W -->|Secondary Capture overlay| O
  W --> D
  A --> D
  V -->|/api| A
  R -->|approve / reject + reason| V
```

Training pipeline (DVC):

```mermaid
flowchart LR
  raw[RSNA DICOMs] --> prepare[prepare<br/>DICOM→PNG, stratified split] --> train[train_cls<br/>DenseNet-121, AMP] --> eval[eval_cls<br/>CI, calibration, subgroups, CAM localization]
  msd[MSD Spleen CT] --> seg[train_seg<br/>MONAI 3D U-Net] --> segm[Dice / HD95]
```

## Quick start

### 1. Train (free Kaggle GPU)
Import `notebooks/kaggle_train.ipynb` into Kaggle, attach the *rsna-pneumonia-detection-challenge* dataset, and Run All.
Download `aiml02_outputs.tgz` and extract it into the repo root (it contains `models/` and `reports/`).

Local training also works (`dvc repro`) but needs `classifier.batch_size: 6` on a 4 GB GPU.

### 2. Run the system
```bash
docker compose up --build
python scripts/upload_to_orthanc.py data/raw/rsna/stage_2_train_images --from-test-split 40
```
Open **http://localhost:8080**. Studies appear in the worklist within a few seconds.

| Service | URL |
|---|---|
| Radiologist viewer | http://localhost:8080 |
| API + OpenAPI docs | http://localhost:8000/docs |
| Orthanc (demo login `orthanc`/`orthanc`) | http://localhost:8042 |

Viewer shortcuts: `J`/`K` next/previous study · `O` toggle overlay · `A` approve · `R` write a rejection reason ·
drag to pan · shift+drag window/level · wheel zoom.

### 3. Develop
```bash
uv venv .venv --python 3.12
uv pip install --python .venv torch torchvision --index-url https://download.pytorch.org/whl/cu124
uv pip install --python .venv -r requirements.txt -r services/inference/requirements.txt -e .
.venv/Scripts/python -m pytest          # CPU-only, no dataset needed
cd web && npm install && npm run dev    # http://localhost:5173, proxies /api to :8000
```

## API

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/health` | model version, worker status, disclaimer |
| GET | `/api/results?status=` | worklist (`pending_review` / `approved` / `rejected`) |
| GET | `/api/results/{id}` | one AI result |
| GET | `/api/results/{id}/image.png`, `/cam.png` | display image and transparent Grad-CAM layer |
| POST | `/api/results/{id}/review` | `{decision, reviewer, reason}`. A reason is required for reject |
| GET | `/api/audit?instance_id=` | audit entries (newest first) |
| GET | `/api/audit/verify` | recompute the hash chain |
| GET | `/api/stats` | pending count, radiologist agreement, mean latency |

## Repository layout
```
src/medimg/            training library (data, models, Grad-CAM, evaluation, 3D segmentation)
services/inference/    FastAPI + Orthanc worker + audit store (Dockerfile)
services/orthanc/      PACS configuration
web/                   React + TypeScript viewer (Dockerfile, nginx)
notebooks/             Kaggle training notebook
scripts/               demo data loader
tests/                 pytest: DICOM handling, Grad-CAM, statistics, audit chain, API
docs/                  model card, architecture notes
params.yaml, dvc.yaml  reproducible pipeline
```

## Limitations
See [docs/MODEL_CARD.md](docs/MODEL_CARD.md). In short: single-source training data, binary label only, no
prospective validation, Grad-CAM is coarse (about 12×12 at the last layer), and performance differs across view positions.
**Not for clinical use.**
