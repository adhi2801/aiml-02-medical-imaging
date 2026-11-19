# One-word entry points.
#   make setup        CPU venv, same resolve as CI (constraints-cpu.txt)
#   make setup-cuda   CUDA 12.4 venv (constraints-cu124.txt)
#   make test         byte-compile + pytest, exactly what CI runs
#   make stack        Orthanc + inference worker + reading-room viewer (docker compose)
#   make demo         the Hugging Face Space app locally (needs models/classifier.pt or MODEL_REPO)
#   make reports      re-run the DVC pipeline stages whose inputs changed
#   make uncertainty  selective prediction, conformal sets, subgroup error rates from reports/predictions_rsna.csv
#   make kaggle       re-run train + eval on a Kaggle T4
PY := .venv/Scripts/python
ifeq ($(OS),)
PY := .venv/bin/python
endif
TORCH_CPU := --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match
DEPS := torch torchvision "monai>=1.4" nibabel "pydicom>=3.0" pandas scikit-learn scipy pillow opencv-python-headless \
	matplotlib pyyaml tqdm pytest -r services/inference/requirements.txt -e .

.PHONY: setup setup-cuda test stack demo reports uncertainty kaggle

setup:
	uv venv --python 3.12
	uv pip install -c constraints-cpu.txt $(TORCH_CPU) $(DEPS)

setup-cuda:
	uv venv --python 3.12
	uv pip install -c constraints-cu124.txt --extra-index-url https://download.pytorch.org/whl/cu124 \
		--index-strategy unsafe-best-match $(DEPS)

test:
	$(PY) -m compileall -q src scripts tests services
	$(PY) -m pytest -q

stack:
	docker compose up --build

demo:
	uv pip install "gradio==5.9.1" requests
	cd space && ../$(PY) app.py

reports:
	dvc repro

uncertainty:
	$(PY) -m medimg.uncertainty_report

kaggle:
	$(PY) scripts/kaggle_push.py
