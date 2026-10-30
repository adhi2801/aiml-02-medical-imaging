# Build from the repo root:  docker build -f services/inference/Dockerfile .
FROM python:3.12-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

# CPU wheels keep the image ~1.5 GB instead of ~6 GB; a DenseNet-121 forward+Grad-CAM is well under 1 s on CPU.
RUN pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
COPY services/inference/requirements.txt /tmp/service-req.txt
RUN pip install -r /tmp/service-req.txt "pydicom>=3.0" numpy opencv-python-headless pyyaml pandas

COPY pyproject.toml /build/
COPY src /build/src
RUN pip install --no-deps /build && rm -rf /build

RUN useradd --create-home --uid 1000 app && mkdir -p /data && chown app /data
USER app
WORKDIR /srv
COPY --chown=app services/inference/app ./app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=30s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
