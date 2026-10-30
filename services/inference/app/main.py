"""REST API for the radiologist viewer. OpenAPI docs are served at /docs."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from .db import Store

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
ARTIFACTS = DATA_DIR / "artifacts"
DISCLAIMER = "For education and research only. Not a medical device. Not for clinical use."

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["store"] = Store(DATA_DIR / "medimg.sqlite")
    if os.environ.get("DISABLE_WORKER") != "1":
        from .engine import Engine
        from .worker import OrthancWorker

        engine = Engine(Path(os.environ.get("MODEL_PATH", "/models/classifier.pt")),
                        Path(os.environ.get("CALIBRATION_PATH", "/models/calibration.json")),
                        os.environ.get("DEVICE", "cpu"))
        auth = (os.environ["ORTHANC_USER"], os.environ["ORTHANC_PASSWORD"]) if os.environ.get("ORTHANC_USER") else None
        worker = OrthancWorker(engine, state["store"], os.environ.get("ORTHANC_URL", "http://orthanc:8042"),
                               ARTIFACTS, auth)
        worker.start()
        state.update(engine=engine, worker=worker)
    yield
    if "worker" in state:
        state["worker"].stop()


app = FastAPI(title="AI-Assisted Medical Imaging API", description=DISCLAIMER, version="0.1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("CORS_ORIGINS", "http://localhost:5173").split(","),
                   allow_methods=["*"], allow_headers=["*"])


class ReviewIn(BaseModel):
    decision: Literal["approve", "reject"]
    reviewer: str = Field(min_length=2, max_length=80)
    reason: str | None = Field(default=None, max_length=1000)

    @model_validator(mode="after")
    def reject_needs_reason(self):
        if self.decision == "reject" and not (self.reason and self.reason.strip()):
            raise ValueError("a reason is required when rejecting an AI finding")
        return self


def store() -> Store:
    return state["store"]


@app.get("/api/health")
def health():
    w = state.get("worker")
    e = state.get("engine")
    return {"status": "ok", "disclaimer": DISCLAIMER,
            "model_version": e.version if e else None,
            "worker": {"alive": w.is_alive(), "processed": w.processed, "last_error": w.last_error} if w else None}


@app.get("/api/results")
def list_results(status: Literal["pending_review", "approved", "rejected"] | None = None):
    return store().list_results(status)


@app.get("/api/results/{instance_id}")
def get_result(instance_id: str):
    r = store().get_result(instance_id)
    if r is None:
        raise HTTPException(404, "no AI result for this instance")
    return r | {"disclaimer": DISCLAIMER}


@app.get("/api/results/{instance_id}/{kind}.png")
def artifact(instance_id: str, kind: Literal["image", "cam"]):
    path = ARTIFACTS / f"{instance_id}_{kind}.png"
    if not path.resolve().is_relative_to(ARTIFACTS.resolve()) or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="image/png")


@app.post("/api/results/{instance_id}/review")
def review(instance_id: str, body: ReviewIn):
    try:
        entry = store().review(instance_id, body.decision, body.reviewer.strip(), body.reason)
    except KeyError:
        raise HTTPException(404, "no AI result for this instance")
    return {"ok": True, "audit": entry, "result": store().get_result(instance_id)}


@app.get("/api/audit")
def audit(instance_id: str | None = None, limit: int = 500):
    return store().audit_log(instance_id, min(limit, 5000))


@app.get("/api/audit/verify")
def verify():
    return store().verify_chain()


@app.get("/api/stats")
def stats():
    rows = store().list_results()
    reviewed = [r for r in rows if r["status"] != "pending_review"]
    agree = sum(r["status"] == "approved" for r in reviewed)
    return {"total": len(rows), "pending": len(rows) - len(reviewed), "reviewed": len(reviewed),
            "radiologist_agreement": round(agree / len(reviewed), 3) if reviewed else None,
            "mean_latency_ms": round(sum(r["latency_ms"] or 0 for r in rows) / len(rows), 1) if rows else None}
