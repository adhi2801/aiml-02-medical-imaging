"""Audit trail + review API: the guarantees a hospital auditor would ask about."""
import sqlite3
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "inference"))

from app.db import Store  # noqa: E402

RESULT = {"instance_id": "inst-1", "study_uid": "1.2", "sop_uid": "1.2.3", "patient_id": "p1", "sex": "F", "age": 58,
          "view": "PA", "probability": 0.81, "threshold": 0.42, "ai_label": "opacity", "model_version": "densenet121-abcd",
          "latency_ms": 120.0}


@pytest.fixture
def store(tmp_path):
    s = Store(tmp_path / "t.sqlite")
    s.add_result(RESULT)
    return s


def test_review_is_logged_and_chain_valid(store):
    store.review("inst-1", "approve", "dr.rao", None)
    store.review("inst-1", "reject", "dr.iyer", "opacity is a nipple shadow")
    assert store.get_result("inst-1")["status"] == "rejected"
    events = [e["event"] for e in store.audit_log()]
    assert events == ["review_reject", "review_approve", "inference"]
    assert store.verify_chain() == {"valid": True, "entries": 3, "head": store.audit_log()[0]["hash"]}


def test_audit_rows_cannot_be_updated_or_deleted(store):
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("UPDATE audit SET actor = 'mallory'")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        store.conn.execute("DELETE FROM audit")


def test_tampering_is_detected(store):
    store.review("inst-1", "approve", "dr.rao", None)
    # An attacker with raw DB access drops the triggers and rewrites history...
    store.conn.execute("DROP TRIGGER audit_no_update")
    store.conn.execute("UPDATE audit SET payload = '{\"decision\":\"reject\"}' WHERE event = 'review_approve'")
    # ...and the hash chain exposes exactly which row was altered.
    v = store.verify_chain()
    assert v["valid"] is False and v["first_bad_id"] == 2


def test_api_requires_reason_for_reject(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("DISABLE_WORKER", "1")
    import importlib

    import app.main as main
    importlib.reload(main)
    from fastapi.testclient import TestClient

    with TestClient(main.app) as client:
        main.state["store"].add_result(RESULT)
        r = client.post("/api/results/inst-1/review", json={"decision": "reject", "reviewer": "dr.rao"})
        assert r.status_code == 422
        r = client.post("/api/results/inst-1/review", json={"decision": "approve", "reviewer": "dr.rao"})
        assert r.status_code == 200 and r.json()["result"]["status"] == "approved"
        assert client.get("/api/audit/verify").json()["valid"] is True
        assert client.post("/api/results/nope/review", json={"decision": "approve", "reviewer": "dr.rao"}).status_code == 404
        assert client.get("/api/stats").json()["radiologist_agreement"] == 1.0


def test_committed_openapi_spec_is_current():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from export_openapi import OUT, render
    assert OUT.read_text(encoding="utf-8") == render(), "API changed: run python scripts/export_openapi.py"
