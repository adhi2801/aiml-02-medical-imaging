"""SQLite storage: AI results (mutable review status) + an append-only, hash-chained audit log.

Tamper evidence: every audit row stores sha256(previous row hash + this row's canonical JSON).
Editing or deleting any past row breaks every hash after it, and `verify_chain()` reports
the first broken row. SQLite triggers additionally reject UPDATE/DELETE on the audit table.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

GENESIS = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    instance_id   TEXT PRIMARY KEY,           -- Orthanc instance id
    study_uid     TEXT, sop_uid TEXT, patient_id TEXT,
    sex TEXT, age INTEGER, view TEXT,
    probability   REAL NOT NULL,              -- temperature-calibrated
    threshold     REAL NOT NULL,
    ai_label      TEXT NOT NULL,              -- 'opacity' | 'no_opacity'
    model_version TEXT NOT NULL,
    latency_ms    REAL,
    created_at    TEXT NOT NULL,
    status        TEXT NOT NULL DEFAULT 'pending_review',  -- pending_review | approved | rejected
    reviewed_by   TEXT, review_reason TEXT, reviewed_at TEXT
);
CREATE TABLE IF NOT EXISTS audit (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, actor TEXT NOT NULL, event TEXT NOT NULL,
    instance_id TEXT, payload TEXT NOT NULL,
    prev_hash TEXT NOT NULL, hash TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS audit_no_update BEFORE UPDATE ON audit
    BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TRIGGER IF NOT EXISTS audit_no_delete BEFORE DELETE ON audit
    BEGIN SELECT RAISE(ABORT, 'audit log is append-only'); END;
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _row_hash(prev_hash: str, ts: str, actor: str, event: str, instance_id: str | None, payload: str) -> str:
    canonical = json.dumps([prev_hash, ts, actor, event, instance_id, payload], separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    # ---- audit ----
    def _append_audit(self, actor: str, event: str, instance_id: str | None, payload: dict) -> dict:
        """Caller must hold self.lock and be inside a transaction."""
        prev = self.conn.execute("SELECT hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
        prev_hash = prev["hash"] if prev else GENESIS
        ts, body = _now(), json.dumps(payload, sort_keys=True, separators=(",", ":"))
        h = _row_hash(prev_hash, ts, actor, event, instance_id, body)
        self.conn.execute("INSERT INTO audit (ts, actor, event, instance_id, payload, prev_hash, hash) "
                          "VALUES (?,?,?,?,?,?,?)", (ts, actor, event, instance_id, body, prev_hash, h))
        return {"ts": ts, "hash": h}

    def audit_log(self, instance_id: str | None = None, limit: int = 500) -> list[dict]:
        q, args = "SELECT * FROM audit", ()
        if instance_id:
            q, args = q + " WHERE instance_id = ?", (instance_id,)
        rows = self.conn.execute(q + " ORDER BY id DESC LIMIT ?", (*args, limit)).fetchall()
        return [dict(r) | {"payload": json.loads(r["payload"])} for r in rows]

    def verify_chain(self) -> dict:
        prev, n = GENESIS, 0
        for r in self.conn.execute("SELECT * FROM audit ORDER BY id"):
            n += 1
            expected = _row_hash(prev, r["ts"], r["actor"], r["event"], r["instance_id"], r["payload"])
            if r["prev_hash"] != prev or r["hash"] != expected:
                return {"valid": False, "entries": n, "first_bad_id": r["id"]}
            prev = r["hash"]
        return {"valid": True, "entries": n, "head": prev}

    # ---- results ----
    def add_result(self, rec: dict) -> None:
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                cols = ",".join(rec)
                self.conn.execute(f"INSERT OR REPLACE INTO results ({cols}, created_at) VALUES "
                                  f"({','.join('?' * len(rec))}, ?)", (*rec.values(), _now()))
                self._append_audit("ai-service", "inference", rec["instance_id"],
                                   {k: rec[k] for k in ("probability", "threshold", "ai_label", "model_version")})
                self.conn.execute("COMMIT")
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def review(self, instance_id: str, decision: str, reviewer: str, reason: str | None) -> dict:
        status = {"approve": "approved", "reject": "rejected"}[decision]
        with self.lock:
            self.conn.execute("BEGIN IMMEDIATE")
            try:
                cur = self.conn.execute("SELECT status, probability, ai_label FROM results WHERE instance_id = ?",
                                        (instance_id,)).fetchone()
                if cur is None:
                    raise KeyError(instance_id)
                ts = _now()
                self.conn.execute("UPDATE results SET status=?, reviewed_by=?, review_reason=?, reviewed_at=? "
                                  "WHERE instance_id=?", (status, reviewer, reason, ts, instance_id))
                entry = self._append_audit(reviewer, f"review_{decision}", instance_id, {
                    "decision": decision, "reason": reason, "previous_status": cur["status"],
                    "ai_label": cur["ai_label"], "ai_probability": cur["probability"]})
                self.conn.execute("COMMIT")
                return entry
            except Exception:
                self.conn.execute("ROLLBACK")
                raise

    def get_result(self, instance_id: str) -> dict | None:
        r = self.conn.execute("SELECT * FROM results WHERE instance_id = ?", (instance_id,)).fetchone()
        return dict(r) if r else None

    def list_results(self, status: str | None = None) -> list[dict]:
        q, args = "SELECT * FROM results", ()
        if status:
            q, args = q + " WHERE status = ?", (status,)
        return [dict(r) for r in self.conn.execute(q + " ORDER BY created_at DESC", args)]

    def has_result(self, instance_id: str) -> bool:
        return self.conn.execute("SELECT 1 FROM results WHERE instance_id = ?", (instance_id,)).fetchone() is not None

    # ---- worker cursor ----
    def get_state(self, key: str, default: str) -> str:
        r = self.conn.execute("SELECT value FROM state WHERE key = ?", (key,)).fetchone()
        return r["value"] if r else default

    def set_state(self, key: str, value: str) -> None:
        self.conn.execute("INSERT OR REPLACE INTO state (key, value) VALUES (?, ?)", (key, value))
