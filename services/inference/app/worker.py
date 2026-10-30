"""Orthanc watcher: follows the /changes feed, analyzes every new instance exactly once.

The feed cursor is persisted, so a restart resumes where it left off instead of
re-analyzing (or missing) studies. Failures are logged to the audit trail, never swallowed.
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import httpx

from .db import Store
from .engine import Engine

log = logging.getLogger("worker")


class OrthancWorker(threading.Thread):
    def __init__(self, engine: Engine, store: Store, orthanc_url: str, artifacts: Path,
                 auth: tuple[str, str] | None = None, poll_seconds: float = 2.0):
        super().__init__(daemon=True, name="orthanc-worker")
        self.engine, self.store, self.artifacts = engine, store, artifacts
        self.client = httpx.Client(base_url=orthanc_url, auth=auth, timeout=30)
        self.poll_seconds = poll_seconds
        self.stop_event = threading.Event()
        self.processed = 0
        self.last_error: str | None = None

    def run(self) -> None:
        since = int(self.store.get_state("orthanc_since", "0"))
        while not self.stop_event.is_set():
            try:
                feed = self.client.get("/changes", params={"since": since, "limit": 100}).json()
                for change in feed["Changes"]:
                    if change["ChangeType"] == "NewInstance":
                        self.process(change["ID"])
                since = feed["Last"]
                self.store.set_state("orthanc_since", str(since))
                self.last_error = None
                if feed["Done"]:
                    self.stop_event.wait(self.poll_seconds)
            except httpx.HTTPError as e:
                self.last_error = f"orthanc unreachable: {e}"
                log.warning(self.last_error)
                self.stop_event.wait(5)

    def process(self, instance_id: str) -> None:
        if self.store.has_result(instance_id):
            return
        try:
            dicom = self.client.get(f"/instances/{instance_id}/file").raise_for_status().content
            out = self.engine.analyze(dicom, instance_id, self.artifacts)
            if out is None:  # our own Secondary Capture
                return
            self.store.add_result(out["record"])
            self.client.post("/instances", content=out["secondary_capture"],
                             headers={"Content-Type": "application/dicom"}).raise_for_status()
            self.processed += 1
            log.info("analyzed %s p=%.3f", instance_id, out["record"]["probability"])
        except Exception as e:  # one bad file must not stop the queue
            log.exception("failed on %s", instance_id)
            with self.store.lock:
                self.store.conn.execute("BEGIN IMMEDIATE")
                self.store._append_audit("ai-service", "inference_error", instance_id, {"error": repr(e)[:500]})
                self.store.conn.execute("COMMIT")

    def stop(self) -> None:
        self.stop_event.set()
