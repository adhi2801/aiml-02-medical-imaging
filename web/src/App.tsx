import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, type AuditEntry, type Result, type Stats, type Status } from "./api";
import { Viewer } from "./Viewer";

const STATUS_LABEL: Record<Status, string> = {
  pending_review: "Pending",
  approved: "Approved",
  rejected: "Rejected",
};

function loadReviewer(): string {
  try {
    return localStorage.getItem("reviewer") ?? "";
  } catch {
    return "";
  }
}

function saveReviewer(name: string) {
  try {
    localStorage.setItem("reviewer", name);
  } catch {
    /* private mode: keep in memory only */
  }
}

const fmtTime = (iso: string) => new Date(iso).toLocaleString(undefined, { dateStyle: "short", timeStyle: "medium" });

export default function App() {
  const [results, setResults] = useState<Result[]>([]);
  const [filter, setFilter] = useState<Status | "all">("pending_review");
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [audit, setAudit] = useState<AuditEntry[]>([]);
  const [stats, setStats] = useState<Stats | null>(null);
  const [chain, setChain] = useState<{ valid: boolean; entries: number } | null>(null);
  const [reviewer, setReviewer] = useState(loadReviewer);
  const [reason, setReason] = useState("");
  const [showOverlay, setShowOverlay] = useState(true);
  const [opacity, setOpacity] = useState(0.45);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const reasonRef = useRef<HTMLTextAreaElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [r, s, c] = await Promise.all([api.results(), api.stats(), api.verify()]);
      setResults(r);
      setStats(s);
      setChain(c);
      setOffline(false);
    } catch {
      setOffline(true);
    }
  }, []);

  useEffect(() => {
    refresh();
    const t = setInterval(refresh, 5000);
    return () => clearInterval(t);
  }, [refresh]);

  const visible = useMemo(
    () => (filter === "all" ? results : results.filter((r) => r.status === filter)),
    [results, filter],
  );
  const selected = results.find((r) => r.instance_id === selectedId) ?? null;

  useEffect(() => {
    if (!selectedId && visible.length) setSelectedId(visible[0].instance_id);
  }, [visible, selectedId]);

  useEffect(() => {
    setReason("");
    setError(null);
    if (selectedId) api.audit(selectedId).then(setAudit).catch(() => setAudit([]));
  }, [selectedId, selected?.status]);

  const move = useCallback(
    (step: number) => {
      if (!visible.length) return;
      const i = visible.findIndex((r) => r.instance_id === selectedId);
      const next = visible[Math.min(visible.length - 1, Math.max(0, (i < 0 ? 0 : i) + step))];
      setSelectedId(next.instance_id);
    },
    [visible, selectedId],
  );

  const decide = useCallback(
    async (decision: "approve" | "reject") => {
      if (!selected || busy) return;
      if (reviewer.trim().length < 2) {
        setError("Enter your name before signing a review.");
        return;
      }
      if (decision === "reject" && !reason.trim()) {
        setError("A reason is required to reject an AI finding.");
        reasonRef.current?.focus();
        return;
      }
      setBusy(true);
      setError(null);
      try {
        await api.review(selected.instance_id, decision, reviewer.trim(), reason);
        saveReviewer(reviewer.trim());
        const nextPending = results.find((r) => r.status === "pending_review" && r.instance_id !== selected.instance_id);
        await refresh();
        if (filter === "pending_review" && nextPending) setSelectedId(nextPending.instance_id);
      } catch (e) {
        setError((e as Error).message);
      } finally {
        setBusy(false);
      }
    },
    [selected, busy, reviewer, reason, results, refresh, filter],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).closest("input, textarea")) return;
      if (e.key === "o") setShowOverlay((s) => !s);
      else if (e.key === "j" || e.key === "ArrowDown") move(1);
      else if (e.key === "k" || e.key === "ArrowUp") move(-1);
      else if (e.key === "a") decide("approve");
      else if (e.key === "r") reasonRef.current?.focus();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [move, decide]);

  return (
    <div className="app">
      <div className="disclaimer" role="alert">
        EDUCATIONAL PROTOTYPE · NOT A MEDICAL DEVICE · NOT FOR CLINICAL USE · AI output must never replace a radiologist
      </div>

      <header className="topbar">
        <div className="brand">
          MedImg <span>Reader</span>
        </div>
        <div className="kpis">
          <Kpi label="Pending" value={stats?.pending ?? "–"} />
          <Kpi label="Reviewed" value={stats?.reviewed ?? "–"} />
          <Kpi
            label="Agreement"
            value={stats?.radiologist_agreement != null ? `${Math.round(stats.radiologist_agreement * 100)}%` : "–"}
          />
          <Kpi label="AI latency" value={stats?.mean_latency_ms != null ? `${Math.round(stats.mean_latency_ms)} ms` : "–"} />
          <div className={`chain ${chain?.valid === false ? "bad" : chain ? "ok" : ""}`} title="SHA-256 hash chain over every audit entry">
            {chain == null ? "Audit: –" : chain.valid ? `Audit chain verified · ${chain.entries}` : "AUDIT CHAIN BROKEN"}
          </div>
        </div>
      </header>

      {offline && <div className="banner">Cannot reach the inference API. Retrying every 5 s…</div>}

      <main className="layout">
        <aside className="worklist">
          <div className="tabs" role="tablist">
            {(["pending_review", "approved", "rejected", "all"] as const).map((s) => (
              <button
                key={s}
                role="tab"
                aria-selected={filter === s}
                className={filter === s ? "active" : ""}
                onClick={() => {
                  setFilter(s);
                  setSelectedId(null);
                }}
              >
                {s === "all" ? "All" : STATUS_LABEL[s]}
              </button>
            ))}
          </div>
          <ul>
            {visible.map((r) => (
              <li key={r.instance_id}>
                <button
                  className={`row ${r.instance_id === selectedId ? "selected" : ""}`}
                  onClick={() => setSelectedId(r.instance_id)}
                >
                  <span className={`dot ${r.ai_label}`} aria-hidden />
                  <span className="row-main">
                    <span className="mono pid">{r.patient_id.slice(0, 12) || r.instance_id.slice(0, 12)}</span>
                    <span className="meta">
                      {r.sex} · {r.age ?? "?"}y · {r.view}
                    </span>
                  </span>
                  <span className="mono prob">{Math.round(r.probability * 100)}%</span>
                  <span className={`chip ${r.status}`}>{STATUS_LABEL[r.status]}</span>
                </button>
              </li>
            ))}
            {!visible.length && <li className="empty">No studies here. Send DICOMs to Orthanc (AET MEDIMG, port 4242).</li>}
          </ul>
        </aside>

        <section className="stage">
          {selected ? (
            <>
              <Viewer
                imageUrl={api.imageUrl(selected.instance_id, "image")}
                camUrl={api.imageUrl(selected.instance_id, "cam")}
                showOverlay={showOverlay}
                opacity={opacity}
              />
              <div className="overlay-controls">
                <label className="toggle">
                  <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} />
                  Grad-CAM overlay <kbd>O</kbd>
                </label>
                <input
                  type="range"
                  min={0.1}
                  max={0.9}
                  step={0.05}
                  value={opacity}
                  onChange={(e) => setOpacity(+e.target.value)}
                  aria-label="Overlay opacity"
                  disabled={!showOverlay}
                />
                <span className="legend">
                  <span className="grad" /> low → high influence on the prediction
                </span>
              </div>
            </>
          ) : (
            <div className="placeholder">Select a study</div>
          )}
        </section>

        <aside className="panel">
          {selected && (
            <>
              <h2>AI finding</h2>
              <div className={`verdict ${selected.ai_label}`}>
                {selected.ai_label === "opacity" ? "Lung opacity suspected" : "No opacity detected"}
              </div>
              <ProbabilityBar p={selected.probability} threshold={selected.threshold} />
              <dl className="facts">
                <dt>Calibrated probability</dt>
                <dd className="mono">{(selected.probability * 100).toFixed(1)}%</dd>
                <dt>Decision threshold</dt>
                <dd className="mono">{(selected.threshold * 100).toFixed(1)}%</dd>
                <dt>Patient</dt>
                <dd>
                  {selected.sex} · {selected.age ?? "?"} y · {selected.view} view
                </dd>
                <dt>Model</dt>
                <dd className="mono">{selected.model_version}</dd>
                <dt>Inference</dt>
                <dd className="mono">{selected.latency_ms?.toFixed(0)} ms</dd>
              </dl>

              {selected.view === "AP" && (
                <p className="caution">
                  AP (portable) film. In training data these come from sicker patients, so the model may lean on
                  view-related cues. Check the heatmap is on lung tissue, not tubes, lines or markers.
                </p>
              )}

              <h2>Radiologist review</h2>
              {selected.status !== "pending_review" && (
                <p className={`signed ${selected.status}`}>
                  {STATUS_LABEL[selected.status]} by <b>{selected.reviewed_by}</b> · {fmtTime(selected.reviewed_at!)}
                  {selected.review_reason && <span className="reason-quote">“{selected.review_reason}”</span>}
                </p>
              )}
              <label className="field">
                Reviewer
                <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="Dr. name" />
              </label>
              <label className="field">
                Reason / comment <span className="muted">(required to reject)</span>
                <textarea
                  ref={reasonRef}
                  rows={3}
                  value={reason}
                  onChange={(e) => setReason(e.target.value)}
                  placeholder="e.g. heatmap focused on ECG lead, not lung"
                />
              </label>
              {error && <p className="error">{error}</p>}
              <div className="actions">
                <button className="approve" disabled={busy} onClick={() => decide("approve")}>
                  Approve <kbd>A</kbd>
                </button>
                <button className="reject" disabled={busy} onClick={() => decide("reject")}>
                  Reject
                </button>
              </div>

              <h2>Audit trail</h2>
              <ol className="audit">
                {audit.map((a) => (
                  <li key={a.id}>
                    <span className="mono ts">{fmtTime(a.ts)}</span>
                    <span>
                      <b>{a.actor}</b> {a.event.replace("_", " ")}
                      {typeof a.payload.reason === "string" && a.payload.reason && <> – “{a.payload.reason}”</>}
                    </span>
                    <span className="mono hash" title={a.hash}>
                      #{a.hash.slice(0, 10)}
                    </span>
                  </li>
                ))}
              </ol>
            </>
          )}
        </aside>
      </main>
    </div>
  );
}

function Kpi({ label, value }: { label: string; value: string | number }) {
  return (
    <div className="kpi">
      <div className="kpi-v mono">{value}</div>
      <div className="kpi-l">{label}</div>
    </div>
  );
}

function ProbabilityBar({ p, threshold }: { p: number; threshold: number }) {
  return (
    <div className="pbar" aria-label={`Probability ${Math.round(p * 100)} percent, threshold ${Math.round(threshold * 100)} percent`}>
      <div className="pbar-fill" style={{ width: `${p * 100}%` }} data-over={p >= threshold} />
      <div className="pbar-thr" style={{ left: `${threshold * 100}%` }} />
    </div>
  );
}
