export type Status = "pending_review" | "approved" | "rejected";

export interface Result {
  instance_id: string;
  study_uid: string;
  patient_id: string;
  sex: string;
  age: number | null;
  view: string;
  probability: number;
  threshold: number;
  ai_label: "opacity" | "no_opacity";
  model_version: string;
  latency_ms: number | null;
  created_at: string;
  status: Status;
  reviewed_by: string | null;
  review_reason: string | null;
  reviewed_at: string | null;
}

export interface AuditEntry {
  id: number;
  ts: string;
  actor: string;
  event: string;
  instance_id: string | null;
  payload: Record<string, unknown>;
  hash: string;
  prev_hash: string;
}

export interface Stats {
  total: number;
  pending: number;
  reviewed: number;
  radiologist_agreement: number | null;
  mean_latency_ms: number | null;
}

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const detail = Array.isArray(body.detail) ? body.detail.map((d: { msg: string }) => d.msg).join("; ") : body.detail;
    throw new Error(detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  results: (status?: Status) => fetch(`/api/results${status ? `?status=${status}` : ""}`).then(json<Result[]>),
  audit: (instanceId: string) => fetch(`/api/audit?instance_id=${encodeURIComponent(instanceId)}`).then(json<AuditEntry[]>),
  verify: () => fetch("/api/audit/verify").then(json<{ valid: boolean; entries: number; first_bad_id?: number }>),
  stats: () => fetch("/api/stats").then(json<Stats>),
  review: (instanceId: string, decision: "approve" | "reject", reviewer: string, reason: string) =>
    fetch(`/api/results/${encodeURIComponent(instanceId)}/review`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ decision, reviewer, reason: reason.trim() || null }),
    }).then(json<{ ok: boolean; result: Result }>),
  imageUrl: (id: string, kind: "image" | "cam") => `/api/results/${encodeURIComponent(id)}/${kind}.png`,
};
