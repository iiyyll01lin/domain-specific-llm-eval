import { getLifecycleConfig } from './config'
import type { DocumentRow, ProcessingJob } from './types'

interface FetchOptions {
  signal: AbortSignal
}

export async function fetchDocuments(options: FetchOptions): Promise<DocumentRow[]> {
  const { ingestionBaseUrl, requestTimeoutMs } = getLifecycleConfig()
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(() => controller.abort(), requestTimeoutMs)
  try {
    const res = await fetch(`${ingestionBaseUrl}/documents`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (!res.ok) throw new Error(`Failed to load documents: ${res.status}`)
    const payload = await res.json().catch(() => ([]))
    const items = Array.isArray(payload) ? payload : Array.isArray((payload as any)?.items) ? (payload as any).items : []
    return items.map(normalizeDocumentRow)
  } finally {
    ;(typeof window === 'undefined' ? clearTimeout : window.clearTimeout)(timer)
  }
}

export async function fetchProcessingJobs(options: FetchOptions): Promise<ProcessingJob[]> {
  const { processingBaseUrl, requestTimeoutMs } = getLifecycleConfig()
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(() => controller.abort(), requestTimeoutMs)
  try {
    const res = await fetch(`${processingBaseUrl}/process-jobs`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (!res.ok) throw new Error(`Failed to load process jobs: ${res.status}`)
    const payload = await res.json().catch(() => ([]))
    const items = Array.isArray(payload) ? payload : Array.isArray((payload as any)?.items) ? (payload as any).items : []
    return items.map(normalizeProcessingJob)
  } finally {
    ;(typeof window === 'undefined' ? clearTimeout : window.clearTimeout)(timer)
  }
}

function normalizeDocumentRow(raw: any): DocumentRow {
  return {
    km_id: String(raw?.km_id ?? ''),
    version: String(raw?.version ?? ''),
    checksum: String(raw?.checksum ?? ''),
    status: String(raw?.status ?? ''),
    size: typeof raw?.size === 'number' ? raw.size : Number(raw?.size_bytes ?? raw?.size) || undefined,
    last_event_ts: typeof raw?.last_event_ts === 'string' ? raw.last_event_ts : raw?.updated_at,
    document_id: typeof raw?.document_id === 'string' ? raw.document_id : undefined,
    source: typeof raw?.source === 'string' ? raw.source : undefined,
  }
}

function normalizeProcessingJob(raw: any): ProcessingJob {
  const progress = typeof raw?.progress === 'number'
    ? clamp(raw.progress, 0, 100)
    : typeof raw?.percentage === 'number'
      ? clamp(raw.percentage, 0, 100)
      : Number.parseFloat(raw?.progress ?? '0')
  const startedAt = typeof raw?.started_at === 'string' ? raw.started_at : raw?.created_at
  const updatedAt = typeof raw?.updated_at === 'string' ? raw.updated_at : raw?.last_event_ts
  return {
    job_id: String(raw?.job_id ?? raw?.id ?? ''),
    document_id: String(raw?.document_id ?? ''),
    status: String(raw?.status ?? ''),
    progress: Number.isFinite(progress) ? progress : 0,
    chunk_count: typeof raw?.chunk_count === 'number' ? raw.chunk_count : undefined,
    embedding_profile_hash: typeof raw?.embedding_profile_hash === 'string' ? raw.embedding_profile_hash : undefined,
    started_at: startedAt,
    updated_at: updatedAt,
    sla_seconds: typeof raw?.sla_seconds === 'number' ? raw.sla_seconds : undefined,
    elapsed_seconds: typeof raw?.elapsed_seconds === 'number' ? raw.elapsed_seconds : undefined,
    error_code: typeof raw?.error_code === 'string' ? raw.error_code : undefined,
    error_message: typeof raw?.error_message === 'string' ? raw.error_message : undefined,
  }
}

function clamp(value: number, min: number, max: number): number {
  if (!Number.isFinite(value)) return min
  return Math.min(Math.max(value, min), max)
}

function mergeSignals(signalA: AbortSignal, signalB: AbortSignal): AbortSignal {
  const controller = new AbortController()
  if (signalA.aborted || signalB.aborted) {
    controller.abort()
    return controller.signal
  }
  const abort = () => controller.abort()
  if (typeof signalA.addEventListener === 'function') signalA.addEventListener('abort', abort, { once: true })
  if (typeof signalB.addEventListener === 'function') signalB.addEventListener('abort', abort, { once: true })
  return controller.signal
}
