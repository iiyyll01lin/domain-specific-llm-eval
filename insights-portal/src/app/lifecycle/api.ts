import { getLifecycleConfig } from './config'
import type { DocumentRow, EvalRun, KgJobItem, KmSummary, ProcessingJob, ReportItem, SubgraphResult, TestsetJob } from './types'

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

export async function fetchTestsetJobs(options: FetchOptions): Promise<TestsetJob[]> {
  const { testsetBaseUrl, requestTimeoutMs } = getLifecycleConfig()
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(() => controller.abort(), requestTimeoutMs)
  try {
    const res = await fetch(`${testsetBaseUrl}/testset-jobs`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (!res.ok) throw new Error(`Failed to load testset jobs: ${res.status}`)
    const payload = await res.json().catch(() => ([]))
    const items = Array.isArray(payload) ? payload : Array.isArray((payload as any)?.items) ? (payload as any).items : []
    return items.map(normalizeTestsetJob)
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

function normalizeTestsetJob(raw: any): TestsetJob {
  return {
    job_id: String(raw?.job_id ?? raw?.id ?? ''),
    status: String(raw?.status ?? ''),
    method: String(raw?.method ?? raw?.config?.method ?? ''),
    config_hash: String(raw?.config_hash ?? ''),
    sample_count: parseOptionalNumber(raw?.sample_count),
    persona_count: parseOptionalNumber(raw?.persona_count),
    scenario_count: parseOptionalNumber(raw?.scenario_count),
    seed: parseOptionalNumber(raw?.seed),
    created_at: typeof raw?.created_at === 'string' ? raw.created_at : undefined,
    updated_at: typeof raw?.updated_at === 'string' ? raw.updated_at : undefined,
    duplicate: typeof raw?.duplicate === 'boolean' ? raw.duplicate : undefined,
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

function parseOptionalNumber(value: unknown): number | undefined {
  if (typeof value === 'number' && Number.isFinite(value)) return value
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseFloat(value)
    if (Number.isFinite(parsed)) return parsed
  }
  return undefined
}

// ---------------------------------------------------------------------------
// Evaluations
// ---------------------------------------------------------------------------

export async function fetchEvalRuns(options: FetchOptions): Promise<EvalRun[]> {
  const { evalBaseUrl, requestTimeoutMs } = getLifecycleConfig()
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(() => controller.abort(), requestTimeoutMs)
  try {
    const res = await fetch(`${evalBaseUrl}/eval-runs`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (!res.ok) throw new Error(`Failed to load eval runs: ${res.status}`)
    const payload = await res.json().catch(() => ([]))
    const items = Array.isArray(payload) ? payload : Array.isArray((payload as any)?.items) ? (payload as any).items : []
    return items.map(normalizeEvalRun)
  } finally {
    ;(typeof window === 'undefined' ? clearTimeout : window.clearTimeout)(timer)
  }
}

// ---------------------------------------------------------------------------
// Reports
// ---------------------------------------------------------------------------

export async function fetchReports(options: FetchOptions): Promise<ReportItem[]> {
  const { reportingBaseUrl, requestTimeoutMs } = getLifecycleConfig()
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(() => controller.abort(), requestTimeoutMs)
  try {
    const res = await fetch(`${reportingBaseUrl}/reports`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (!res.ok) throw new Error(`Failed to load reports: ${res.status}`)
    const payload = await res.json().catch(() => ([]))
    const items = Array.isArray(payload) ? payload : Array.isArray((payload as any)?.items) ? (payload as any).items : []
    return items.map(normalizeReportItem)
  } finally {
    ;(typeof window === 'undefined' ? clearTimeout : window.clearTimeout)(timer)
  }
}

// ---------------------------------------------------------------------------
// KM Summaries (adapter service)
// ---------------------------------------------------------------------------

export async function fetchKmSummaries(options: FetchOptions): Promise<KmSummary[]> {
  const { adapterBaseUrl, requestTimeoutMs } = getLifecycleConfig()
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(() => controller.abort(), requestTimeoutMs)
  try {
    const res = await fetch(`${adapterBaseUrl}/km-summaries`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (!res.ok) throw new Error(`Failed to load KM summaries: ${res.status}`)
    const payload = await res.json().catch(() => ([]))
    const items = Array.isArray(payload) ? payload : Array.isArray((payload as any)?.items) ? (payload as any).items : []
    return items.map(normalizeKmSummary)
  } finally {
    ;(typeof window === 'undefined' ? clearTimeout : window.clearTimeout)(timer)
  }
}

// ---------------------------------------------------------------------------
// Normalizers for new types
// ---------------------------------------------------------------------------

function normalizeEvalRun(raw: any): EvalRun {
  return {
    run_id: String(raw?.run_id ?? raw?.id ?? ''),
    testset_id: String(raw?.testset_id ?? ''),
    status: String(raw?.status ?? ''),
    evaluation_item_count: parseOptionalNumber(raw?.evaluation_item_count),
    metrics_version: typeof raw?.metrics_version === 'string' ? raw.metrics_version : undefined,
    created_at: typeof raw?.created_at === 'string' ? raw.created_at : undefined,
    completed_at: typeof raw?.completed_at === 'string' ? raw.completed_at : undefined,
    error_code: typeof raw?.error_code === 'string' ? raw.error_code : undefined,
    error_message: typeof raw?.error_message === 'string' ? raw.error_message : undefined,
  }
}

function normalizeReportItem(raw: any): ReportItem {
  return {
    run_id: String(raw?.run_id ?? ''),
    template: String(raw?.template ?? ''),
    html_available: Boolean(raw?.html_available ?? raw?.html_path),
    pdf_available: Boolean(raw?.pdf_available ?? raw?.pdf_path),
    html_path: typeof raw?.html_path === 'string' ? raw.html_path : undefined,
    pdf_path: typeof raw?.pdf_path === 'string' ? raw.pdf_path : undefined,
    created_at: typeof raw?.created_at === 'string' ? raw.created_at : undefined,
  }
}

function normalizeKmSummary(raw: any): KmSummary {
  return {
    schema: String(raw?.schema ?? raw?.schema_version ?? ''),
    testset_id: typeof raw?.testset_id === 'string' ? raw.testset_id : undefined,
    kg_id: typeof raw?.kg_id === 'string' ? raw.kg_id : undefined,
    sample_count: parseOptionalNumber(raw?.sample_count),
    node_count: parseOptionalNumber(raw?.node_count),
    relationship_count: parseOptionalNumber(raw?.relationship_count),
    created_at: typeof raw?.created_at === 'string' ? raw.created_at : undefined,
  }
}

// ---------------------------------------------------------------------------
// KG Jobs (kg-service)
// ---------------------------------------------------------------------------

export async function fetchKgJobs(options: FetchOptions): Promise<KgJobItem[]> {
  const { kgBaseUrl, requestTimeoutMs } = getLifecycleConfig()
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(() => controller.abort(), requestTimeoutMs)
  try {
    const res = await fetch(`${kgBaseUrl}/kg-jobs`, {
      method: 'GET',
      headers: { Accept: 'application/json' },
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (!res.ok) throw new Error(`Failed to load KG jobs: ${res.status}`)
    const payload = await res.json().catch(() => ([]))
    const items = Array.isArray(payload) ? payload : Array.isArray((payload as any)?.items) ? (payload as any).items : []
    return items.map(normalizeKgJobItem)
  } finally {
    ;(typeof window === 'undefined' ? clearTimeout : window.clearTimeout)(timer)
  }
}

function normalizeKgJobItem(raw: any): KgJobItem {
  return {
    kg_id: String(raw?.kg_id ?? ''),
    status: String(raw?.status ?? ''),
    doc_count: parseOptionalNumber(raw?.doc_count),
    node_count: parseOptionalNumber(raw?.node_count),
    edge_count: parseOptionalNumber(raw?.edge_count),
    created_at: typeof raw?.created_at === 'string' ? raw.created_at : undefined,
    updated_at: typeof raw?.updated_at === 'string' ? raw.updated_at : undefined,
    error_message: typeof raw?.error_message === 'string' ? raw.error_message : undefined,
  }
}

export interface FetchSubgraphOptions {
  signal: AbortSignal
  kgId: string
  seedNode: string
  depth?: number
  maxNodes?: number
}

export async function fetchSubgraph(options: FetchSubgraphOptions): Promise<SubgraphResult> {
  const { kgBaseUrl, requestTimeoutMs } = getLifecycleConfig()
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(
    () => controller.abort(),
    requestTimeoutMs,
  )
  try {
    const res = await fetch(`${kgBaseUrl}/kg-jobs/${encodeURIComponent(options.kgId)}/subgraph`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        seed_node: options.seedNode,
        depth: options.depth ?? 2,
        max_nodes: options.maxNodes ?? 50,
      }),
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (res.status === 404) throw new Error(`KG job not found: ${options.kgId}`)
    if (!res.ok) throw new Error(`Subgraph request failed: ${res.status}`)
    return await res.json()
  } finally {
    ;(typeof window === 'undefined' ? clearTimeout : window.clearTimeout)(timer)
  }
}

// ---------------------------------------------------------------------------
// Auto-Insights: LLM-powered executive summary
// ---------------------------------------------------------------------------

export interface AiInsightsPayload {
  run_id?: string
  kpis: Record<string, number>
  counts?: Record<string, unknown>
  verdict?: string
  failing_metrics?: string[]
  thresholds?: Record<string, { warning: number; critical: number }>
  model?: string
}

export interface AiInsightsResult {
  run_id?: string
  summary: string
  model_used: string
  prompt_tokens?: number
  completion_tokens?: number
}

/** Error code surfaced when the backend has no LLM API key configured. */
export const AI_INSIGHTS_NO_KEY_CODE = 'NO_API_KEY' as const

export class AiInsightsError extends Error {
  constructor(
    message: string,
    public readonly code: typeof AI_INSIGHTS_NO_KEY_CODE | 'API_ERROR' | 'NETWORK_ERROR',
    public readonly statusCode?: number,
  ) {
    super(message)
    this.name = 'AiInsightsError'
  }
}

export async function generateAiInsights(
  payload: AiInsightsPayload,
  options: { signal: AbortSignal },
): Promise<AiInsightsResult> {
  const { reportingBaseUrl } = getLifecycleConfig()
  // LLM calls can take 20–60 s; use a generous timeout
  const LLM_TIMEOUT_MS = 90_000
  const controller = new AbortController()
  const timer = (typeof window === 'undefined' ? setTimeout : window.setTimeout)(
    () => controller.abort(),
    LLM_TIMEOUT_MS,
  )
  try {
    const res = await fetch(`${reportingBaseUrl}/api/v1/insights/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify(payload),
      signal: mergeSignals(options.signal, controller.signal),
    })
    if (res.status === 503) {
      const body = await res.json().catch(() => ({}))
      throw new AiInsightsError(
        (body as any).detail || 'LLM API key not configured.',
        AI_INSIGHTS_NO_KEY_CODE,
        503,
      )
    }
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new AiInsightsError(
        (body as any).detail || `Insights request failed: ${res.status}`,
        'API_ERROR',
        res.status,
      )
    }
    return (await res.json()) as AiInsightsResult
  } catch (err) {
    if (err instanceof AiInsightsError) throw err
    throw new AiInsightsError(
      err instanceof Error ? err.message : 'Network error calling insights service.',
      'NETWORK_ERROR',
    )
  } finally {
    ;(typeof window === 'undefined' ? clearTimeout : window.clearTimeout)(timer)
  }
}
