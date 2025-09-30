export interface DocumentRow {
  km_id: string
  version: string
  checksum: string
  status: string
  size?: number
  last_event_ts?: string
  document_id?: string
  source?: string
}

export interface ProcessingJob {
  job_id: string
  document_id: string
  status: string
  progress: number
  chunk_count?: number
  embedding_profile_hash?: string
  started_at?: string
  updated_at?: string
  sla_seconds?: number
  elapsed_seconds?: number
  error_code?: string
  error_message?: string
}

export interface LifecycleConfig {
  ingestionBaseUrl: string
  processingBaseUrl: string
  testsetBaseUrl: string
  pollIntervalMs: number
  requestTimeoutMs: number
}

export interface TestsetJob {
  job_id: string
  status: string
  method: string
  config_hash: string
  sample_count?: number
  persona_count?: number
  scenario_count?: number
  seed?: number
  created_at?: string
  updated_at?: string
  duplicate?: boolean
  error_code?: string
  error_message?: string
}
