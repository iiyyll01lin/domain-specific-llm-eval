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
  evalBaseUrl: string
  reportingBaseUrl: string
  adapterBaseUrl: string
  kgBaseUrl: string
  pollIntervalMs: number
  requestTimeoutMs: number
}

export interface KgJobItem {
  kg_id: string
  status: string
  doc_count?: number
  node_count?: number
  edge_count?: number
  created_at?: string
  updated_at?: string
  error_message?: string
}

export interface EvalRun {
  run_id: string
  testset_id: string
  status: string
  evaluation_item_count?: number
  metrics_version?: string
  created_at?: string
  completed_at?: string
  error_code?: string
  error_message?: string
}

export interface ReportItem {
  run_id: string
  template: string
  html_available: boolean
  pdf_available: boolean
  html_path?: string
  pdf_path?: string
  created_at?: string
}

export interface KmSummary {
  schema: string
  testset_id?: string
  kg_id?: string
  sample_count?: number
  node_count?: number
  relationship_count?: number
  created_at?: string
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

/** Node in a KG subgraph response. */
export interface SubgraphNode {
  id: string
  label: string
  type?: string
  truncated?: boolean
}

/** Edge in a KG subgraph response. */
export interface SubgraphEdge {
  source: string
  target: string
  relation?: string
}

/** Full subgraph payload returned by POST /kg-jobs/{kg_id}/subgraph. */
export interface SubgraphResult {
  kg_id: string
  seed_node: string
  depth: number
  nodes: SubgraphNode[]
  edges: SubgraphEdge[]
  sampled?: boolean
  total_nodes?: number
}
