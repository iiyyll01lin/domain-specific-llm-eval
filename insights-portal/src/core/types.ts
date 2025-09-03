export type MetricKey =
  | 'ContextPrecision'
  | 'ContextRecall'
  | 'Faithfulness'
  | 'AnswerRelevancy'
  | 'AnswerSimilarity'
  | 'ContextualKeywordMean'
  | (string & Record<never, never>)

export interface EvaluationItem {
  id: string
  language?: string | null
  latencyMs?: number | null
  metrics: Partial<Record<MetricKey, number | null>>
  user_input?: string
  reference?: string
  rag_answer?: string
  reference_contexts?: string[]
  rag_contexts?: string[]
  extra?: Record<string, unknown>
}

export interface RunArtifacts {
  summaryJson?: File
  configYaml?: File
}

export interface RunParsed {
  /** Optional identifier for the run (e.g., file name or directory path) */
  id?: string
  items: EvaluationItem[]
  kpis: Partial<Record<MetricKey, number>>
  counts: { total: number }
  latencies?: { avg?: number; p50?: number; p90?: number; p99?: number }
  /** Optional artifacts used to build this run in the UI (for provenance/KPI info) */
  artifacts?: RunArtifacts
}

export interface ThresholdLevel {
  warning: number
  critical: number
}

export type Thresholds = Partial<Record<MetricKey, ThresholdLevel>>

export type Verdict = 'Ready' | 'At Risk' | 'Blocked'

export interface VerdictResult {
  verdict: Verdict
  triggeredRuleId?: string
  failingMetrics?: MetricKey[]
}

export interface ThresholdProfile {
  id: string
  name: string
  thresholds: Thresholds
  rules: Array<{
    id: string
    when: 'any_critical' | 'any_warning' | 'all_warning_or_above'
    verdict: Verdict
  }>
}
