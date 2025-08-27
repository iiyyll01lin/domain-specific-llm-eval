import { z } from 'zod'
import type { EvaluationItem, MetricKey } from './types'

// Flexible item schema to accommodate various input shapes
export const RawItemSchema = z
  .object({
    id: z.string().or(z.number()).optional().nullable(),
    language: z.string().optional().nullable(),
    latencyMs: z.number().optional().nullable(),
  })
  .passthrough()

export type RawItem = z.infer<typeof RawItemSchema>

const knownMetricKeys: MetricKey[] = [
  'ContextPrecision',
  'ContextRecall',
  'Faithfulness',
  'AnswerRelevancy',
  'AnswerSimilarity',
  'ContextualKeywordMean',
]

export function normalizeItem(input: RawItem): EvaluationItem {
  // Determine a robust id, falling back to common alternatives
  const idCandidate =
    (input as any).id ??
    (input as any).index ??
    (input as any).sample_id ??
    (input as any).row_id ??
    (input as any).question_id

  const metrics: EvaluationItem['metrics'] = {}
  for (const k of knownMetricKeys) {
    const v = (input as any)[k]
    if (v === undefined) continue
    metrics[k] = typeof v === 'number' ? v : Number(v)
  }
  // Common alternative keys mapping
  const altMap: Record<string, MetricKey> = {
    context_precision: 'ContextPrecision',
    context_recall: 'ContextRecall',
    faithfulness: 'Faithfulness',
    answer_relevancy: 'AnswerRelevancy',
    answer_similarity: 'AnswerSimilarity',
    contextual_keyword_mean: 'ContextualKeywordMean',
  }
  for (const [k, mk] of Object.entries(altMap)) {
    const v = (input as any)[k]
    if (v !== undefined && metrics[mk] === undefined) {
      metrics[mk] = typeof v === 'number' ? v : Number(v)
    }
  }

  // Normalize latency using common alternative field names
  const latencyCandidates = [
    (input as any).latencyMs,
    (input as any).latency_ms,
    (input as any).latency,
    (input as any).response_time_ms,
    (input as any).inference_latency_ms,
  ]
  let latency: number | null = null
  for (const v of latencyCandidates) {
    if (v == null) continue
    const n = typeof v === 'number' ? v : Number(v)
    if (!Number.isNaN(n)) {
      latency = n
      break
    }
  }

  return {
    id: String(idCandidate ?? ''),
    language: (input as any).language ?? null,
    latencyMs: latency,
    metrics,
    user_input: (input as any).user_input,
    reference: (input as any).reference,
    rag_answer: (input as any).rag_answer,
    reference_contexts: (input as any).reference_contexts,
    rag_contexts: (input as any).rag_contexts,
    extra: input,
  }
}

export function aggregateKpis(items: EvaluationItem[]) {
  const acc: Record<string, { sum: number; n: number }> = {}
  for (const it of items) {
    for (const [k, v] of Object.entries(it.metrics)) {
      if (v == null || isNaN(v)) continue
      const a = (acc[k] = acc[k] || { sum: 0, n: 0 })
      a.sum += v
      a.n += 1
    }
  }
  const out: Record<string, number> = {}
  for (const [k, a] of Object.entries(acc)) {
    out[k] = a.n ? a.sum / a.n : NaN
  }
  return out
}

// Compute basic latency statistics. Assumes latencyMs in milliseconds.
export function computeLatencyStats(items: EvaluationItem[]) {
  const vals: number[] = []
  let sum = 0
  for (const it of items) {
    const v = it.latencyMs
    if (v == null || Number.isNaN(v)) continue
    vals.push(v)
    sum += v
  }
  if (!vals.length) return { avg: undefined, p50: undefined, p90: undefined, p99: undefined }
  vals.sort((a, b) => a - b)
  // Nearest-rank method: rank = ceil(p/100 * N)
  const pct = (p: number) => {
    if (!vals.length) return undefined
    const rank = Math.ceil((p / 100) * vals.length)
    const idx = Math.min(vals.length - 1, Math.max(0, rank - 1))
    return vals[idx]
  }
  return {
    avg: sum / vals.length,
    p50: pct(50),
    p90: pct(90),
    p99: pct(99),
  }
}
