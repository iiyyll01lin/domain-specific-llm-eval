/// <reference lib="webworker" />
import { RawItemSchema, normalizeItem, aggregateKpis, computeLatencyStats } from '@/core/schemas'
import type { EvaluationItem } from '@/core/types'
import { applyFilters, aggregateKpisFiltered } from '@/core/analysis/filters'
// PapaParse is used for chunked CSV parsing inside the worker
// Typings are optional; declare module shim is provided to avoid hard dependency on @types
import Papa from 'papaparse'
import { sampleItems } from '@/core/analysis/sampling'

type InMsg =
  | { type: 'parse-summary-json'; file: File }
  | { type: 'parse-csv'; file: File }
  | { type: 'fast-scan'; file: File; sample?: number }
    | { type: 'aggregate'; items: EvaluationItem[]; filters: any; sample?: { count?: number; pct?: number; method?: 'first' | 'random' } }
  | { type: 'config'; coalesceMs?: number }

// Note: union type for outgoing messages is intentionally omitted to avoid unused type warnings in strict lint settings.

function extractArrayFromJson(data: any): any[] {
  // 1) Direct array
  if (Array.isArray(data)) return data
  if (!data || typeof data !== 'object') return []
  // 2) Common container keys
  const candidates = ['items', 'results', 'evaluations', 'records', 'rows', 'data']
  for (const k of candidates) {
    const v = (data as any)[k]
    if (Array.isArray(v)) return v
    // nested container: e.g., { data: { items: [...] } }
    if (v && typeof v === 'object') {
      for (const kk of candidates) {
        const vv = (v as any)[kk]
        if (Array.isArray(vv)) return vv
      }
    }
  }
  // 3) First array value in object
  for (const v of Object.values(data)) {
    if (Array.isArray(v)) return v
  }
  return []
}

async function parseFile(file: File): Promise<EvaluationItem[]> {
  const text = await file.text()
  let data: unknown
  try {
    data = JSON.parse(text)
  } catch (e: any) {
  const msg = String(e?.message ?? e)
  // Best-effort offset/position extraction from error message if available
  // e.g., Unexpected token } in JSON at position 1234
  const m = /position\s+(\d+)/i.exec(msg)
  const pos = m ? Number(m[1]) : undefined
  throw new Error(`JSON 解析失敗: ${file.name}${pos != null ? ` @ offset ${pos}` : ''}: ${msg}`)
  }
  const arr = extractArrayFromJson(data)
  const items: EvaluationItem[] = []
  let i = 0
  for (const raw of arr) {
    try {
      const z = RawItemSchema.parse(raw)
      items.push(normalizeItem(z))
    } catch (e: any) {
      // skip invalid rows but continue
    }
    i++
    if (i % 200 === 0) postMessage({ type: 'progress', phase: 'parse', current: i, total: arr.length })
  }
  return items
}

async function parseCsvFile(file: File): Promise<EvaluationItem[]> {
  // Chunked CSV parsing using PapaParse
  const items: EvaluationItem[] = []
  let processed = 0
  await new Promise<void>((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      worker: false, // already in WebWorker
      skipEmptyLines: 'greedy',
      chunkSize: 1024 * 128,
      // Use explicit any annotations to satisfy strict settings without external type defs
  chunk: (results: any) => {
        const rows = results.data as any[]
        for (const raw of rows) {
          try {
            const z = RawItemSchema.parse(raw)
            items.push(normalizeItem(z))
          } catch {
            // ignore invalid row
          }
        }
        processed += rows.length
        postMessage({ type: 'progress', phase: 'parse-csv', current: processed, total: processed })
      },
      complete: () => resolve(),
      error: (err: any, fileArg?: any, inputElem?: any, reason?: any) => {
        try {
          const row = (err && (err.row ?? err.code ?? undefined)) as number | undefined
          // PapaParse puts cursor/bytes into meta when possible; not always exposed here
          const metaCursor = (err && (err.meta?.cursor ?? undefined)) as number | undefined
          const msg = String(err?.message ?? reason ?? err)
          reject(new Error(`CSV 解析失敗: ${file.name}${row != null ? ` @ row ${row}` : ''}${metaCursor != null ? `, offset ${metaCursor}` : ''}: ${msg}`))
        } catch (e) {
          reject(err)
        }
      },
    })
  })
  return items
}

// Simple coalescing for aggregate messages: keep last and process with a short micro-batch delay.
let pendingAggregate: { items: EvaluationItem[]; filters: any } | null = null
let aggregateTimer: number | undefined
let coalesceMs = 100 // default window in ms; can be updated via 'config' message

self.onmessage = async (ev: MessageEvent<InMsg>) => {
  const msg = ev.data
  try {
    if (msg.type === 'parse-summary-json') {
      postMessage({ type: 'progress', phase: 'read', current: 0, total: 1 })
      const items = await parseFile(msg.file)
      const kpis = aggregateKpis(items)
      const lat = computeLatencyStats(items)
      postMessage({ type: 'parsed', items, kpis, total: items.length, latencies: lat })
    } else if (msg.type === 'parse-csv') {
      postMessage({ type: 'progress', phase: 'read', current: 0, total: 1 })
      const items = await parseCsvFile(msg.file)
      const kpis = aggregateKpis(items)
      const lat = computeLatencyStats(items)
      postMessage({ type: 'parsed', items, kpis, total: items.length, latencies: lat })
    } else if (msg.type === 'fast-scan') {
      // Fast scan: only count length and detect metric keys from first N items
      const text = await msg.file.text()
      let data: any
      try {
        data = JSON.parse(text)
      } catch (e: any) {
        throw new Error(`JSON 解析失敗: ${e?.message ?? e}`)
      }
      const arr: any[] = Array.isArray(data) ? data : data?.items ?? []
      const sample = Math.min(msg.sample ?? 50, arr.length)
      const head = arr.slice(0, sample)
      const metricsCount: Record<string, number> = {}
      for (const raw of head) {
        const z: any = raw || {}
        for (const k of Object.keys(z)) {
          // naive detection: keys with numeric values between 0..1 are likely metrics
          const v = (z as any)[k]
          if (typeof v === 'number' && v >= 0 && v <= 1) metricsCount[k] = (metricsCount[k] || 0) + 1
        }
      }
      const coverage: Record<string, number> = {}
      for (const [k, n] of Object.entries(metricsCount)) {
        coverage[k] = sample ? n / sample : 0
      }
      postMessage({ type: 'scan', total: arr.length, metricsCoverage: coverage })
    } else if (msg.type === 'aggregate') {
      pendingAggregate = { items: msg.items, filters: msg.filters || {} }
      if (aggregateTimer) clearTimeout(aggregateTimer)
      // Coalesce rapid calls within the configured window
        aggregateTimer = setTimeout(() => {
        if (!pendingAggregate) return
          const { items, filters } = pendingAggregate
          const t0 = Date.now()
        pendingAggregate = null
          const BIG_N = 100_000
          const useChunked = (items?.length || 0) > BIG_N && !msg.sample
          if (!useChunked) {
            const filtered = applyFilters(items, filters)
            const t1 = Date.now()
            const sampled = msg.sample ? sampleItems(filtered, msg.sample) : filtered
            const t2 = Date.now()
            const kpis = aggregateKpisFiltered(sampled)
            const lat = computeLatencyStats(sampled)
            const t3 = Date.now()
            postMessage({ type: 'aggregated', kpis, total: filtered.length, latencies: lat, timings: { filterMs: t1 - t0, sampleMs: t2 - t1, aggregateMs: t3 - t2 } })
            return
          }
          // Chunked pipeline for very large datasets to avoid long single-run loops
          const CHUNK = 10_000
          const sums: Record<string, number> = {}
          const counts: Record<string, number> = {}
          const latencies: number[] = []
          let totalFiltered = 0
          let i = 0
          const t1Start = Date.now()
          const step = () => {
            const start = i
            const end = Math.min(items.length, i + CHUNK)
            for (let j = start; j < end; j++) {
              const it = items[j] as any
              // inline filter (language, latency, metric ranges)
              if (filters?.language) {
                const lang = (it.language || '').toLowerCase()
                if (lang !== String(filters.language).toLowerCase()) continue
              }
              if (filters?.latencyRange) {
                const [lo, hi] = filters.latencyRange
                const v = it.latencyMs
                if (lo != null && (v == null || v < lo)) continue
                if (hi != null && (v == null || v > hi)) continue
              }
              if (filters?.metricRanges) {
                let ok = true
                for (const k in filters.metricRanges) {
                  const r = filters.metricRanges[k]
                  const lo = r?.[0]
                  const hi = r?.[1]
                  const v = (it.metrics || {})[k]
                  if (lo != null && (v == null || v < lo)) { ok = false; break }
                  if (hi != null && (v == null || v > hi)) { ok = false; break }
                }
                if (!ok) continue
              }
              totalFiltered++
              if (typeof it.latencyMs === 'number') latencies.push(it.latencyMs)
              const m = it.metrics || {}
              for (const k in m) {
                const v = m[k]
                if (v == null || Number.isNaN(v)) continue
                sums[k] = (sums[k] || 0) + v
                counts[k] = (counts[k] || 0) + 1
              }
            }
            i = end
            if (i < items.length) {
              // Yield back to event loop in worker to remain responsive
              setTimeout(step, 0)
            } else {
              // No sampling in chunked path (already optimized); compute kpis
              const kpis: Record<string, number> = {}
              for (const k in sums) kpis[k] = counts[k] ? sums[k] / counts[k] : NaN
              // Compute latency stats from collected latencies
              const latSorted = latencies.slice().sort((a, b) => a - b)
              const pick = (p: number) => latSorted.length ? latSorted[Math.max(0, Math.min(latSorted.length - 1, Math.floor(p * (latSorted.length - 1))))] : null
              const lat = {
                avg: latSorted.length ? latSorted.reduce((s, x) => s + x, 0) / latSorted.length : null,
                p50: pick(0.5), p90: pick(0.9), p99: pick(0.99)
              }
              const t3 = Date.now()
              postMessage({ type: 'aggregated', kpis, total: totalFiltered, latencies: lat, timings: { filterMs: t1Start - t0, sampleMs: 0, aggregateMs: t3 - t1Start } })
            }
          }
          step()
      }, coalesceMs) as unknown as number
    } else if (msg.type === 'config') {
      if (typeof msg.coalesceMs === 'number' && Number.isFinite(msg.coalesceMs) && msg.coalesceMs >= 0) {
        coalesceMs = msg.coalesceMs
      }
    }
  } catch (e: any) {
    postMessage({ type: 'error', message: e?.message ?? String(e) })
  }
}

export {}
