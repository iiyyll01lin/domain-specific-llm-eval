/// <reference lib="webworker" />
import { RawItemSchema, normalizeItem, aggregateKpis, computeLatencyStats } from '@/core/schemas'
import type { EvaluationItem } from '@/core/types'
// PapaParse is used for chunked CSV parsing inside the worker
// Typings are optional; declare module shim is provided to avoid hard dependency on @types
import Papa from 'papaparse'

type InMsg =
  | { type: 'parse-summary-json'; file: File }
  | { type: 'parse-csv'; file: File }
  | { type: 'fast-scan'; file: File; sample?: number }

type OutMsg =
  | { type: 'progress'; phase: string; current: number; total: number }
  | { type: 'parsed'; items: EvaluationItem[]; kpis: Record<string, number>; total: number; latencies?: { avg?: number; p50?: number; p90?: number; p99?: number } }
  | { type: 'scan'; total: number; metricsCoverage?: Record<string, number> }
  | { type: 'error'; message: string }

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
    throw new Error(`JSON 解析失敗: ${e?.message ?? e}`)
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
      chunk: (results: any, parser: any) => {
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
      error: (err: any) => reject(err),
    })
  })
  return items
}

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
    }
  } catch (e: any) {
    postMessage({ type: 'error', message: e?.message ?? String(e) })
  }
}

export {}
