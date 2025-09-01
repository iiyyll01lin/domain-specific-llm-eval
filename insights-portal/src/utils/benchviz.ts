// Utilities for benchmark visualization and summaries.
// Comments in English only.

export type TimingSample = { at: number; filterMs: number; sampleMs: number; aggregateMs: number; total: number }
export type BenchResult = { size: number; samplePct: number | null; coalesceMs: number; filterMs: number; sampleMs: number; aggregateMs: number }

export type BoxStats = {
  n: number
  min: number
  q1: number
  median: number
  q3: number
  max: number
}

export function toTotals(bench: BenchResult[]): number[] {
  return bench.map((b) => (b.filterMs || 0) + (b.sampleMs || 0) + (b.aggregateMs || 0))
}

export function computeBox(numbers: number[]): BoxStats {
  if (!numbers.length) return { n: 0, min: 0, q1: 0, median: 0, q3: 0, max: 0 }
  const arr = numbers.slice().sort((a, b) => a - b)
  const n = arr.length
  const q = (p: number) => {
    const idx = (n - 1) * p
    const lo = Math.floor(idx)
    const hi = Math.ceil(idx)
    const h = idx - lo
    return (1 - h) * arr[lo] + h * arr[hi]
  }
  return {
    n,
    min: arr[0],
    q1: q(0.25),
    median: q(0.5),
    q3: q(0.75),
    max: arr[n - 1],
  }
}

export function groupBySize(bench: BenchResult[]): Record<number, BenchResult[]> {
  return bench.reduce<Record<number, BenchResult[]>>((acc, r) => {
    (acc[r.size] ||= []).push(r)
    return acc
  }, {})
}

export function topKByTotal(bench: BenchResult[], k = 5): BenchResult[] {
  return bench
    .slice()
    .sort((a, b) => (a.filterMs + a.sampleMs + a.aggregateMs) - (b.filterMs + b.sampleMs + b.aggregateMs))
    .slice(0, k)
}
