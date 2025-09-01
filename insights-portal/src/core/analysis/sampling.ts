// Simple sampling utilities for large datasets. All comments must be in English.

export type SampleOptions = {
  count?: number
  pct?: number // 0..1
  method?: 'first' | 'random'
}

export function sampleItems<T>(items: T[], opts?: SampleOptions): T[] {
  if (!opts) return items
  const method = opts.method || 'first'
  let n = typeof opts.count === 'number' ? Math.max(0, Math.floor(opts.count)) : -1
  if (n < 0 && typeof opts.pct === 'number') n = Math.floor(items.length * Math.min(1, Math.max(0, opts.pct)))
  if (n < 0) return items
  if (n >= items.length) return items
  if (method === 'first') return items.slice(0, n)
  // random sample without replacement using Fisher-Yates partial shuffle
  const arr = items.slice()
  for (let i = 0; i < n; i++) {
    const j = i + Math.floor(Math.random() * (arr.length - i))
    const tmp = arr[i]
    arr[i] = arr[j]
    arr[j] = tmp
  }
  return arr.slice(0, n)
}
