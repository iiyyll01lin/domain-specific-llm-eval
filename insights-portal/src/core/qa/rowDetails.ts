// Utilities for QA row details interactions.
// All comments in English.

export type RowDetailsLoader<T> = (id: string) => Promise<T>

export async function measureRowDetailsLatency<T>(loader: RowDetailsLoader<T>, id: string): Promise<{ durationMs: number; data: T }>{
  const t0 = performance.now()
  const data = await loader(id)
  const t1 = performance.now()
  return { durationMs: t1 - t0, data }
}

// A trivial default loader that simulates immediate availability to allow SLA tests.
export async function immediateRowDetails<T>(data: T): Promise<T> {
  return data
}
