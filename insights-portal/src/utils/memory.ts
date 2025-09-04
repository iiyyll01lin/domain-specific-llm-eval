// Simple heuristic memory pressure detection using performance.memory if available.
// In browsers without support, returns undefined.

export interface MemoryStatus {
  usedMB: number
  totalMB?: number
  pressure?: 'low'|'medium'|'high'
}

export function sampleMemory(): MemoryStatus | undefined {
  const anyPerf: any = performance as any
  if (!anyPerf || !anyPerf.memory) return undefined
  const { usedJSHeapSize, jsHeapSizeLimit } = anyPerf.memory
  const usedMB = usedJSHeapSize / (1024*1024)
  const totalMB = jsHeapSizeLimit / (1024*1024)
  const pct = usedJSHeapSize / jsHeapSizeLimit
  let pressure: MemoryStatus['pressure'] = 'low'
  if (pct > 0.85) pressure = 'high'
  else if (pct > 0.65) pressure = 'medium'
  return { usedMB, totalMB, pressure }
}