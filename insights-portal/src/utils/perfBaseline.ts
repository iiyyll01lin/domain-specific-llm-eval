// Persist and compare performance baselines. Comments in English only.

export type PerfSample = { name: string; value: number }

const KEY = 'portal.perf.baseline'

export function saveBaseline(samples: PerfSample[]) {
  localStorage.setItem(KEY, JSON.stringify({ at: Date.now(), samples }))
}

export function loadBaseline(): { at: number; samples: PerfSample[] } | null {
  try {
    const raw = localStorage.getItem(KEY)
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

export function compareWithBaseline(current: PerfSample[], tolerancePct = 0.1): { regressions: PerfSample[]; ok: boolean } {
  const base = loadBaseline()
  if (!base) return { regressions: [], ok: true }
  const byName = new Map(base.samples.map((s) => [s.name, s.value]))
  const regressions: PerfSample[] = []
  for (const s of current) {
    const b = byName.get(s.name)
    if (typeof b === 'number' && s.value > b * (1 + tolerancePct)) {
      regressions.push({ name: s.name, value: s.value })
    }
  }
  return { regressions, ok: regressions.length === 0 }
}
