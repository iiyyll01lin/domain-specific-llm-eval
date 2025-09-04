import { create } from 'zustand'
import type { RunParsed, Thresholds } from '@/core/types'

export interface FiltersState {
  language?: string | null
  latencyRange?: [number | null, number | null]
  metricRanges?: Record<string, [number | null, number | null]>
}

export type Locale = 'zh-TW'|'en-US'
export type ThemeMode = 'dark'|'light'

interface PortalState {
  locale: Locale
  setLocale: (l: Locale) => void
  personaId?: string
  setPersonaId: (id: string | undefined) => void
  theme: ThemeMode
  setTheme: (m: ThemeMode) => void
  toggleTheme: () => void
  run?: RunParsed
  runs?: Record<string, RunParsed>
  selectedRuns?: string[]
  setRuns: (runs: Record<string, RunParsed>) => void
  setSelectedRuns: (ids: string[]) => void
  setRunData: (run: RunParsed) => void
  /** Executive Overview panel expanded map scoped per run: map<runId, Record<panelId, expanded>> */
  overviewPanels: Record<string, Record<string, boolean>>
  setPanelExpanded: (runId: string, panelId: string, expanded: boolean) => void
  filters: FiltersState
  setFilters: (f: Partial<FiltersState>) => void
  clearFilters: () => void
  clearMetricRange: (key: string) => void
  thresholds: Thresholds
  setThresholds: (t: Thresholds) => void
  defaultThresholds?: Thresholds
  setDefaultThresholds: (t: Thresholds) => void
}

export const usePortalStore = create<PortalState>((set) => ({
  locale: (localStorage.getItem('portal.lang') as Locale) || 'zh-TW',
  setLocale: (l) => {
    localStorage.setItem('portal.lang', l)
    set({ locale: l })
  },
  personaId: undefined,
  setPersonaId: (id) => {
    if (id) localStorage.setItem('portal.persona', id)
    else localStorage.removeItem('portal.persona')
    set({ personaId: id })
  },
  theme: (localStorage.getItem('portal.theme') as ThemeMode) || 'dark',
  setTheme: (m) => {
    localStorage.setItem('portal.theme', m)
    // Update document data-theme here so non-React parts also reflect immediately
    document.documentElement.setAttribute('data-theme', m)
    set({ theme: m })
  },
  toggleTheme: () => {
    const current = (localStorage.getItem('portal.theme') as ThemeMode) || 'dark'
    const next = current === 'dark' ? 'light' : 'dark'
    localStorage.setItem('portal.theme', next)
    document.documentElement.setAttribute('data-theme', next)
    set({ theme: next })
  },
  run: undefined,
  runs: {},
  selectedRuns: [],
  setRuns: (runs) => set({ runs }),
  setSelectedRuns: (ids) => set({ selectedRuns: ids }),
  setRunData: (run) => set({ run }),
  overviewPanels: (() => {
    try {
      const raw = localStorage.getItem('portal.overviewPanels')
      return raw ? (JSON.parse(raw) as Record<string, Record<string, boolean>>) : {}
    } catch {
      return {}
    }
  })(),
  setPanelExpanded: (runId, panelId, expanded) => set((s) => {
    const map = { ...(s.overviewPanels || {}) }
    const rid = runId || 'default'
    map[rid] = { ...(map[rid] || {}), [panelId]: expanded }
    try { localStorage.setItem('portal.overviewPanels', JSON.stringify(map)) } catch {
      // ignore persistence error
    }
    return { overviewPanels: map }
  }),
  filters: { language: null, latencyRange: [null, null], metricRanges: {} },
  setFilters: (f) => set((s) => ({ filters: { ...s.filters, ...f, metricRanges: { ...(s.filters.metricRanges||{}), ...(f.metricRanges||{}) } } })),
  clearFilters: () => set({ filters: { language: null, latencyRange: [null, null], metricRanges: {} } }),
  clearMetricRange: (key: string) => set((s) => {
    const mr = { ...(s.filters.metricRanges || {}) }
    delete mr[key]
    return { filters: { ...s.filters, metricRanges: mr } }
  }),
  thresholds: {
    ContextPrecision: { warning: 0.9, critical: 0.8 },
    ContextRecall: { warning: 0.9, critical: 0.8 },
    Faithfulness: { warning: 0.35, critical: 0.3 },
    AnswerRelevancy: { warning: 0.7, critical: 0.6 },
  },
  setThresholds: (t) => set({ thresholds: t }),
  defaultThresholds: undefined,
  setDefaultThresholds: (t) => set({ defaultThresholds: t }),
}))
