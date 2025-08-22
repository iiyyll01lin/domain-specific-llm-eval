import { create } from 'zustand'
import type { RunParsed, Thresholds } from '@/core/types'

export type Locale = 'zh-TW'|'en-US'

interface PortalState {
  locale: Locale
  setLocale: (l: Locale) => void
  run?: RunParsed
  setRunData: (run: RunParsed) => void
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
  run: undefined,
  setRunData: (run) => set({ run }),
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
