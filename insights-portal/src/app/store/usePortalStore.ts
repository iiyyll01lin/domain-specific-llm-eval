import { create } from 'zustand'
import type { RunParsed, Thresholds } from '@/core/types'

export type Locale = 'zh-TW'|'en-US'
export type ThemeMode = 'dark'|'light'

interface PortalState {
  locale: Locale
  setLocale: (l: Locale) => void
  theme: ThemeMode
  setTheme: (m: ThemeMode) => void
  toggleTheme: () => void
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
