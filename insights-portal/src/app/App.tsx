import React from 'react'
import { useTranslation } from 'react-i18next'
import { ExecutiveOverview } from './routes/ExecutiveOverview'
import { QAFailureExplorer } from './routes/QAFailureExplorer'
import { AnalyticsDistribution } from './routes/AnalyticsDistribution'
import { LangSwitcher } from '@/components/LangSwitcher'
import { usePortalStore } from '@/app/store/usePortalStore'
import { ThemeSwitcher } from './ThemeSwitcher'

export const App: React.FC = () => {
  const { t } = useTranslation()
  const [route, setRoute] = React.useState<'executive'|'qa'|'analytics'>('executive')
  const setThresholds = usePortalStore((s) => s.setThresholds)
  const setDefaultThresholds = usePortalStore((s) => s.setDefaultThresholds)
  const setRunData = usePortalStore((s) => s.setRunData)

  React.useEffect(() => {
    // Load bundled thresholds profile once on app start
    let cancelled = false
    async function loadProfile() {
      try {
        const res = await fetch('/profiles/thresholds.standard.json', { cache: 'no-store' })
        if (!res.ok) return
        const data = await res.json()
        const gates = data?.gates || {}
        const out: any = {}
        for (const [k, v] of Object.entries(gates as Record<string, any>)) {
          if (v && typeof v === 'object' && 'warning' in v && 'critical' in v) {
            out[k] = { warning: Number((v as any).warning), critical: Number((v as any).critical) }
          }
        }
        if (!cancelled && Object.keys(out).length) {
          setThresholds(out)
          setDefaultThresholds(out)
        }
      } catch {
        // ignore
      }
    }
    loadProfile()
    return () => { cancelled = true }
  }, [setThresholds, setDefaultThresholds])

  React.useEffect(() => {
    // In dev, auto-load a fixed JSON so reviewers don't need to pick a file.
    // Use Vite /@fs to fetch from absolute path. No effect in production.
    if (!import.meta.env.DEV) return
    const abs = '/@fs/mnt/d/workspace/domain-specific-llm-eval/eval-pipeline/outputs/run_20250709_160725_85a5ba54/evaluations-pre/ragas_enhanced_evaluation_results_20250709_205451_fixed.json'
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch(abs)
        if (!res.ok) return
        const data = await res.json()
        // Normalize in the same way worker does: accept array or {items}
        const arr = Array.isArray(data) ? data : (data as any)?.items ?? []
        // Reuse schema logic inside the main thread for preinstall only
        const mod = await import('@/core/schemas')
        const items = arr.map((r: any) => {
          try { return mod.normalizeItem(mod.RawItemSchema.parse(r) as any) } catch { return undefined }
        }).filter(Boolean)
        const kpis = mod.aggregateKpis(items as any)
        const latencies = mod.computeLatencyStats(items as any)
        if (!cancelled) setRunData({ items: items as any, kpis, counts: { total: (items as any).length }, latencies })
      } catch {
        // ignore in dev
      }
    })()
    return () => { cancelled = true }
  }, [setRunData])
  return (
    <div style={{ fontFamily: 'system-ui, sans-serif', padding: 16 }}>
      <header style={{ display: 'flex', gap: 12, alignItems: 'center' }}>
        <h1 style={{ marginRight: 'auto' }}>{t('appTitle')}</h1>
        <button onClick={() => setRoute('executive')}>{t('nav.executive')}</button>
        <button onClick={() => setRoute('qa')}>{t('nav.qa')}</button>
        <button onClick={() => setRoute('analytics')}>{t('nav.analytics')}</button>
        <ThemeSwitcher />
        <LangSwitcher />
      </header>
      <main style={{ marginTop: 16 }}>
        {route === 'executive' && <ExecutiveOverview />}
        {route === 'qa' && <QAFailureExplorer />}
        {route === 'analytics' && <AnalyticsDistribution />}
      </main>
    </div>
  )
}
