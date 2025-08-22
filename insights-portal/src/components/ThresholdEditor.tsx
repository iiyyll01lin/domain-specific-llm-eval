import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import type { Thresholds } from '@/core/types'

export const ThresholdEditor: React.FC = () => {
  const thresholds = usePortalStore((s) => s.thresholds)
  const setThresholds = usePortalStore((s) => s.setThresholds)

  const onChange = (k: string, field: 'warning' | 'critical', v: string) => {
    const num = Number(v)
    if (Number.isNaN(num)) return
    const next = { ...thresholds, [k]: { ...(thresholds as any)[k], [field]: num } } as any
    setThresholds(next)
  }

  const onReset = async () => {
    // Try to load bundled profile thresholds first
    const profile = await loadProfileThresholds()
    if (profile) {
      setThresholds(profile)
      return
    }
    // Fallback to sensible defaults
    setThresholds({
      ContextPrecision: { warning: 0.9, critical: 0.8 },
      ContextRecall: { warning: 0.9, critical: 0.8 },
      Faithfulness: { warning: 0.35, critical: 0.3 },
      AnswerRelevancy: { warning: 0.7, critical: 0.6 },
    })
  }

  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
        <h3 style={{ margin: 0 }}>Thresholds</h3>
        <button onClick={onReset}>重置</button>
      </div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(220px, 1fr))', gap: 8, marginTop: 8 }}>
        {Object.entries(thresholds).map(([k, lv]) => (
          <div key={k} style={{ border: '1px solid #ddd', borderRadius: 8, padding: 8 }}>
            <div style={{ fontWeight: 600, marginBottom: 6 }}>{k}</div>
            <label>
              warning
              <input
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={lv?.warning ?? ''}
                onChange={(e) => onChange(k, 'warning', e.target.value)}
                style={{ marginLeft: 6, width: 90 }}
              />
            </label>
            <label style={{ marginLeft: 12 }}>
              critical
              <input
                type="number"
                step="0.01"
                min={0}
                max={1}
                value={lv?.critical ?? ''}
                onChange={(e) => onChange(k, 'critical', e.target.value)}
                style={{ marginLeft: 6, width: 90 }}
              />
            </label>
          </div>
        ))}
      </div>
    </div>
  )
}

async function loadProfileThresholds(): Promise<Thresholds | undefined> {
  try {
    const res = await fetch('/profiles/thresholds.standard.json', { cache: 'no-store' })
    if (!res.ok) return undefined
    const data = await res.json()
    const gates = (data && data.gates) || {}
    // Convert to Thresholds shape
    const out: Thresholds = {}
    for (const [k, v] of Object.entries(gates as Record<string, any>)) {
      if (v && typeof v === 'object' && 'warning' in v && 'critical' in v) {
        out[k as keyof Thresholds] = { warning: Number((v as any).warning), critical: Number((v as any).critical) } as any
      }
    }
    return out
  } catch {
    return undefined
  }
}
