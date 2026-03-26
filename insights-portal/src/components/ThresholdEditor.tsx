import React from 'react'
import { usePortalStore } from '@/app/store/usePortalStore'
import type { Thresholds } from '@/core/types'
import { validateThresholdValue, validateThresholdsShape } from '@/core/verdict'
import { useTranslation } from 'react-i18next'

export const ThresholdEditor: React.FC = () => {
  const thresholds = usePortalStore((s) => s.thresholds)
  const setThresholds = usePortalStore((s) => s.setThresholds)
  const { t } = useTranslation()

  // Used to add unknown indicators
  const [isAdding, setIsAdding] = React.useState(false)                   
  const [newKey, setNewKey] = React.useState('')                         
  const [newWarn, setNewWarn] = React.useState<string>('0.80')            
  const [newCrit, setNewCrit] = React.useState<string>('0.60')            
  const [addErr, setAddErr] = React.useState<string>('')                  

  const onChange = (k: string, field: 'warning' | 'critical', v: string) => {
  const num = Number(v)
  if (!validateThresholdValue(num)) return
  const next = { ...thresholds, [k]: { ...(thresholds as any)[k], [field]: num } } as any
  const chk = validateThresholdsShape(next)
  if (!chk.ok) return
  setThresholds(next)
  }

  // Add unknown directly to the panel
  const onAddMetric = () => {                                             
    setAddErr('')
    const rawKey = (newKey || '').trim()
    const key = rawKey
      .replace(/\s+/g, '')            
      .replace(/[^\w.\-]/g, '')       
    if (!key) { setAddErr(t('errors.enterMetricKey')); return }

    if (key !== rawKey) {
      setAddErr(t('errors.metricSanitized', { key }))  // Silent Sanitization message
    }

    if (thresholds && Object.prototype.hasOwnProperty.call(thresholds, key)) {
      setAddErr(t('errors.metricExists')); return
    }
    const w = Number(newWarn)
    const c = Number(newCrit)
    if (!validateThresholdValue(w) || !validateThresholdValue(c)) {
      setAddErr(t('errors.range01')); return
    }
    const next = { 
      ...(thresholds as Thresholds), 
      [key]: { warning: w, critical: c } 
    } as Thresholds
    const chk = validateThresholdsShape(next)
    if (!chk.ok) { setAddErr(chk.errors.join(', ')); return }
    setThresholds(next)
    // reset UI
    setIsAdding(false)
    setNewKey('')
    setNewWarn('0.80')
    setNewCrit('0.60')
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
        <h3 style={{ margin: 0 }}>{t('thresholds.title')}</h3>
        <button onClick={onReset}>{t('thresholds.reset')}</button>
        {/* enter add mode */}
        <button onClick={() => setIsAdding(v => !v)} aria-expanded={isAdding} aria-controls="add-threshold-row">
          {isAdding ? t('thresholds.cancelAdd') : t('thresholds.addMetric')}
        </button>
      </div>

      {/*add unknown metric enter*/}
      {isAdding && (
        <div id="add-threshold-row" 
             style={{ marginTop: 8, padding: 8, border: '1px dashed #bbb', borderRadius: 8, display: 'flex', gap: 12, alignItems: 'center' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
             {t('thresholds.key')}
            <input
              value={newKey}
              onChange={(e) => setNewKey(e.target.value)}
              placeholder={t('thresholds.placeholderKey') as string}
              aria-label="new-metric-key"
              style={{ width: 200 }}
            />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {t('thresholds.warning')}
            <input
              type="number" step="0.01" min={0} max={1}
              value={newWarn}
              onChange={(e) => setNewWarn(e.target.value)}
              aria-label="new-metric-warning"
              style={{ width: 90 }}
            />
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            {t('thresholds.critical')}
            <input
              type="number" step="0.01" min={0} max={1}
              value={newCrit}
              onChange={(e) => setNewCrit(e.target.value)}
              aria-label="new-metric-critical"
              style={{ width: 90 }}
            />
          </label>
          <button onClick={onAddMetric}>{t('thresholds.confirmAdd')}</button>
          {addErr && (
            <div role="alert" style={{ color: 'crimson', fontSize: 12 }}>{addErr}</div>
          )}
        </div>
      )}

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
                aria-label={`threshold-${k}-warning`}
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
                aria-label={`threshold-${k}-critical`}
              />
            </label>
            {(() => {
              const chk = validateThresholdsShape({ [k]: lv } as any)
              return chk.ok ? null : (
                <div style={{ color: 'crimson', marginTop: 4, fontSize: 12 }} aria-live="polite">
                  {chk.errors.join(', ')}
                </div>
              )
            })()}
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
