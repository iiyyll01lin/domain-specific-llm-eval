import React from 'react'
import { useTranslation } from 'react-i18next'
import { usePortalStore } from '@/app/store/usePortalStore'
import { fetchKmSummaries } from './api'
import { usePollingResource } from './usePollingResource'
import type { KmSummary } from './types'

const formatDateTime = (value?: string) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, { hour12: false, dateStyle: 'short', timeStyle: 'medium' }).format(date)
}

const formatNumber = (value?: number) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return value.toLocaleString()
}

function resolveId(summary: KmSummary): string {
  return summary.testset_id ?? summary.kg_id ?? '—'
}

export const KmSummariesPanel: React.FC = () => {
  const { t } = useTranslation()
  const kmSummaries = usePortalStore((s) => s.kmSummaries)
  const setKmSummaries = usePortalStore((s) => s.setKmSummaries)
  const { data, isLoading, error, lastUpdated, refetch } = usePollingResource<KmSummary[]>(kmSummaries, React.useCallback(
    (signal) => fetchKmSummaries({ signal }),
    []
  ))

  React.useEffect(() => {
    setKmSummaries(data)
  }, [data, setKmSummaries])

  const lastUpdatedLabel = lastUpdated
    ? t('lifecycle.shared.lastUpdated', { time: formatDateTime(new Date(lastUpdated).toISOString()) })
    : t('lifecycle.shared.awaiting')

  return (
    <section style={{ border: '1px solid #263238', borderRadius: 12, padding: 16 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0 }}>{t('lifecycle.kmSummaries.title')}</h2>
          <p style={{ margin: '4px 0 0', color: '#78909c', fontSize: 12 }}>{t('lifecycle.kmSummaries.subtitle')}</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button type="button" onClick={refetch} disabled={isLoading}>
            {t('lifecycle.shared.refresh')}
          </button>
          <span style={{ fontSize: 12, color: '#90a4ae' }}>{lastUpdatedLabel}</span>
        </div>
      </header>
      {error && (
        <div style={{ background: '#ffebee', color: '#c62828', padding: 12, borderRadius: 8, marginBottom: 12 }}>
          {t('lifecycle.kmSummaries.error')} {error.message}
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #455a64' }}>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kmSummaries.columns.schema')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kmSummaries.columns.id')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kmSummaries.columns.sampleCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kmSummaries.columns.nodeCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kmSummaries.columns.relCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kmSummaries.columns.created')}</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && !kmSummaries.length && (
              <tr>
                <td colSpan={6} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.kmSummaries.loading')}
                </td>
              </tr>
            )}
            {!isLoading && !kmSummaries.length && (
              <tr>
                <td colSpan={6} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.kmSummaries.empty')}
                </td>
              </tr>
            )}
            {kmSummaries.map((summary, index) => (
              // schema + id combo is unique; fall back to index for safety
              <tr key={`${summary.schema}-${resolveId(summary)}-${index}`} style={{ borderBottom: '1px solid rgba(69,90,100,0.4)' }}>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{summary.schema}</td>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{resolveId(summary)}</td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(summary.sample_count)}</td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(summary.node_count)}</td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(summary.relationship_count)}</td>
                <td style={{ padding: '8px 4px' }}>{formatDateTime(summary.created_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
