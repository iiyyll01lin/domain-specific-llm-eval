import React from 'react'
import { useTranslation } from 'react-i18next'
import { usePortalStore } from '@/app/store/usePortalStore'
import { fetchKgJobs } from './api'
import { usePollingResource } from './usePollingResource'
import { StatusBadge } from './StatusBadge'
import type { KgJobItem } from './types'

/** Feature flag: set window.ENABLE_KG_PANEL = true to show this panel. */
function isKgPanelEnabled(): boolean {
  if (typeof window === 'undefined') return false
  return Boolean((window as any).ENABLE_KG_PANEL)
}

const formatDateTime = (value?: string) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, {
    hour12: false,
    dateStyle: 'short',
    timeStyle: 'medium',
  }).format(date)
}

const formatNumber = (value?: number) => {
  if (typeof value !== 'number' || !Number.isFinite(value)) return '—'
  return value.toLocaleString()
}

export const KgPanel: React.FC = () => {
  const { t } = useTranslation()
  const kgJobs = usePortalStore((s) => s.kgJobs)
  const setKgJobs = usePortalStore((s) => s.setKgJobs)

  const { data, isLoading, error, lastUpdated, refetch } = usePollingResource<KgJobItem[]>(
    kgJobs,
    React.useCallback((signal) => fetchKgJobs({ signal }), []),
  )

  React.useEffect(() => {
    setKgJobs(data)
  }, [data, setKgJobs])

  const lastUpdatedLabel = lastUpdated
    ? t('lifecycle.shared.lastUpdated', {
        time: formatDateTime(new Date(lastUpdated).toISOString()),
      })
    : t('lifecycle.shared.awaiting')

  if (!isKgPanelEnabled()) {
    return (
      <section
        style={{ border: '1px solid #263238', borderRadius: 12, padding: 16, color: '#78909c' }}
      >
        <h2 style={{ margin: 0 }}>{t('lifecycle.kgJobs.title')}</h2>
        <p style={{ marginTop: 8 }}>{t('lifecycle.kgJobs.subtitle')}</p>
        <p style={{ fontStyle: 'italic', fontSize: 12 }}>
          KG panel is feature-flagged. Set <code>window.ENABLE_KG_PANEL = true</code> to enable.
        </p>
      </section>
    )
  }

  return (
    <section style={{ border: '1px solid #263238', borderRadius: 12, padding: 16 }}>
      <header
        style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          marginBottom: 12,
        }}
      >
        <div>
          <h2 style={{ margin: 0 }}>{t('lifecycle.kgJobs.title')}</h2>
          <p style={{ margin: '4px 0 0', color: '#78909c', fontSize: 12 }}>
            {t('lifecycle.kgJobs.subtitle')}
          </p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <button type="button" onClick={refetch} disabled={isLoading}>
            {t('lifecycle.shared.refresh')}
          </button>
          <span style={{ fontSize: 12, color: '#90a4ae' }}>{lastUpdatedLabel}</span>
        </div>
      </header>

      {error && (
        <div
          style={{
            background: '#ffebee',
            color: '#c62828',
            padding: 12,
            borderRadius: 8,
            marginBottom: 12,
          }}
        >
          {t('lifecycle.kgJobs.error')} {error.message}
        </div>
      )}

      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #455a64' }}>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.kgId')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.status')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.docCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.nodeCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.edgeCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.created')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.kgJobs.columns.updated')}</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && !kgJobs.length && (
              <tr>
                <td
                  colSpan={7}
                  style={{ padding: 12, textAlign: 'center', color: '#78909c' }}
                >
                  {t('lifecycle.kgJobs.loading')}
                </td>
              </tr>
            )}
            {!isLoading && !kgJobs.length && (
              <tr>
                <td
                  colSpan={7}
                  style={{ padding: 12, textAlign: 'center', color: '#78909c' }}
                >
                  {t('lifecycle.kgJobs.empty')}
                </td>
              </tr>
            )}
            {kgJobs.map((job, index) => (
              <tr
                key={`${job.kg_id}-${index}`}
                style={{ borderBottom: '1px solid rgba(69,90,100,0.4)' }}
              >
                <td
                  style={{
                    padding: '8px 4px',
                    fontFamily: 'monospace',
                    fontSize: 12,
                    maxWidth: 180,
                    overflow: 'hidden',
                    textOverflow: 'ellipsis',
                    whiteSpace: 'nowrap',
                  }}
                  title={job.kg_id}
                >
                  {job.kg_id}
                </td>
                <td style={{ padding: '8px 4px' }}>
                  <StatusBadge status={job.status} />
                </td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(job.doc_count)}</td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(job.node_count)}</td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(job.edge_count)}</td>
                <td style={{ padding: '8px 4px', fontSize: 12 }}>
                  {formatDateTime(job.created_at)}
                </td>
                <td style={{ padding: '8px 4px', fontSize: 12 }}>
                  {formatDateTime(job.updated_at)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
