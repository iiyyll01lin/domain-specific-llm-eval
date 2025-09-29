import React from 'react'
import { useTranslation } from 'react-i18next'
import { usePortalStore } from '@/app/store/usePortalStore'
import { fetchDocuments } from './api'
import { usePollingResource } from './usePollingResource'
import type { DocumentRow } from './types'

const formatDateTime = (value?: string) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, { hour12: false, dateStyle: 'short', timeStyle: 'medium' }).format(date)
}

const formatSize = (bytes?: number) => {
  if (!bytes || !Number.isFinite(bytes)) return '—'
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`
}

const statusColor: Record<string, string> = {
  queued: '#0277bd',
  running: '#fbc02d',
  completed: '#2e7d32',
  duplicate: '#6a1b9a',
  error: '#c62828',
}

function StatusBadge({ status, label }: { status: string; label: string }) {
  const color = statusColor[status] || '#37474f'
  return (
    <span style={{ background: color, color: '#fff', padding: '2px 8px', borderRadius: 8, fontSize: 12, textTransform: 'capitalize' }}>{label}</span>
  )
}

export const DocumentsPanel: React.FC = () => {
  const { t } = useTranslation()
  const documents = usePortalStore((s) => s.documents)
  const setDocuments = usePortalStore((s) => s.setDocuments)
  const { data, isLoading, error, lastUpdated, refetch } = usePollingResource<DocumentRow[]>(documents, React.useCallback(
    (signal) => fetchDocuments({ signal }),
    []
  ))

  React.useEffect(() => {
    setDocuments(data)
  }, [data, setDocuments])

  const translateStatus = React.useCallback(
    (status: string) => t(`lifecycle.documents.status.${status}`, { defaultValue: status.replace(/_/g, ' ') }),
    [t]
  )
  const lastUpdatedLabel = lastUpdated
    ? t('lifecycle.shared.lastUpdated', { time: formatDateTime(new Date(lastUpdated).toISOString()) })
    : t('lifecycle.shared.awaiting')

  return (
    <section style={{ border: '1px solid #263238', borderRadius: 12, padding: 16 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0 }}>{t('lifecycle.documents.title')}</h2>
          <p style={{ margin: '4px 0 0', color: '#78909c', fontSize: 12 }}>{t('lifecycle.documents.subtitle')}</p>
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
          {t('lifecycle.documents.error')} {error.message}
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #455a64' }}>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.documents.columns.kmId')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.documents.columns.version')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.documents.columns.checksum')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.documents.columns.size')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.documents.columns.status')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.documents.columns.lastEvent')}</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && !documents.length && (
              <tr>
                <td colSpan={6} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.documents.loading')}
                </td>
              </tr>
            )}
            {!isLoading && !documents.length && (
              <tr>
                <td colSpan={6} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.documents.empty')}
                </td>
              </tr>
            )}
            {documents.map((row) => (
              <tr key={`${row.km_id}-${row.version}`} style={{ borderBottom: '1px solid rgba(69,90,100,0.4)' }}>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{row.km_id}</td>
                <td style={{ padding: '8px 4px' }}>{row.version}</td>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{row.checksum}</td>
                <td style={{ padding: '8px 4px' }}>{formatSize(row.size)}</td>
                <td style={{ padding: '8px 4px' }}><StatusBadge status={row.status} label={translateStatus(row.status)} /></td>
                <td style={{ padding: '8px 4px' }}>{formatDateTime(row.last_event_ts)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
