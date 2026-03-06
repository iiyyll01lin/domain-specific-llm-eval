import React from 'react'
import { useTranslation } from 'react-i18next'
import { usePortalStore } from '@/app/store/usePortalStore'
import { fetchReports } from './api'
import { usePollingResource } from './usePollingResource'
import { getLifecycleConfig } from './config'
import type { ReportItem } from './types'

const formatDateTime = (value?: string) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, { hour12: false, dateStyle: 'short', timeStyle: 'medium' }).format(date)
}

function AvailableBadge({ available, label }: { available: boolean; label: string }) {
  return (
    <span
      style={{
        background: available ? '#1b5e20' : '#37474f',
        color: '#fff',
        padding: '2px 8px',
        borderRadius: 8,
        fontSize: 12,
      }}
    >
      {label}
    </span>
  )
}

export const ReportsPanel: React.FC = () => {
  const { t } = useTranslation()
  const reports = usePortalStore((s) => s.reports)
  const setReports = usePortalStore((s) => s.setReports)
  const { data, isLoading, error, lastUpdated, refetch } = usePollingResource<ReportItem[]>(reports, React.useCallback(
    (signal) => fetchReports({ signal }),
    []
  ))

  React.useEffect(() => {
    setReports(data)
  }, [data, setReports])

  const lastUpdatedLabel = lastUpdated
    ? t('lifecycle.shared.lastUpdated', { time: formatDateTime(new Date(lastUpdated).toISOString()) })
    : t('lifecycle.shared.awaiting')

  const handleOpenHtml = React.useCallback((item: ReportItem) => {
    if (!item.html_available) return
    const { reportingBaseUrl } = getLifecycleConfig()
    const url = `${reportingBaseUrl}/reports/${encodeURIComponent(item.run_id)}/${encodeURIComponent(item.template)}/html`
    window.open(url, '_blank', 'noopener,noreferrer')
  }, [])

  return (
    <section style={{ border: '1px solid #263238', borderRadius: 12, padding: 16 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0 }}>{t('lifecycle.reports.title')}</h2>
          <p style={{ margin: '4px 0 0', color: '#78909c', fontSize: 12 }}>{t('lifecycle.reports.subtitle')}</p>
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
          {t('lifecycle.reports.error')} {error.message}
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #455a64' }}>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.reports.columns.runId')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.reports.columns.template')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.reports.columns.html')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.reports.columns.pdf')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.reports.columns.created')}</th>
              <th style={{ padding: '8px 4px' }}></th>
            </tr>
          </thead>
          <tbody>
            {isLoading && !reports.length && (
              <tr>
                <td colSpan={6} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.reports.loading')}
                </td>
              </tr>
            )}
            {!isLoading && !reports.length && (
              <tr>
                <td colSpan={6} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.reports.empty')}
                </td>
              </tr>
            )}
            {reports.map((item) => (
              <tr key={`${item.run_id}-${item.template}`} style={{ borderBottom: '1px solid rgba(69,90,100,0.4)' }}>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{item.run_id}</td>
                <td style={{ padding: '8px 4px' }}>{item.template}</td>
                <td style={{ padding: '8px 4px' }}>
                  <AvailableBadge
                    available={item.html_available}
                    label={item.html_available ? t('lifecycle.reports.available') : t('lifecycle.reports.unavailable')}
                  />
                </td>
                <td style={{ padding: '8px 4px' }}>
                  <AvailableBadge
                    available={item.pdf_available}
                    label={item.pdf_available ? t('lifecycle.reports.available') : t('lifecycle.reports.unavailable')}
                  />
                </td>
                <td style={{ padding: '8px 4px' }}>{formatDateTime(item.created_at)}</td>
                <td style={{ padding: '8px 4px' }}>
                  {item.html_available && (
                    <button
                      type="button"
                      onClick={() => handleOpenHtml(item)}
                      style={{ fontSize: 12, padding: '2px 10px', cursor: 'pointer' }}
                    >
                      {t('lifecycle.reports.openHtml')}
                    </button>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
