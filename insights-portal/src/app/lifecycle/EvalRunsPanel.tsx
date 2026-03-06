import React from 'react'
import { useTranslation } from 'react-i18next'
import { usePortalStore } from '@/app/store/usePortalStore'
import { fetchEvalRuns } from './api'
import { usePollingResource } from './usePollingResource'
import type { EvalRun } from './types'

const statusColor: Record<string, string> = {
  queued: '#0277bd',
  running: '#fbc02d',
  completed: '#2e7d32',
  error: '#c62828',
}

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

function StatusBadge({ status, label }: { status: string; label: string }) {
  const color = statusColor[status] || '#37474f'
  return (
    <span style={{ background: color, color: '#fff', padding: '2px 8px', borderRadius: 8, fontSize: 12, textTransform: 'capitalize' }}>
      {label}
    </span>
  )
}

export const EvalRunsPanel: React.FC = () => {
  const { t } = useTranslation()
  const evalRuns = usePortalStore((s) => s.evalRuns)
  const setEvalRuns = usePortalStore((s) => s.setEvalRuns)
  const { data, isLoading, error, lastUpdated, refetch } = usePollingResource<EvalRun[]>(evalRuns, React.useCallback(
    (signal) => fetchEvalRuns({ signal }),
    []
  ))

  React.useEffect(() => {
    setEvalRuns(data)
  }, [data, setEvalRuns])

  const translateStatus = React.useCallback(
    (status: string) => t(`lifecycle.evals.status.${status}`, { defaultValue: status.replace(/_/g, ' ') }),
    [t]
  )

  const lastUpdatedLabel = lastUpdated
    ? t('lifecycle.shared.lastUpdated', { time: formatDateTime(new Date(lastUpdated).toISOString()) })
    : t('lifecycle.shared.awaiting')

  return (
    <section style={{ border: '1px solid #263238', borderRadius: 12, padding: 16 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0 }}>{t('lifecycle.evals.title')}</h2>
          <p style={{ margin: '4px 0 0', color: '#78909c', fontSize: 12 }}>{t('lifecycle.evals.subtitle')}</p>
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
          {t('lifecycle.evals.error')} {error.message}
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #455a64' }}>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.evals.columns.runId')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.evals.columns.testsetId')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.evals.columns.status')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.evals.columns.itemCount')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.evals.columns.metricsVersion')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.evals.columns.created')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.evals.columns.completed')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.evals.columns.error')}</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && !evalRuns.length && (
              <tr>
                <td colSpan={8} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.evals.loading')}
                </td>
              </tr>
            )}
            {!isLoading && !evalRuns.length && (
              <tr>
                <td colSpan={8} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.evals.empty')}
                </td>
              </tr>
            )}
            {evalRuns.map((run) => (
              <tr key={run.run_id} style={{ borderBottom: '1px solid rgba(69,90,100,0.4)' }}>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{run.run_id}</td>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{run.testset_id || '—'}</td>
                <td style={{ padding: '8px 4px' }}>
                  <StatusBadge status={run.status} label={translateStatus(run.status)} />
                </td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(run.evaluation_item_count)}</td>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{run.metrics_version || '—'}</td>
                <td style={{ padding: '8px 4px' }}>{formatDateTime(run.created_at)}</td>
                <td style={{ padding: '8px 4px' }}>{formatDateTime(run.completed_at)}</td>
                <td style={{ padding: '8px 4px', color: run.error_code ? '#c62828' : '#546e7a' }}>
                  {run.error_code ? `${run.error_code}${run.error_message ? `: ${run.error_message}` : ''}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
