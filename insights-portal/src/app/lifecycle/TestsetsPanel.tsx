import React from 'react'
import { useTranslation } from 'react-i18next'
import { usePortalStore } from '@/app/store/usePortalStore'
import { fetchTestsetJobs } from './api'
import { usePollingResource } from './usePollingResource'
import type { TestsetJob } from './types'

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

const formatSeed = (seed?: number) => {
  if (typeof seed !== 'number' || !Number.isFinite(seed)) return '—'
  return seed.toString()
}

function StatusBadge({ status, label }: { status: string; label: string }) {
  const color = statusColor[status] || '#37474f'
  return (
    <span style={{ background: color, color: '#fff', padding: '2px 8px', borderRadius: 8, fontSize: 12, textTransform: 'capitalize' }}>{label}</span>
  )
}

function DuplicateTag({ visible, label }: { visible: boolean; label: string }) {
  if (!visible) return null
  return (
    <span style={{ marginLeft: 8, padding: '2px 6px', borderRadius: 6, fontSize: 11, background: 'rgba(55,71,79,0.2)', color: '#37474f' }}>{label}</span>
  )
}

export const TestsetsPanel: React.FC = () => {
  const { t } = useTranslation()
  const testsetJobs = usePortalStore((s) => s.testsetJobs)
  const setTestsetJobs = usePortalStore((s) => s.setTestsetJobs)
  const { data, isLoading, error, lastUpdated, refetch } = usePollingResource<TestsetJob[]>(testsetJobs, React.useCallback(
    (signal) => fetchTestsetJobs({ signal }),
    []
  ))

  React.useEffect(() => {
    setTestsetJobs(data)
  }, [data, setTestsetJobs])

  const translateStatus = React.useCallback(
    (status: string) => t(`lifecycle.testsets.status.${status}`, { defaultValue: status.replace(/_/g, ' ') }),
    [t]
  )

  const lastUpdatedLabel = lastUpdated
    ? t('lifecycle.shared.lastUpdated', { time: formatDateTime(new Date(lastUpdated).toISOString()) })
    : t('lifecycle.shared.awaiting')

  return (
    <section style={{ border: '1px solid #263238', borderRadius: 12, padding: 16 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0 }}>{t('lifecycle.testsets.title')}</h2>
          <p style={{ margin: '4px 0 0', color: '#78909c', fontSize: 12 }}>{t('lifecycle.testsets.subtitle')}</p>
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
          {t('lifecycle.testsets.error')} {error.message}
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #455a64' }}>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.jobId')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.method')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.status')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.samples')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.personas')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.scenarios')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.seed')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.configHash')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.created')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.updated')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.testsets.columns.error')}</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && !testsetJobs.length && (
              <tr>
                <td colSpan={11} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.testsets.loading')}
                </td>
              </tr>
            )}
            {!isLoading && !testsetJobs.length && (
              <tr>
                <td colSpan={11} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.testsets.empty')}
                </td>
              </tr>
            )}
            {testsetJobs.map((job) => (
              <tr key={job.job_id} style={{ borderBottom: '1px solid rgba(69,90,100,0.4)' }}>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{job.job_id}</td>
                <td style={{ padding: '8px 4px' }}>{job.method || '—'}</td>
                <td style={{ padding: '8px 4px', whiteSpace: 'nowrap' }}>
                  <StatusBadge status={job.status} label={translateStatus(job.status)} />
                  <DuplicateTag visible={Boolean(job.duplicate)} label={t('lifecycle.testsets.duplicateTag')} />
                </td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(job.sample_count)}</td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(job.persona_count)}</td>
                <td style={{ padding: '8px 4px' }}>{formatNumber(job.scenario_count)}</td>
                <td style={{ padding: '8px 4px' }}>{formatSeed(job.seed)}</td>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{job.config_hash || '—'}</td>
                <td style={{ padding: '8px 4px' }}>{formatDateTime(job.created_at)}</td>
                <td style={{ padding: '8px 4px' }}>{formatDateTime(job.updated_at)}</td>
                <td style={{ padding: '8px 4px', color: job.error_code ? '#c62828' : '#546e7a' }}>
                  {job.error_code ? `${job.error_code}${job.error_message ? `: ${job.error_message}` : ''}` : '—'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
