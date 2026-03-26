import React from 'react'
import { useTranslation } from 'react-i18next'
import { usePortalStore } from '@/app/store/usePortalStore'
import { fetchProcessingJobs } from './api'
import { usePollingResource } from './usePollingResource'
import type { ProcessingJob } from './types'

const statusColor: Record<string, string> = {
  queued: '#0277bd',
  running: '#fbc02d',
  completed: '#2e7d32',
  error: '#c62828',
}

const formatDuration = (seconds?: number) => {
  if (!seconds || seconds <= 0) return '—'
  if (seconds < 60) return `${seconds.toFixed(1)}s`
  const minutes = Math.floor(seconds / 60)
  const remainder = seconds % 60
  if (minutes < 60) return `${minutes}m ${remainder.toFixed(0)}s`
  const hours = Math.floor(minutes / 60)
  const remMinutes = minutes % 60
  return `${hours}h ${remMinutes}m`
}

const formatTimestamp = (value?: string) => {
  if (!value) return '—'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  return new Intl.DateTimeFormat(undefined, { hour12: false, dateStyle: 'short', timeStyle: 'medium' }).format(date)
}

function ProgressBar({ value }: { value: number }) {
  const pct = Math.max(0, Math.min(100, value || 0))
  return (
    <div style={{ width: '100%', background: 'rgba(55,71,79,0.4)', borderRadius: 6, height: 10 }}>
      <div style={{ width: `${pct}%`, background: '#26a69a', height: '100%', borderRadius: 6 }} />
    </div>
  )
}

function StatusBadge({ status, label }: { status: string; label: string }) {
  const color = statusColor[status] || '#37474f'
  return (
    <span style={{ background: color, color: '#fff', padding: '2px 8px', borderRadius: 8, fontSize: 12, textTransform: 'capitalize' }}>{label}</span>
  )
}

function SlaBadge({ job, okLabel, breachedLabel }: { job: ProcessingJob; okLabel: string; breachedLabel: string }) {
  if (!job.sla_seconds || !job.elapsed_seconds) return null
  const breached = job.elapsed_seconds > job.sla_seconds
  if (!breached) return (
    <span style={{ color: '#2e7d32', fontSize: 12 }}>{okLabel}</span>
  )
  return (
    <span style={{ color: '#c62828', fontSize: 12 }}>{breachedLabel}</span>
  )
}

export const ProcessingPanel: React.FC = () => {
  const { t } = useTranslation()
  const processingJobs = usePortalStore((s) => s.processingJobs)
  const setProcessingJobs = usePortalStore((s) => s.setProcessingJobs)
  const { data, isLoading, error, lastUpdated, refetch } = usePollingResource<ProcessingJob[]>(processingJobs, React.useCallback(
    (signal) => fetchProcessingJobs({ signal }),
    []
  ))

  React.useEffect(() => {
    setProcessingJobs(data)
  }, [data, setProcessingJobs])

  const translateStatus = React.useCallback(
    (status: string) => t(`lifecycle.processing.status.${status}`, { defaultValue: status.replace(/_/g, ' ') }),
    [t]
  )
  const lastUpdatedLabel = lastUpdated
    ? t('lifecycle.shared.lastUpdated', { time: formatTimestamp(new Date(lastUpdated).toISOString()) })
    : t('lifecycle.shared.awaiting')

  return (
    <section style={{ border: '1px solid #263238', borderRadius: 12, padding: 16 }}>
      <header style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 12 }}>
        <div>
          <h2 style={{ margin: 0 }}>{t('lifecycle.processing.title')}</h2>
          <p style={{ margin: '4px 0 0', color: '#78909c', fontSize: 12 }}>{t('lifecycle.processing.subtitle')}</p>
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
          {t('lifecycle.processing.error')} {error.message}
        </div>
      )}
      <div style={{ overflowX: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ textAlign: 'left', borderBottom: '1px solid #455a64' }}>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.jobId')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.documentId')}</th>
              <th style={{ padding: '8px 4px', width: 200 }}>{t('lifecycle.processing.columns.progress')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.status')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.chunks')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.profileHash')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.started')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.updated')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.elapsed')}</th>
              <th style={{ padding: '8px 4px' }}>{t('lifecycle.processing.columns.sla')}</th>
            </tr>
          </thead>
          <tbody>
            {isLoading && !processingJobs.length && (
              <tr>
                <td colSpan={10} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.processing.loading')}
                </td>
              </tr>
            )}
            {!isLoading && !processingJobs.length && (
              <tr>
                <td colSpan={10} style={{ padding: 12, textAlign: 'center', color: '#78909c' }}>
                  {t('lifecycle.processing.empty')}
                </td>
              </tr>
            )}
            {processingJobs.map((job) => (
              <tr key={job.job_id} style={{ borderBottom: '1px solid rgba(69,90,100,0.4)' }}>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{job.job_id}</td>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{job.document_id}</td>
                <td style={{ padding: '8px 4px' }}>
                  <ProgressBar value={job.progress} />
                </td>
                <td style={{ padding: '8px 4px' }}><StatusBadge status={job.status} label={translateStatus(job.status)} /></td>
                <td style={{ padding: '8px 4px' }}>{job.chunk_count ?? '—'}</td>
                <td style={{ padding: '8px 4px', fontFamily: 'monospace' }}>{job.embedding_profile_hash ?? '—'}</td>
                <td style={{ padding: '8px 4px' }}>{formatTimestamp(job.started_at)}</td>
                <td style={{ padding: '8px 4px' }}>{formatTimestamp(job.updated_at)}</td>
                <td style={{ padding: '8px 4px' }}>{formatDuration(job.elapsed_seconds)}</td>
                <td style={{ padding: '8px 4px' }}><SlaBadge job={job} okLabel={t('lifecycle.processing.sla.ok')} breachedLabel={t('lifecycle.processing.sla.breached')} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
