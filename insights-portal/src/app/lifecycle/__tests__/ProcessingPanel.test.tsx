// @vitest-environment jsdom
import * as React from 'react'
import { beforeEach, afterEach, describe, expect, it, vi, type SpyInstance } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import { ProcessingPanel } from '../ProcessingPanel'
import i18n from '@/app/i18n'
import { usePortalStore } from '@/app/store/usePortalStore'
import * as lifecycleConfig from '../config'
import * as api from '../api'

const baseConfig = lifecycleConfig.getLifecycleConfig()

describe('ProcessingPanel', () => {
  let configSpy: SpyInstance<[], ReturnType<typeof lifecycleConfig.getLifecycleConfig>>

  beforeEach(() => {
    configSpy = vi.spyOn(lifecycleConfig, 'getLifecycleConfig').mockReturnValue({
      ...baseConfig,
      pollIntervalMs: 25,
      requestTimeoutMs: 1_000,
    })
    usePortalStore.setState({ processingJobs: [] })
  })

  afterEach(() => {
    configSpy?.mockRestore()
    vi.restoreAllMocks()
  })

  it('renders processing job details with SLA badge', async () => {
    const payload = [
      {
        job_id: 'job-001',
        document_id: 'doc-123',
        status: 'running',
        progress: 42,
        chunk_count: 8,
        embedding_profile_hash: 'hash-xyz',
        started_at: '2025-09-28T10:00:00Z',
        updated_at: '2025-09-28T10:05:00Z',
        elapsed_seconds: 900,
        sla_seconds: 600,
      },
    ]
    const fetchSpy = vi.spyOn(api, 'fetchProcessingJobs').mockResolvedValue(payload as any)

    render(
      <I18nextProvider i18n={i18n}>
        <ProcessingPanel />
      </I18nextProvider>
    )

    await screen.findByText('job-001')
    screen.getByText('doc-123')
    screen.getByText(i18n.t('lifecycle.processing.sla.breached'))
    expect(fetchSpy).toHaveBeenCalled()
  })

  it('polls the processing endpoint every 10 seconds', async () => {
    const fetchSpy = vi.spyOn(api, 'fetchProcessingJobs').mockResolvedValue([])

    render(
      <I18nextProvider i18n={i18n}>
        <ProcessingPanel />
      </I18nextProvider>
    )

    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(2), { timeout: 1_000 })
  })

  it('shows error message when fetch fails', async () => {
    vi.spyOn(api, 'fetchProcessingJobs').mockRejectedValue(new Error('boom'))

    render(
      <I18nextProvider i18n={i18n}>
        <ProcessingPanel />
      </I18nextProvider>
    )

    await screen.findByText(/Failed to load processing jobs/i)
  })
})
