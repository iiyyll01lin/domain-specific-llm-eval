// @vitest-environment jsdom
import * as React from 'react'
import { beforeEach, afterEach, describe, expect, it, vi, type SpyInstance } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import { TestsetsPanel } from '../TestsetsPanel'
import i18n from '@/app/i18n'
import { usePortalStore } from '@/app/store/usePortalStore'
import * as lifecycleConfig from '../config'
import * as api from '../api'

const baseConfig = lifecycleConfig.getLifecycleConfig()

describe('TestsetsPanel', () => {
  let configSpy: SpyInstance<[], ReturnType<typeof lifecycleConfig.getLifecycleConfig>>

  beforeEach(() => {
    configSpy = vi.spyOn(lifecycleConfig, 'getLifecycleConfig').mockReturnValue({
      ...baseConfig,
      pollIntervalMs: 25,
      requestTimeoutMs: 1_000,
    })
    usePortalStore.setState({ testsetJobs: [] })
  })

  afterEach(() => {
    configSpy?.mockRestore()
    vi.restoreAllMocks()
  })

  it('renders rows fetched from testset service', async () => {
    const payload = [
      {
        job_id: 'job-001',
        method: 'ragas',
        status: 'completed',
        sample_count: 42,
        persona_count: 5,
        scenario_count: 12,
        seed: 123,
        config_hash: 'abc123def456',
        updated_at: '2025-09-30T10:00:00Z',
        created_at: '2025-09-30T09:55:00Z',
      },
    ]
    const fetchSpy = vi.spyOn(api, 'fetchTestsetJobs').mockResolvedValue(payload as any)

    render(
      <I18nextProvider i18n={i18n}>
        <TestsetsPanel />
      </I18nextProvider>
    )

    await screen.findByText('job-001')
    screen.getByText('ragas')
    screen.getByText(i18n.t('lifecycle.testsets.status.completed'))
    screen.getByText('42')
    screen.getByText('abc123def456')
    expect(fetchSpy).toHaveBeenCalled()
  })

  it('polls the endpoint periodically', async () => {
    const fetchSpy = vi.spyOn(api, 'fetchTestsetJobs').mockResolvedValue([])

    render(
      <I18nextProvider i18n={i18n}>
        <TestsetsPanel />
      </I18nextProvider>
    )

    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(2), { timeout: 1_000 })
  })

  it('displays error banner when fetch fails', async () => {
    vi.spyOn(api, 'fetchTestsetJobs').mockRejectedValue(new Error('boom'))

    render(
      <I18nextProvider i18n={i18n}>
        <TestsetsPanel />
      </I18nextProvider>
    )

    await screen.findByText(/Failed to load testset jobs/i)
  })
})
