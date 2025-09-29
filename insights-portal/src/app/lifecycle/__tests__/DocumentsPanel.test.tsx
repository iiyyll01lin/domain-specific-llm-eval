// @vitest-environment jsdom
import * as React from 'react'
import { beforeEach, afterEach, describe, expect, it, vi, type SpyInstance } from 'vitest'
import { render, screen, waitFor } from '@testing-library/react'
import { I18nextProvider } from 'react-i18next'
import { DocumentsPanel } from '../DocumentsPanel'
import i18n from '@/app/i18n'
import { usePortalStore } from '@/app/store/usePortalStore'
import * as lifecycleConfig from '../config'
import * as api from '../api'

const baseConfig = lifecycleConfig.getLifecycleConfig()

describe('DocumentsPanel', () => {
  let configSpy: SpyInstance<[], ReturnType<typeof lifecycleConfig.getLifecycleConfig>>

  beforeEach(() => {
    configSpy = vi.spyOn(lifecycleConfig, 'getLifecycleConfig').mockReturnValue({
      ...baseConfig,
      pollIntervalMs: 25,
      requestTimeoutMs: 1_000,
    })
    usePortalStore.setState({ documents: [] })
  })

  afterEach(() => {
    configSpy?.mockRestore()
    vi.restoreAllMocks()
  })

  it('renders rows fetched from ingestion service', async () => {
    const payload = [
      {
        km_id: 'KM-001',
        version: 'v1',
        checksum: 'abc123',
        status: 'completed',
        size: 2048,
        last_event_ts: '2025-09-28T12:34:56Z',
      },
    ]
    const fetchSpy = vi.spyOn(api, 'fetchDocuments').mockResolvedValue(payload as any)

    render(
      <I18nextProvider i18n={i18n}>
        <DocumentsPanel />
      </I18nextProvider>
    )

    await screen.findByText('KM-001')
    screen.getByText('abc123')
    screen.getByText(i18n.t('lifecycle.documents.status.completed'))
    expect(fetchSpy).toHaveBeenCalled()
  })

  it('polls the endpoint every 10 seconds', async () => {
    const fetchSpy = vi.spyOn(api, 'fetchDocuments').mockResolvedValue([])

    render(
      <I18nextProvider i18n={i18n}>
        <DocumentsPanel />
      </I18nextProvider>
    )

    await waitFor(() => expect(fetchSpy).toHaveBeenCalledTimes(2), { timeout: 1_000 })
  })

  it('displays error banner when fetch fails', async () => {
    vi.spyOn(api, 'fetchDocuments').mockRejectedValue(new Error('boom'))

    render(
      <I18nextProvider i18n={i18n}>
        <DocumentsPanel />
      </I18nextProvider>
    )

    await screen.findByText(/Failed to load documents/i)
  })
})
