import { describe, it, expect, vi, beforeEach } from 'vitest'
import React from 'react'
import { render, fireEvent } from '@testing-library/react'
import * as exporter from '@/core/exporter'
// Mock echarts to avoid canvas usage under jsdom
vi.mock('echarts', () => ({
  init: () => ({ setOption: () => {}, dispose: () => {}, getDataURL: () => 'data:', on: () => {}, off: () => {}, resize: () => {} }),
  getInstanceByDom: () => ({ getDataURL: () => 'data:' })
}))
import { AnalyticsDistribution } from '../AnalyticsDistribution'
import { usePortalStore } from '@/app/store/usePortalStore'

describe('AnalyticsDistribution export', () => {
  beforeEach(() => {
    // Seed run in store
    usePortalStore.setState({
      run: {
        items: [
          { id: '1', language: 'en', metrics: { Faithfulness: 0.5 } },
          { id: '2', language: 'zh', metrics: { Faithfulness: 0.7 } },
        ],
        kpis: { Faithfulness: 0.6 },
        counts: { total: 2 },
      } as any,
    })
  })

  it('CSV export includes branding/meta and XLSX uses multi-sheets', async () => {
    const csvSpy = vi.spyOn(exporter, 'exportTableToCSV').mockReturnValue()
    const multiSpy = vi.spyOn(exporter, 'exportMultipleSheetsXLSX').mockResolvedValue()
    const { getByLabelText } = render(<AnalyticsDistribution />)
    fireEvent.click(getByLabelText('export-analytics-csv'))
    expect(csvSpy).toHaveBeenCalled()
    const csvArgs = csvSpy.mock.calls[0]
    expect(csvArgs[2]?.branding?.brand).toBeTruthy()
    fireEvent.click(getByLabelText('export-analytics-xlsx'))
    await Promise.resolve()
    expect(multiSpy).toHaveBeenCalled()
    const xlsxArgs = multiSpy.mock.calls[0]
    expect(Array.isArray(xlsxArgs[1])).toBe(true)
    expect(xlsxArgs[2]?.branding?.title).toBe('Analytics Export')
  })
})
