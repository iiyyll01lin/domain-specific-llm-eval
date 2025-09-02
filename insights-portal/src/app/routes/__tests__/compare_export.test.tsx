// @vitest-environment jsdom
import { describe, it, expect, vi, beforeEach } from 'vitest'
import React from 'react'
import { render, fireEvent, screen } from '@testing-library/react'
import * as exporter from '@/core/exporter'
import CompareView from '../CompareView'
import { usePortalStore } from '@/app/store/usePortalStore'

describe('CompareView exports', () => {
  const exportCsvSpy = vi.spyOn(exporter, 'exportTableToCSV').mockImplementation(() => {})
  const exportMultiXlsxSpy = vi.spyOn(exporter, 'exportMultipleSheetsXLSX').mockResolvedValue()

  beforeEach(() => {
    exportCsvSpy.mockClear()
  exportMultiXlsxSpy.mockClear()
    const set = usePortalStore.getState()
    set.setRuns({
      a: { items: [
        { id: '1', language: 'en', latencyMs: 10, metrics: { Faithfulness: 0.5, AnswerRelevancy: 0.7 } },
        { id: '2', language: 'en', latencyMs: 12, metrics: { Faithfulness: 0.6 } },
        { id: '3', language: 'zh', latencyMs: 15, metrics: { /* missing Faithfulness */ AnswerRelevancy: 0.8 } },
      ], kpis: { Faithfulness: 0.55 }, counts: { total: 3 } } as any,
      b: { items: [
        { id: '4', language: 'en', latencyMs: 9, metrics: { Faithfulness: 0.4, AnswerRelevancy: 0.6 } },
        { id: '5', language: 'zh', latencyMs: 13, metrics: { Faithfulness: 0.65, AnswerRelevancy: 0.75 } },
      ], kpis: { Faithfulness: 0.525 }, counts: { total: 2 } } as any,
    })
    set.setSelectedRuns(['a', 'b'])
  })

  it('exports CSV with samples and naPct columns per run', async () => {
    render(<CompareView />)
    fireEvent.click(screen.getByLabelText('export-compare-csv'))
    expect(exportCsvSpy).toHaveBeenCalled()
    const [filename, rows, meta] = exportCsvSpy.mock.calls[0]
    expect(filename).toBe('compare.csv')
    expect(Array.isArray(rows)).toBe(true)
    // Find a Faithfulness row and check presence of samples and naPct keys
    const faith = (rows as any[]).find((r) => r.metric === 'Faithfulness')
    expect(faith).toBeTruthy()
    expect(Object.keys(faith)).toEqual(expect.arrayContaining([
      'a.mean','a.median','a.p50','a.p90','a.deltaAbs','a.deltaPct','a.samples','a.naPct',
      'b.mean','b.median','b.p50','b.p90','b.deltaAbs','b.deltaPct','b.samples','b.naPct',
    ]))
    expect(meta?.filters).toBeDefined()
    expect(meta?.thresholds).toBeDefined()
    expect(meta?.branding?.title).toBe('Compare Report')
  })

  it('exports XLSX with same schema (multi-sheet)', async () => {
    render(<CompareView />)
    await fireEvent.click(screen.getByLabelText('export-compare-xlsx'))
    expect(exportMultiXlsxSpy).toHaveBeenCalled()
    const [filename, sheets, meta] = exportMultiXlsxSpy.mock.calls[0]
    expect(filename).toBe('compare.xlsx')
    // Ensure sheets contain data and overview
    const dataSheet = (sheets as any[]).find((s) => s.name === 'data')
    const overviewSheet = (sheets as any[]).find((s) => s.name === 'overview')
    expect(dataSheet).toBeTruthy()
    expect(overviewSheet).toBeTruthy()
    const faith = (dataSheet.rows as any[]).find((r) => r.metric === 'Faithfulness')
    expect(faith).toBeTruthy()
    expect(meta?.branding?.brand).toBe('Insights Portal')
  })

  it('cohort panel exports CSV', async () => {
    render(<CompareView />)
    // Expand cohort panel
    fireEvent.click(screen.getByText('Cohort Compare'))
    // Click export CSV inside the cohort panel
    fireEvent.click(screen.getByLabelText('export-cohort-csv'))
    expect(exportCsvSpy).toHaveBeenCalled()
  const call = exportCsvSpy.mock.calls.find((c: any) => String(c[0]).startsWith('cohort_'))
    expect(call).toBeTruthy()
    expect(call?.[2]?.branding?.title).toBe('Compare Report')
  })
})
