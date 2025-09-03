import { describe, it, expect, vi, beforeEach } from 'vitest'

// Mock xlsx module to capture workbook content
const appended: Array<{ name: string; rows: any[] }> = []
vi.mock('xlsx', () => {
  return {
    utils: {
      json_to_sheet: (rows: any[]) => ({ __rows: rows }),
      book_new: () => ({ __sheets: [] }),
      book_append_sheet: (wb: any, ws: any, name: string) => { appended.push({ name, rows: ws.__rows || [] }) },
    },
  write: () => new Uint8Array([1,2,3]),
  }
})

import { exportTableToXLSX, exportMultipleSheetsXLSX } from '@/core/exporter'

describe('exporter XLSX meta and multiple sheets', () => {
  beforeEach(() => { appended.length = 0 })

  it('appends meta and branding sheets when meta provided', async () => {
    const rows = [{ a: 1 }]
    await exportTableToXLSX('t.xlsx', rows, { timestamp: '2025-09-02T00:00:00Z', filters: { lang: 'en' }, thresholds: { Faithfulness: { warning: 0.3, critical: 0.2 } }, branding: { brand: 'ACME', title: 'Eval', footer: '©' } })
    const names = appended.map((s) => s.name)
    expect(names).toContain('data')
    expect(names).toContain('meta')
    expect(names).toContain('branding')
  })

  it('supports multiple custom sheets and meta', async () => {
    await exportMultipleSheetsXLSX('multi.xlsx', [
      { name: 'data', rows: [{ x: 1 }] },
      { name: 'overview', rows: [{ y: 2 }] },
    ], { timestamp: '2025-09-02T00:00:00Z', branding: { brand: 'Portal' } })
    const names = appended.map((s) => s.name)
    expect(names).toEqual(expect.arrayContaining(['data','overview','meta','branding']))
    const overview = appended.find((s) => s.name === 'overview')
    expect(overview?.rows?.[0]?.y).toBe(2)
  })
})
