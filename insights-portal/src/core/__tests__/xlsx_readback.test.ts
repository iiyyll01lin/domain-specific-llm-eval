import { describe, it, expect } from 'vitest'
import * as XLSX from 'xlsx'
import { buildMultipleSheetsXlsxArray } from '@/core/exporter'

// Real XLSX read-back test (no download), using SheetJS to parse workbook and assert sheets

describe('XLSX real read-back (data/meta/branding/overview)', () => {
  it('creates workbook with expected sheets and first rows', async () => {
    const dataRows = [
      { metric: 'Faithfulness', 'runA.mean': 0.5, 'runB.mean': 0.6, 'runA.samples': 3, 'runB.samples': 2 },
      { metric: 'AnswerRelevancy', 'runA.mean': 0.7, 'runB.mean': 0.68, 'runA.samples': 3, 'runB.samples': 2 },
    ]
    const overviewRows = [
      { metric: 'Faithfulness', 'runA.n': 3, 'runA.naPct': 0, 'runB.n': 2, 'runB.naPct': 0 },
      { metric: 'AnswerRelevancy', 'runA.n': 3, 'runA.naPct': 0, 'runB.n': 2, 'runB.naPct': 0 },
    ]
    const meta = { timestamp: '2025-09-02T00:00:00Z', filters: { lang: 'en' }, thresholds: { Faithfulness: { warning: 0.3, critical: 0.2 } }, branding: { brand: 'Insights Portal', title: 'Compare Report', footer: 'offline' } }
    const bytes = await buildMultipleSheetsXlsxArray([
      { name: 'data', rows: dataRows },
      { name: 'overview', rows: overviewRows },
    ], meta)
    const wb = XLSX.read(bytes, { type: 'array' })
    expect(wb.SheetNames).toEqual(expect.arrayContaining(['data','overview','meta','branding']))
    const dataSheet = XLSX.utils.sheet_to_json<any>(wb.Sheets['data'])
    expect(dataSheet[0].metric).toBe('Faithfulness')
    const metaSheet = XLSX.utils.sheet_to_json<any>(wb.Sheets['meta'])
    const metaMap = Object.fromEntries(metaSheet.map((r: any) => [r.key, r.value]))
    expect(metaMap.timestamp).toBe('2025-09-02T00:00:00Z')
    const brandingSheet = XLSX.utils.sheet_to_json<any>(wb.Sheets['branding'])
    const brandMap = Object.fromEntries(brandingSheet.map((r: any) => [r.key, r.value]))
    expect(brandMap.brand).toBe('Insights Portal')
    expect(brandMap.title).toBe('Compare Report')
  })
})
