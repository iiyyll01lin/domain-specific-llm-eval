import pw from '@playwright/test'
const { test, expect } = pw

const enabled = process.env.PW_E2E_ENABLED === '1'

test.describe('QA export and interactions', () => {
  test.skip(!enabled, 'Playwright E2E disabled by default. Set PW_E2E_ENABLED=1 to run.')

  test('navigate to QA, search filters rows, export CSV/XLSX triggers downloads', async ({ page }) => {
    await page.goto('/?sample=run_minimal')
    await page.getByTestId('nav-qa').click()
    await page.getByTestId('qa-table').waitFor()

    // Search with a term that likely yields zero
    const search = page.getByTestId('qa-search')
    await search.fill('no_such_term_zzzzzz')
    // Give time for filtering
    await page.waitForTimeout(100)
    const rowCount = await page.locator('[data-testid^="qa-row-"]').count()
    expect(rowCount).toBe(0)

    // Clear search
    await search.fill('')
    await page.waitForTimeout(50)
    const rowCount2 = await page.locator('[data-testid^="qa-row-"]').count()
    expect(rowCount2).toBeGreaterThan(0)

    // Capture CSV download
    const [csv] = await Promise.all([
      page.waitForEvent('download'),
      page.getByTestId('qa-export-csv').click(),
    ])
    const csvName = csv.suggestedFilename()
    expect(csvName.toLowerCase()).toContain('.csv')

    // Capture XLSX download
    const [xlsx] = await Promise.all([
      page.waitForEvent('download'),
      page.getByTestId('qa-export-xlsx').click(),
    ])
    const xlsxName = xlsx.suggestedFilename()
    expect(xlsxName.toLowerCase()).toContain('.xlsx')
  })
})
