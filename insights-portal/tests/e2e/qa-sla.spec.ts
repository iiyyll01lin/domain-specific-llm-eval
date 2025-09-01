import pw from '@playwright/test'
const { test, expect } = pw

const enabled = process.env.PW_E2E_ENABLED === '1'

test.describe('QA details SLA using stable test ids', () => {
  test.skip(!enabled, 'Playwright E2E disabled by default. Set PW_E2E_ENABLED=1 to run.')

  test('first open ≤200ms on long context sample', async ({ page }) => {
    await page.goto('/?sample=run_minimal')
    // Go to QA
    await page.getByTestId('nav-qa').click()
    await page.getByTestId('qa-table').waitFor()
    // Click first visible row details
    const firstRow = page.getByTestId('qa-row-0')
    await firstRow.waitFor()
    const t0 = Date.now()
    await firstRow.getByRole('button', { name: 'Details' }).click()
    await page.getByTestId('qa-details').waitFor()
    const dt = Date.now() - t0
    expect(dt).toBeLessThanOrEqual(200)
  })
})
