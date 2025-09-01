import pw from '@playwright/test'
const { test, expect } = pw

const enabled = process.env.PW_E2E_ENABLED === '1'

test.describe('QA Row Details SLA (dev)', () => {
  test.skip(!enabled, 'Playwright E2E disabled by default. Set PW_E2E_ENABLED=1 to run.')

  test('opens row details within 200ms for long context fixture', async ({ page }) => {
    await page.goto('/')
    // Navigate to QA
    await page.getByRole('button', { name: /qa/i }).click()
    // Open the first row details
    const t0 = Date.now()
    await page.getByRole('button', { name: /details/i }).first().click()
    await page.getByText(/contexts/i).first().waitFor()
    const t1 = Date.now()
    expect(t1 - t0).toBeLessThan(200)
  })
})