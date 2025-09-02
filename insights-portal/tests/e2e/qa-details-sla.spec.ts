import pw from '@playwright/test'
const { test, expect } = pw

const enabled = process.env.PW_E2E_ENABLED === '1'

test.describe('QA Row Details SLA (dev)', () => {
  test.skip(!enabled, 'Playwright E2E disabled by default. Set PW_E2E_ENABLED=1 to run.')

  test('opens row details within 200ms for long context fixture', async ({ page }) => {
  await page.goto('/?sample=run_minimal')
  // Navigate to QA via stable test id and wait for table
  await page.getByTestId('nav-qa').click()
  await page.getByTestId('qa-table').waitFor()
  // Open the first visible row's details using stable row test id
  const firstRow = page.getByTestId('qa-row-0')
  await firstRow.waitFor()
  const t0 = Date.now()
  await firstRow.getByRole('button', { name: 'Details' }).click()
  await page.getByTestId('qa-details').waitFor()
  const t1 = Date.now()
  expect(t1 - t0).toBeLessThanOrEqual(200)
  })
})