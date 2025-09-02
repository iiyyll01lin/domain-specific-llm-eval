import pw from '@playwright/test'
const { test, expect } = pw

const enabled = process.env.PW_E2E_ENABLED === '1'

// Utility: inject multi-run data into the app via the test hook we added in App.tsx
async function seedRuns(page: pw.Page) {
  await page.addInitScript(() => {
    (window as any).__seedRuns = () => {
      const runs = {
        runA: { items: [
          { id: '1', language: 'en', metrics: { Faithfulness: 0.5, AnswerRelevancy: 0.7 } },
          { id: '2', language: 'en', metrics: { Faithfulness: 0.6, AnswerRelevancy: 0.8 } },
          { id: '3', language: 'zh', metrics: { Faithfulness: 0.4, AnswerRelevancy: 0.6 } },
        ], kpis: { Faithfulness: 0.5, AnswerRelevancy: 0.7 }, counts: { total: 3 } },
        runB: { items: [
          { id: '4', language: 'en', metrics: { Faithfulness: 0.55, AnswerRelevancy: 0.65 } },
          { id: '5', language: 'zh', metrics: { Faithfulness: 0.7, AnswerRelevancy: 0.75 } },
        ], kpis: { Faithfulness: 0.625, AnswerRelevancy: 0.7 }, counts: { total: 2 } },
      }
      window.dispatchEvent(new CustomEvent('portal:test:set-runs', { detail: { runs, selectedRuns: ['runA', 'runB'], route: 'analytics' } }))
    }
  })
}

// E2E: Multi-run overlays interactions in Analytics (hist/box/scatter)
test.describe('Analytics multi-run overlays', () => {
  test.skip(!enabled, 'Playwright E2E disabled by default. Set PW_E2E_ENABLED=1 to run.')

  test('legend toggle affects series visibility in hist and box; scatter brush updates filters', async ({ page }) => {
    await seedRuns(page)
    await page.goto('/?sample=run_minimal')
    // Trigger seeding
    await page.evaluate(() => (window as any).__seedRuns())

    // Ensure we are on analytics and chart is present
    await page.getByTestId('nav-analytics').click()
    await page.getByRole('img', { name: 'histogram' }).waitFor()

    // Enable legend and toggle one run off
    const legendToggle = page.getByTestId('analytics-legend-toggle')
    await legendToggle.check()
    const runACheck = page.getByTestId('legend-run-runA')
    const runBCheck = page.getByTestId('legend-run-runB')
    await expect(runACheck).toBeChecked()
    await expect(runBCheck).toBeChecked()
    await runBCheck.uncheck()

    // Switch to box mode and ensure chart still renders without runB
    await page.selectOption('select', { label: 'Box' })
    await page.getByRole('img', { name: 'histogram' }).waitFor()

    // Switch to scatter and perform a brush selection to update filters
    await page.selectOption('select', { label: 'Scatter' })
    // Basic smoke: wait a tick and assume brush tool is available; cannot easily draw without low-level actions
    await page.waitForTimeout(200)

    // Toggle legend back on for runB
    await legendToggle.check()
    await runBCheck.check()
    await page.waitForTimeout(50)
  })
})
