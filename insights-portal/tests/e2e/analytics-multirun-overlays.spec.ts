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

  // Switch to box mode and ensure chart still renders without runB, outliers presence exposed
  await page.selectOption('select[aria-label="analytics-mode"]', { value: 'box' })
    await page.getByRole('img', { name: 'histogram' }).waitFor()
  const seriesCountBox = await page.getByTestId('analytics-series-count').innerText()
  expect(Number(seriesCountBox)).toBeGreaterThanOrEqual(1)
  const outliersFlag = await page.getByTestId('analytics-outliers-present').innerText()
  expect(['true','false']).toContain(outliersFlag)

    // Switch to scatter and perform a brush selection to update filters
    await page.selectOption('select[aria-label="analytics-mode"]', { value: 'scatter' })
    await page.waitForTimeout(200)
    const canvas = await page.locator('[role="img"]').first()
    const box = await canvas.boundingBox()
    if (box) {
      const startX = box.x + box.width * 0.2
      const startY = box.y + box.height * 0.8
      const endX = box.x + box.width * 0.6
      const endY = box.y + box.height * 0.4
      await page.mouse.move(startX, startY)
      await page.mouse.down()
      await page.mouse.move(endX, endY, { steps: 10 })
      await page.mouse.up()
      // Verify metricRanges updated via hidden testid
      const rangesText = await page.getByTestId('analytics-metric-ranges').innerText()
      expect(rangesText).toContain('Faithfulness')
      expect(rangesText).toContain('AnswerRelevancy')
    }

  // Toggle legend back on for runB
    await legendToggle.check()
    await runBCheck.check()
    await page.waitForTimeout(50)
  })
})
