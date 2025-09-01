// Stable test ids for Playwright/RTL. Comments in English only.

export const TID = {
  nav: {
    executive: 'nav-executive',
    qa: 'nav-qa',
    analytics: 'nav-analytics',
  },
  qa: {
    table: 'qa-table',
    row: (idx: number) => `qa-row-${idx}`,
    search: 'qa-search',
    detailsBtn: (id: string | number) => `qa-details-btn-${id}`,
    detailsDrawer: 'qa-details',
    detailsClose: 'qa-details-close',
  },
  dev: {
    runBench: 'dev-bench-run-all',
    matrix: 'dev-bench-matrix',
  },
} as const

export type TestIds = typeof TID
