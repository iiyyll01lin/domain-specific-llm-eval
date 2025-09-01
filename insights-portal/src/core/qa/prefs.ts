// QA preferences persistence helpers. Comments in English only.

const KEY_BOOKMARKS = 'portal.qa.bookmarks'
const KEY_VISIBLE_COLS = 'portal.qa.visibleCols'
const KEY_VISIBLE_METRICS = 'portal.qa.visibleMetrics'

export type VisibleCols = Record<string, boolean>
export type VisibleMetrics = Record<string, boolean>

export function loadBookmarks(): Set<string> {
  try {
    const raw = localStorage.getItem(KEY_BOOKMARKS)
    if (!raw) return new Set()
    return new Set(JSON.parse(raw) as string[])
  } catch {
    return new Set()
  }
}

export function saveBookmarks(bm: Set<string>) {
  try { localStorage.setItem(KEY_BOOKMARKS, JSON.stringify(Array.from(bm))) } catch {/* ignore */}
}

export function loadVisibleCols(defaults: VisibleCols = { question: true, answer: false, reference: false }): VisibleCols {
  try {
    const raw = localStorage.getItem(KEY_VISIBLE_COLS)
    return raw ? (JSON.parse(raw) as VisibleCols) : defaults
  } catch {
    return defaults
  }
}

export function saveVisibleCols(v: VisibleCols) {
  try { localStorage.setItem(KEY_VISIBLE_COLS, JSON.stringify(v)) } catch {/* ignore */}
}

export function loadVisibleMetrics(): VisibleMetrics {
  try {
    const raw = localStorage.getItem(KEY_VISIBLE_METRICS)
    return raw ? (JSON.parse(raw) as VisibleMetrics) : {}
  } catch {
    return {}
  }
}

export function saveVisibleMetrics(v: VisibleMetrics) {
  try { localStorage.setItem(KEY_VISIBLE_METRICS, JSON.stringify(v)) } catch {/* ignore */}
}
