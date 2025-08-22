// Minimal YAML parsing for simple key:value and nested objects used for thresholds in config.yaml.
// This is intentionally lightweight to avoid adding a heavy dependency for v1.
// Supports:
// - indentation-based nesting with spaces (2 spaces typical)
// - string/number/boolean/null scalars
// - simple objects; lists are not needed for thresholds structure

export function parseSimpleYAML(yamlText: string): any {
  const lines = yamlText.replace(/\r\n?/g, '\n').split('\n')
  type Node = { indent: number; key?: string; value?: any; parent?: Node }
  const root: Node = { indent: -1, value: {} }
  let current: Node = root

  for (let raw of lines) {
    const line = raw.replace(/#.*$/, '').trimEnd()
    if (!line.trim()) continue
    const leading = raw.length - raw.trimStart().length
    while (current && leading <= current.indent) current = current.parent || root

    const kv = line.split(/:\s*/, 2)
    if (kv.length === 1) continue
    const key = kv[0].trim()
    const rest = kv[1]
    if (rest === '' || rest === undefined) {
      // start of nested object
      const node: Node = { indent: leading, key, value: {}, parent: current }
      ;(current.value ||= {})[key] = node.value
      current = node
    } else {
      const value = parseScalar(rest)
      ;(current.value ||= {})[key] = value
    }
  }

  return root.value
}

function parseScalar(text: string): any {
  const t = text.trim()
  if (t === 'null' || t === 'Null' || t === 'NULL' || t === '~') return null
  if (t === 'true' || t === 'True' || t === 'TRUE') return true
  if (t === 'false' || t === 'False' || t === 'FALSE') return false
  if (/^[-+]?[0-9]+(\.[0-9]+)?$/.test(t)) return Number(t)
  return t.replace(/^['"]|['"]$/g, '')
}

export type SimpleThresholds = {
  thresholds?: Record<string, number | { warning?: number; critical?: number }>
}

export function extractThresholdsFromConfig(config: any): Record<string, { warning: number; critical: number }> | undefined {
  // Look for common shapes: config.thresholds.metric.warning/critical or flat numbers requiring mapping.
  const src = config?.thresholds
  if (!src) return undefined
  const out: Record<string, { warning: number; critical: number }> = {}
  for (const [k, v] of Object.entries(src)) {
    if (v && typeof v === 'object' && ('warning' in (v as any) || 'critical' in (v as any))) {
      const wRaw = (v as any).warning
      const cRaw = (v as any).critical
      const w = wRaw != null ? Number(wRaw) : undefined
      const c = cRaw != null ? Number(cRaw) : undefined
      if (w != null && !Number.isNaN(w) && c != null && !Number.isNaN(c)) {
        out[k] = { warning: w, critical: c }
      } else if (w != null && !Number.isNaN(w)) {
        // If only warning provided, set critical slightly lower as fallback
        out[k] = { warning: w, critical: Math.max(0, w - 0.05) }
      } else if (c != null && !Number.isNaN(c)) {
        // If only critical provided, set warning slightly higher as fallback
        out[k] = { warning: Math.min(1, c + 0.05), critical: c }
      }
    } else if (typeof v === 'number') {
      // Flat numeric means warning=critical=v (or we can treat critical<=warning=v)
      out[k] = { warning: Number(v), critical: Number(v) }
    }
  }
  return Object.keys(out).length ? out : undefined
}
