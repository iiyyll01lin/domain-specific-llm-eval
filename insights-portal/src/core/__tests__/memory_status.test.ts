import { describe, it, expect } from 'vitest'
import { sampleMemory } from '@/utils/memory'

describe('memory status (non-fatal in unsupported env)', () => {
  it('returns undefined or object', () => {
    const m = sampleMemory()
    if (m) {
      expect(m.usedMB).toBeGreaterThan(0)
    } else {
      expect(m).toBeUndefined()
    }
  })
})
