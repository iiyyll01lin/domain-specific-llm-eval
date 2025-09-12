import React from 'react'

export type FeatureFlags = {
  kgVisualization: boolean
  multiRunCompare: boolean
  experimentalMetricViz: boolean
  lifecycleConsole: boolean
}

const defaultFlags: FeatureFlags = {
  kgVisualization: false,
  multiRunCompare: false,
  experimentalMetricViz: false,
  lifecycleConsole: false,
}

const Ctx = React.createContext<FeatureFlags>(defaultFlags)

interface FFProps { children: React.ReactNode }
export const FeatureFlagsProvider: React.FC<FFProps> = ({ children }: FFProps) => {
  const [flags, setFlags] = React.useState<FeatureFlags>(defaultFlags)

  React.useEffect(() => {
    let cancelled = false
    ;(async () => {
      try {
        const res = await fetch('/config/feature-flags.json', { cache: 'no-store' })
        if (!res.ok) return
        const data = await res.json()
        if (!cancelled) setFlags({ ...defaultFlags, ...data })
      } catch {
        // silent fallback
      }
    })()
    return () => { cancelled = true }
  }, [])

  return <Ctx.Provider value={flags}>{children}</Ctx.Provider>
}

export function useFeatureFlags() {
  return React.useContext(Ctx)
}
