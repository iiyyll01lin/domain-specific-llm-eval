import React from 'react'
import { useFeatureFlags } from '@/core/featureFlags'
import { DocumentsPanel } from '@/app/lifecycle/DocumentsPanel'
import { ProcessingPanel } from '@/app/lifecycle/ProcessingPanel'

const Placeholder: React.FC<{ title: string }> = ({ title }) => (
  <div style={{ padding: 24 }}>
    <h2>{title}</h2>
    <p>Coming soon – spike placeholder component.</p>
  </div>
)

export const LifecycleRoutes: React.FC = () => {
  const flags = useFeatureFlags()
  if (!flags.lifecycleConsole) return <div style={{ padding: 24 }}>Lifecycle console disabled.</div>
  return (
    <div style={{ display: 'grid', gap: 16 }}>
      <DocumentsPanel />
      <ProcessingPanel />
      <Placeholder title="Knowledge Graph" />
      <Placeholder title="Testsets" />
      <Placeholder title="Evaluations" />
      <Placeholder title="Reports" />
      <Placeholder title="Admin" />
    </div>
  )
}
