import React from 'react'
import { useFeatureFlags } from '@/core/featureFlags'
import { DocumentsPanel } from '@/app/lifecycle/DocumentsPanel'
import { ProcessingPanel } from '@/app/lifecycle/ProcessingPanel'
import { TestsetsPanel } from '@/app/lifecycle/TestsetsPanel'
import { EvalRunsPanel } from '@/app/lifecycle/EvalRunsPanel'
import { ReportsPanel } from '@/app/lifecycle/ReportsPanel'
import { KmSummariesPanel } from '@/app/lifecycle/KmSummariesPanel'
import { KgPanel } from '@/app/lifecycle/KgPanel'

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
      <KgPanel />
      <TestsetsPanel />
      <EvalRunsPanel />
      <ReportsPanel />
      <KmSummariesPanel />
      <Placeholder title="Admin" />
    </div>
  )
}
