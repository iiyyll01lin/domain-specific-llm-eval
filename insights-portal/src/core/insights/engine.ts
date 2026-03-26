import type { Thresholds, MetricKey } from '@/core/types'

export interface Insight {
  id: string
  title: string
  description: string
  evidence: Array<string>
  actions: Array<string>
  severity: 'info'|'warn'|'error'
}

interface GenerateParams {
  kpis: Record<string, number | undefined>
  thresholds: Thresholds
}

const pct = (v: number | undefined) => (v == null || Number.isNaN(v) ? 'N/A' : (v*100).toFixed(1)+'%')

export function generateInsights(params: GenerateParams): Insight[] {
  const { kpis, thresholds } = params
  const out: Insight[] = []
  const f = kpis['Faithfulness']
  const cp = kpis['ContextPrecision']
  const cr = kpis['ContextRecall']
  const ar = kpis['AnswerRelevancy']
  const keyword = kpis['ContextualKeywordMean']

  // Rule: High retrieval (precision+recall) but low faithfulness => hallucination risk
  if (num(cp) >= safeThresh(thresholds,'ContextPrecision','warning') && num(cr) >= safeThresh(thresholds,'ContextRecall','warning') && num(f) < safeThresh(thresholds,'Faithfulness','warning')) {
    out.push({
      id: 'hallucination_risk',
      title: 'Potential hallucination risk',
      description: 'Retrieval quality is strong but answers are not consistently grounded.',
      evidence: [
        `ContextPrecision ${pct(cp)} vs warning ${pct(safeThresh(thresholds,'ContextPrecision','warning'))}`,
        `ContextRecall ${pct(cr)} vs warning ${pct(safeThresh(thresholds,'ContextRecall','warning'))}`,
        `Faithfulness ${pct(f)} below warning ${pct(safeThresh(thresholds,'Faithfulness','warning'))}`,
      ],
      actions: [
        'Review grounding prompts and enforce citation requirements',
        'Audit retrieval context relevance for edge questions',
        'Introduce answer post-validation or re-ranking step'
      ],
      severity: 'warn'
    })
  }

  // Rule: Low keyword mean => retrieval query / keyword strategy issue
  if (keyword != null && keyword < safeThresh(thresholds,'ContextualKeywordMean','warning')) {
    out.push({
      id: 'keyword_low',
      title: 'Contextual keyword score below target',
      description: 'Low average contextual keyword score may reduce retrieval breadth or precision.',
      evidence: [
        `ContextualKeywordMean ${pct(keyword)} below warning ${pct(safeThresh(thresholds,'ContextualKeywordMean','warning'))}`
      ],
      actions: [
        'Refine query expansion or keyword extraction heuristics',
        'Enable lemmatization/stemming and domain synonym lists',
        'Add negative keyword filtering to reduce noise'
      ],
      severity: 'info'
    })
  }

  // Rule: Answer relevancy high but faithfulness low => grounding mismatch
  if (ar != null && f != null && ar >= safeThresh(thresholds,'AnswerRelevancy','warning') && f < safeThresh(thresholds,'Faithfulness','warning')) {
    out.push({
      id: 'relevancy_grounding_gap',
      title: 'Relevancy vs faithfulness gap',
      description: 'Answers are relevant to the query but may include unsupported claims.',
      evidence: [
        `AnswerRelevancy ${pct(ar)} >= warning`,
        `Faithfulness ${pct(f)} below warning`],
      actions: [
        'Introduce stricter answer validation against retrieved context',
        'Adjust prompt to discourage speculative content',
        'Consider retrieval augmentation with higher quality sources'
      ],
      severity: 'warn'
    })
  }

  return out.slice(0,3)
}

function safeThresh(th: Thresholds, key: MetricKey, level: 'warning'|'critical') {
  return th?.[key]?.[level] ?? (level === 'warning' ? 0.5 : 0.3)
}
function num(v: any): number { return typeof v === 'number' ? v : NaN }