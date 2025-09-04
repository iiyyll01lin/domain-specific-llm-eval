import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

const resources = {
  'zh-TW': {
    translation: {
      appTitle: 'Insights Portal',
    nav: { executive: '總覽', qa: 'QA', analytics: '分析', compare: '比較' },
  overview: { sortByGap: '依與門檻差距排序', pickHint: '請選擇 JSON/CSV 檔以載入 run。' },
  analytics: { title: '分析分佈', mode: '模式', cohort: '分組', metric: '指標', exportCsv: '匯出 CSV', exportXlsx: '匯出 XLSX', exportPng: '匯出 PNG', legend: '圖例', compareTable: '比較 (指標: {{metric}})' },
  compare: { title: '比較多個執行', baseline: '基準', exportCsv: '匯出 CSV', exportXlsx: '匯出 XLSX', cohortBtn: '群組比較', cohortMetric: '指標', cohortGroup: '群組', cohortDelta: 'Δ 相對基準', cohortExpand: '展開/收合' },
      metrics: {
  ContextPrecision: { label: 'Context Precision', help: '檢索內容的精準度' },
  ContextRecall: { label: 'Context Recall', help: '檢索內容的召回率' },
  Faithfulness: { label: 'Faithfulness', help: '答案與來源一致性' },
  AnswerRelevancy: { label: 'Answer Relevancy', help: '答案與問題的相關性' },
  AnswerSimilarity: { label: 'Answer Similarity', help: '答案與參考答案相似度' },
  ContextualKeywordMean: { label: 'Contextual Keyword Mean', help: '上下文關鍵字平均分' },
      },
    },
  },
  'en-US': {
    translation: {
      appTitle: 'Insights Portal',
    nav: { executive: 'Executive', qa: 'QA', analytics: 'Analytics', compare: 'Compare' },
  overview: { sortByGap: 'Sort by gap to thresholds', pickHint: 'Pick a JSON/CSV file to load a run.' },
  analytics: { title: 'Analytics Distribution', mode: 'Mode', cohort: 'Cohort', metric: 'Metric', exportCsv: 'Export CSV', exportXlsx: 'Export XLSX', exportPng: 'Export PNG', legend: 'Legend', compareTable: 'Compare (metric: {{metric}})' },
  compare: { title: 'Compare Runs', baseline: 'Baseline', exportCsv: 'Export CSV', exportXlsx: 'Export XLSX', cohortBtn: 'Cohort Compare', cohortMetric: 'Metric', cohortGroup: 'Group', cohortDelta: 'Δ vs base', cohortExpand: 'Toggle' },
      metrics: {
  ContextPrecision: { label: 'Context Precision', help: 'Precision of retrieved contexts' },
  ContextRecall: { label: 'Context Recall', help: 'Recall of retrieved contexts' },
  Faithfulness: { label: 'Faithfulness', help: 'Answer grounded in sources' },
  AnswerRelevancy: { label: 'Answer Relevancy', help: 'Answer relevance to the question' },
  AnswerSimilarity: { label: 'Answer Similarity', help: 'Similarity against reference answer' },
  ContextualKeywordMean: { label: 'Contextual Keyword Mean', help: 'Contextual keyword average score' },
      },
    },
  },
}

i18n.use(initReactI18next).init({
  resources,
  lng: (localStorage.getItem('portal.lang') as 'zh-TW'|'en-US') || 'zh-TW',
  fallbackLng: 'en-US',
  interpolation: { escapeValue: false },
})

export default i18n
