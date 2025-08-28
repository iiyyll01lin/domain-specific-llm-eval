import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

const resources = {
  'zh-TW': {
    translation: {
      appTitle: 'Insights Portal',
      nav: { executive: '總覽', qa: 'QA', analytics: '分析' },
  overview: { sortByGap: '依與門檻差距排序', pickHint: '請選擇 JSON/CSV 檔以載入 run。' },
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
      nav: { executive: 'Executive', qa: 'QA', analytics: 'Analytics' },
  overview: { sortByGap: 'Sort by gap to thresholds', pickHint: 'Pick a JSON/CSV file to load a run.' },
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
