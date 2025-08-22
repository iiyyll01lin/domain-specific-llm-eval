import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

const resources = {
  'zh-TW': {
    translation: {
      appTitle: 'Insights Portal',
      nav: { executive: '總覽', qa: 'QA', analytics: '分析' },
      metrics: {
        ContextPrecision: { label: 'Context Precision' },
        ContextRecall: { label: 'Context Recall' },
        Faithfulness: { label: 'Faithfulness' },
        AnswerRelevancy: { label: 'Answer Relevancy' },
        AnswerSimilarity: { label: 'Answer Similarity' },
        ContextualKeywordMean: { label: 'Contextual Keyword Mean' },
      },
    },
  },
  'en-US': {
    translation: {
      appTitle: 'Insights Portal',
      nav: { executive: 'Executive', qa: 'QA', analytics: 'Analytics' },
      metrics: {
        ContextPrecision: { label: 'Context Precision' },
        ContextRecall: { label: 'Context Recall' },
        Faithfulness: { label: 'Faithfulness' },
        AnswerRelevancy: { label: 'Answer Relevancy' },
        AnswerSimilarity: { label: 'Answer Similarity' },
        ContextualKeywordMean: { label: 'Contextual Keyword Mean' },
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
