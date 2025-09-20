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
  lifecycle: {
    shared: {
      refresh: '刷新',
      lastUpdated: '上次更新 {{time}}',
      awaiting: '等待第一筆資料…',
    },
    documents: {
      title: '文件',
      subtitle: 'KM 文件佇列與去重狀態。',
      loading: '載入文件中…',
      empty: '尚無文件已匯入。',
      columns: { kmId: 'KM ID', version: '版本', checksum: '雜湊', size: '大小', status: '狀態', lastEvent: '最新事件' },
      status: { queued: '排隊中', running: '處理中', completed: '完成', duplicate: '重複', error: '錯誤' },
      error: '文件載入失敗：',
    },
    processing: {
      title: '處理作業',
      subtitle: '分塊、嵌入與清單生成狀態。',
      loading: '載入處理作業中…',
      empty: '目前沒有處理作業。',
      columns: {
        jobId: '作業 ID',
        documentId: '文件 ID',
        progress: '進度',
        status: '狀態',
        chunks: '區塊數',
        profileHash: '設定雜湊',
        started: '開始時間',
        updated: '更新時間',
        elapsed: '耗時',
        sla: 'SLA',
      },
      sla: { ok: 'SLA 正常', breached: 'SLA 超時' },
      status: { queued: '排隊中', running: '處理中', completed: '完成', error: '錯誤' },
      error: '處理作業載入失敗：',
    },
  },
      metrics: {
  ContextPrecision: { label: 'Context Precision', help: '檢索內容的精準度' },
  ContextRecall: { label: 'Context Recall', help: '檢索內容的召回率' },
  Faithfulness: { label: 'Faithfulness', help: '答案與來源一致性' },
  AnswerRelevancy: { label: 'Answer Relevancy', help: '答案與問題的相關性' },
  AnswerSimilarity: { label: 'Answer Similarity', help: '答案與參考答案相似度' },
  ContextualKeywordMean: { label: 'Contextual Keyword Mean', help: '上下文關鍵字平均分' },
      },

      errors: {
        jsonParse: 'JSON 解析失敗：{{file}}（位移 {{offset}}，第 {{line}} 行，第 {{column}} 欄）。',
        csvParse: 'CSV 解析失敗：{{file}}（第 {{row}} 列，位移 {{offset}}）。',
        workerRuntime: '工作緒執行錯誤：{{message}}',
        unknown: '發生未知錯誤。',
        noCompatibleFiles: '沒有相容的檔案可供載入。',

        enterMetricKey: '請輸入 metric key',
        metricExists: '此 metric 已存在',
        range01: 'warning/critical 必須介於 0~1',
        metricSanitized: '已將輸入清理為 "{{key}}"'
      },

      thresholds: {
        title: 'Thresholds',
        reset: '重置',
        addMetric: '+ Add Metric',
        cancelAdd: '取消新增',
        confirmAdd: '加入',
        key: 'key',
        warning: 'warning',
        critical: 'critical',
        placeholderKey: '例如：AnswerFluency'
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
  lifecycle: {
    shared: {
      refresh: 'Refresh',
      lastUpdated: 'Last updated {{time}}',
      awaiting: 'Awaiting first sync…',
    },
    documents: {
      title: 'Documents',
      subtitle: 'Knowledge management ingestion queue and dedupe status.',
      loading: 'Loading documents…',
      empty: 'No documents ingested yet.',
      columns: { kmId: 'KM ID', version: 'Version', checksum: 'Checksum', size: 'Size', status: 'Status', lastEvent: 'Last Event' },
      status: { queued: 'Queued', running: 'Running', completed: 'Completed', duplicate: 'Duplicate', error: 'Error' },
      error: 'Failed to load documents:',
    },
    processing: {
      title: 'Processing Jobs',
      subtitle: 'Chunking, embeddings, and manifest generation status.',
      loading: 'Loading processing jobs…',
      empty: 'No active processing jobs.',
      columns: {
        jobId: 'Job ID',
        documentId: 'Document ID',
        progress: 'Progress',
        status: 'Status',
        chunks: 'Chunks',
        profileHash: 'Profile Hash',
        started: 'Started',
        updated: 'Updated',
        elapsed: 'Elapsed',
        sla: 'SLA',
      },
      sla: { ok: 'On-track', breached: 'SLA exceeded' },
      status: { queued: 'Queued', running: 'Running', completed: 'Completed', error: 'Error' },
      error: 'Failed to load processing jobs:',
    },
  },
      metrics: {
  ContextPrecision: { label: 'Context Precision', help: 'Precision of retrieved contexts' },
  ContextRecall: { label: 'Context Recall', help: 'Recall of retrieved contexts' },
  Faithfulness: { label: 'Faithfulness', help: 'Answer grounded in sources' },
  AnswerRelevancy: { label: 'Answer Relevancy', help: 'Answer relevance to the question' },
  AnswerSimilarity: { label: 'Answer Similarity', help: 'Similarity against reference answer' },
  ContextualKeywordMean: { label: 'Contextual Keyword Mean', help: 'Contextual keyword average score' },
      },
      errors: {
        jsonParse: 'JSON parse failed: {{file}} (offset {{offset}}, line {{line}}, column {{column}}).',
        csvParse: 'CSV parse failed: {{file}} (row {{row}}, offset {{offset}}).',
        workerRuntime: 'Worker runtime error: {{message}}',
        unknown: 'Unknown error occurred.',
        noCompatibleFiles: 'No compatible files.',

        enterMetricKey: 'Please enter a metric key',
        metricExists: 'This metric already exists',
        range01: 'warning/critical must be between 0 and 1',
        metricSanitized: 'Input sanitized to "{{key}}"'
      },

       thresholds: {
        title: 'Thresholds',
        reset: 'Reset',
        addMetric: '+ Add Metric',
        cancelAdd: 'Cancel',
        confirmAdd: 'Add',
        key: 'key',
        warning: 'warning',
        critical: 'critical',
        placeholderKey: 'e.g., AnswerFluency'
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
