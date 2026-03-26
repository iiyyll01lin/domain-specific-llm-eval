import type { LifecycleConfig } from './types'

declare global {
  interface Window {
    RAG_EVAL_PORTAL_CONFIG?: {
      services?: {
        ingestion?: string
        processing?: string
        testset?: string
        eval?: string
        reporting?: string
        adapter?: string
        kg?: string
      }
      polling?: {
        intervalMs?: number
        requestTimeoutMs?: number
      }
    }
  }
}

const DEFAULT_CONFIG: LifecycleConfig = {
  ingestionBaseUrl: 'http://localhost:8001',
  processingBaseUrl: 'http://localhost:8002',
  testsetBaseUrl: 'http://localhost:8003',
  evalBaseUrl: 'http://localhost:8004',
  reportingBaseUrl: 'http://localhost:8005',
  adapterBaseUrl: 'http://localhost:8006',
  kgBaseUrl: 'http://localhost:8007',
  pollIntervalMs: 10_000,
  requestTimeoutMs: 10_000,
}

const envIngestion = typeof import.meta !== 'undefined' ? (import.meta as any)?.env?.VITE_INGESTION_BASE : undefined
const envProcessing = typeof import.meta !== 'undefined' ? (import.meta as any)?.env?.VITE_PROCESSING_BASE : undefined
const envTestset = typeof import.meta !== 'undefined' ? (import.meta as any)?.env?.VITE_TESTSET_BASE : undefined
const envEval = typeof import.meta !== 'undefined' ? (import.meta as any)?.env?.VITE_EVAL_BASE : undefined
const envReporting = typeof import.meta !== 'undefined' ? (import.meta as any)?.env?.VITE_REPORTING_BASE : undefined
const envAdapter = typeof import.meta !== 'undefined' ? (import.meta as any)?.env?.VITE_ADAPTER_BASE : undefined
const envKg = typeof import.meta !== 'undefined' ? (import.meta as any)?.env?.VITE_KG_BASE : undefined
const envInterval = typeof import.meta !== 'undefined' ? Number((import.meta as any)?.env?.VITE_LIFECYCLE_POLL_MS) : NaN
const envTimeout = typeof import.meta !== 'undefined' ? Number((import.meta as any)?.env?.VITE_LIFECYCLE_TIMEOUT_MS) : NaN

const globalCfg = typeof window !== 'undefined' ? window.RAG_EVAL_PORTAL_CONFIG : undefined

const config: LifecycleConfig = {
  ingestionBaseUrl: sanitizeBase(globalCfg?.services?.ingestion) || sanitizeBase(envIngestion) || DEFAULT_CONFIG.ingestionBaseUrl,
  processingBaseUrl: sanitizeBase(globalCfg?.services?.processing) || sanitizeBase(envProcessing) || DEFAULT_CONFIG.processingBaseUrl,
  testsetBaseUrl: sanitizeBase(globalCfg?.services?.testset) || sanitizeBase(envTestset) || DEFAULT_CONFIG.testsetBaseUrl,
  evalBaseUrl: sanitizeBase(globalCfg?.services?.eval) || sanitizeBase(envEval) || DEFAULT_CONFIG.evalBaseUrl,
  reportingBaseUrl: sanitizeBase(globalCfg?.services?.reporting) || sanitizeBase(envReporting) || DEFAULT_CONFIG.reportingBaseUrl,
  adapterBaseUrl: sanitizeBase(globalCfg?.services?.adapter) || sanitizeBase(envAdapter) || DEFAULT_CONFIG.adapterBaseUrl,
  kgBaseUrl: sanitizeBase(globalCfg?.services?.kg) || sanitizeBase(envKg) || DEFAULT_CONFIG.kgBaseUrl,
  pollIntervalMs: resolveNumber(DEFAULT_CONFIG.pollIntervalMs, globalCfg?.polling?.intervalMs, envInterval),
  requestTimeoutMs: resolveNumber(DEFAULT_CONFIG.requestTimeoutMs, globalCfg?.polling?.requestTimeoutMs, envTimeout),
}

export function getLifecycleConfig(): LifecycleConfig {
  return config
}

function sanitizeBase(input?: string | null): string | undefined {
  if (!input) return undefined
  const trimmed = input.trim()
  if (!trimmed) return undefined
  return trimmed.replace(/\/$/, '')
}

function resolveNumber(fallback: number, ...candidates: Array<number | undefined | null>): number {
  for (const candidate of candidates) {
    if (typeof candidate === 'number' && Number.isFinite(candidate) && candidate > 0) {
      return candidate
    }
  }
  return fallback
}
