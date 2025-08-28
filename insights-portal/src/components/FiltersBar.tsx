import React, { useMemo } from 'react';
import { buildActiveChips, ChipItem, FiltersState } from './filters/chips';

interface Props {
  metrics: string[]; // available metric keys in [0,1]
  filters: FiltersState;
  onChange: (next: FiltersState) => void;
  locale?: string;
}

// Lightweight slider component to avoid extra deps.
function RangeInput({
  value,
  min = 0,
  max = 1,
  step = 0.01,
  onChange,
}: {
  value: number | undefined;
  min?: number;
  max?: number;
  step?: number;
  onChange: (v: number | undefined) => void;
}) {
  return (
    <input
      type="range"
      min={min}
      max={max}
      step={step}
      value={value == null ? (min + max) / 2 : value}
      onChange={(e) => onChange(Number(e.target.value))}
      style={{ width: 120 }}
      aria-label="metric-range-slider"
    />
  );
}

export function FiltersBar({ metrics, filters, onChange, locale = 'en-US' }: Props) {
  const chips: ChipItem[] = useMemo(() => buildActiveChips(filters, locale), [filters, locale]);

  // Helpers to mutate state safely
  const updateMetricRange = (key: string, part: 'min' | 'max', val: number | undefined) => {
    const next: FiltersState = { ...filters, metrics: { ...(filters.metrics || {}) } };
    const current = next.metrics![key] || {};
    next.metrics![key] = { ...current, [part]: val };
    onChange(next);
  };

  const updateLatency = (part: 'min' | 'max', val: number | undefined) => {
    const next: FiltersState = { ...filters, latencyMs: { ...(filters.latencyMs || {}) } };
    (next.latencyMs as any)[part] = val;
    onChange(next);
  };

  const clearChip = (chip: ChipItem) => {
    if (chip.type === 'language') {
      onChange({ ...filters, language: undefined });
      return;
    }
    if (chip.type === 'latency') {
      onChange({ ...filters, latencyMs: undefined });
      return;
    }
    if (chip.type === 'metric' && chip.metricKey) {
      const nextMetrics = { ...(filters.metrics || {}) };
      delete nextMetrics[chip.metricKey];
      onChange({ ...filters, metrics: nextMetrics });
      return;
    }
  };

  const clearAll = () => {
    onChange({ language: undefined, latencyMs: undefined, metrics: {} });
  };

  return (
    <div style={{ display: 'grid', gap: 12 }}>
      {/* Language filter */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <label style={{ minWidth: 80 }}>Language</label>
        <input
          type="text"
          placeholder="e.g., zh, en"
          value={filters.language || ''}
          onChange={(e) => onChange({ ...filters, language: e.target.value || undefined })}
          aria-label="language-filter-input"
        />
      </div>

      {/* Latency range */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
        <label style={{ minWidth: 80 }}>Latency (ms)</label>
        <input
          type="number"
          placeholder="min"
          value={filters.latencyMs?.min ?? ''}
          onChange={(e) => updateLatency('min', e.target.value === '' ? undefined : Number(e.target.value))}
          aria-label="latency-min-input"
          style={{ width: 100 }}
        />
        <span>–</span>
        <input
          type="number"
          placeholder="max"
          value={filters.latencyMs?.max ?? ''}
          onChange={(e) => updateLatency('max', e.target.value === '' ? undefined : Number(e.target.value))}
          aria-label="latency-max-input"
          style={{ width: 100 }}
        />
      </div>

      {/* Metric sliders */}
      <div style={{ display: 'grid', gap: 6 }}>
        {metrics.map((m) => {
          const range = filters.metrics?.[m] || {};
          return (
            <div key={m} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <div style={{ minWidth: 160 }}>{m}</div>
              <RangeInput value={range.min} onChange={(v) => updateMetricRange(m, 'min', v)} />
              <span>–</span>
              <RangeInput value={range.max} onChange={(v) => updateMetricRange(m, 'max', v)} />
              <input
                type="number"
                placeholder="min"
                value={range.min ?? ''}
                onChange={(e) => updateMetricRange(m, 'min', e.target.value === '' ? undefined : Number(e.target.value))}
                aria-label={`metric-${m}-min-input`}
                style={{ width: 80 }}
              />
              <input
                type="number"
                placeholder="max"
                value={range.max ?? ''}
                onChange={(e) => updateMetricRange(m, 'max', e.target.value === '' ? undefined : Number(e.target.value))}
                aria-label={`metric-${m}-max-input`}
                style={{ width: 80 }}
              />
            </div>
          );
        })}
      </div>

      {/* Chips */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
        {chips.map((c) => (
          <button
            key={c.key}
            onClick={() => clearChip(c)}
            title="Click to clear this filter"
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: 6,
              padding: '2px 8px',
              borderRadius: 12,
              border: '1px solid var(--border-color, #444)',
              background: 'var(--chip-bg, #222)',
              color: 'var(--chip-fg, #ddd)',
            }}
          >
            <span>✕</span>
            <span>{c.label}</span>
          </button>
        ))}
        {chips.length > 0 && (
          <button onClick={clearAll} style={{ marginLeft: 8 }} aria-label="clear-all-filters">
            Clear All
          </button>
        )}
      </div>
    </div>
  );
}

export default FiltersBar;
