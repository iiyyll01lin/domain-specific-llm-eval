## Feature Flags Configuration API (Spec Phase)

Endpoint: GET /config/feature-flags

Purpose:
Provide UI with a stable, cache-aware JSON describing frontend‑toggleable capabilities and experimental modules. Designed for CDN caching (short TTL) and graceful fallback.

Response Schema (JSON):
```
{
	"kgVisualization": true,              // Enables KG summary & future graph pane
	"multiRunCompare": false,            // Enables multi-run comparison overlay
	"experimentalMetricViz": false,      // Enables experimental metric visualizers
	"lifecycleConsole": true,            // Enables lifecycle console routes
	"refreshIntervalSeconds": 300,       // Recommended client re-fetch cadence
	"schemaVersion": "v1",              // Contract version for backward compatibility
	"generatedAt": "2025-09-10T09:00:00Z",
	"overrides": {                       // Optional per-env or rollout keys
		"tenant-default": { "multiRunCompare": true }
	}
}
```

Contract Rules:
- Missing flag ⇒ client defaults to `false`.
- Unknown additional keys must be ignored (forward compatibility).
- `schemaVersion` change MAY introduce new flags; removal of existing flags requires deprecation window ≥1 release.

Caching:
- Recommended headers: `Cache-Control: public, max-age=60, stale-while-revalidate=120`.
- UI may include ETag and send `If-None-Match`; server should return 304 when unchanged.

Error Handling:
- 5xx or network error ⇒ UI falls back to built-in defaults (all false except lifecycleConsole may remain cached last-known state).
- 4xx (e.g., 403) ⇒ treat as all false; surface a non-blocking telemetry warning.

Security:
- Phase 1: public (no auth) read-only; values are non-sensitive toggles.
- Phase 2: optional signed response (HMAC header `x-flags-signature`) for tamper detection when flags influence security posture.

Operational Considerations:
- Flags stored in config map / environment and hot-reload with process signal (SIGHUP) or periodic poll of central config service.
- Provide CLI admin helper (future) to render current effective flag set.

Test Cases (Planned):
1. Default fetch returns valid JSON with required keys.
2. Unknown key doesn’t break client (add `futureFlag` on server, ignored by UI).
3. 304 Not Modified path uses cached value.
4. 5xx fallback applies safe defaults.
5. ETag changes when any flag boolean flips.

