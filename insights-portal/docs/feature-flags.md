# Feature Flags Architecture (Portal)

Status: Spec (no backend code yet)

## Purpose
Provide a lightweight, cache-aware mechanism to toggle optional or experimental UI capabilities without forcing redeploys.

## Current Flags (schemaVersion v1)
| Flag                  | Default | Description                                             | Related UI-FR             |
|-----------------------|---------|---------------------------------------------------------|---------------------------|
| kgVisualization       | false   | Enables knowledge graph summary & (future) visual panel | UI-FR-016..018            |
| multiRunCompare       | false   | Enables multi-run comparison overlays                   | UI-FR-029                 |
| experimentalMetricViz | false   | Enables experimental metric visualizers sandbox         | UI-FR-044 (sandbox)       |
| lifecycleConsole      | false   | Enables lifecycle console placeholder routes            | UI-FR-009..032 (indirect) |

## Fetch Flow
1. App boot mounts `FeatureFlagsProvider`.
2. Provider GET /config/feature-flags (JSON) (no auth Phase 1).
3. Merge server result onto default object; missing keys remain default.
4. UI rerenders consumers via React context.

Fallback Logic:
- Network / 5xx: keep defaults (all false) and log `console.debug` once.
- Partial response: ignore unknown keys, preserve future compatibility.

## Caching & Refresh
- Initial load only; optional refresh on interval = `refreshIntervalSeconds` if present (Phase 2).
- Recommended server headers: `Cache-Control: public, max-age=60, stale-while-revalidate=120`.

## Extensibility Pattern
1. Add flag to interface `FeatureFlags` in `src/core/featureFlags.tsx`.
2. Document in this file (table above).
3. Guard UI feature with `const { flagName } = useFeatureFlags()`.
4. Avoid deep conditional trees; prefer early returns or lazy imports:
```ts
if (kgVisualization) {
  const CytoscapeView = React.lazy(() => import('../viz/KgView'))
  // render suspense boundary
}
```

## Security Considerations
- Flags are non-sensitive in Phase 1. Do not gate security-critical paths.
- Phase 2: If flag influences data exposure, sign payload (HMAC) and verify in client (optional).

## Testing Strategy (See separate test plan doc)
- Unit test: provider returns defaults when fetch mocked to 500.
- Unit test: new flag key merges without affecting existing ones.
- E2E (future): toggle flag via test server fixture and assert component presence.

## Open Questions
- Should we allow URL param overrides for rapid demos? (Proposed: `?ff=kgVisualization,multiRunCompare` in dev only.)
- Add localStorage caching to survive network blips? (Low priority.)

---
End of document.
