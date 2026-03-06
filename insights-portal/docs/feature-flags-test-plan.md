# Feature Flags Test Plan (Spec)

Scope: Validate correctness, resilience, and forward compatibility of the portal feature flag provider without relying on backend implementation.

## 1. Objectives
- Ensure default flags applied when network fails.
- Ensure server-provided partial overrides merge safely.
- Guarantee unknown future flags do not break consumer code.
- Validate re-render propagation to subscribed components.
- Confirm lazy-loaded visualization code not bundled when flag=false.

## 2. In-Scope Components
- `FeatureFlagsProvider` (context)
- `useFeatureFlags` hook consumers (sample stub component in tests)

## 3. Out-of-Scope
- Backend service availability (covered in integration tests later)
- Security signature (Phase 2 feature)

## 4. Test Categories
### 4.1 Unit Tests (Vitest)
| ID        | Title                        | Steps                                             | Expected                                  |
|-----------|------------------------------|---------------------------------------------------|-------------------------------------------|
| UT-FF-001 | Defaults on fetch error      | Mock fetch reject                                 | All flags false                           |
| UT-FF-002 | Merge server flags           | Mock fetch returns { kgVisualization:true }       | kgVisualization true; others false        |
| UT-FF-003 | Ignore unknown flag          | Mock fetch returns { futureFlag:true }            | futureFlag not in context; no crash       |
| UT-FF-004 | Re-render propagation        | After mock resolves, read hook value in component | Updated value observed                    |
| UT-FF-005 | Idempotent state             | Provide same payload twice (simulate 304)         | No additional re-render (optional assert) |
| UT-FF-006 | Refresh interval placeholder | Simulate adding refreshIntervalSeconds later      | Hook schedule set (pending Phase 2)       |

### 4.2 Bundle / Build Checks
| ID        | Title                                           | Method                                             | Expected                                      |
|-----------|-------------------------------------------------|----------------------------------------------------|-----------------------------------------------|
| BD-FF-001 | Cytoscape excluded by default                   | Analyze build stats (e.g., `vite build --analyze`) | No cytoscape chunk when kgVisualization=false |
| BD-FF-002 | Cytoscape included when dynamic import executed | Simulate flag true & dynamic import route          | Separate async chunk appears                  |

### 4.3 E2E (Future Playwright)
| ID         | Title                     | Scenario                                     | Expected                             |
|------------|---------------------------|----------------------------------------------|--------------------------------------|
| E2E-FF-001 | Enable KG viz flag        | Serve fixture JSON with kgVisualization true | KG panel placeholder visible         |
| E2E-FF-002 | Disable lifecycle console | lifecycleConsole false                       | Lifecycle routes hidden text appears |

## 5. Mocking Strategy
Use `vi.spyOn(global, 'fetch')` returning controlled `Response` objects; restore after each test. For error path, `Promise.reject(new Error('net'))`.

## 6. Risk & Mitigation
| Risk                     | Impact                 | Mitigation                                                |
|--------------------------|------------------------|-----------------------------------------------------------|
| Silent schema drift      | Wrong defaults applied | Assert presence of `schemaVersion` key                    |
| Over-fetching            | Performance overhead   | Add (Phase 2) interval logic with clearTimeout on unmount |
| Unexpected large payload | Memory overhead        | Enforce max JSON size (basic length check) in tests       |

## 7. Exit Criteria
- All UT-FF and BD-FF tests implemented & passing locally.
- E2E placeholders scripted (can be skipped until backend ready).

---
End of document.
