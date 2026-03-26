# ADR-006: Event Schema Versioning Strategy

Status: Accepted  
Date: 2025-09-10  

## 1. Context
Multiple event families (document.ingested, document.processed, testset.created, run.completed, report.completed, kg.*) power UI updates and downstream analytics. Uncontrolled schema drift causes consumer breakage and silent data quality issues.

## 2. Problem
Current design references event schemas but lacks an explicit versioning and compatibility lifecycle. Need a low-friction approach allowing additive evolution while protecting consumers.

## 3. Decision
Adopt per-event-type semantic schema versioning (major.minor) with registry governance:
- minor increment: additive (new nullable field)
- major increment: breaking (rename/remove/semantics change)
- patch (optional) for documentation-only adjustments (no code changes)

Schema registry file: `events/registry.json` containing array of entries:
```
{
  "event": "run.completed",
  "version": "1.0",
  "schema_sha256": "...",
  "status": "active",
  "introduced": "2025-09-10",
  "deprecated": null
}
```

## 4. Rationale
- Explicit contract: Consumers validate against declared version.
- Hash ensures content integrity & drift detection in CI.
- Granular per-event version avoids global churn.

## 5. Alternatives
| Approach                            | Pros           | Cons                                    | Verdict  |
|-------------------------------------|----------------|-----------------------------------------|----------|
| Global version for all events       | Simple         | Forces broad bumps for isolated changes | Rejected |
| No version, rely on optional fields | Fast initially | Silent breaks & ambiguity               | Rejected |
| Date-stamped versions               | Human friendly | Lacks change semantics                  | Rejected |

## 6. Compatibility Policy
- Minor updates: Consumers must ignore unknown fields.
- Major updates: Old version retained for 2 sprints (dual emission) before deprecation.
- Deprecation path: status moves active → deprecated → removed.

## 7. CI Enforcement
1. On PR, compute schema hash for each changed event JSON.  
2. If content changed w/out version bump → fail.  
3. If major bump, ensure prior version still present (dual emission guard toggle).  
4. Emit changelog snippet in PR comment (automation future).  

## 8. Tooling
- `validate_event(event_name, payload)` uses cached schema; outputs list of violations.
- `list_event_versions(event_name)` for consumers.
- Optional FastAPI middleware logs version mismatch if producer tag differs from registry.

## 9. Risks & Mitigations
| Risk                   | Impact              | Mitigation                          |
|------------------------|---------------------|-------------------------------------|
| Excessive major bumps  | Consumer churn      | Review gate requiring justification |
| Skipped version bump   | Undetected drift    | CI hash comparison                  |
| Dual emission overhead | Slight CPU/log cost | Time-box deprecation window         |

## 10. Success Criteria
- Zero CI failures due to untracked drift after adoption (post Sprint 6).
- 100% events include version field at emission.
- Average deprecation window ≤ 2 sprints.

## 11. Future Enhancements
- Auto-generated Markdown changelog from registry diffs.
- Deprecation dashboard (age, last seen timestamp).
- Schema evolution simulation tests (fuzz) in CI.

---
