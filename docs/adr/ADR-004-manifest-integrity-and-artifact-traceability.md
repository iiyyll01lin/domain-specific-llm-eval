# ADR-004: Manifest Integrity & Artifact Traceability

Status: Accepted  
Date: 2025-09-10  
Decision Owners: Platform Lead, QA Lead  
Reviewers: Security, SRE, Data Engineering  

## 1. Context
The evaluation pipeline emits multiple artifacts per run (chunks.jsonl, evaluation_items.json, kpis.json, run_meta.json, testset_summary_v0.json, kg_summary_v0.json, export_summary.json, report HTML/PDF). Need: provable completeness, tamper detection, and future reproducibility hooks. Early introduction reduces retrofitting complexity and supports SMART traceability objective.

## 2. Problem Statement
Without a consolidated manifest, partial deletions or silent overwrites may go undetected, harming trust and regression analysis. Need a lightweight, extensible integrity layer without heavy blockchain-like overhead.

## 3. Decision
Introduce a versioned manifest file per evaluation run: `manifest.json` (v1 prototype in TASK-083) containing:
- run_id, testset_id, created_at (ISO8601)
- artifacts[]: { name, path, sha256, size_bytes, content_type?, produced_at }
- metrics: { generation_time_ms?, artifact_count }
- schema_version: "1.0"

Validation CLI / library routine verifies presence + hashes on demand (CI or troubleshooting).

## 4. Rationale
- Centralizes artifact inventory for governance & audits.
- Hashes enable integrity verification pre/post transport.
- Provides anchor for caching (hash of sorted artifact hashes) in later optimization.

## 5. Alternatives Considered
| Option             | Pros                | Cons                              | Verdict            |
|--------------------|---------------------|-----------------------------------|--------------------|
| No Manifest        | Simplicity          | Integrity blind spot              | Rejected           |
| DB Catalog Table   | Queryable           | Adds DB coupling, version drift   | Deferred (phase 2) |
| Signed TAR Archive | Strong immutability | Higher storage & complexity early | Deferred           |

## 6. Data Model (Draft)
```
{
  "run_id": "...",
  "testset_id": "...",
  "created_at": "2025-09-10T12:34:56Z",
  "schema_version": "1.0",
  "artifacts": [
    {"name":"chunks","path":"chunks.jsonl","sha256":"...","size_bytes":12345,"produced_at":"..."},
    {"name":"evaluation_items","path":"evaluation_items.json","sha256":"...","size_bytes":67890,"produced_at":"..."}
  ],
  "metrics": {"generation_time_ms":1234, "artifact_count": 8}
}
```

## 7. Integrity Process
1. After each artifact finalization append entry with hash & size.  
2. On run completion finalize manifest, compute overall hash = SHA256(sorted(artifact.sha256)).  
3. Provide `validate_manifest(run_dir)` helper for CI / manual.  

## 8. Risks & Mitigations
| Risk                       | Impact                   | Mitigation                             |
|----------------------------|--------------------------|----------------------------------------|
| Partial write race         | Missing artifact entry   | Write temp + atomic rename             |
| Hash collision (practical) | Integrity false positive | SHA256 adequate; monitor for anomalies |
| Manifest drift             | Stale view               | Regenerate routine warns on mismatch   |

## 9. Success Metrics
- 100% runs produce manifest.json.  
- Validation passes on fresh outputs; corruption simulation fails.  
- Overhead: generation <1% total run time.  

## 10. Future Enhancements
- v2: Signed manifest (Ed25519) + public key distribution.
- v3: DB-backed manifest index for cross-run lineage queries.
- Selective artifact compression policy referencing manifest entries.

---
Mark Accepted once TASK-083 prototype stabilizes and validation integrated into CI.
