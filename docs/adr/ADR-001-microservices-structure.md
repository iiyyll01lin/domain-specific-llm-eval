# ADR-001: Microservices Structure for RAG Evaluation Platform

Status: Accepted  
Date: 2025-09-10  
Decision Owners: Platform Lead, Backend Architect  
Reviewers: Frontend Lead, Data Engineering, SRE  

## 1. Context
The system must support ingestion → processing → testset generation → evaluation → reporting with modular evolution, independent scaling, and selective deployment of optional (flagged) features like Knowledge Graph (KG) building and Subgraph API. Monolith would accelerate initial delivery but risks entangled deployments and slower iteration on performance-critical stages (processing, evaluation). Observability, idempotency anchors, and artifact lineage are core non-functionals.

## 2. Decision
Adopt a lightweight microservices approach with the following initial logical services (deployment units may be co-located early):
- ingestion-service
- processing-service
- testset-service
- evaluation-service
- reporting-service
- insights-adapter (normalization & export)
- kg-service (feature flagged)
- websocket-gateway (real-time events multiplex)

Shared library: common/ (config, logging, auth stub, error envelope, storage abstraction, event schema utilities).

## 3. Rationale
- Independent Scaling: processing & evaluation can horizontally scale without affecting reporting.
- Failure Isolation: KG experimental features can fail/rollback independently.
- Deploy Cadence: security patches or metric tweaks in one service avoid full-system redeploy.
- Observability Clarity: per-service metrics expose bottlenecks faster.
- Future Extensibility: swap evaluation engine or add alternative testset generators with minimal cross-impact.

## 4. Alternatives Considered
| Option               | Pros                                            | Cons                                                             | Reason Rejected                    |
|----------------------|-------------------------------------------------|------------------------------------------------------------------|------------------------------------|
| Single Monolith      | Simplest initial deployment, fewer network hops | Harder scaling hotspots, entangled releases, larger blast radius | Long-term agility risk             |
| 2-Tier (Core + Aux)  | Reduced service count                           | Still couples unrelated lifecycles                               | Insufficient isolation             |
| Functions/Serverless | Elastic scaling                                 | Cold starts, complexity in sequence orchestration                | Latency + observability complexity |

## 5. Implications
- Requires lightweight service templates & shared tooling consistency.
- Slight initial overhead in infra (routing, deployment scripts).
- Enables early introduction of canary deploy for evaluation-service.

## 6. Decision Drivers
1. Deterministic artifact lineage (traceability objective SMART#4).  
2. Performance variability across pipeline stages.  
3. Feature flag isolation (KG, subgraph).  
4. Operational debugging clarity.  

## 7. Risks & Mitigations
| Risk               | Impact                  | Mitigation                                                           |
|--------------------|-------------------------|----------------------------------------------------------------------|
| Over-fragmentation | Cognitive load          | Limit to enumerated list; avoid splitting until justified by metrics |
| Network overhead   | Slight latency increase | Co-locate (pod) early; internal fast networking                      |
| Config drift       | Inconsistent behavior   | Central shared config lib + CI schema check                          |

## 8. Adoption Plan
- Sprint 1: Scaffold services (TASK-001) with shared common package.
- Sprint 2+: Evaluate consolidation if p95 inter-service latency > threshold.

## 9. Status Tracking
Open tasks: TASK-001, TASK-002, TASK-003, TASK-004, TASK-005.

## 10. Future Revisit Triggers
- If average deployment batch < 2 services for 3 consecutive sprints → consider merging.
- If inter-service call latency > 50ms p95 internally → profile and revisit topology.

---
Decision recorded for transparency; update status to Accepted after initial implementation stability review.
