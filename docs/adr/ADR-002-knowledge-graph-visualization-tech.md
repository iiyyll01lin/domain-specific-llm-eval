# ADR-002: Knowledge Graph Visualization Technology (Cytoscape.js)

Status: Accepted  
Date: 2025-09-10  
Decision Owners: Frontend Lead, Platform Lead  
Reviewers: Data Engineering, UX, Performance SME  

## 1. Context
The UI needs to optionally (feature-flagged) visualize a Knowledge Graph (KG) derived from ingested documents. Graph scale (hundreds to low thousands nodes typical) must not degrade baseline portal performance. Requirements include lazy loading, deterministic subgraph sampling, interaction (focus/expand), and theming alignment. Candidate libs: Cytoscape.js, D3 force layouts, Vis.js, Sigma.js.

## 2. Decision
Adopt Cytoscape.js with dynamic import (code splitting) and a dedicated chunk gated by feature flag `kgVisualization`.

## 3. Rationale
- Performance: Efficient handling of medium graphs; supports WebGL extensions if needed later.
- Ecosystem: Mature layout algorithms (cose, concentric, breadthfirst) suitable for knowledge inspection.
- Extensibility: Style API + events enabling future overlay of metric annotations.
- Bundle Isolation: Can be lazy-loaded, aligning with UI-NFR-006 (avoid penalizing non-KG users).

## 4. Alternatives Compared
| Library      | Pros                                      | Cons                                                | Verdict                       |
|--------------|-------------------------------------------|-----------------------------------------------------|-------------------------------|
| Cytoscape.js | Efficient, rich styling, plugin ecosystem | Larger base size than minimal D3 subset             | Chosen                        |
| D3 (custom)  | Flexible, lower base size                 | Higher custom code cost (layouts, interactions)     | Rejected (engineering effort) |
| Sigma.js     | WebGL optimized                           | Less flexible styling complexity for edge semantics | Rejected (theming limits)     |
| Vis.js       | Simple API                                | Layout & styling less granular                      | Rejected (control)            |

## 5. Implications
- Introduces separate JS bundle (~tens of KB gzipped) loaded only when flag active.
- Requires sampling cap (TASK-065) to guard performance.
- Encourages consistent data schema for nodes/edges from KG summary & subgraph API.

## 6. Data & API Shape (Initial)
Node: { id, label, entity_type?, degree, sample_hash }
Edge: { id, source, target, rel_type, score }

## 7. Risks & Mitigations
| Risk              | Impact             | Mitigation                                             |
|-------------------|--------------------|--------------------------------------------------------|
| Bundle bloat      | Slower first paint | Dynamic import + TASK-082 bundle guard                 |
| Very large graphs | UI freeze          | Enforce node cap + server-side sampling (TASK-066)     |
| Layout jitter     | User confusion     | Cache layout params; deterministic seed when supported |

## 8. Adoption Plan
- Sprint 4: Implement lazy component (TASK-065), configure basic layout.
- Sprint 5+: Introduce subgraph focus (TASK-067) once API stable.

## 9. Metrics / Success Criteria
- KG chunk gz size < 300KB (CI enforced).
- Initial render (500 nodes) < 2s on reference hardware.
- No regression to core portal TTI when flag disabled.

## 10. Future Considerations
- WebGL renderer switch if node cap needs >2k with acceptable fps.
- Annotation overlays (evaluation metrics mapped onto nodes).
- Accessibility audit (keyboard navigation & color contrast) by Sprint 6.

---
Pending validation via prototype spike before marking Accepted.
