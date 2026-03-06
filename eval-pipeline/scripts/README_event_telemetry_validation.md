# Event Schema & Telemetry Taxonomy Validation

## Overview
Two validation scripts ensure governance rules from ADR-005 (Telemetry Taxonomy) and ADR-006 (Event Schema Versioning) are enforced in CI.

## Files
- events/schema_registry.json – Canonical registry (name, version, schema path, sha256)
- events/schemas/*.json – Individual JSON Schemas (draft-07)
- telemetry/telemetry_taxonomy.json – Machine-readable taxonomy
- scripts/compute_event_schema_hashes.py – Updates sha256 when schemas change
- scripts/validate_event_schemas.py – Fails if hash mismatch, duplicate key, or schema invalid
- scripts/validate_telemetry_taxonomy.py – Fails on naming, depth, duplicate path, invalid level

## Typical Workflow
1. Modify or add a schema in events/schemas/
2. Run compute_event_schema_hashes.py to refresh hashes
3. Run validation scripts locally
4. Commit changes (registry + schemas)

## CI Integration (example)
```yaml
jobs:
  governance-validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - name: Install deps
        run: pip install jsonschema
      - name: Validate event schemas
        run: python eval-pipeline/scripts/validate_event_schemas.py
      - name: Validate telemetry taxonomy
        run: python eval-pipeline/scripts/validate_telemetry_taxonomy.py
```

## Failure Modes
- Hash mismatch: Forgot to run compute_event_schema_hashes.py
- Duplicate event key: Reused (name, version)
- Invalid taxonomy level: Level not in allowed_levels
- Depth exceeded: Namespace nesting too deep (violates rules.max_depth)

## Future Enhancements
- Add semantic diff classification (BREAKING/MINOR) for schema changes
- Emit markdown summary artifact for governance dashboards
- Auto-bump taxonomy_version / registry_version on change detection
