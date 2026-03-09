# Container Security Checks

Status: TASK-124 Complete – 2025-09-25

## 1. Automated Scans (GitHub Actions)
- Workflow: `.github/workflows/build-governance.yml`
- Jobs:
  - **security-scan**: `aquasecurity/trivy-action` (filesystem mode) scanning repo sources.
  - **build-and-push**: image build gated by Trivy image scan before GHCR push.
- Both scans fail the pipeline when HIGH/CRITICAL vulnerabilities are detected (`exit-code: 1`).
- SARIF reports are uploaded as workflow artifacts (`trivy-fs-sarif`, `trivy-image-sarif`).

## 2. Local Reproduction
```
# Filesystem scan
docker run --rm -v $(pwd):/repo aquasec/trivy fs --severity HIGH,CRITICAL /repo

# Image scan (assumes rag-eval:dev built)
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock aquasec/trivy image --severity HIGH,CRITICAL rag-eval:dev
```

## 3. Waivers & Allowlist Policy
- Temporary waivers require approval from `platform-secops@team`.
- Document any suppression in `docs/security.md` with justification and expiration date.
- Prefer upgrading base images or dependencies before applying waivers.

## 4. Secrets Scan Gate
- Workflow job `secrets-scan` runs `gitleaks` with `.gitleaks.toml`.
- Repository-specific allowlist entries must be documented in `.gitleaks.toml` with rationale.
- Any new hard-coded credential should fail pull request validation before merge.

## 5. SBOM, Signing, and Provenance
- `build-governance.yml` generates CycloneDX SBOM output at `sbom/sbom-main.json` and a component diff report at `sbom/sbom-diff.json`.
- Provenance metadata is emitted to `attest/provenance.intoto.jsonl` for each build.
- If `COSIGN_PRIVATE_KEY` is configured, pushed images are signed via `cosign`; otherwise the workflow records an unsigned attestation and emits a warning.

Local verification example:
```
cosign verify --key cosign.pub ghcr.io/<owner>/rag-eval:v<version>
```

For the full prebuilt deployment flow, including GHCR login and compose-backed smoke validation without local builds, see [docs/prebuilt_image_workflow.md](docs/prebuilt_image_workflow.md).

## 6. Next Steps
- Integrate SARIF uploads with GitHub Security Alerts (enable on repository settings).
- Tighten unsigned-image handling from warning to hard-fail once repository secrets are provisioned.
