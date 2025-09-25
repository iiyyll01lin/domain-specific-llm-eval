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

## 4. Next Steps
- Integrate SARIF uploads with GitHub Security Alerts (enable on repository settings).
- Extend scans with SBOM comparison (see TASK-130) once pipeline is wired.
