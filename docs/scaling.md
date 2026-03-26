# Kubernetes Horizontal Scaling (HPA) Guide

## Overview

Stateless services — `ingestion`, `processing`, and `eval` — support Horizontal Pod Autoscaler (HPA) driven by **CPU utilisation** and (optionally) a **custom request-rate metric** exported via Prometheus.

The Helm chart ships with HPA manifests disabled by default. Enable per service by setting `<service>.autoscaling.enabled=true` in your values override.

---

## Quick Start

```bash
# Enable HPA for the processing service (CPU-based, 2–10 replicas)
helm upgrade --install my-release deploy/helm \
  --set processing.autoscaling.enabled=true \
  --set processing.autoscaling.minReplicas=2 \
  --set processing.autoscaling.maxReplicas=10 \
  --set processing.autoscaling.targetCPUUtilizationPercentage=70
```

Verify the HPA was created:

```bash
kubectl get hpa -n <namespace>
# Example output:
# NAME                          REFERENCE               TARGETS   MINPODS  MAXPODS  REPLICAS
# my-release-processing         Deployment/processing   32%/70%   2        10       3
```

---

## Helm Values Reference

```yaml
# values.yaml defaults (autoscaling section)
defaults:
  autoscaling:
    enabled: false
    minReplicas: 1
    maxReplicas: 5
    targetCPUUtilizationPercentage: 80
```

Per-service override example (`values-prod.yaml`):

```yaml
processing:
  autoscaling:
    enabled: true
    minReplicas: 2
    maxReplicas: 12
    targetCPUUtilizationPercentage: 65

eval:
  autoscaling:
    enabled: true
    minReplicas: 1
    maxReplicas: 6
    targetCPUUtilizationPercentage: 75
```

---

## Custom Request-Rate Metric (Optional)

Each service exposes a Prometheus counter `svc_<name>_requests_total`. To scale on
request rate instead of CPU, configure the `metrics-server` and a `Prometheus Adapter`:

1. Install `prometheus-adapter` pointing at your Prometheus endpoint.
2. Add an `externalMetric` rule mapping `svc_processing_requests_total` → `requests_per_second`.
3. Reference it in your values:

```yaml
processing:
  autoscaling:
    enabled: true
    targetCPUUtilizationPercentage: null   # disable CPU metric
    customMetric:
      name: requests_per_second
      target: 50    # scale when rate > 50 req/s per pod
```

> **Note**: Custom metric autoscaling requires the Prometheus Adapter to be installed
> and the `autoscaling/v2` API available in your cluster (Kubernetes ≥ 1.23).

---

## Tuning Guidelines

| Service     | Recommended Trigger         | Min Replicas | Max Replicas | Notes                                        |
|-------------|------------------------------|:------------:|:------------:|----------------------------------------------|
| `ingestion` | CPU 70%                      | 1            | 4            | I/O-bound; rarely needs >4                   |
| `processing`| CPU 65% or 50 req/s          | 2            | 12           | Embedding is CPU-heavy; keep warm replicas   |
| `eval`      | CPU 75%                      | 1            | 6            | LLM calls are bursty; generous max           |
| `testset`   | CPU 80% (not recommended)    | 1            | 3            | Calls external LLM; rate-limit aware         |
| `kg`        | CPU 80%                      | 1            | 4            | Graph builds are memory-intensive            |

### General Rules

- **Scale-down stabilization**: Set `stabilizationWindowSeconds: 300` to avoid thrashing after brief spikes.
- **Resource requests**: Always set `resources.requests.cpu` so the HPA has a denominator. If unset, HPA cannot compute CPU utilisation percentage.
- **Readiness gate**: HPA only counts pods whose `/readyz` returns 200. Pods failing readiness are excluded from the replica count — ensure `/readyz` accurately reflects service health.

---

## Startup Probe (Heavy Init Services)

The `kg` service (and `processing` when `LOAD_EMBEDDINGS=true`) loads large embedding models at startup. Add a startup probe to prevent premature liveness kills:

```yaml
# values-prod.yaml
kg:
  startupProbe:
    httpGet:
      path: /readyz
      port: 8000
    failureThreshold: 30
    periodSeconds: 10
```

This allows up to **5 minutes** (`30 × 10s`) for model loading before the liveness probe kicks in.

---

## HPA Template Reference

The Helm chart template is at [`deploy/helm/templates/hpa.yaml`](../deploy/helm/templates/hpa.yaml).

Rendered example for `processing` with CPU HPA enabled:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: my-release-processing
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: my-release-processing
  minReplicas: 2
  maxReplicas: 12
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 65
```

---

## Dry-Run Test

Validate HPA resources are syntactically correct without a cluster:

```bash
/tmp/linux-amd64/helm template test-release deploy/helm \
  --set processing.autoscaling.enabled=true \
  --set processing.autoscaling.minReplicas=2 \
  --set processing.autoscaling.maxReplicas=10 | \
  grep -A 20 "HorizontalPodAutoscaler"
```
