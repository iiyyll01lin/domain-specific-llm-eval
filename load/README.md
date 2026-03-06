# Load Test Configuration — TASK-103

## Overview

Load tests for the processing and eval services using [Locust](https://locust.io/).

## Setup

```bash
pip install locust
```

## Running Against Local Stack

```bash
# Start services first
docker compose -f docker-compose.services.yml up -d

# Run load test (headless, 2 minutes)
locust -f load/locustfile.py \
    --host http://localhost:8002 \
    --users 20 --spawn-rate 5 --run-time 2m --headless

# Run with Locust web UI (http://localhost:8089)
locust -f load/locustfile.py --host http://localhost:8002
```

## SLA Thresholds

| Endpoint                  | Service    | p95 Budget |
|---------------------------|------------|-----------|
| `POST /processing-jobs`   | processing | 5 000 ms  |
| `POST /eval-jobs`         | eval       | 8 000 ms  |

Any threshold violation causes the Locust run to exit with code 1.

## CI Integration

Add to CI pipeline after services are healthy:

```yaml
- name: Load test (smoke)
  run: |
    locust -f load/locustfile.py \
      --host http://localhost:8002 \
      --users 5 --spawn-rate 2 --run-time 30s --headless
```
