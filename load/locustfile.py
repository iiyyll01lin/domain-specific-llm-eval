"""
TASK-103: Locust load test for processing & eval service concurrency.

Run:
    locust -f load/locustfile.py --host http://localhost:8002 \
           --users 20 --spawn-rate 5 --run-time 2m --headless

Target thresholds (NFR):
    processing-service  p95 < 5 000 ms
    eval-service        p95 < 8 000 ms
"""
import json
import uuid

from locust import HttpUser, TaskSet, between, events, task


# ---------------------------------------------------------------------------
# Processing service load tasks
# ---------------------------------------------------------------------------

class ProcessingTasks(TaskSet):
    """Simulate concurrent processing job submissions."""

    @task(3)
    def create_processing_job(self):
        run_id = f"load-{uuid.uuid4().hex[:8]}"
        with self.client.post(
            "/processing-jobs",
            json={"run_id": run_id, "chunk_size": 512, "chunk_overlap": 64},
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="/processing-jobs [POST]",
        ) as resp:
            if resp.status_code not in (200, 201, 202, 429):
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(1)
    def list_processing_jobs(self):
        self.client.get("/", name="/ [GET]")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health [GET]")


class ProcessingUser(HttpUser):
    tasks = [ProcessingTasks]
    wait_time = between(0.5, 2.0)
    host = "http://localhost:8002"


# ---------------------------------------------------------------------------
# Eval service load tasks
# ---------------------------------------------------------------------------

class EvalTasks(TaskSet):
    """Simulate concurrent evaluation job submissions."""

    @task(3)
    def create_eval_job(self):
        run_id = f"load-{uuid.uuid4().hex[:8]}"
        testset_id = f"ts-{uuid.uuid4().hex[:8]}"
        with self.client.post(
            "/eval-jobs",
            json={
                "run_id": run_id,
                "testset_id": testset_id,
                "metrics": ["faithfulness", "answer_relevancy"],
            },
            headers={"Content-Type": "application/json"},
            catch_response=True,
            name="/eval-jobs [POST]",
        ) as resp:
            if resp.status_code not in (200, 201, 202, 429):
                resp.failure(f"Unexpected status {resp.status_code}")

    @task(1)
    def list_eval_jobs(self):
        self.client.get("/", name="/ [GET]")

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health [GET]")


class EvalUser(HttpUser):
    tasks = [EvalTasks]
    wait_time = between(1.0, 3.0)
    host = "http://localhost:8005"


# ---------------------------------------------------------------------------
# SLA thresholds — checked in the stop hook
# ---------------------------------------------------------------------------

THRESHOLDS = {
    "/processing-jobs [POST]": {"p95_ms": 5_000},
    "/eval-jobs [POST]":       {"p95_ms": 8_000},
}


@events.quitting.add_listener
def check_thresholds(environment, **kwargs):
    """Fail the run if any endpoint exceeds its p95 threshold."""
    failed = False
    stats = environment.runner.stats

    for endpoint, limits in THRESHOLDS.items():
        entry = stats.get(endpoint, "POST")
        if entry is None:
            continue
        p95 = entry.get_response_time_percentile(0.95)
        budget = limits["p95_ms"]
        status = "PASS" if p95 <= budget else "FAIL"
        print(f'[load-check] {status:4}  {endpoint}  p95={p95:.0f}ms  budget={budget}ms')
        if p95 > budget:
            failed = True

    if failed:
        environment.process_exit_code = 1
