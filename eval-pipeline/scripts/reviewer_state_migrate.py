from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.reviewer_state_repository import build_reviewer_state_repository


def main() -> int:
    parser = argparse.ArgumentParser(description="Run reviewer state storage migrations")
    parser.add_argument("--backend", choices=["sqlite", "postgres"], default="sqlite")
    parser.add_argument("--state-store-path", default="outputs/human_feedback/reviewer_state.db")
    parser.add_argument("--state-store-dsn", default=None)
    parser.add_argument("--review-queue-snapshot", default="outputs/human_feedback/review_queue.jsonl")
    parser.add_argument("--reviewer-results-snapshot", default="outputs/human_feedback/reviewer_results.jsonl")
    args = parser.parse_args()

    repository = build_reviewer_state_repository(
        backend=args.backend,
        state_store_path=Path(args.state_store_path),
        review_queue_snapshot_path=Path(args.review_queue_snapshot),
        reviewer_results_snapshot_path=Path(args.reviewer_results_snapshot),
        state_store_dsn=args.state_store_dsn,
    )
    health = repository.health()
    print(health)
    return 0 if health.get("status") in {"ok", "ready"} else 1


if __name__ == "__main__":
    raise SystemExit(main())