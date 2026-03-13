from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.human_feedback_manager import HumanFeedbackManager


def test_process_feedback_queues_low_confidence_reviews(tmp_path: Path) -> None:
    manager = HumanFeedbackManager(
        {
            "evaluation": {
                "human_feedback": {
                    "enabled": True,
                    "threshold": 0.7,
                    "review_queue_dir": str(tmp_path),
                }
            }
        }
    )

    result = manager.process_feedback(
        {"questions": ["What happened?"]},
        [
            {
                "answer": "Too short",
                "confidence": 0.4,
                "ragas_score": 0.3,
                "keyword_score": 0.9,
                "domain_score": 0.3,
            }
        ],
    )

    assert result["feedback_candidates"] == 1
    assert result["queued_reviews"][0]["priority"] == "high"
    queue_lines = (tmp_path / "review_queue.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(queue_lines) == 1
    assert json.loads(queue_lines[0])["reason"].startswith("Low confidence")


def test_process_testset_marks_feedback_required(tmp_path: Path) -> None:
    manager = HumanFeedbackManager(
        {
            "evaluation": {
                "human_feedback": {
                    "enabled": True,
                    "threshold": 0.8,
                    "review_queue_dir": str(tmp_path),
                }
            }
        }
    )

    results = manager.process_testset(
        {
            "qa_pairs": [
                {
                    "user_input": "Question 1",
                    "reference": "short",
                    "ragas_score": 0.2,
                    "keyword_score": 0.9,
                }
            ]
        }
    )

    assert results[0]["feedback_required"] is True
    assert results[0]["feedback_priority"] == "high"