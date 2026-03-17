"""Human feedback queueing and review recommendation helpers."""

from __future__ import annotations

import json
import logging
import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class HumanFeedbackManager:
    """Manages human feedback collection and processing."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the human feedback manager with configuration."""
        self.config = config
        self.feedback_config = config.get("evaluation", {}).get("human_feedback", {})

        # Configuration settings
        self.feedback_enabled = self.feedback_config.get("enabled", False)
        self.feedback_threshold = float(
            self.feedback_config.get(
                "threshold", self.feedback_config.get("initial_threshold", 0.7)
            )
        )
        self.sampling_rate = self.feedback_config.get("sampling_rate", 0.2)
        self.dynamic_uncertainty_enabled = bool(
            self.feedback_config.get("dynamic_uncertainty", True)
        )
        bounds = self.feedback_config.get("uncertainty_bounds", {})
        self.uncertainty_min = float(bounds.get("min", 0.3))
        self.uncertainty_max = float(bounds.get("max", 0.9))
        self.uncertainty_buffer = float(bounds.get("buffer", 0.1))
        self.diverse_sample_rate = float(
            self.feedback_config.get("diverse_sample_rate", 0.1)
        )
        self.min_scores_for_uncertainty = int(
            self.feedback_config.get("min_scores_for_uncertainty", 5)
        )
        self.min_window_size = int(self.feedback_config.get("min_window_size", 3))
        self.max_window_size = int(self.feedback_config.get("max_window_size", 10))
        self.feedback_consistency_threshold = float(
            self.feedback_config.get("feedback_consistency_threshold", 0.2)
        )
        self.variance_smoothing_alpha = float(
            self.feedback_config.get("variance_smoothing_alpha", 0.3)
        )
        self.random_seed = self.feedback_config.get("random_seed")
        self._rng = random.Random(self.random_seed)
        self.review_queue_dir = Path(
            self.feedback_config.get("review_queue_dir", "outputs/human_feedback")
        )
        self.review_queue_dir.mkdir(parents=True, exist_ok=True)
        self.review_queue_file = self.review_queue_dir / "review_queue.jsonl"
        self.policy_state_file = self.review_queue_dir / "feedback_policy_state.json"
        reviewer_results_file = self.feedback_config.get("reviewer_results_file")
        self.reviewer_results_file = (
            Path(reviewer_results_file)
            if reviewer_results_file
            else self.review_queue_dir / "reviewer_results.jsonl"
        )

        logger.info(
            f"🤖 Human feedback manager initialized (enabled: {self.feedback_enabled})"
        )

    def process_feedback(
        self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Process human feedback for RAG responses.

        Args:
            testset: The test dataset
            rag_responses: RAG system responses

        Returns:
            Dictionary containing human feedback results
        """
        logger.info("👥 Processing human feedback...")

        if not self.feedback_enabled:
            return {
                "enabled": False,
                "samples_processed": len(rag_responses),
                "feedback_candidates": 0,
                "recommendations": {
                    "total_recommendations": 0,
                    "high_priority": 0,
                    "medium_priority": 0,
                    "low_priority": 0,
                    "recommendations": [],
                },
                "summary": {
                    "status": "disabled",
                    "total_feedback": 0,
                    "positive_feedback": 0,
                    "negative_feedback": 0,
                },
            }

        try:
            # Identify samples that need human feedback
            feedback_candidates = self._identify_feedback_candidates(
                testset, rag_responses
            )

            # Process existing feedback if available
            feedback_results = self._process_existing_feedback(feedback_candidates)

            # Generate feedback recommendations
            recommendations = self._generate_feedback_recommendations(
                feedback_candidates
            )
            queued_items = self._persist_review_queue(recommendations, feedback_candidates)
            policy_state = self._build_policy_state(rag_responses, feedback_results)
            self.policy_state_file.write_text(
                json.dumps(policy_state, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            results = {
                "enabled": True,
                "samples_processed": len(feedback_candidates),
                "feedback_candidates": len(feedback_candidates),
                "existing_feedback": feedback_results,
                "recommendations": recommendations,
                "queued_reviews": queued_items,
                "review_queue_path": str(self.review_queue_file),
                "policy_state": policy_state,
                "policy_state_path": str(self.policy_state_file),
                "summary": self._generate_feedback_summary(
                    feedback_results, recommendations
                ),
            }

            logger.info(
                f"✅ Human feedback processing completed ({len(feedback_candidates)} candidates)"
            )
            return results

        except Exception as e:
            logger.error(f"❌ Human feedback processing failed: {e}")
            return {
                "enabled": True,
                "error": str(e),
                "message": "Human feedback processing encountered an error",
            }

    def _identify_feedback_candidates(
        self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify samples that need human feedback based on uncertainty or other criteria."""
        candidates = []
        score_history = self._collect_response_scores(rag_responses)

        questions = testset.get("questions", [])
        qa_pairs = testset.get("qa_pairs", [])

        for i, response in enumerate(rag_responses):
            # Check if this response needs human feedback
            reason = self._get_feedback_reason(response, i, score_history)
            needs_feedback = reason is not None

            if needs_feedback:
                candidate = {
                    "index": i,
                    "question": (
                        questions[i]
                        if i < len(questions)
                        else qa_pairs[i].get("user_input", f"Question {i+1}")
                        if i < len(qa_pairs)
                        else f"Question {i+1}"
                    ),
                    "response": response,
                    "reason": reason,
                }
                candidates.append(candidate)

        # Apply sampling if too many candidates
        sample_limit = max(1, int(len(rag_responses) * self.sampling_rate))
        if len(candidates) > sample_limit:
            candidates = self._rng.sample(candidates, sample_limit)

        return candidates

    def _get_feedback_reason(
        self,
        response: Dict[str, Any],
        index: int,
        score_history: Optional[List[float]] = None,
    ) -> Optional[str]:
        """Get the reason why this response needs human feedback."""
        confidence = self._safe_float(response.get("confidence", 1.0), 1.0)
        if confidence < self.feedback_threshold:
            return f"Low confidence score: {confidence:.2f}"

        answer = str(response.get("answer", "")).strip()
        if len(answer) < 40:
            return f"Short answer length: {len(answer)}"

        domain_score = self._extract_response_score(response)
        if domain_score < self.feedback_threshold:
            return f"Low evaluation score: {domain_score:.2f}"

        if "ragas_score" in response and "keyword_score" in response:
            ragas_score = self._safe_float(response.get("ragas_score", 0.5), 0.5)
            keyword_score = self._safe_float(response.get("keyword_score", 0.5), 0.5)

            if abs(ragas_score - keyword_score) > 0.3:
                return f"Conflicting metrics - RAGAS: {ragas_score:.2f}, Keyword: {keyword_score:.2f}"

        bounds = self._compute_dynamic_uncertainty_bounds(score_history or [])
        if bounds is not None:
            uncertainty_low, uncertainty_high = bounds
            if uncertainty_low <= domain_score <= uncertainty_high:
                return (
                    "Dynamic uncertainty range match: "
                    f"{domain_score:.2f} within {uncertainty_low:.2f}-{uncertainty_high:.2f}"
                )
            if domain_score > uncertainty_high and self.diverse_sample_rate > 0:
                if self._rng.random() < self.diverse_sample_rate:
                    return "Diverse confident-sample audit"

        return None

    def _collect_response_scores(self, rag_responses: List[Dict[str, Any]]) -> List[float]:
        scores: List[float] = []
        for response in rag_responses:
            score = self._extract_response_score(response)
            if 0.0 <= score <= 1.0:
                scores.append(score)
        return scores

    def _extract_response_score(self, response: Dict[str, Any]) -> float:
        return self._safe_float(
            response.get("domain_score", response.get("ragas_score", 1.0)), 1.0
        )

    def _safe_float(self, value: Any, default: float) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    def _compute_dynamic_uncertainty_bounds(
        self, score_history: List[float]
    ) -> Optional[tuple[float, float]]:
        if not self.dynamic_uncertainty_enabled or len(score_history) < self.min_scores_for_uncertainty:
            return None

        sorted_scores = sorted(score_history)
        q1 = self._percentile(sorted_scores, 0.25)
        q3 = self._percentile(sorted_scores, 0.75)
        low = max(self.uncertainty_min, q1 - self.uncertainty_buffer)
        high = min(self.uncertainty_max, q3 + self.uncertainty_buffer)
        return (low, high)

    def _percentile(self, sorted_scores: List[float], percentile: float) -> float:
        if not sorted_scores:
            return self.feedback_threshold
        if len(sorted_scores) == 1:
            return sorted_scores[0]
        position = (len(sorted_scores) - 1) * percentile
        lower_index = int(position)
        upper_index = min(lower_index + 1, len(sorted_scores) - 1)
        if upper_index == lower_index:
            return sorted_scores[lower_index]
        weight = position - lower_index
        return (sorted_scores[lower_index] * (1.0 - weight)) + (sorted_scores[upper_index] * weight)

    def _smoothed_feedback_variance(self, feedback_values: List[float]) -> float:
        if len(feedback_values) < 2:
            return 0.0
        window = feedback_values[-self.max_window_size :]
        smoothed: List[float] = []
        last_value = window[0]
        alpha = min(max(self.variance_smoothing_alpha, 0.01), 1.0)
        for value in window:
            last_value = (alpha * value) + ((1.0 - alpha) * last_value)
            smoothed.append(last_value)
        return statistics.pvariance(smoothed) if len(smoothed) > 1 else 0.0

    def _adaptive_window_size(self, feedback_values: List[float]) -> int:
        variance = self._smoothed_feedback_variance(feedback_values)
        if variance <= self.feedback_consistency_threshold:
            return self.min_window_size
        normalized_variance = min(1.0, variance / max(self.feedback_consistency_threshold, 0.001))
        return int(
            round(
                self.min_window_size
                + normalized_variance * (self.max_window_size - self.min_window_size)
            )
        )

    def _build_policy_state(
        self,
        rag_responses: List[Dict[str, Any]],
        feedback_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        scores = self._collect_response_scores(rag_responses)
        bounds = self._compute_dynamic_uncertainty_bounds(scores)
        feedback_values = [
            1.0 for _ in range(int(feedback_results.get("positive_feedback", 0)))
        ] + [0.0 for _ in range(int(feedback_results.get("negative_feedback", 0)))]
        adaptive_window = self._adaptive_window_size(feedback_values or [1.0])
        recent_scores = scores[-adaptive_window:] if scores else []
        recommended_threshold = self.feedback_threshold
        if recent_scores:
            recommended_threshold = min(max(statistics.median(recent_scores), 0.0), 1.0)

        return {
            "current_threshold": self.feedback_threshold,
            "recommended_threshold": round(recommended_threshold, 4),
            "dynamic_uncertainty_enabled": self.dynamic_uncertainty_enabled,
            "uncertainty_range": {
                "low": round(bounds[0], 4) if bounds else None,
                "high": round(bounds[1], 4) if bounds else None,
            },
            "score_count": len(scores),
            "score_summary": {
                "min": round(min(scores), 4) if scores else None,
                "max": round(max(scores), 4) if scores else None,
                "median": round(statistics.median(scores), 4) if scores else None,
            },
            "diverse_sample_rate": self.diverse_sample_rate,
            "adaptive_window_size": adaptive_window,
            "smoothed_feedback_variance": round(
                self._smoothed_feedback_variance(feedback_values), 6
            ),
        }

    def _persist_review_queue(
        self,
        recommendations: Dict[str, Any],
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidate_by_index = {candidate["index"]: candidate for candidate in candidates}
        queued_items: List[Dict[str, Any]] = []

        if not recommendations.get("recommendations"):
            return queued_items

        with open(self.review_queue_file, "a", encoding="utf-8") as handle:
            for recommendation in recommendations["recommendations"]:
                candidate = candidate_by_index.get(recommendation["index"], {})
                response = candidate.get("response", {})
                queue_item = {
                    "index": recommendation["index"],
                    "question": recommendation["question"],
                    "reason": recommendation["reason"],
                    "priority": recommendation["priority"],
                    "status": "pending",
                    "suggested_action": recommendation["suggested_action"],
                    "answer": response.get("answer", ""),
                    "confidence": response.get("confidence"),
                    "ragas_score": response.get("ragas_score"),
                    "keyword_score": response.get("keyword_score"),
                }
                handle.write(json.dumps(queue_item, ensure_ascii=False) + "\n")
                queued_items.append(queue_item)

        return queued_items

    def _process_existing_feedback(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Process any existing human feedback."""
        reviewer_results = self._load_reviewer_results()
        if not reviewer_results:
            return {
                "total_feedback": 0,
                "positive_feedback": 0,
                "negative_feedback": 0,
                "resolved_candidates": 0,
                "feedback_items": [],
            }

        candidate_indexes = {candidate["index"] for candidate in candidates}
        candidate_questions = {
            str(candidate.get("question", "")).strip(): candidate["index"]
            for candidate in candidates
        }
        matched_feedback = []
        for item in reviewer_results:
            item_index = item.get("index")
            item_question = str(item.get("question", "")).strip()
            if item_index in candidate_indexes or item_question in candidate_questions:
                matched_feedback.append(item)

        positive_feedback = sum(
            1 for item in matched_feedback if bool(item.get("approved", item.get("is_correct", False)))
        )
        negative_feedback = len(matched_feedback) - positive_feedback

        return {
            "total_feedback": len(matched_feedback),
            "positive_feedback": positive_feedback,
            "negative_feedback": negative_feedback,
            "resolved_candidates": len({item.get("index") for item in matched_feedback}),
            "feedback_items": matched_feedback,
        }

    def ingest_reviewer_results(self, reviewer_results: Any) -> Dict[str, Any]:
        """Persist normalized reviewer labels so future runs can consume them."""
        normalized_items = self._coerce_reviewer_results(reviewer_results)
        if not normalized_items:
            return {
                "ingested": 0,
                "results_file": str(self.reviewer_results_file),
            }

        self.reviewer_results_file.parent.mkdir(parents=True, exist_ok=True)
        existing = self._load_reviewer_results()
        dedupe_keys = {
            self._reviewer_result_key(item) for item in existing
        }
        ingested = 0
        with open(self.reviewer_results_file, "a", encoding="utf-8") as handle:
            for item in normalized_items:
                item_key = self._reviewer_result_key(item)
                if item_key in dedupe_keys:
                    continue
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
                dedupe_keys.add(item_key)
                ingested += 1

        return {
            "ingested": ingested,
            "results_file": str(self.reviewer_results_file),
        }

    def _coerce_reviewer_results(self, reviewer_results: Any) -> List[Dict[str, Any]]:
        if reviewer_results is None:
            return []
        if isinstance(reviewer_results, (str, Path)):
            path = Path(reviewer_results)
            if not path.exists():
                return []
            if path.suffix.lower() == ".json":
                payload = json.loads(path.read_text(encoding="utf-8"))
                return self._coerce_reviewer_results(payload)
            items: List[Dict[str, Any]] = []
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                items.append(self._normalize_reviewer_result(json.loads(line)))
            return items
        if isinstance(reviewer_results, dict):
            return [self._normalize_reviewer_result(reviewer_results)]
        if isinstance(reviewer_results, list):
            return [self._normalize_reviewer_result(item) for item in reviewer_results if isinstance(item, dict)]
        return []

    def _normalize_reviewer_result(self, item: Dict[str, Any]) -> Dict[str, Any]:
        approved = bool(item.get("approved", item.get("is_correct", False)))
        score = item.get("score")
        if score is None:
            score = 1.0 if approved else 0.0
        return {
            "index": item.get("index"),
            "question": str(item.get("question", "")).strip(),
            "approved": approved,
            "score": self._safe_float(score, 0.0),
            "notes": str(item.get("notes", "")).strip(),
            "reviewer": str(item.get("reviewer", "unknown")).strip(),
            "resolution": str(item.get("resolution", "resolved")).strip() or "resolved",
        }

    def _reviewer_result_key(self, item: Dict[str, Any]) -> tuple[Any, str, str]:
        return (
            item.get("index"),
            str(item.get("question", "")).strip(),
            str(item.get("reviewer", "unknown")).strip(),
        )

    def _load_reviewer_results(self) -> List[Dict[str, Any]]:
        if not self.reviewer_results_file.exists():
            return []
        results: List[Dict[str, Any]] = []
        for line in self.reviewer_results_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                results.append(self._normalize_reviewer_result(json.loads(line)))
            except Exception:
                continue
        return results

    def _generate_feedback_recommendations(
        self, candidates: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate recommendations for human feedback collection."""
        recommended_threshold = None
        if candidates:
            scores = [
                self._extract_response_score(candidate.get("response", {}))
                for candidate in candidates
            ]
            if scores:
                recommended_threshold = round(statistics.median(scores), 4)
        if not candidates:
            return {
                "total_recommendations": 0,
                "high_priority": 0,
                "medium_priority": 0,
                "low_priority": 0,
                "recommended_threshold": recommended_threshold,
                "recommendations": [],
            }

        recommendations = []
        high_priority = 0
        medium_priority = 0
        low_priority = 0

        for candidate in candidates:
            priority = self._determine_priority(candidate)

            rec = {
                "index": candidate["index"],
                "question": candidate["question"],
                "reason": candidate["reason"],
                "priority": priority,
                "suggested_action": self._suggest_action(candidate, priority),
            }

            recommendations.append(rec)

            if priority == "high":
                high_priority += 1
            elif priority == "medium":
                medium_priority += 1
            else:
                low_priority += 1

        return {
            "total_recommendations": len(recommendations),
            "high_priority": high_priority,
            "medium_priority": medium_priority,
            "low_priority": low_priority,
            "recommended_threshold": recommended_threshold,
            "recommendations": recommendations,
        }

    def _determine_priority(self, candidate: Dict[str, Any]) -> str:
        """Determine the priority level for feedback collection."""
        reason = candidate["reason"]

        if "Low confidence" in reason:
            return "high"
        elif "Conflicting metrics" in reason:
            return "medium"
        else:
            return "low"

    def _suggest_action(self, candidate: Dict[str, Any], priority: str) -> str:
        """Suggest action for feedback collection."""
        if priority == "high":
            return "Immediate review recommended - low confidence in response quality"
        elif priority == "medium":
            return "Review recommended - conflicting evaluation metrics detected"
        else:
            return "Optional review for quality assurance"

    def _generate_feedback_summary(
        self, feedback_results: Dict[str, Any], recommendations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate summary of feedback processing."""
        return {
            "existing_feedback_count": feedback_results.get("total_feedback", 0),
            "new_recommendations": recommendations.get("total_recommendations", 0),
            "high_priority_items": recommendations.get("high_priority", 0),
            "feedback_coverage": f"{feedback_results.get('total_feedback', 0)} existing + {recommendations.get('total_recommendations', 0)} recommended",
            "recommended_threshold": recommendations.get("recommended_threshold"),
        }

    def evaluate_testset(self, testset_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process testset data for human feedback evaluation.

        Args:
            testset_data: Testset data containing qa_pairs

        Returns:
            List of evaluation results
        """
        logger.info("🔄 Processing testset for human feedback evaluation...")

        try:
            qa_pairs = testset_data.get("qa_pairs", [])
            if not qa_pairs:
                logger.warning("No QA pairs found in testset data")
                return []

            # Convert qa_pairs into reviewable responses with real scoring hints
            mock_rag_responses = []
            for qa_pair in qa_pairs:
                mock_response = {
                    "question": qa_pair.get("user_input", qa_pair.get("question", "")),
                    "answer": qa_pair.get("reference", ""),
                    "contexts": qa_pair.get("contexts", []),
                    "ground_truth": qa_pair.get("reference", ""),
                    "confidence": float(qa_pair.get("confidence", qa_pair.get("ragas_score", 0.8))),
                    "ragas_score": float(qa_pair.get("ragas_score", qa_pair.get("domain_score", 0.8))),
                    "keyword_score": float(qa_pair.get("keyword_score", qa_pair.get("ragas_score", 0.8))),
                    "domain_score": float(qa_pair.get("domain_score", qa_pair.get("ragas_score", 0.8))),
                }
                mock_rag_responses.append(mock_response)

            # Process feedback
            feedback_result = self.process_feedback(testset_data, mock_rag_responses)

            # Convert to list format expected by stage factory
            results = []
            recommendations = {
                item["index"]: item for item in feedback_result.get("queued_reviews", [])
            }
            reviewer_results = {
                item.get("index"): item
                for item in feedback_result.get("existing_feedback", {}).get("feedback_items", [])
                if item.get("index") is not None
            }
            for i, qa_pair in enumerate(qa_pairs):
                review_item = recommendations.get(i)
                reviewer_item = reviewer_results.get(i)
                result_item = {
                    **qa_pair,
                    "human_feedback_score": (
                        reviewer_item.get("score", 0.0) if reviewer_item else 0.0
                    ),
                    "feedback_required": review_item is not None and reviewer_item is None,
                    "feedback_reason": review_item.get("reason") if review_item else None,
                    "feedback_priority": review_item.get("priority") if review_item else None,
                    "review_resolution": reviewer_item.get("resolution") if reviewer_item else None,
                    "reviewer_notes": reviewer_item.get("notes") if reviewer_item else None,
                    "human_feedback_data": feedback_result,
                }
                results.append(result_item)

            logger.info(
                f"✅ Human feedback processing completed for {len(results)} items"
            )
            return results

        except Exception as e:
            logger.error(f"❌ Human feedback processing failed: {e}")
            return []

    def process_testset(self, testset_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Compatibility wrapper used by the stage executor."""
        return self.evaluate_testset(testset_data)

    def is_enabled(self) -> bool:
        """Check if human feedback processing is enabled."""
        return self.feedback_enabled
