import os

# Phase 5: Distributed Tracing with LangSmith
if os.environ.get("LANGCHAIN_TRACING_V2") == "true":
    # Usually handled inherently by Langchain via env vars.
    print("LangSmith Tracing Enabled globally.")

"""
RAGAS Evaluator for RAG Evaluation Pipeline

Handles RAGAS-based evaluation of RAG systems.
"""

import sys
from pathlib import Path

# Add utils directory to Python path for local imports
current_file_dir = Path(__file__).parent
utils_dir = current_file_dir.parent / "utils"
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))
import langchain
from langchain_community.cache import SQLiteCache

# Create cache directory if it doesn't exist
os.makedirs(".cache", exist_ok=True)
try:
    langchain.llm_cache = SQLiteCache(database_path=".cache/langchain_llm.db")
except Exception as e:
    print(f"Failed to setup SQLiteCache: {e}")

import logging
import math
import re
# Apply global tiktoken patch BEFORE any other imports
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import numpy as np

sys.path.append(str(Path(__file__).parent.parent.parent))
from global_tiktoken_patch import apply_global_tiktoken_patch

apply_global_tiktoken_patch()

# Import RAGAS model_dump fix
from .ragas_model_dump_fix import RagasModelDumpFix, apply_ragas_model_dump_fix
from evaluation.spatial_rag_evaluator import MixedRealityMultimodalEval
from evaluation.multimodal_metrics import MultimodalResponseEvaluator
from evaluation.swarm_agent import SwarmSynthesizer
from evaluation.symbolic_evaluator import SymbolicEvaluator
from evaluation.temporal_causality_evaluator import TemporalCausalityEvaluator
from evaluation.telepathic_intent_evaluator import TelepathicIntentAlignment
from generation.neuro_symbolic_rag import NeuroSymbolicRAGEngine
from optimization.dpo_alignment import DirectPreferenceOptimizationPipeline
from .evaluation_result_contract import attach_result_contract, evaluation_error_result

try:
    import pandas as pd
except ImportError:
    pd = None

logger = logging.getLogger(__name__)


@dataclass
class LLMRoleBinding:
    role: str
    endpoint: str
    model_name: str
    temperature: float
    max_tokens: int
    enabled: bool = False


class _ActorIdentityLLM:
    """Fallback actor that preserves answers when actor preprocessing is disabled."""

    def invoke(self, payload: Dict[str, Any]) -> str:
        return str(payload.get("answer", ""))


class DomainRegexHeuristic:
    """Custom evaluation metric that checks for domain-specific vocabulary constraints."""

    def __init__(
        self,
        required_terms: Optional[List[str]] = None,
        regex_patterns: Optional[List[str]] = None,
        case_sensitive: bool = False,
    ):
        self.required_terms = required_terms or []
        flags = 0 if case_sensitive else re.IGNORECASE
        self.regex_patterns = [re.compile(pattern, flags) for pattern in regex_patterns or []]
        self.case_sensitive = case_sensitive

    def is_enabled(self) -> bool:
        return bool(self.required_terms or self.regex_patterns)

    def score(self, text: str) -> float:
        if not self.is_enabled():
            return 1.0

        normalized_text = text if self.case_sensitive else text.lower()
        term_matches = sum(
            1
            for term in self.required_terms
            if (term if self.case_sensitive else term.lower()) in normalized_text
        )
        regex_matches = sum(1 for pattern in self.regex_patterns if pattern.search(text))
        total_rules = len(self.required_terms) + len(self.regex_patterns)
        return (term_matches + regex_matches) / total_rules if total_rules else 1.0


class RagasEvaluator:
    """RAGAS-based evaluator for RAG systems."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAGAS evaluator with configuration."""
        self.config = config
        evaluation_config = config.get("evaluation") if isinstance(config, dict) else None
        if isinstance(evaluation_config, dict):
            self.ragas_config = evaluation_config.get("ragas", {})
            self.ragas_metrics_config = evaluation_config.get("ragas_metrics", {})
        else:
            self.ragas_config = config.get("ragas", {}) if isinstance(config, dict) else {}
            self.ragas_metrics_config = (
                config.get("ragas_metrics", config) if isinstance(config, dict) else {}
            )
        self.ragas_available = False
        self.metric_weights = self._load_metric_weights()
        self.heuristic = self._build_domain_heuristic()
        self.multimodal_evaluator = MultimodalResponseEvaluator()
        self.swarm_synthesizer = SwarmSynthesizer()
        self.symbolic_engine = NeuroSymbolicRAGEngine()
        self.symbolic_evaluator = SymbolicEvaluator()
        self.spatial_evaluator = MixedRealityMultimodalEval()
        self.temporal_evaluator = TemporalCausalityEvaluator()
        self.intent_evaluator = TelepathicIntentAlignment()
        alignment_config = self.ragas_metrics_config.get("alignment", {})
        self.alignment_pipeline = DirectPreferenceOptimizationPipeline(alignment_config)
        self.actor_llm: Any = _ActorIdentityLLM()
        self.critic_llm: Any = None
        self.llm_roles: Dict[str, LLMRoleBinding] = {
            "actor": LLMRoleBinding("actor", "", "identity", 0.0, 0, enabled=False),
            "critic": LLMRoleBinding("critic", "", "default-ragas", 0.0, 0, enabled=False),
        }
        self.actor_preprocessing_enabled = False

        self._setup_ragas()

    def _build_domain_heuristic(self) -> DomainRegexHeuristic:
        domain_regex_config = self.ragas_metrics_config.get("domain_regex", {})
        required_terms = domain_regex_config.get("required_terms", [])
        regex_patterns = domain_regex_config.get("regex_patterns", [])
        case_sensitive = domain_regex_config.get("case_sensitive", False)
        return DomainRegexHeuristic(
            required_terms=required_terms,
            regex_patterns=regex_patterns,
            case_sensitive=case_sensitive,
        )

    def _load_metric_weights(self) -> Dict[str, float]:
        configured_weights = self.ragas_metrics_config.get("metric_weights", {})
        return {
            "context_precision": float(configured_weights.get("context_precision", 0.5)),
            "faithfulness": float(configured_weights.get("faithfulness", 0.5)),
            "domain_regex": float(configured_weights.get("domain_regex", 0.0)),
        }

    def _setup_ragas(self) -> None:
        """Setup RAGAS evaluation components with custom LLM."""
        try:
            # Apply RAGAS model_dump compatibility fix first
            apply_ragas_model_dump_fix()

            from datasets import Dataset
            from ragas import evaluate  # type: ignore[attr-defined]
            # Default to offline deterministic metrics — no LLM endpoint required.
            # LLM-based metrics (context_precision, faithfulness) are appended by
            # _setup_custom_llm() only when a valid endpoint is explicitly configured.
            from ragas.metrics import (
                NonLLMContextPrecisionWithReference,
                NonLLMContextRecall,
                BleuScore,
                RougeScore,
            )

            self.evaluate_func = evaluate
            self.metrics = [
                NonLLMContextPrecisionWithReference(),
                NonLLMContextRecall(),
                BleuScore(),
                RougeScore(),
            ]

            # Default metrics
            self.Dataset = Dataset

            # Optionally append LLM-based metrics if endpoint is configured
            self._setup_custom_llm()

            self.ragas_available = True
            logger.info(
                "✅ RAGAS evaluation components loaded successfully with model_dump fix"
            )

        except ImportError as e:
            logger.warning(f"⚠️ RAGAS not available: {e}")
            self.ragas_available = False
        except Exception as e:
            logger.error(f"❌ Error setting up RAGAS: {e}")
            self.ragas_available = False

    def _setup_custom_llm(self) -> None:
        """Setup custom LLM for RAGAS evaluation."""
        try:
            llm_config = self.ragas_metrics_config.get("llm", {})
            self.actor_preprocessing_enabled = bool(
                llm_config.get("enable_actor_preprocessing", False)
                or llm_config.get("fill_missing_answers", False)
            )

            # Check if we should use custom LLM
            use_custom_llm = llm_config.get("use_custom_llm", False)

            if not use_custom_llm:
                logger.info("💡 Using default RAGAS LLM configuration")
                return

            logger.info("🔧 Setting up custom LLM for RAGAS evaluation...")

            # Import necessary RAGAS components for custom LLM
            try:
                from langchain_openai import ChatOpenAI
                from ragas import RunConfig  # type: ignore[attr-defined]
                from ragas.llms import LangchainLLMWrapper

                actor_config = dict(llm_config)
                actor_config.update(llm_config.get("actor", {}))
                critic_config = dict(llm_config)
                critic_config.update(llm_config.get("critic", {}))

                # Create custom LLM using your API
                endpoint = self._normalize_chat_endpoint(
                    str(actor_config.get("endpoint", "") or "")
                )

                logger.info(f"🔧 Using cleaned endpoint: {endpoint}")

                actor_max_tokens = int(actor_config.get("max_length", 512) or 512)
                custom_llm = ChatOpenAI(  # type: ignore[call-arg]
                    base_url=endpoint,
                    api_key=actor_config.get("api_key"),
                    model=actor_config.get("model_name", "gpt-4o"),
                    temperature=actor_config.get("temperature", 0.1),
                    max_tokens=actor_max_tokens,
                    request_timeout=60,
                    max_retries=3,
                )

                # Wrap for RAGAS
                self.custom_llm = LangchainLLMWrapper(custom_llm)
                self.actor_llm = self.custom_llm
                self.llm_roles["actor"] = LLMRoleBinding(
                    role="actor",
                    endpoint=endpoint,
                    model_name=actor_config.get("model_name", "gpt-4o"),
                    temperature=float(actor_config.get("temperature", 0.1)),
                    max_tokens=actor_max_tokens,
                    enabled=True,
                )

                # Setup Critic LLM (Independent model for Evaluation bias reduction)
                critic_endpoint = self._normalize_chat_endpoint(
                    str(critic_config.get("endpoint", endpoint) or endpoint)
                )

                critic_max_tokens = int(
                    critic_config.get("max_length", actor_config.get("max_length", 512))
                    or actor_config.get("max_length", 512)
                    or 512
                )
                critic_llm = ChatOpenAI(  # type: ignore[call-arg]
                    api_key=critic_config.get("api_key", actor_config.get("api_key")),
                    model=critic_config.get("model_name", "gpt-4-turbo"),
                    temperature=critic_config.get("temperature", 0.0),
                    max_tokens=critic_max_tokens,
                    base_url=critic_endpoint if critic_endpoint else None,
                )
                self.critic_llm = LangchainLLMWrapper(critic_llm)
                self.llm_roles["critic"] = LLMRoleBinding(
                    role="critic",
                    endpoint=critic_endpoint,
                    model_name=critic_config.get("model_name", "gpt-4-turbo"),
                    temperature=float(critic_config.get("temperature", 0.0)),
                    max_tokens=critic_max_tokens,
                    enabled=True,
                )

                # Set custom LLM for each metric
                for metric in self.metrics:
                    if hasattr(metric, "llm"):
                        metric.llm = (
                            self.critic_llm
                        )  # Use critic specifically for metrics

                # Append LLM-based metrics now that a valid endpoint is configured
                try:
                    from ragas.metrics import context_precision, faithfulness

                    self.metrics = self.metrics + [context_precision, faithfulness]
                    logger.info(
                        "✅ LLM-based metrics appended (context_precision, faithfulness)"
                    )
                except ImportError as _e:
                    logger.warning(f"⚠️ Could not import LLM-based metrics: {_e}")

                logger.info("✅ Actor/Critic LLM roles configured for RAGAS evaluation")
                logger.info(
                    "   🎭 Actor: %s via %s",
                    self.llm_roles["actor"].model_name,
                    self.llm_roles["actor"].endpoint,
                )
                logger.info(
                    "   🧪 Critic: %s via %s",
                    self.llm_roles["critic"].model_name,
                    self.llm_roles["critic"].endpoint,
                )

            except ImportError as e:
                logger.warning(f"⚠️ Cannot setup custom LLM, missing dependencies: {e}")
                logger.info("💡 Falling back to default RAGAS configuration")

        except Exception as e:
            logger.error(f"❌ Error setting up custom LLM: {e}")
            logger.info("💡 Continuing with default RAGAS configuration")

    def _normalize_chat_endpoint(self, endpoint: str) -> str:
        normalized = endpoint.strip()
        while normalized.endswith("/"):
            normalized = normalized[:-1]
        if normalized.endswith("/chat/completions"):
            normalized = normalized[: -len("/chat/completions")]
        normalized = normalized.replace("/chat/completions", "")
        if normalized.endswith("/v1"):
            normalized = normalized[: -len("/v1")]
        return normalized

    def _merge_metric_payloads(
        self,
        base_result: Dict[str, Any],
        extra_metrics: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not extra_metrics:
            return base_result
        base_result.setdefault("metrics", {}).update(extra_metrics)
        valid_means = [
            metric.get("mean", 0.0)
            for metric in base_result["metrics"].values()
            if metric.get("mean") is not None
        ]
        base_result.setdefault("summary", {})
        base_result["summary"]["total_metrics"] = len(base_result["metrics"])
        base_result["summary"]["average_score"] = (
            sum(valid_means) / len(valid_means) if valid_means else 0.0
        )
        base_result["summary"]["metric_names"] = list(base_result["metrics"].keys())
        base_result["summary"]["valid_metrics"] = len(valid_means)
        base_result["summary"]["domain_score"] = self._compute_weighted_domain_score(
            base_result["metrics"]
        )
        return base_result

    def _collect_alignment_failures(
        self, rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        threshold = float(
            self.ragas_metrics_config.get("alignment", {}).get(
                "confidence_threshold", 0.6
            )
        )
        queued = 0
        for response in rag_responses:
            answer = str(response.get("answer") or response.get("rag_answer") or "").strip()
            reference = str(
                response.get("ground_truth")
                or response.get("reference")
                or response.get("expected_answer")
                or ""
            ).strip()
            confidence = response.get("confidence", response.get("rag_confidence", 1.0))
            try:
                confidence_value = float(confidence)
            except (TypeError, ValueError):
                confidence_value = 1.0
            if not reference:
                continue
            if answer and confidence_value >= threshold:
                continue
            self.alignment_pipeline.ingest_failure(
                prompt=str(response.get("question") or response.get("user_input") or ""),
                bad_response=answer,
                expected_ideal=reference,
                metadata={
                    "confidence": confidence_value,
                    "response_id": response.get("id"),
                },
            )
            queued += 1

        result: Dict[str, Any] = {"queued_failures": queued}
        if queued and self.alignment_pipeline.should_auto_run():
            result["training_run"] = self.alignment_pipeline.run_alignment_cycle()
        elif queued:
            result["training_run"] = {
                "executed": False,
                "sample_count": len(self.alignment_pipeline.failure_queue),
                "dataset_path": str(self.alignment_pipeline.dataset_file),
            }
        else:
            result["training_run"] = {
                "executed": False,
                "sample_count": 0,
                "dataset_path": str(self.alignment_pipeline.dataset_file),
            }
        return result

    def _evaluate_agentic_metrics(
        self, rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        def _safe_mean(values: List[float]) -> float:
            return sum(values) / len(values) if values else 0.0

        tool_selection_scores: List[float] = []
        tool_efficiency_scores: List[float] = []

        for response in rag_responses:
            tool_calls = response.get("tool_calls") or response.get("agent_trace") or []
            expected_tools = response.get("expected_tools") or []

            normalized_tools: List[str] = []
            for tool_call in tool_calls:
                if isinstance(tool_call, dict):
                    tool_name = tool_call.get("tool") or tool_call.get("name") or tool_call.get("id")
                else:
                    tool_name = tool_call
                if tool_name:
                    normalized_tools.append(str(tool_name))

            if expected_tools:
                expected = {str(tool) for tool in expected_tools}
                used = set(normalized_tools)
                tool_selection_scores.append(len(expected & used) / len(expected))

            if normalized_tools:
                successful_calls = 0
                for tool_call in tool_calls:
                    if isinstance(tool_call, dict):
                        status = str(tool_call.get("status", "success")).lower()
                        if status not in {"error", "failed", "failure"}:
                            successful_calls += 1
                    else:
                        successful_calls += 1

                unique_tools = len(set(normalized_tools))
                total_calls = len(normalized_tools)
                tool_efficiency_scores.append(
                    (successful_calls / total_calls) * (unique_tools / total_calls)
                )

        metrics: Dict[str, Dict[str, Any]] = {}
        if tool_selection_scores:
            metrics["tool_selection_accuracy"] = {
                "mean": _safe_mean(tool_selection_scores),
                "std": float(np.std(tool_selection_scores)) if len(tool_selection_scores) > 1 else 0.0,
                "min": min(tool_selection_scores),
                "max": max(tool_selection_scores),
                "valid_count": len(tool_selection_scores),
                "total_count": len(tool_selection_scores),
                "scores": tool_selection_scores,
            }
        if tool_efficiency_scores:
            metrics["tool_use_efficiency"] = {
                "mean": _safe_mean(tool_efficiency_scores),
                "std": float(np.std(tool_efficiency_scores)) if len(tool_efficiency_scores) > 1 else 0.0,
                "min": min(tool_efficiency_scores),
                "max": max(tool_efficiency_scores),
                "valid_count": len(tool_efficiency_scores),
                "total_count": len(tool_efficiency_scores),
                "scores": tool_efficiency_scores,
            }
        return metrics

    def _evaluate_swarm_metrics(
        self, rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        agreement_scores: List[float] = []
        revision_scores: List[float] = []

        for response in rag_responses:
            question = str(response.get("question") or response.get("user_input") or "")
            answer = str(response.get("answer") or response.get("rag_answer") or "")
            if not question or not answer:
                continue

            verdict = self.swarm_synthesizer.debate_answer(question, answer)
            agreement_scores.append(float(verdict.get("agreement_rate", 0.0)))
            revision_scores.append(1.0 if verdict.get("dissent_reasons") else 0.0)

        metrics: Dict[str, Dict[str, Any]] = {}
        if agreement_scores:
            metrics["swarm_agreement_rate"] = {
                "mean": sum(agreement_scores) / len(agreement_scores),
                "std": float(np.std(agreement_scores)) if len(agreement_scores) > 1 else 0.0,
                "min": min(agreement_scores),
                "max": max(agreement_scores),
                "valid_count": len(agreement_scores),
                "total_count": len(agreement_scores),
                "scores": agreement_scores,
            }
        if revision_scores:
            metrics["swarm_revision_rate"] = {
                "mean": sum(revision_scores) / len(revision_scores),
                "std": float(np.std(revision_scores)) if len(revision_scores) > 1 else 0.0,
                "min": min(revision_scores),
                "max": max(revision_scores),
                "valid_count": len(revision_scores),
                "total_count": len(revision_scores),
                "scores": revision_scores,
            }
        return metrics

    def _evaluate_symbolic_metrics(
        self, rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        proof_scores: List[float] = []

        for response in rag_responses:
            question = str(response.get("question") or response.get("user_input") or "")
            answer = str(response.get("answer") or response.get("rag_answer") or "")
            contexts = response.get("contexts") or response.get("rag_contexts") or []
            if isinstance(contexts, list):
                context_text = " ".join(
                    str(item.get("content", item)) if isinstance(item, dict) else str(item)
                    for item in contexts
                )
            else:
                context_text = str(contexts)

            symbolic_answer, was_proven = self.symbolic_engine.generate(question, context_text)
            proof_score = self.symbolic_evaluator.evaluate_proof(answer, context_text, was_proven)
            if was_proven and "Formally Proven Fact:" in symbolic_answer:
                proven_fact = symbolic_answer.split("Formally Proven Fact:", 1)[1].strip().lower()
                if proven_fact and proven_fact not in answer.lower():
                    proof_score = 0.0
            proof_scores.append(float(proof_score))

        if not proof_scores:
            return {}
        return {
            "symbolic_proof_score": {
                "mean": sum(proof_scores) / len(proof_scores),
                "std": float(np.std(proof_scores)) if len(proof_scores) > 1 else 0.0,
                "min": min(proof_scores),
                "max": max(proof_scores),
                "valid_count": len(proof_scores),
                "total_count": len(proof_scores),
                "scores": proof_scores,
            }
        }

    def _evaluate_spatial_metrics(
        self, rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        spatial_scores: List[float] = []
        for response in rag_responses:
            coordinates = response.get("coordinates")
            if not isinstance(coordinates, (list, tuple)) or len(coordinates) != 3:
                continue
            question = str(response.get("question") or response.get("user_input") or "")
            answer = str(response.get("answer") or response.get("rag_answer") or "")
            spatial_scores.append(
                float(
                    self.spatial_evaluator.evaluate_spatial_reasoning(
                        question,
                        (int(coordinates[0]), int(coordinates[1]), int(coordinates[2])),
                        answer,
                    )
                )
            )

        if not spatial_scores:
            return {}
        return {
            "spatial_reasoning_score": {
                "mean": sum(spatial_scores) / len(spatial_scores),
                "std": float(np.std(spatial_scores)) if len(spatial_scores) > 1 else 0.0,
                "min": min(spatial_scores),
                "max": max(spatial_scores),
                "valid_count": len(spatial_scores),
                "total_count": len(spatial_scores),
                "scores": spatial_scores,
            }
        }

    def _evaluate_intent_metrics(
        self, rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        alignment_scores: List[float] = []
        for response in rag_responses:
            eeg_signal = response.get("eeg_signal")
            if not isinstance(eeg_signal, list) or not eeg_signal:
                continue
            intent = self.intent_evaluator.decode_eeg([float(value) for value in eeg_signal])
            answer = str(response.get("answer") or response.get("rag_answer") or "")
            alignment_scores.append(float(self.intent_evaluator.calculate_alignment(intent, answer)))

        if not alignment_scores:
            return {}
        return {
            "intent_alignment_score": {
                "mean": sum(alignment_scores) / len(alignment_scores),
                "std": float(np.std(alignment_scores)) if len(alignment_scores) > 1 else 0.0,
                "min": min(alignment_scores),
                "max": max(alignment_scores),
                "valid_count": len(alignment_scores),
                "total_count": len(alignment_scores),
                "scores": alignment_scores,
            }
        }

    def _evaluate_temporal_metrics(
        self, rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Dict[str, Any]]:
        temporal_scores: List[float] = []
        for response in rag_responses:
            timeline = response.get("timeline") or response.get("current_events")
            if not isinstance(timeline, list) or not timeline:
                continue
            anomaly = response.get("temporal_anomaly") or response.get("anomaly")
            prediction = str(
                response.get("prediction")
                or response.get("answer")
                or response.get("rag_answer")
                or ""
            )
            if not prediction:
                continue
            normalized_timeline = [str(item) for item in timeline]
            if anomaly:
                normalized_timeline = self.temporal_evaluator.inject_temporal_perturbation(
                    normalized_timeline, str(anomaly)
                )
            temporal_scores.append(
                float(
                    self.temporal_evaluator.score_prediction(
                        normalized_timeline, prediction
                    )
                )
            )

        if not temporal_scores:
            return {}
        return {
            "temporal_causality_score": {
                "mean": sum(temporal_scores) / len(temporal_scores),
                "std": float(np.std(temporal_scores)) if len(temporal_scores) > 1 else 0.0,
                "min": min(temporal_scores),
                "max": max(temporal_scores),
                "valid_count": len(temporal_scores),
                "total_count": len(temporal_scores),
                "scores": temporal_scores,
            }
        }

    def evaluate(
        self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate RAG responses using RAGAS metrics.

        Args:
            testset: The test dataset
            rag_responses: RAG system responses

        Returns:
            Dictionary containing RAGAS evaluation results
        """
        if not self.ragas_available:
            logger.warning("⚠️ RAGAS evaluation skipped - RAGAS not available")
            return attach_result_contract({
                "error": "RAGAS not available",
                "available": False,
                "message": "RAGAS evaluation requires ragas library installation",
            }, result_source="ragas_unavailable", success=False, error_stage="ragas_import", mock_data=False)

        logger.info("🔍 Starting RAGAS evaluation...")

        try:
            # Prepare data for RAGAS evaluation
            evaluation_data = self._prepare_evaluation_data(testset, rag_responses)

            if not evaluation_data:
                return attach_result_contract({
                    "error": "No valid data for evaluation",
                    "available": True,
                    "message": "Could not prepare data for RAGAS evaluation",
                }, result_source="ragas_input_validation", success=False, error_stage="prepare_evaluation_data", mock_data=False)

            # Create proper RAGAS dataset
            try:
                # Use the datasets.Dataset.from_dict approach which works with RAGAS evaluate
                # Fix data format for RAGAS 0.2.x compatibility
                fixed_data = RagasModelDumpFix.fix_ragas_dataset_format(evaluation_data)
                dataset = RagasModelDumpFix.create_safe_ragas_dataset(fixed_data)

                logger.info(f"📊 Created RAGAS dataset with {len(dataset)} samples")
                logger.info(f"📊 Dataset columns: {dataset.column_names}")

            except Exception as e:
                logger.error(f"❌ Failed to create RAGAS dataset: {e}")
                return attach_result_contract({
                    "error": f"Dataset creation failed: {e}",
                    "available": True,
                    "message": "Could not create RAGAS dataset",
                }, result_source="ragas_dataset_creation_error", success=False, error_stage="create_safe_ragas_dataset", mock_data=False)

            # Try RAGAS evaluation with proper error handling and fallback
            try:
                logger.info("🤖 Running RAGAS evaluation with model_dump fix...")

                # Use safe RAGAS evaluation with model_dump fix
                results = RagasModelDumpFix.safe_ragas_evaluate(
                    dataset=dataset, metrics=self.metrics
                )

                if results is None:
                    logger.error(
                        "❌ RAGAS offline evaluation returned None — "
                        "no mock fallback in CI/offline mode"
                    )
                    return evaluation_error_result(
                        result_source="ragas_safe_evaluate_failed",
                        error_stage="ragas_safe_evaluate",
                        error=(
                            "Offline RAGAS evaluation returned None. "
                            "Check that datasets has required columns and metric "
                            "dependencies (sacrebleu, rouge_score) are installed."
                        ),
                        mock_data=False,
                        extra={"available": True},
                    )

                logger.info("✅ RAGAS evaluation completed successfully")

            except Exception as e:
                logger.error(
                    f"❌ RAGAS offline evaluation raised an exception: {e}. "
                    "No mock fallback — returning error result for CI reliability."
                )
                return evaluation_error_result(
                    result_source="ragas_exception_fallback",
                    error_stage="ragas_exception_fallback",
                    error=str(e),
                    mock_data=False,
                    extra={
                        "available": True,
                        "message": "RAGAS offline evaluation failed — no mock fallback",
                    },
                )

            domain_regex_scores = self._score_domain_regex(evaluation_data["answer"])

            # Format results
            formatted_results = self._format_results(results, domain_regex_scores)

            multimodal_results = self.multimodal_evaluator.evaluate_responses(
                rag_responses
            )
            formatted_results = self._merge_metric_payloads(
                formatted_results,
                cast(Dict[str, Dict[str, Any]], multimodal_results.get("metrics", {})),
            )
            formatted_results = self._merge_metric_payloads(
                formatted_results,
                self._evaluate_agentic_metrics(rag_responses),
            )
            formatted_results = self._merge_metric_payloads(
                formatted_results,
                self._evaluate_swarm_metrics(rag_responses),
            )
            formatted_results = self._merge_metric_payloads(
                formatted_results,
                self._evaluate_symbolic_metrics(rag_responses),
            )
            formatted_results = self._merge_metric_payloads(
                formatted_results,
                self._evaluate_spatial_metrics(rag_responses),
            )
            formatted_results = self._merge_metric_payloads(
                formatted_results,
                self._evaluate_intent_metrics(rag_responses),
            )
            formatted_results = self._merge_metric_payloads(
                formatted_results,
                self._evaluate_temporal_metrics(rag_responses),
            )
            formatted_results["multimodal"] = {
                "modalities_present": multimodal_results.get("modalities_present", {})
            }
            formatted_results["agentic"] = {
                "responses_with_tool_calls": sum(
                    1
                    for response in rag_responses
                    if response.get("tool_calls") or response.get("agent_trace")
                ),
                "responses_with_expected_tools": sum(
                    1 for response in rag_responses if response.get("expected_tools")
                ),
            }
            formatted_results["alignment"] = self._collect_alignment_failures(
                rag_responses
            )

            logger.info("✅ RAGAS evaluation completed successfully")
            return attach_result_contract(
                formatted_results,
                result_source="ragas_evaluator",
                success=True,
                error_stage=None,
                mock_data=False,
            )

        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            return evaluation_error_result(
                result_source="ragas_evaluator_error",
                error_stage="evaluate",
                error=str(e),
                mock_data=False,
                extra={
                    "available": True,
                    "message": "RAGAS evaluation encountered an error",
                },
            )

    def _prepare_evaluation_data(
        self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]
    ) -> Dict[str, List]:
        """Prepare data for RAGAS evaluation with proper type conversion."""
        questions = testset.get("questions", [])
        ground_truths = testset.get("ground_truths", [])
        contexts = testset.get("contexts", [])

        # Ensure we have matching data
        min_length = (
            min(len(questions), len(rag_responses))
            if questions and rag_responses
            else 0
        )

        if min_length == 0:
            logger.warning("⚠️ No matching data for RAGAS evaluation")
            return {}

        # Prepare evaluation data with strict type checking and conversion
        evaluation_data: Dict[str, List[Any]] = {
            "question": [],
            "answer": [],
            "contexts": [],
            "ground_truth": [],
        }

        # Process questions - ensure they are strings and handle concatenated strings
        for i in range(min_length):
            q = questions[i] if i < len(questions) else ""

            # Handle cases where multiple questions are concatenated
            if isinstance(q, str):
                # Clean and split concatenated questions - take the first complete question
                q = q.strip()

                # Check for concatenated questions (multiple question marks)
                if q.count("?") > 1:
                    # Split by question mark and take the first complete question
                    import re

                    # Split on question mark followed by capital letter or common question words
                    question_parts = re.split(
                        r"\?(?=\s*[A-Z]|What|How|Why|When|Where|Who|Which)", q
                    )
                    if question_parts:
                        q = (
                            question_parts[0].strip() + "?"
                            if not question_parts[0].endswith("?")
                            else question_parts[0].strip()
                        )

                # Limit length to prevent very long concatenated text
                if len(q) > 200:
                    q = q[:197] + "..."

            # Convert to string and clean
            q_str = str(q).strip()
            if not q_str or q_str == "?":
                q_str = f"What is the main topic discussed in section {i+1}?"

            evaluation_data["question"].append(q_str)

        # Process answers - ensure they are strings
        for i in range(min_length):
            resp = rag_responses[i] if i < len(rag_responses) else {}
            answer = resp.get("answer", "") if isinstance(resp, dict) else str(resp)
            question = evaluation_data["question"][i] if i < len(evaluation_data["question"]) else ""

            # Convert to string and clean
            answer_str = str(answer).strip()
            if not answer_str:
                answer_str = self._prepare_answer_for_evaluation(
                    question=question,
                    contexts=contexts[i] if i < len(contexts) else [],
                    fallback_text=f"No answer available for question {i+1}",
                )
            elif self.actor_preprocessing_enabled:
                answer_str = self._prepare_answer_for_evaluation(
                    question=question,
                    answer=answer_str,
                    contexts=contexts[i] if i < len(contexts) else [],
                    fallback_text=answer_str,
                )

            evaluation_data["answer"].append(answer_str)

        # Process contexts - ensure they are lists of strings, handle RAGAS Document objects
        for i in range(min_length):
            if i < len(contexts):
                ctx = contexts[i]
                if isinstance(ctx, str):
                    evaluation_data["contexts"].append([ctx.strip()])
                elif isinstance(ctx, list):
                    # Handle list of contexts - convert each to string
                    ctx_list = []
                    for c in ctx:
                        if hasattr(c, "page_content"):  # RAGAS Document object
                            ctx_list.append(str(c.page_content).strip())
                        else:
                            ctx_list.append(str(c).strip())
                    evaluation_data["contexts"].append([c for c in ctx_list if c])
                elif hasattr(ctx, "page_content"):  # Single RAGAS Document object
                    evaluation_data["contexts"].append([str(ctx.page_content).strip()])
                else:
                    evaluation_data["contexts"].append([str(ctx).strip()])
            else:
                evaluation_data["contexts"].append([f"Context for question {i+1}"])

        # Process ground truths - ensure they are strings
        for i in range(min_length):
            if i < len(ground_truths):
                gt = ground_truths[i]
                gt_str = str(gt).strip()
                if not gt_str:
                    gt_str = f"Expected answer for question {i+1}"
                evaluation_data["ground_truth"].append(gt_str)
            else:
                evaluation_data["ground_truth"].append(
                    f"Expected answer for question {i+1}"
                )

        # Validate data consistency
        lengths = [len(evaluation_data[key]) for key in evaluation_data.keys()]
        if len(set(lengths)) > 1:
            logger.warning(
                f"⚠️ Inconsistent data lengths: {dict(zip(evaluation_data.keys(), lengths))}"
            )
            # Trim all to minimum length
            min_len = min(lengths)
            for key in evaluation_data.keys():
                evaluation_data[key] = evaluation_data[key][:min_len]

        logger.info(
            f"📊 Prepared {len(evaluation_data['question'])} samples for RAGAS evaluation"
        )
        if evaluation_data["question"]:
            logger.debug(
                f"🔍 Sample question: {evaluation_data['question'][0][:100]}..."
            )
            logger.debug(f"🔍 Sample answer: {evaluation_data['answer'][0][:100]}...")

        return evaluation_data

    def _prepare_answer_for_evaluation(
        self,
        question: str,
        contexts: Any,
        fallback_text: str,
        answer: str = "",
    ) -> str:
        if not self.actor_preprocessing_enabled:
            return answer or fallback_text

        actor = getattr(self, "actor_llm", None)
        if actor is None or not hasattr(actor, "invoke"):
            return answer or fallback_text

        try:
            prepared = actor.invoke(
                {
                    "question": question,
                    "answer": answer,
                    "contexts": contexts,
                    "fallback_text": fallback_text,
                }
            )
        except Exception as exc:
            logger.warning("⚠️ Actor preprocessing failed, using fallback answer: %s", exc)
            return answer or fallback_text

        prepared_text = str(prepared).strip()
        return prepared_text or answer or fallback_text

    def _score_domain_regex(self, answers: List[str]) -> List[float]:
        if not self.heuristic.is_enabled():
            return []
        return [self.heuristic.score(answer) for answer in answers]

    def _format_results(
        self, results: Any, domain_regex_scores: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Format RAGAS results for consistent output with robust NaN handling."""

        def safe_mean(values: List[float]) -> float:
            """Calculate mean safely handling NaN values"""
            if not values:
                return 0.0
            valid_values = [
                v
                for v in values
                if v is not None and not (isinstance(v, float) and math.isnan(v))
            ]
            return sum(valid_values) / len(valid_values) if valid_values else 0.0

        def calculate_robust_summary_stats(
            scores: List[Any], metric_name: str
        ) -> Dict[str, Any]:
            """Calculate robust summary statistics"""
            if not scores:
                return {
                    "mean_score": 0.0,
                    "std_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0,
                    "valid_count": 0,
                    "total_count": 0,
                    "individual_scores": [],
                }

            # Filter valid scores
            valid_scores = []
            for score in scores:
                if score is not None and not (
                    isinstance(score, float) and math.isnan(score)
                ):
                    valid_scores.append(score)

            if not valid_scores:
                return {
                    "mean_score": 0.0,
                    "std_score": 0.0,
                    "min_score": 0.0,
                    "max_score": 0.0,
                    "valid_count": 0,
                    "total_count": len(scores),
                    "individual_scores": [],
                }

            return {
                "mean_score": safe_mean(valid_scores),
                "std_score": (
                    float(np.std(valid_scores)) if len(valid_scores) > 1 else 0.0
                ),
                "min_score": min(valid_scores),
                "max_score": max(valid_scores),
                "valid_count": len(valid_scores),
                "total_count": len(scores),
                "individual_scores": valid_scores,
            }

        formatted: Dict[str, Any] = {
            "available": True,
            "metrics": {},
            "summary": {},
            "llm_roles": {
                name: {
                    "model_name": binding.model_name,
                    "endpoint": binding.endpoint,
                    "result_source": "ragas_fallback_mock",
                    "error_stage": "ragas_exception_fallback",
                    "temperature": binding.temperature,
                    "max_tokens": binding.max_tokens,
                    "enabled": binding.enabled,
                }
                for name, binding in self.llm_roles.items()
            },
        }

        # Extract metric scores
        if hasattr(results, "to_pandas"):
            df = results.to_pandas()
            for column in df.columns:
                if column not in [
                    "question",
                    "answer",
                    "contexts",
                    "ground_truth",
                    "user_input",
                    "response",
                    "retrieved_contexts",
                    "reference",
                ]:
                    # Get all scores for this metric
                    scores = df[column].tolist()

                    # Calculate robust statistics
                    stats = calculate_robust_summary_stats(scores, column)

                    formatted["metrics"][column] = {
                        "mean": stats["mean_score"],
                        "std": stats["std_score"],
                        "min": stats["min_score"],
                        "max": stats["max_score"],
                        "valid_count": stats["valid_count"],
                        "total_count": stats["total_count"],
                        "scores": stats["individual_scores"],
                    }

        if domain_regex_scores:
            regex_stats = calculate_robust_summary_stats(domain_regex_scores, "domain_regex")
            formatted["metrics"]["domain_regex"] = {
                "mean": regex_stats["mean_score"],
                "std": regex_stats["std_score"],
                "min": regex_stats["min_score"],
                "max": regex_stats["max_score"],
                "valid_count": regex_stats["valid_count"],
                "total_count": regex_stats["total_count"],
                "scores": regex_stats["individual_scores"],
            }

        # Calculate overall summary with robust handling
        if formatted["metrics"]:
            valid_means = [
                m["mean"]
                for m in formatted["metrics"].values()
                if m["mean"] is not None
            ]

            formatted["summary"] = {
                "total_metrics": len(formatted["metrics"]),
                "average_score": safe_mean(valid_means),
                "metric_names": list(formatted["metrics"].keys()),
                "valid_metrics": len(valid_means),
                "domain_score": self._compute_weighted_domain_score(formatted["metrics"]),
            }

        return formatted

    def _compute_weighted_domain_score(self, metrics: Dict[str, Dict[str, Any]]) -> float:
        weighted_total = 0.0
        total_weight = 0.0
        for metric_name, weight in self.metric_weights.items():
            metric_result = metrics.get(metric_name)
            if not metric_result or weight <= 0:
                continue
            weighted_total += metric_result.get("mean", 0.0) * weight
            total_weight += weight

        if total_weight == 0:
            available_means = [metric.get("mean", 0.0) for metric in metrics.values()]
            return sum(available_means) / len(available_means) if available_means else 0.0

        return weighted_total / total_weight

    def evaluate_testset(self, testset_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate testset data directly using RAGAS metrics.

        Args:
            testset_data: Testset data containing qa_pairs

        Returns:
            List of evaluation results
        """
        logger.info("🔄 Running RAGAS evaluation on testset...")

        try:
            qa_pairs = testset_data.get("qa_pairs", [])
            if not qa_pairs:
                logger.warning("No QA pairs found in testset data")
                return []

            # Convert qa_pairs to RAGAS expected format
            questions = []
            ground_truths = []
            contexts = []
            mock_rag_responses = []

            for qa_pair in qa_pairs:
                question = qa_pair.get("user_input", qa_pair.get("question", ""))
                reference = qa_pair.get("reference", "")
                context = qa_pair.get("contexts", "")

                questions.append(question)
                ground_truths.append(reference)

                # Handle contexts - ensure it's a list
                if isinstance(context, str):
                    contexts.append([context])
                elif isinstance(context, list):
                    contexts.append(context)
                else:
                    contexts.append([str(context)])

                # Mock RAG response
                mock_response = {
                    "question": question,
                    "answer": reference,  # Use reference as answer for evaluation
                    "contexts": contexts[-1],
                    "ground_truth": reference,
                }
                mock_rag_responses.append(mock_response)

            # Create proper testset format for RAGAS
            formatted_testset = {
                "questions": questions,
                "ground_truths": ground_truths,
                "contexts": contexts,
            }

            # Run RAGAS evaluation with model_dump fix
            evaluation_result = self.evaluate(formatted_testset, mock_rag_responses)

            # Convert to list format expected by stage factory
            results = []
            for i, qa_pair in enumerate(qa_pairs):
                # Extract score from evaluation result
                ragas_score = 0.0
                if "summary" in evaluation_result:
                    ragas_score = evaluation_result["summary"].get("average_score", 0.0)
                elif "average_score" in evaluation_result:
                    ragas_score = evaluation_result.get("average_score", 0.0)

                result_item = {
                    **qa_pair,
                    "ragas_score": ragas_score,
                    "ragas_metrics": evaluation_result.get("metrics", {}),
                    "ragas_evaluation": evaluation_result,
                }
                results.append(result_item)

            logger.info(f"✅ RAGAS evaluation completed for {len(results)} items")
            return results

        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            return []

    def is_available(self) -> bool:
        """Check if RAGAS evaluation is available."""
        return self.ragas_available

    def evaluate_responses(self, rag_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate already collected RAG responses from the pipeline coordinator."""
        questions: List[str] = []
        ground_truths: List[str] = []
        contexts: List[List[Any]] = []
        normalized_responses: List[Dict[str, Any]] = []

        for response in rag_responses:
            question = str(response.get("question") or response.get("user_input") or "")
            answer = str(response.get("answer") or response.get("rag_answer") or "")
            ground_truth = str(
                response.get("ground_truth") or response.get("reference") or ""
            )
            context_payload = (
                response.get("contexts")
                or response.get("rag_contexts")
                or response.get("retrieved_contexts")
                or []
            )
            if isinstance(context_payload, list):
                contexts_list = context_payload
            else:
                contexts_list = [context_payload]

            questions.append(question)
            ground_truths.append(ground_truth)
            contexts.append(contexts_list)
            normalized_responses.append(
                {
                    **response,
                    "answer": answer,
                    "contexts": contexts_list,
                }
            )

        return self.evaluate(
            {
                "questions": questions,
                "ground_truths": ground_truths,
                "contexts": contexts,
            },
            normalized_responses,
        )
