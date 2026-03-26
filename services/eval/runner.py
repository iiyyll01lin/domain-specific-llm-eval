from __future__ import annotations

import logging
from dataclasses import replace
from time import perf_counter
from typing import Callable, Iterable, Iterator, MutableMapping, Optional

from ragas.dataset_schema import MultiTurnSample, SingleTurnSample
from ragas.testset.synthesizers.testset_schema import TestsetSample

from services.eval.context_capture import CapturedEvaluationItem, ContextCapture
from services.eval.metrics import PrometheusRAGRequestMetrics, RAGRequestMetricsRecorder
from services.eval.rag_interface import RAGAdapter, RAGRequest, RAGResponse
from services.eval.retry_policy import RAGInvocationError, RetryPolicy

logger = logging.getLogger(__name__)


def _default_retry_policy_factory(max_attempts: int) -> RetryPolicy:
    return RetryPolicy(max_attempts=max_attempts)


class EvaluationRunner:
    """Coordinate evaluation item creation via a RAG adapter (TASK-031).

    The runner is intentionally side-effect free: persistence, streaming, and
    metric aggregation are handled by downstream tasks (TASK-033 onwards). This
    class focuses on invoking the configured :class:`RAGAdapter`, capturing the
    resulting contexts, and surfacing deterministic metadata required by later
    stages.
    """

    def __init__(
        self,
        *,
        adapter: RAGAdapter,
        context_capture: Optional[ContextCapture] = None,
        metrics_recorder: Optional[RAGRequestMetricsRecorder] = None,
        retry_policy_factory: Optional[Callable[[int], RetryPolicy]] = None,
        default_max_retries: int = 3,
    ) -> None:
        if adapter is None:
            raise ValueError("adapter must be provided")
        if default_max_retries < 1:
            raise ValueError("default_max_retries must be >= 1")

        self._adapter = adapter
        self._context_capture = context_capture or ContextCapture()
        self._metrics = metrics_recorder or PrometheusRAGRequestMetrics()
        self._retry_policy_factory = retry_policy_factory or _default_retry_policy_factory
        self._default_max_retries = default_max_retries

    def run(
        self,
        *,
        run_id: str,
        samples: Iterable[TestsetSample],
        timeout_seconds: Optional[float] = None,
        max_retries: Optional[int] = None,
        trace_id: Optional[str] = None,
    ) -> Iterator[CapturedEvaluationItem]:
        """Evaluate *samples* and yield :class:`CapturedEvaluationItem` results."""

        attempts = max(1, max_retries or self._default_max_retries)
        retry_policy = self._retry_policy_factory(attempts)

        for index, sample in enumerate(samples):
            question = self._extract_question(sample)
            sample_id = self._derive_sample_id(index, sample)
            metadata = self._build_metadata(
                sample=sample,
                index=index,
                sample_id=sample_id,
                timeout_seconds=timeout_seconds,
                max_retries=attempts,
            )

            request = RAGRequest(
                question=question,
                run_id=run_id,
                sample_id=sample_id,
                metadata=metadata,
                timeout_seconds=timeout_seconds,
                max_retries=attempts,
            )

            response, rag_attempts, latency_seconds, outcome, error_code = self._invoke_with_retry(
                request=request,
                retry_policy=retry_policy,
                trace_id=trace_id,
            )

            captured = self._context_capture.capture(request, response)
            enriched_metadata = dict(captured.metadata)
            enriched_metadata.update(
                {
                    "rag_attempts": rag_attempts,
                    "rag_outcome": outcome,
                }
            )
            if error_code:
                enriched_metadata["rag_error_code"] = error_code
            captured = replace(captured, metadata=enriched_metadata)

            yield captured

    def _invoke_with_retry(
        self,
        *,
        request: RAGRequest,
        retry_policy: RetryPolicy,
        trace_id: Optional[str],
    ) -> tuple[RAGResponse, int, float, str, Optional[str]]:
        """Call the adapter with retries and metric emission."""

        def _operation() -> tuple[RAGResponse, float]:
            start = perf_counter()
            response = self._adapter.invoke(request).ensure_contexts()
            elapsed = perf_counter() - start
            return response, elapsed

        attempts_used = 0
        latency_seconds = 0.0
        outcome = "unknown"
        error_code: Optional[str] = None

        try:
            (response, latency_seconds), telemetry = retry_policy.execute(_operation)
            attempts_used = telemetry.attempts
            outcome = "success" if response.success else "failure"
            error_code = response.error_code
        except TimeoutError as exc:
            attempts_used = retry_policy.max_attempts
            outcome = "failure"
            error_code = "timeout"
            response = RAGResponse.failure(
                error_code=error_code,
                raw={
                    "message": str(exc),
                    "type": exc.__class__.__name__,
                },
            ).ensure_contexts()
            logger.warning(
                "rag.invoke.timeout",
                extra={
                    "trace_id": trace_id,
                    "context": {
                        "sample_id": request.sample_id,
                        "run_id": request.run_id,
                        "attempts": attempts_used,
                    },
                },
            )
        except RAGInvocationError as exc:
            attempts_used = retry_policy.max_attempts
            outcome = "failure"
            error_code = exc.error_code or "rag_invocation_error"
            response = RAGResponse.failure(
                error_code=error_code,
                raw={
                    "message": str(exc),
                    "status_code": exc.status_code,
                },
            ).ensure_contexts()
            logger.warning(
                "rag.invoke.error",
                extra={
                    "trace_id": trace_id,
                    "context": {
                        "sample_id": request.sample_id,
                        "run_id": request.run_id,
                        "attempts": attempts_used,
                        "status_code": exc.status_code,
                        "error_code": exc.error_code,
                    },
                },
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            attempts_used = retry_policy.max_attempts
            outcome = "failure"
            error_code = "unexpected_error"
            response = RAGResponse.failure(
                error_code=error_code,
                raw={
                    "message": str(exc),
                    "type": exc.__class__.__name__,
                },
            ).ensure_contexts()
            logger.exception(
                "rag.invoke.unexpected",
                extra={
                    "trace_id": trace_id,
                    "context": {
                        "sample_id": request.sample_id,
                        "run_id": request.run_id,
                        "attempts": attempts_used,
                    },
                },
            )

        self._metrics.record(
            outcome=outcome,
            latency_seconds=latency_seconds,
            attempts=attempts_used,
            trace_id=trace_id,
            error_code=error_code,
        )
        return response, attempts_used, latency_seconds, outcome, error_code

    @staticmethod
    def _extract_question(sample: TestsetSample) -> str:
        eval_sample = sample.eval_sample
        if isinstance(eval_sample, SingleTurnSample):
            return eval_sample.user_input or ""
        if isinstance(eval_sample, MultiTurnSample):
            return eval_sample.to_string()
        return ""

    @staticmethod
    def _derive_sample_id(index: int, sample: TestsetSample) -> str:
        base = getattr(sample.eval_sample, "sample_id", None)
        if isinstance(base, str) and base:
            return base
        return f"sample-{index:06d}"

    @staticmethod
    def _build_metadata(
        *,
        sample: TestsetSample,
        index: int,
        sample_id: str,
        timeout_seconds: Optional[float],
        max_retries: int,
    ) -> MutableMapping[str, object]:
        metadata: MutableMapping[str, object] = {
            "sample_index": index,
            "sample_id": sample_id,
            "synthesizer_name": sample.synthesizer_name,
            "rag_config": {
                "timeout_seconds": timeout_seconds,
                "max_retries": max_retries,
            },
        }

        eval_sample = sample.eval_sample
        reference = getattr(eval_sample, "reference", None)
        if reference is not None:
            metadata["reference_answer"] = reference

        reference_contexts = getattr(eval_sample, "reference_contexts", None)
        if reference_contexts:
            metadata["reference_contexts"] = list(reference_contexts)

        rubrics = getattr(eval_sample, "rubrics", None)
        if rubrics:
            metadata["rubrics"] = dict(rubrics)

        retrieved_contexts = getattr(eval_sample, "retrieved_contexts", None)
        if retrieved_contexts:
            metadata["retrieved_contexts"] = list(retrieved_contexts)

        return metadata


__all__ = ["EvaluationRunner"]
