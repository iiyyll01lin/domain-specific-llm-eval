from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

from ragas.testset.synthesizers.testset_schema import TestsetSample

from services.eval.aggregation.aggregator import KPIAggregationResult
from services.eval.context_capture import CapturedEvaluationItem
from services.eval.metrics.interface import MetricInput, MetricValue
from services.eval.metrics.loader import MetricRegistry
from services.eval.persistence_pipeline import EvaluationPersistencePipeline
from services.eval.runner import EvaluationRunner

logger = logging.getLogger(__name__)


def _flatten_metric_results(
    plugin_results: Mapping[str, Tuple[MetricValue, ...]],
    metric_versions: Mapping[str, str],
) -> Tuple[MutableMapping[str, float | None], MutableMapping[str, Dict[str, object]]]:
    """Normalise plugin outputs into numeric scores and metadata payloads."""

    scores: MutableMapping[str, float | None] = {}
    details: MutableMapping[str, Dict[str, object]] = {}
    for plugin_name, values in plugin_results.items():
        version = metric_versions.get(plugin_name, "unknown")
        for value in values:
            key = value.key or plugin_name
            detail: Dict[str, object] = {
                "plugin": plugin_name,
                "version": version,
            }
            if value.confidence is not None:
                detail["confidence"] = value.confidence
            if value.metadata:
                detail["metadata"] = dict(value.metadata)
            try:
                numeric_value = float(value.value)
            except (TypeError, ValueError):  # pragma: no cover - defensive guard
                numeric_value = None
            scores[key] = numeric_value
            details[key] = detail
    return scores, details


@dataclass(frozen=True)
class EvaluationExecutionSummary:
    """Aggregate output produced after completing an evaluation run."""

    run_id: str
    item_count: int
    metric_versions: Mapping[str, str]
    artifacts: Mapping[str, Path]
    aggregation: KPIAggregationResult


def execute_evaluation_run(
    *,
    run_id: str,
    samples: Iterable[TestsetSample],
    runner: EvaluationRunner,
    metric_registry: MetricRegistry,
    pipeline: EvaluationPersistencePipeline,
    timeout_seconds: Optional[float] = None,
    max_retries: Optional[int] = None,
    trace_id: Optional[str] = None,
    drain_timeout: Optional[float] = 30.0,
) -> EvaluationExecutionSummary:
    """Execute a full evaluation run and persist artefacts via the pipeline."""

    metric_versions = {
        getattr(plugin, "name", plugin.__class__.__name__): getattr(plugin, "version", "unknown")
        for plugin in metric_registry.plugins
    }

    processed = 0
    finalized = False
    aggregation: Optional[KPIAggregationResult] = None

    try:
        for item in runner.run(
            run_id=run_id,
            samples=samples,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
            trace_id=trace_id,
        ):
            metadata_snapshot = dict(item.metadata)
            raw_snapshot = dict(item.raw)

            metric_input = MetricInput(
                run_id=item.run_id,
                sample_id=item.sample_id,
                question=item.question,
                answer=item.answer,
                reference_answer=metadata_snapshot.get("reference_answer"),
                contexts=item.contexts,
                metadata=metadata_snapshot,
                raw_response=raw_snapshot,
            )

            plugin_results = metric_registry.evaluate(metric_input)
            metric_scores, metric_details = _flatten_metric_results(plugin_results, metric_versions)

            extended_metadata = dict(metadata_snapshot)
            extended_metadata["metrics"] = dict(metric_scores)
            if metric_details:
                extended_metadata["metric_details"] = dict(metric_details)
            if "metric_versions" not in extended_metadata:
                extended_metadata["metric_versions"] = dict(metric_versions)

            enriched_item: CapturedEvaluationItem = replace(item, metadata=extended_metadata, raw=raw_snapshot)

            pipeline.submit(enriched_item, metric_scores)
            processed += 1

        drained = pipeline.wait_until_drained(timeout=drain_timeout)
        if not drained:
            logger.warning(
                "evaluation pipeline drain timed out",
                extra={
                    "context": {
                        "run_id": run_id,
                        "timeout_seconds": drain_timeout,
                    }
                },
            )

        aggregation = pipeline.finalize()
        finalized = True
        artifacts = dict(pipeline.artifacts())
        logger.info(
            "evaluation run completed",
            extra={
                "context": {
                    "run_id": run_id,
                    "item_count": processed,
                    "metrics": sorted(aggregation.metrics.keys()),
                }
            },
        )

        return EvaluationExecutionSummary(
            run_id=run_id,
            item_count=processed,
            metric_versions=dict(metric_versions),
            artifacts=artifacts,
            aggregation=aggregation,
        )
    finally:
        if not finalized:
            try:
                pipeline.wait_until_drained(timeout=drain_timeout)
            except Exception:  # pragma: no cover - defensive cleanup
                logger.exception("evaluation pipeline drain during cleanup failed")
            try:
                pipeline.finalize()
            except Exception:  # pragma: no cover - defensive cleanup
                logger.exception("evaluation pipeline finalize during cleanup failed")


__all__ = ["EvaluationExecutionSummary", "execute_evaluation_run"]
