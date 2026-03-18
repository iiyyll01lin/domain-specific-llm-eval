import logging
from typing import Any, Dict, List

from .deterministic_backends import DeterministicTemporalBackend

logger = logging.getLogger(__name__)


class TemporalCausalityEvaluator:
    """Evaluates the LLM's future prediction and causality tracing."""

    def __init__(self, backend: DeterministicTemporalBackend | None = None):
        self.backend = backend or DeterministicTemporalBackend()

    def inject_temporal_perturbation(
        self, current_events: List[str], anomaly: str
    ) -> List[str]:
        """Injects a butterfly-effect event into the timeline."""
        perturbed = current_events.copy()
        perturbed.append(anomaly)
        logger.info(f"Injected temporal anomaly: {anomaly}")
        return perturbed

    def score_prediction(self, timeline: List[str], prediction: str) -> float:
        """Scores prediction based on deterministic causal contracts."""
        result = self.backend.evaluate(timeline, prediction)
        if result["score"] >= 0.9:
            logger.info("High causal coherence score.")
        return float(result["score"])
