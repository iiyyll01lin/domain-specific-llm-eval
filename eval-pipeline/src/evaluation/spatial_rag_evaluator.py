import logging
from typing import Dict, Tuple

from .deterministic_backends import DeterministicSpatialBackend

logger = logging.getLogger(__name__)


class MixedRealityMultimodalEval:
    """Evaluates Spatial and 3D Context RAG."""

    def __init__(self, backend: DeterministicSpatialBackend | None = None):
        # Mock 3D point cloud context
        self.point_cloud_map: Dict[Tuple[int, int, int], str] = {
            (0, 0, 0): "Center of the room, empty.",
            (10, 5, 2): "A robotic assembly arm.",
            (100, 100, 50): "Drone flying zone.",
        }
        self.backend = backend or DeterministicSpatialBackend()

    def retrieve_spatial_context(self, coordinates: Tuple[int, int, int]) -> str:
        return self.point_cloud_map.get(coordinates, "Unknown space.")

    def evaluate_spatial_reasoning(
        self, query: str, coordinates: Tuple[int, int, int], answer: str
    ) -> float:
        """Scores LLM ability to understand spatial context."""
        context = self.retrieve_spatial_context(coordinates)
        result = self.backend.evaluate(query, coordinates, answer, context)
        if result["score"] == 1.0:
            logger.info("Spatial reasoning is correct.")
        return float(result["score"])
