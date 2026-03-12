import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


class SwarmTelemetryIngestor:
    """Ingests ROS/Drone visual streams to provide Real-time Spatial Eval Context"""

    def __init__(self):
        self.active_drones = 0

    def connect_ros_node(self, node_id: str) -> bool:
        self.active_drones += 1
        logger.info(f"ROS Drone Node '{node_id}' connected.")
        return True

    def ingest_visual_feed(self, node_id: str, telemetry: Dict[str, Any]) -> str:
        """Converts telemetry bounds to spatial RAG strings."""
        if not telemetry:
            return "Empty Feed"

        objects = telemetry.get("detected_objects", [])
        return f"Node {node_id} sees: {', '.join(objects)}"
