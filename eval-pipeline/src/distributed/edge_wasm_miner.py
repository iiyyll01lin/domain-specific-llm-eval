import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DecentralizedEdgeMiner:
    """Distributes evaluation workloads to edge devices via WASM."""

    def __init__(self):
        self.nodes: List[str] = []

    def register_edge_node(self, node_id: str) -> None:
        self.nodes.append(node_id)
        logger.info(f"Edge node {node_id} registered.")

    def compile_to_wasm(self, eval_function: str) -> str:
        """Mocks compilation of an evaluation logic chunk to WebAssembly."""
        return f"WASM_BIN[{eval_function}]"

    def distribute_workload(self, task: str) -> Dict[str, str]:
        """Distributes the task across all registered nodes."""
        results: Dict[str, str] = {}
        wasm_payload = self.compile_to_wasm(task)

        for node in self.nodes:
            results[node] = f"Processed {wasm_payload} successfully"
            logger.info(f"Task sent to {node}")

        return results
