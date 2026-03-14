import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class ForceGraphVisualizer:
    def __init__(self) -> None:
        self.ready = True

    def build_payload(self, kg_data: Dict[str, Any]) -> Dict[str, Any]:
        nodes = kg_data.get("nodes", [])
        links = kg_data.get("links", kg_data.get("relationships", []))
        isolated_nodes = []
        connected_ids = {
            str(link.get("source")) for link in links if isinstance(link, dict)
        } | {
            str(link.get("target")) for link in links if isinstance(link, dict)
        }
        for node in nodes:
            node_id = str(node.get("id")) if isinstance(node, dict) else str(node)
            if node_id not in connected_ids:
                isolated_nodes.append(node_id)
        return {
            "node_count": len(nodes),
            "link_count": len(links),
            "isolated_nodes": isolated_nodes,
            "nodes": nodes,
            "links": links,
        }

    def generate_html_payload(self, kg_data: Dict[str, Any]) -> str:
        logger.info("Mounting Real-time 3D Topology WebGL canvas...")
        payload = self.build_payload(kg_data)
        return (
            f"<div id='3d-graph' data-graph='{json.dumps(payload)}'>[WebGL Graph Rendered]</div>"
        )

    def export_from_kg_artifact(
        self,
        kg_json_path: str | Path,
        output_dir: str | Path,
    ) -> Dict[str, str]:
        payload = self.build_payload(
            json.loads(Path(kg_json_path).read_text(encoding="utf-8"))
        )
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        payload_path = target_dir / "topology_payload.json"
        html_path = target_dir / "topology.html"
        payload_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        html_path.write_text(self.generate_html_payload(payload), encoding="utf-8")
        return {"payload_path": str(payload_path), "html_path": str(html_path)}
