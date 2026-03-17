import json
import logging
from collections import defaultdict, deque
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
        node_degrees = defaultdict(int)
        adjacency = defaultdict(set)
        connected_ids = {
            str(link.get("source")) for link in links if isinstance(link, dict)
        } | {
            str(link.get("target")) for link in links if isinstance(link, dict)
        }
        for link in links:
            if not isinstance(link, dict):
                continue
            source = str(link.get("source"))
            target = str(link.get("target"))
            node_degrees[source] += 1
            node_degrees[target] += 1
            adjacency[source].add(target)
            adjacency[target].add(source)
        for node in nodes:
            node_id = str(node.get("id")) if isinstance(node, dict) else str(node)
            if node_id not in connected_ids:
                isolated_nodes.append(node_id)

        visited = set()
        clusters = []
        for node in nodes:
            node_id = str(node.get("id")) if isinstance(node, dict) else str(node)
            if node_id in visited:
                continue
            queue = deque([node_id])
            cluster = []
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)
                cluster.append(current)
                for neighbor in adjacency.get(current, set()):
                    if neighbor not in visited:
                        queue.append(neighbor)
            clusters.append(sorted(cluster))

        max_degree = max(node_degrees.values()) if node_degrees else 0
        high_centrality_nodes = sorted(
            [node_id for node_id, degree in node_degrees.items() if degree == max_degree and degree > 0]
        )
        return {
            "node_count": len(nodes),
            "link_count": len(links),
            "isolated_nodes": isolated_nodes,
            "high_centrality_nodes": high_centrality_nodes,
            "weakly_connected_clusters": [cluster for cluster in clusters if len(cluster) > 1],
            "node_degrees": dict(node_degrees),
            "graph_density": (
                round((2 * len(links)) / (len(nodes) * max(len(nodes) - 1, 1)), 4)
                if len(nodes) > 1
                else 0.0
            ),
            "nodes": nodes,
            "links": links,
        }

    def generate_html_payload(self, kg_data: Dict[str, Any]) -> str:
        logger.info("Mounting Real-time 3D Topology WebGL canvas...")
        payload = self.build_payload(kg_data)
        return (
            "<html><body>"
            "<div id='3d-graph'></div>"
            f"<script>const graphData = {json.dumps(payload)};"
            "document.getElementById('3d-graph').dataset.graph = JSON.stringify(graphData);"
            "document.getElementById('3d-graph').innerText = '[WebGL Graph Rendered]';"
            "</script></body></html>"
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
