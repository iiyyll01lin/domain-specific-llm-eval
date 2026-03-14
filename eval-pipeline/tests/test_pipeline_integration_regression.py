from __future__ import annotations

import json
from pathlib import Path

from src.pipeline.orchestrator import PipelineOrchestrator
from src.ui.app_store_marketplace import UnifiedAppStore
from src.ui.force_graph_viewer import ForceGraphVisualizer


class _FederatedResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class _FederatedSession:
    def post(self, endpoint, json=None, timeout=0):
        return _FederatedResponse({"accepted": True, "endpoint": endpoint})


def test_orchestrator_taxonomy_topology_appstore_and_federated_helpers(tmp_path: Path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "demo.json").write_text(
        json.dumps(
            {
                "id": "demo-runbook",
                "name": "Demo Runbook",
                "author": "Example",
                "version": "1.0.0",
                "trust": "verified",
                "dependencies": "base-eval",
            }
        ),
        encoding="utf-8",
    )
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("patient_id,organization\n1,HealthOrg\n", encoding="utf-8")

    kg_dir = tmp_path / "testsets" / "knowledge_graphs"
    kg_dir.mkdir(parents=True)
    (kg_dir / "knowledge_graph_test.json").write_text(
        json.dumps(
            {
                "nodes": [{"id": "A"}, {"id": "B"}],
                "relationships": [{"source": "A", "target": "B"}],
                "metadata": {"run_id": "test", "created_at": "2026-03-14T00:00:00"},
            }
        ),
        encoding="utf-8",
    )

    orchestrator = object.__new__(PipelineOrchestrator)
    orchestrator.run_id = "run-test"
    orchestrator.config = {
        "taxonomy_discovery": {"enabled": True},
        "topology": {"enabled": True},
        "app_store": {"enabled": True, "manifest_dir": str(manifest_dir), "auto_install": ["demo-runbook"]},
        "data_sources": {"csv": {"csv_files": [str(csv_path)]}},
        "distributed": {"federated_learning": {"enabled": True, "tenant": "tenant-a"}},
    }
    orchestrator.output_dirs = {
        "metadata": tmp_path / "metadata",
        "reports": tmp_path / "reports",
    }
    orchestrator.output_dirs["metadata"].mkdir(parents=True, exist_ok=True)
    orchestrator.output_dirs["reports"].mkdir(parents=True, exist_ok=True)
    orchestrator.taxonomy_discoverer = __import__(
        "src.loaders.taxonomy_discovery", fromlist=["ZeroShotTaxonomyDiscoverer"]
    ).ZeroShotTaxonomyDiscoverer()
    orchestrator.topology_visualizer = ForceGraphVisualizer()
    orchestrator.app_store = UnifiedAppStore(
        manifest_dir=manifest_dir,
        install_dir=tmp_path / "metadata" / "installed_runbooks",
    )
    orchestrator.federated_client = __import__(
        "src.distributed.federated_learning", fromlist=["FederatedLearningClient"]
    ).FederatedLearningClient(session=_FederatedSession(), spool_dir=tmp_path / "spool")

    taxonomy = orchestrator._run_taxonomy_discovery()
    installed = orchestrator._install_requested_runbooks()
    topology = orchestrator._export_topology_artifact()
    federated = orchestrator._submit_federated_summary({"success_rate": 0.75, "total_queries": 4})

    assert taxonomy is not None
    assert "Patient" in taxonomy["entities"]
    assert installed == ["demo-runbook"]
    assert topology is not None
    assert Path(topology["payload_path"]).exists()
    assert federated is not None
    assert federated["submitted"] is True