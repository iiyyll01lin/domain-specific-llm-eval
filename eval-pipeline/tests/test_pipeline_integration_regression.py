from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from src.pipeline.orchestrator import PipelineOrchestrator

from src.evaluation.evaluation_result_contract import (
    EVALUATION_RESULT_CONTRACT_VERSION,
    attach_result_contract,
    evaluation_error_result,
)

# ---------------------------------------------------------------------------
# Contract assertion helpers
# ---------------------------------------------------------------------------

_REQUIRED_CONTRACT_KEYS = {
    "success",
    "result_source",
    "error_stage",
    "mock_data",
    "contract_version",
}


def _assert_contract(result: Dict[str, Any], *, source: str, success: bool) -> None:
    """Shared helper: verify all mandatory result-contract fields are present."""
    missing = _REQUIRED_CONTRACT_KEYS - result.keys()
    assert not missing, f"Missing contract keys: {missing}"
    assert result["contract_version"] == EVALUATION_RESULT_CONTRACT_VERSION
    assert result["result_source"] == source
    assert result["success"] is success
    assert isinstance(result["mock_data"], bool)


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


class _HardwareResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class _HardwareSession:
    def get(self, endpoint, timeout=0):
        if endpoint.endswith("/models"):
            return _HardwareResponse({"data": [{"id": "model-a"}]})
        return _HardwareResponse({"gpu_utilization": 0.8, "memory_utilization": 0.5})

    def post(self, endpoint, json=None, timeout=0):
        return _HardwareResponse(
            {"choices": [{"text": f"Hardware Accelerated Response for: {(json or {}).get('prompt', '')}"}]}
        )


def test_orchestrator_taxonomy_topology_appstore_and_federated_helpers(tmp_path: Path) -> None:
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy: avoids importlib hang at collection time
    from src.ui.app_store_marketplace import UnifiedAppStore
    from src.ui.force_graph_viewer import ForceGraphVisualizer
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


def test_orchestrator_collects_hardware_acceleration_telemetry() -> None:
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy
    orchestrator = object.__new__(PipelineOrchestrator)
    orchestrator.config = {
        "inference": {
            "hardware_acceleration": {
                "enabled": True,
                "benchmark_prompts": ["benchmark prompt"],
                "benchmark_repeats": 1,
            }
        }
    }
    orchestrator.hardware_acceleration_client = __import__(
        "src.inference.vllm_client", fromlist=["vLLMInferenceClient"]
    ).vLLMInferenceClient(session=_HardwareSession())

    telemetry = orchestrator._collect_hardware_acceleration_telemetry()

    assert telemetry is not None
    assert telemetry["capabilities"]["connected"] is True
    assert telemetry["benchmarks"][0]["median_latency_seconds"] >= 0.0
    assert telemetry["observability"]["gpu_saturation"]["saturation_level"] == "moderate"
    assert telemetry["request_distribution"]["total_requests"] >= 1
    assert telemetry["benchmarks"][0]["latency_samples_seconds"]

# ---------------------------------------------------------------------------
# _run_evaluation() — end-to-end contract assertions
# ---------------------------------------------------------------------------

def _make_eval_orchestrator(tmp_path: Path) -> PipelineOrchestrator:
    """Build a minimal PipelineOrchestrator stub for evaluation-stage tests."""
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy
    orchestrator = object.__new__(PipelineOrchestrator)
    orchestrator.run_id = "run-eval-contract"
    orchestrator.config = {}
    orchestrator.output_dirs = {
        "testsets": tmp_path / "testsets",
        "metadata": tmp_path / "metadata",
        "evaluations": tmp_path / "evaluations",
    }
    for d in orchestrator.output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    orchestrator.memory_tracker = MagicMock()
    orchestrator.hyperparameter_optimizer = None
    orchestrator.federated_client = MagicMock()
    orchestrator.federated_client  # silence "unused" warning
    orchestrator.hardware_acceleration_client = None
    orchestrator.evaluation_dispatcher = MagicMock()
    return orchestrator


def test_run_evaluation_success_contract_shape(tmp_path: Path) -> None:
    """_run_evaluation() success path must carry all normalised contract fields."""
    orchestrator = _make_eval_orchestrator(tmp_path)

    # Plant a fake testset xlsx so the stage can find it
    testset_file = orchestrator.output_dirs["testsets"] / "testset.xlsx"
    testset_file.write_bytes(b"fake-xlsx")

    mock_eval_result = {
        "success": True,
        "total_queries": 5,
        "keyword_metrics": {"pass_rate": 0.8},
        "ragas_metrics": {"average_score": 0.7},
        "feedback_metrics": {"requests": 1},
        "output_file": "eval_out.json",
    }
    orchestrator.evaluation_dispatcher.evaluate_testsets.return_value = mock_eval_result

    result = orchestrator._run_evaluation()

    _assert_contract(result, source="pipeline_orchestrator_evaluation", success=True)
    assert result["testsets_evaluated"] == 1
    assert result["queries_executed"] == 5
    assert isinstance(result["duration"], float)
    assert isinstance(result["metadata_file"], str)


def test_run_evaluation_failure_contract_shape(tmp_path: Path) -> None:
    """_run_evaluation() failure path must carry all normalised contract fields."""
    orchestrator = _make_eval_orchestrator(tmp_path)
    # No testset files -> will raise immediately

    result = orchestrator._run_evaluation()

    _assert_contract(result, source="pipeline_orchestrator_evaluation", success=False)
    assert result["error_stage"] == "evaluation"
    assert "error" in result
    assert isinstance(result["error"], str)


def test_run_evaluation_evaluator_raises_contract_shape(tmp_path: Path) -> None:
    """_run_evaluation() propagates evaluator exceptions via the contract."""
    orchestrator = _make_eval_orchestrator(tmp_path)

    testset_file = orchestrator.output_dirs["testsets"] / "testset.xlsx"
    testset_file.write_bytes(b"fake-xlsx")

    orchestrator.evaluation_dispatcher.evaluate_testsets.side_effect = RuntimeError("eval-boom")

    result = orchestrator._run_evaluation()

    _assert_contract(result, source="pipeline_orchestrator_evaluation", success=False)
    assert result["error_stage"] == "evaluation"
    assert "eval-boom" in result["error"]


# ---------------------------------------------------------------------------
# _run_reporting() — end-to-end contract assertions
# ---------------------------------------------------------------------------

def _make_report_orchestrator(tmp_path: Path) -> PipelineOrchestrator:
    """Build a minimal PipelineOrchestrator stub for reporting-stage tests."""
    from src.pipeline.orchestrator import PipelineOrchestrator  # lazy
    from src.ui.force_graph_viewer import ForceGraphVisualizer  # lazy
    orchestrator = object.__new__(PipelineOrchestrator)
    orchestrator.run_id = "run-report-contract"
    orchestrator.config = {}
    orchestrator.output_dirs = {
        "metadata": tmp_path / "metadata",
        "evaluations": tmp_path / "evaluations",
        "reports": tmp_path / "reports",
    }
    for d in orchestrator.output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    orchestrator.memory_tracker = MagicMock()
    orchestrator.topology_visualizer = ForceGraphVisualizer()
    return orchestrator


def test_run_reporting_success_contract_shape(tmp_path: Path) -> None:
    """_run_reporting() success path must carry all normalised contract fields."""
    orchestrator = _make_report_orchestrator(tmp_path)

    # Plant a fake eval-results file in metadata dir
    eval_file = orchestrator.output_dirs["metadata"] / "evaluation_results_test.json"
    eval_file.write_text(json.dumps({"results": []}), encoding="utf-8")

    mock_reports = [
        {"type": "html", "file_path": str(tmp_path / "report.html")},
        {"type": "json", "file_path": str(tmp_path / "report.json")},
    ]
    orchestrator.report_generator = MagicMock()
    orchestrator.report_generator.generate_reports.return_value = mock_reports

    result = orchestrator._run_reporting()

    _assert_contract(result, source="pipeline_orchestrator_reporting", success=True)
    assert result["reports_generated"] == mock_reports
    assert isinstance(result["report_directory"], str)
    assert isinstance(result["metadata_file"], str)
    assert isinstance(result["duration"], float)


def test_run_reporting_failure_contract_shape(tmp_path: Path) -> None:
    """_run_reporting() failure path must carry all normalised contract fields."""
    orchestrator = _make_report_orchestrator(tmp_path)
    # No eval files in any directory -> will raise

    result = orchestrator._run_reporting()

    _assert_contract(result, source="pipeline_orchestrator_reporting", success=False)
    assert result["error_stage"] == "reporting"
    assert "error" in result
    assert isinstance(result["error"], str)


def test_run_reporting_generator_raises_contract_shape(tmp_path: Path) -> None:
    """_run_reporting() propagates report generator exceptions via the contract."""
    orchestrator = _make_report_orchestrator(tmp_path)

    eval_file = orchestrator.output_dirs["metadata"] / "evaluation_results_test.json"
    eval_file.write_text(json.dumps({"results": []}), encoding="utf-8")

    orchestrator.report_generator = MagicMock()
    orchestrator.report_generator.generate_reports.side_effect = RuntimeError(
        "report-boom"
    )

    result = orchestrator._run_reporting()

    _assert_contract(result, source="pipeline_orchestrator_reporting", success=False)
    assert result["error_stage"] == "reporting"
    assert "report-boom" in result["error"]


def test_run_evaluation_contract_missing_testsets_dir(tmp_path: Path) -> None:
    """_run_evaluation(): missing testsets dir is handled gracefully via contract."""
    orchestrator = _make_eval_orchestrator(tmp_path)
    # Remove testsets dir to simulate missing directory
    import shutil
    shutil.rmtree(orchestrator.output_dirs["testsets"])

    result = orchestrator._run_evaluation()

    # Should produce a failure contract, not an unhandled exception
    _assert_contract(result, source="pipeline_orchestrator_evaluation", success=False)


def test_run_reporting_contract_error_stage_is_string(tmp_path: Path) -> None:
    """_run_reporting(): error_stage is always a non-empty string on failure."""
    orchestrator = _make_report_orchestrator(tmp_path)

    result = orchestrator._run_reporting()

    assert isinstance(result.get("error_stage"), str)
    assert len(result["error_stage"]) > 0


# ---------------------------------------------------------------------------
# Hardware observability artifact — E2E path
# ---------------------------------------------------------------------------

def test_run_evaluation_writes_hardware_observability_artifact(tmp_path: Path) -> None:
    """When hardware_acceleration_client is active, _run_evaluation() must persist
    hardware_observability_<run_id>.json to output_dirs["metadata"]."""
    orchestrator = _make_eval_orchestrator(tmp_path)
    orchestrator.run_id = "hw-obs-test"
    orchestrator.config = {
        "inference": {
            "hardware_acceleration": {
                "enabled": True,
                "benchmark_prompts": ["hw benchmark prompt"],
                "benchmark_repeats": 1,
            }
        }
    }

    # Wire the hardware client with the session stub defined in this module
    orchestrator.hardware_acceleration_client = __import__(
        "src.inference.vllm_client", fromlist=["vLLMInferenceClient"]
    ).vLLMInferenceClient(session=_HardwareSession())

    # Plant a fake testset so the stage doesn't fail early
    testset_file = orchestrator.output_dirs["testsets"] / "testset.xlsx"
    testset_file.write_bytes(b"fake-xlsx")

    mock_eval_result = {
        "success": True,
        "total_queries": 2,
        "keyword_metrics": {"pass_rate": 1.0},
        "ragas_metrics": {"average_score": 0.9},
        "feedback_metrics": {"requests": 0},
        "output_file": "eval_out.json",
    }
    orchestrator.evaluation_dispatcher.evaluate_testsets.return_value = mock_eval_result

    result = orchestrator._run_evaluation()

    # Evaluation must succeed
    assert result["success"] is True, f"Unexpected failure: {result.get('error')}"

    # The artifact file must exist on disk
    artifact_path = (
        orchestrator.output_dirs["metadata"] / f"hardware_observability_{orchestrator.run_id}.json"
    )
    assert artifact_path.exists(), (
        f"hardware_observability artifact not written to {artifact_path}"
    )

    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert "capabilities" in payload
    assert payload["capabilities"]["connected"] is True
