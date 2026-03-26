import pytest

from src.evaluation.dspy_autocorrect import DSPyHallucinationCorrector
from src.inference.vllm_client import vLLMInferenceClient
from src.optimization.hyperparam_search import OptunaOptimizer
from src.ui.force_graph_viewer import ForceGraphVisualizer


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, url, timeout=0):
        if url.endswith("/metrics"):
            return _FakeResponse({"gpu_utilization": 0.72, "kv_cache_utilization": 0.33})
        return _FakeResponse(self.payload)

    def post(self, url, json=None, timeout=0):
        assert url.endswith("/completions")
        prompt = (json or {}).get("prompt", "")
        return _FakeResponse({"choices": [{"text": f"Hardware Accelerated Response for: {prompt}"}]})


def test_optuna_optimizer(tmp_path) -> None:
    opt = OptunaOptimizer(output_dir=str(tmp_path), n_trials=6)
    res = opt.optimize()
    assert res["trial_count"] == 6
    assert (tmp_path / "trial_history.json").exists()
    assert (tmp_path / "best_config.json").exists()
    assert res["best_f1"] > 0


def test_vllm_client() -> None:
    client = vLLMInferenceClient(session=_FakeSession({"data": [{"id": "model-a"}]}))
    caps = client.get_capabilities()
    ans = client.generate("test")
    telemetry = client.collect_hardware_telemetry(["test"], repeats=1)
    assert caps["connected"] is True
    assert caps["model_count"] == 1
    assert caps["runtime_metrics"]["gpu_utilization"] == 0.72
    assert client.is_connected is True
    assert "Accelerated" in ans
    assert telemetry["benchmarks"][0]["median_latency_seconds"] >= 0.0
    assert telemetry["last_generation_telemetry"]["throughput_tokens_per_second"] > 0
    assert telemetry["gpu_saturation"]["saturation_level"] == "moderate"
    assert telemetry["request_distribution"]["total_requests"] >= 1
    assert telemetry["fallback_paths"]["direct_vllm"] >= 1
    assert telemetry["benchmarks"][0]["latency_samples_seconds"]


def test_force_graph() -> None:
    vis = ForceGraphVisualizer()
    html = vis.generate_html_payload({"nodes": [{"id": "A"}, {"id": "B"}], "links": [{"source": "A", "target": "B"}]})
    assert "WebGL" in html
    assert '"link_count": 1' in html
    assert "high_centrality_nodes" in html


def test_force_graph_export_from_kg_artifact(tmp_path) -> None:
    vis = ForceGraphVisualizer()
    kg_path = tmp_path / "kg.json"
    kg_path.write_text(
        '{"nodes": [{"id": "A"}, {"id": "B"}, {"id": "C"}], "relationships": [{"source": "A", "target": "B"}]}',
        encoding="utf-8",
    )

    exported = vis.export_from_kg_artifact(kg_path, tmp_path / "topology")

    assert (tmp_path / "topology" / "topology_payload.json").exists()
    assert (tmp_path / "topology" / "topology.html").exists()
    assert exported["html_path"].endswith("topology.html")
    assert exported["payload_path"].endswith("topology_payload.json")


def test_dspy_corrector() -> None:
    corr = DSPyHallucinationCorrector()
    ans1 = corr.autocorrect("ans", "ctx", 0.9)
    assert ans1 == "ans"
    ans2 = corr.autocorrect("ans", "ctx", 0.3)
    assert "ctx" in ans2
    assert "citation" in ans2
