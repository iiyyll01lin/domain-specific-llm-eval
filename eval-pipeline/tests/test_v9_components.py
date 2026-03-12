import pytest

from src.evaluation.dspy_autocorrect import DSPyHallucinationCorrector
from src.inference.vllm_client import vLLMInferenceClient
from src.optimization.hyperparam_search import OptunaOptimizer
from src.ui.force_graph_viewer import ForceGraphVisualizer


def test_optuna_optimizer() -> None:
    opt = OptunaOptimizer()
    res = opt.optimize()
    assert res["best_chunk_size"] == 512


def test_vllm_client() -> None:
    client = vLLMInferenceClient()
    ans = client.generate("test")
    assert client.is_connected is True
    assert "Accelerated" in ans


def test_force_graph() -> None:
    vis = ForceGraphVisualizer()
    html = vis.generate_html_payload({"nodes": [1, 2]})
    assert "WebGL" in html


def test_dspy_corrector() -> None:
    corr = DSPyHallucinationCorrector()
    ans1 = corr.autocorrect("ans", "ctx", 0.9)
    assert ans1 == "ans"
    ans2 = corr.autocorrect("ans", "ctx", 0.3)
    assert "ctx" in ans2
