import pytest

from src.distributed.edge_wasm_miner import DecentralizedEdgeMiner
from src.evaluation.spatial_rag_evaluator import MixedRealityMultimodalEval
from src.optimization.dpo_alignment import DirectPreferenceOptimizationPipeline
from src.orchestration.omni_cloud_provisioner import OmniCloudAutoProvisioner


def test_omni_cloud_provisioner():
    # 0.45 a pop
    provisioner = OmniCloudAutoProvisioner(budget_limit=1.0)
    assert provisioner.replicate() is True
    assert provisioner.active_nodes == 1
    assert provisioner.replicate() is True
    assert provisioner.active_nodes == 2
    # Third one pushes to 1.35, exceeding 1.0 budget
    assert provisioner.replicate() is False

def test_dpo_alignment():
    pipeline = DirectPreferenceOptimizationPipeline()
    assert pipeline.run_dpo_finetuning() is False
    
    pipeline.ingest_failure("What is 1+1?", "3", "2")
    assert len(pipeline.failure_queue) == 1
    assert pipeline.run_dpo_finetuning() is True
    assert len(pipeline.failure_queue) == 0

def test_edge_wasm_miner():
    miner = DecentralizedEdgeMiner()
    miner.register_edge_node("iphone-15-pro")
    miner.register_edge_node("tesla-fsd-chip")
    
    res = miner.distribute_workload("eval_ragas_context")
    assert len(res) == 2
    assert "WASM_BIN" in res["iphone-15-pro"]
    assert "iphone-15-pro" in res

def test_spatial_rag():
    evaluator = MixedRealityMultimodalEval()
    ctx = evaluator.retrieve_spatial_context((10, 5, 2))
    assert "robotic" in ctx
    
    score = evaluator.evaluate_spatial_reasoning("What is here?", (10, 5, 2), "I see a robotic assembly arm.")
    assert score == 1.0
