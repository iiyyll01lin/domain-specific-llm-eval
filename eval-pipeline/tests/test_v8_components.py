import pytest
from src.distributed.federated_learning import FederatedLearningClient
from src.evaluation.swarm_agent import SwarmSynthesizer
from src.loaders.multimodal_loader import MultimodalDocumentLoader
from src.security.threat_intel import ThreatIntelligenceAPI


def test_federated_client() -> None:
    client = FederatedLearningClient()
    scores = [{"score": 0.9}]
    res = client.aggregate_gradients(scores)
    assert res["status"] == "success"


def test_multimodal_loader() -> None:
    loader = MultimodalDocumentLoader()
    docs = loader.load_document("test.pdf")
    assert len(docs) == 3


def test_threat_intel() -> None:
    api = ThreatIntelligenceAPI()
    jb = api.get_latest_jailbreak()
    assert isinstance(jb, str)


def test_swarm() -> None:
    swarm = SwarmSynthesizer()
    res = swarm.debate_answer("test?", "ans")
    assert "Debated" in res["final_answer"]
