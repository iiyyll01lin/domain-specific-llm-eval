import pytest

from src.loaders.taxonomy_discovery import ZeroShotTaxonomyDiscoverer
from src.orchestration.multi_agent_router import LangGraphEvalOrchestrator
from src.security.quantum_pii_tokenizer import QuantumResistantTokenizer
from src.ui.app_store_marketplace import UnifiedAppStore


def test_multi_agent_router() -> None:
    orchestrator = LangGraphEvalOrchestrator()
    res = orchestrator.provision_test_job({"job_id": "12345"})
    assert "12345" in res


def test_quantum_tokenizer() -> None:
    tokenizer = QuantumResistantTokenizer()
    token = tokenizer.tokenize_pii("Sensitive Data")
    assert token.startswith("QTK-")
    assert len(tokenizer.tokenize_pii("")) == 0


def test_app_store() -> None:
    store = UnifiedAppStore()
    assert store.install_runbook("legal-tax-suite") is True
    assert store.install_runbook("invalid-id") is False


def test_taxonomy_discovery() -> None:
    discoverer = ZeroShotTaxonomyDiscoverer()
    ontology = discoverer.extract_ontology("id,name,value\n1,test,yes")
    assert "Organization" in ontology["entities"]
    assert "AFFECTS" in ontology["relations"]
