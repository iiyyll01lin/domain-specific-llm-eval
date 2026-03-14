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
    assert tokenizer.detokenize(token, access_granted=False) is None
    assert tokenizer.detokenize(token, access_granted=True) == "Sensitive Data"


def test_app_store() -> None:
    store = UnifiedAppStore()
    manifest = store.get_runbook_manifest("legal-tax-suite")
    assert store.install_runbook("legal-tax-suite") is True
    assert store.install_runbook("invalid-id") is False
    assert manifest is not None
    assert manifest["trust"] == "verified"


def test_taxonomy_discovery() -> None:
    discoverer = ZeroShotTaxonomyDiscoverer()
    ontology = discoverer.extract_ontology(
        "patient_id,policy_name,organization\n1,HIPAA,HealthOrg"
    )
    assert "Patient" in ontology["entities"]
    assert "Policy" in ontology["entities"]
    assert "Organization" in ontology["entities"]
    assert "AFFECTS" in ontology["relations"]
    assert ontology["proposals"]

