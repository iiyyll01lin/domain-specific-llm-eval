import pytest
import json
from pathlib import Path

from src.loaders.taxonomy_discovery import ZeroShotTaxonomyDiscoverer
from src.orchestration.multi_agent_router import LangGraphEvalOrchestrator
from src.security.quantum_pii_tokenizer import QuantumResistantTokenizer
from src.ui.app_store_marketplace import UnifiedAppStore


class _BackendResponse:
    def __init__(self, payload):
        self.payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self.payload


class _BackendSession:
    def __init__(self, payload):
        self.payload = payload

    def post(self, endpoint, json=None, timeout=0):
        return _BackendResponse(self.payload)


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


def test_app_store_file_backed_registry_and_receipt(tmp_path) -> None:
    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir()
    (manifest_dir / "custom.json").write_text(
        json.dumps(
            {
                "id": "custom-suite",
                "name": "Custom Suite",
                "author": "Example",
                "version": "1.0.0",
                "trust": "verified",
                "dependencies": "base-eval",
            }
        ),
        encoding="utf-8",
    )
    store = UnifiedAppStore(manifest_dir=manifest_dir, install_dir=tmp_path / "installed")

    assert store.get_runbook_manifest("custom-suite") is not None
    assert store.install_runbook("custom-suite") is True
    assert (tmp_path / "installed" / "custom-suite.json").exists()


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


def test_taxonomy_discovery_file_and_backend_enrichment(tmp_path) -> None:
    csv_path = tmp_path / "sample.csv"
    csv_path.write_text("patient_id,organization\n1,HealthOrg\n", encoding="utf-8")
    discoverer = ZeroShotTaxonomyDiscoverer(
        llm_endpoint="http://backend.example/taxonomy",
        session=_BackendSession({"entities": ["Case"], "relations": ["BELONGS_TO"]}),
    )

    result = discoverer.extract_ontology_from_csv_file(
        csv_path, persist_path=tmp_path / "proposal.json"
    )
    approved_path = discoverer.approve_taxonomy_proposal(
        result, approved_path=tmp_path / "approved.json"
    )

    assert result["backend_used"] is True
    assert "Case" in result["entities"]
    assert (tmp_path / "proposal.json").exists()
    assert Path(approved_path).exists()

