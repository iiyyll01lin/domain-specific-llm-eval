import pytest

from src.evaluation.symbolic_evaluator import SymbolicEvaluator
from src.generation.neuro_symbolic_rag import NeuroSymbolicRAGEngine
from src.loaders.wikidata_sync import WikiDataKnowledgeGraphSync
from src.orchestration.meta_learning_agent import MetaLearningAgent
from src.security.web3_leaderboard import Web3LeaderboardConsortia


class _FakeWikiResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeWikiSession:
    def __init__(self, payload):
        self.payload = payload

    def get(self, endpoint, params=None, timeout=0):
        return _FakeWikiResponse(self.payload)


class DummyModule:
    def synthesize(self):
        return "Old Synthesis"


def test_meta_learning_agent():
    target = DummyModule()
    agent = MetaLearningAgent(target)

    assert target.synthesize() == "Old Synthesis"

    new_code = """
def synthesize():
    return "New dynamically patched synthesis"
"""
    success = agent.rewrite_method_ast("synthesize", new_code)

    assert success is True
    assert target.synthesize() == "New dynamically patched synthesis"


def test_web3_leaderboard():
    board = Web3LeaderboardConsortia()

    hash1 = board.submit_metrics("Bank_A", {"ragas_context_precision": 0.95}, 0.9)
    hash2 = board.submit_metrics("Hospital_B", {"ragas_faithfulness": 0.98}, 0.92)

    assert len(board.ledger) == 2
    assert board.verify_ledger_integrity() is True


def test_neuro_symbolic_rag():
    engine = NeuroSymbolicRAGEngine()
    ans, proven = engine.generate(
        "What is the capital of France?", "Some neural context"
    )
    assert proven is True
    assert "Paris" in ans


def test_symbolic_evaluator():
    evaluator = SymbolicEvaluator()
    score = evaluator.evaluate_proof(
        "Neural Context... Formally Proven Fact: Paris", "context", True
    )
    assert score == 1.0


def test_wikidata_sync():
    sync = WikiDataKnowledgeGraphSync(
        session=_FakeWikiSession(
            {
                "search": [
                    {
                        "id": "Q28865",
                        "label": "Python",
                        "description": "high-level programming language",
                    }
                ]
            }
        )
    )
    data = sync.sync_node("Python v3.10")
    assert data["wikidata_id"] == "Q28865"
    assert data["source"] == "wikidata-live"
