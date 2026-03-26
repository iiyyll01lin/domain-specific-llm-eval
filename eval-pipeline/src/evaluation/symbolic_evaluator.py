import logging

from .deterministic_backends import DeterministicSymbolicBackend

logger = logging.getLogger(__name__)


class SymbolicEvaluator:
    """Evaluates answers using absolute Symbolic Proof Checks."""

    def __init__(self, backend: DeterministicSymbolicBackend | None = None):
        self.backend = backend or DeterministicSymbolicBackend()

    def evaluate_proof(
        self, answer: str, context: str, was_symbolically_proven: bool
    ) -> float:
        """Calculates the Symbolic Proof Check metric."""
        result = self.backend.evaluate(answer, context, was_symbolically_proven)
        if was_symbolically_proven:
            logger.info("Answer is mathematically proven via Symbolic Logic.")
        return float(result["score"])

    def evaluate_proof_contract(
        self,
        answer: str,
        context: str,
        was_symbolically_proven: bool,
        expected_fact: str = "",
    ) -> dict:
        return self.backend.evaluate(answer, context, was_symbolically_proven, expected_fact)
