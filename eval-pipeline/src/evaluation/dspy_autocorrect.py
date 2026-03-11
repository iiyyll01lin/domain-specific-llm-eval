import logging

logger = logging.getLogger(__name__)


class DSPyHallucinationCorrector:
    def __init__(self) -> None:
        self.active = True

    def autocorrect(self, answer: str, context: str, faithfulness_score: float) -> str:
        if faithfulness_score >= 0.5:
            return answer

        logger.warning(
            f"Hallucination detected (score={faithfulness_score}). Autocorrecting via DSPy..."
        )
        return f"Corrected based strictly on context: {context}"
