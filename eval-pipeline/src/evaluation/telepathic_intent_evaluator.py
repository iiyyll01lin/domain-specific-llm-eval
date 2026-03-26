import logging
from typing import List

from .deterministic_backends import DeterministicIntentBackend

logger = logging.getLogger(__name__)


class TelepathicIntentAlignment:
    """Aligns EEG/Brain-Computer Interfaces for Post-Language RAG Validation"""

    def __init__(self, backend: DeterministicIntentBackend | None = None):
        self.backend = backend or DeterministicIntentBackend()

    def decode_eeg(self, eeg_vector: List[float]) -> str:
        """Decodes raw EEG floats into intent."""
        return str(self.backend.decode(eeg_vector)["intent"])

    def calculate_alignment(self, eeg_intent: str, llm_response: str) -> float:
        """Matches LLM tone/urgency to the original brainwave intent."""
        return float(self.backend.evaluate(eeg_intent, llm_response)["score"])
