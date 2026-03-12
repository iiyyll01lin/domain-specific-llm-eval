import logging
from typing import List

logger = logging.getLogger(__name__)


class TelepathicIntentAlignment:
    """Aligns EEG/Brain-Computer Interfaces for Post-Language RAG Validation"""

    def __init__(self):
        pass

    def decode_eeg(self, eeg_vector: List[float]) -> str:
        """Decodes raw EEG floats into intent."""
        if sum(eeg_vector) > 5.0:
            return "URGENT_REQUEST"
        return "GENERAL_INQUIRY"

    def calculate_alignment(self, eeg_intent: str, llm_response: str) -> float:
        """Matches LLM tone/urgency to the original brainwave intent."""
        if eeg_intent == "URGENT_REQUEST" and "immediately" in llm_response.lower():
            return 1.0
        elif (
            eeg_intent == "GENERAL_INQUIRY"
            and "immediately" not in llm_response.lower()
        ):
            return 0.9

        return 0.2
