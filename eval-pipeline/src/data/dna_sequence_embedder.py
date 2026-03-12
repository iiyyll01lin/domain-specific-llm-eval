import logging
from typing import Dict, List

logger = logging.getLogger(__name__)

class DNASequenceEmbedder:
    """Encodes Vector Embeddings into Synthetic DNA Base Pairs (A, C, G, T)"""

    def __init__(self):
        # A simple mock mapping matrix
        self.bases = ['A', 'C', 'G', 'T']
        
    def _float_to_bases(self, value: float) -> str:
        """Mocks converting a float to a 4-base sequence."""
        # Normalize -1.0 to 1.0 into 0 to 255
        norm = int((value + 1.0) / 2.0 * 255)
        # Generate a deterministic sequence based on value
        seq = ""
        for _ in range(4):
            seq += self.bases[norm % 4]
            norm //= 4
        return seq

    def encode_vector(self, vector: List[float]) -> str:
        """Converts a high dimensional array into a DNA string."""
        encoded = "".join([self._float_to_bases(v) for v in vector])
        logger.info(f"Encoded vector of length {len(vector)} into {len(encoded)} base pairs.")
        return encoded

    def find_nearest_neighbor(self, target_dna: str, dna_vault: Dict[str, str]) -> str:
        """Mocks enzymatic restriction/matching to find closest sequence."""
        best_match = None
        highest_overlap = -1
        
        for name, seq in dna_vault.items():
            # Simple mock similarity (Hamming distance reverse)
            overlap = sum(1 for a, b in zip(target_dna, seq) if a == b)
            if overlap > highest_overlap:
                highest_overlap = overlap
                best_match = name
                
        return best_match or ""
