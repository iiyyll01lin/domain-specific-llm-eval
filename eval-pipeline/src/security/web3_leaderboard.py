import hashlib
import json
import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


class Web3LeaderboardConsortia:
    """Stores evaluation metrics in a simulated immutable Web3 ledger (IPFS/Blockchain)."""

    def __init__(self):
        self.ledger: List[Dict[str, Any]] = []

    def submit_metrics(
        self, organization: str, metrics: Dict[str, float], dpo_alignment: float
    ) -> str:
        """Submits metrics and returns a cryptographic hash representing the block."""

        payload = {
            "organization": organization,
            "metrics": metrics,
            "dpo_alignment": dpo_alignment,
            "previous_hash": self.ledger[-1]["hash"] if self.ledger else "GENESIS",
        }

        # Calculate block hash (mock IPFS CID / Blockchain Tx Hash)
        payload_str = json.dumps(payload, sort_keys=True)
        block_hash = hashlib.sha256(payload_str.encode()).hexdigest()

        block = {"payload": payload, "hash": block_hash}

        self.ledger.append(block)
        logger.info(f"Stored Web3 metrics block: {block_hash}")
        return block_hash

    def verify_ledger_integrity(self) -> bool:
        """Verifies the cryptographically linked ledger."""
        for i in range(1, len(self.ledger)):
            prev_block = self.ledger[i - 1]
            curr_block = self.ledger[i]

            if curr_block["payload"]["previous_hash"] != prev_block["hash"]:
                return False

        return True
