import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class DirectPreferenceOptimizationPipeline:
    """Native LLM Alignment Pipeline utilizing DPO/PPO for failed responses."""

    def __init__(self):
        self.failure_queue: List[Dict[str, str]] = []

    def ingest_failure(
        self, prompt: str, bad_response: str, expected_ideal: str
    ) -> None:
        """Queues a failed evaluation for RLHF/DPO finetuning."""
        self.failure_queue.append(
            {"prompt": prompt, "chosen": expected_ideal, "rejected": bad_response}
        )
        logger.info("Ingested failed response into DPO queue.")

    def run_dpo_finetuning(self) -> bool:
        """Mocks Unsloth/trl DPO training loop."""
        if not self.failure_queue:
            return False

        logger.info(f"Running DPO finetuning on {len(self.failure_queue)} samples...")
        self.failure_queue.clear()
        return True
