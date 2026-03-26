from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DirectPreferenceOptimizationPipeline:
    """Native LLM Alignment Pipeline utilizing DPO/PPO for failed responses."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.output_dir = Path(self.config.get("output_dir", "outputs/alignment"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.queue_file = self.output_dir / self.config.get(
            "queue_filename", "dpo_failure_queue.jsonl"
        )
        self.dataset_file = self.output_dir / self.config.get(
            "dataset_filename", "dpo_training_dataset.jsonl"
        )
        self.trainer_command = self.config.get("trainer_command")
        self.trainer_env = dict(self.config.get("trainer_env", {}))
        self.auto_run_threshold = int(self.config.get("auto_run_threshold", 0) or 0)
        self.failure_queue: List[Dict[str, Any]] = []
        self.last_run_metadata: Dict[str, Any] = {}
        self._load_queue_from_disk()

    def _load_queue_from_disk(self) -> None:
        if not self.queue_file.exists():
            return
        try:
            self.failure_queue = [
                json.loads(line)
                for line in self.queue_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        except Exception as exc:
            logger.warning("Failed to load DPO queue from disk: %s", exc)
            self.failure_queue = []

    def _persist_queue(self) -> None:
        payload = "\n".join(
            json.dumps(item, ensure_ascii=False, sort_keys=True)
            for item in self.failure_queue
        )
        self.queue_file.write_text(
            f"{payload}\n" if payload else "", encoding="utf-8"
        )

    def _export_training_dataset(self) -> str:
        dataset_lines = [
            json.dumps(
                {
                    "prompt": item["prompt"],
                    "chosen": item["chosen"],
                    "rejected": item["rejected"],
                    "metadata": item.get("metadata", {}),
                    "captured_at": item.get("captured_at"),
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            for item in self.failure_queue
        ]
        payload = "\n".join(dataset_lines)
        self.dataset_file.write_text(
            f"{payload}\n" if dataset_lines else "",
            encoding="utf-8",
        )
        return str(self.dataset_file)

    def ingest_failure(
        self,
        prompt: str,
        bad_response: str,
        expected_ideal: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Queues a failed evaluation for RLHF/DPO finetuning."""
        self.failure_queue.append(
            {
                "prompt": prompt,
                "chosen": expected_ideal,
                "rejected": bad_response,
                "metadata": metadata or {},
                "captured_at": datetime.utcnow().isoformat(),
            }
        )
        self._persist_queue()
        logger.info("Ingested failed response into DPO queue.")

    def run_alignment_cycle(self) -> Dict[str, Any]:
        """Export queued failures and optionally invoke a configured trainer."""
        if not self.failure_queue:
            self.last_run_metadata = {
                "executed": False,
                "sample_count": 0,
                "dataset_path": str(self.dataset_file),
                "command": None,
            }
            return dict(self.last_run_metadata)

        dataset_path = self._export_training_dataset()
        result: Dict[str, Any] = {
            "executed": True,
            "sample_count": len(self.failure_queue),
            "dataset_path": dataset_path,
            "command": self.trainer_command,
            "completed_at": datetime.utcnow().isoformat(),
        }

        if self.trainer_command:
            command = self.trainer_command
            if isinstance(command, str):
                command_parts = shlex.split(command)
            else:
                command_parts = [str(part) for part in command]
            command_parts = [part.replace("{dataset_path}", dataset_path) for part in command_parts]
            if not any(dataset_path == part or "{dataset_path}" in str(self.trainer_command) for part in command_parts):
                command_parts.append(dataset_path)
            completed = subprocess.run(
                command_parts,
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, **self.trainer_env},
            )
            result.update(
                {
                    "returncode": completed.returncode,
                    "stdout": completed.stdout,
                    "stderr": completed.stderr,
                }
            )
            if completed.returncode != 0:
                logger.warning(
                    "DPO trainer command exited with code %s", completed.returncode
                )

        logger.info(
            "Running DPO finetuning on %s samples...", len(self.failure_queue)
        )
        self.failure_queue.clear()
        self._persist_queue()
        self.last_run_metadata = result
        return result

    def should_auto_run(self) -> bool:
        if self.auto_run_threshold <= 0:
            return False
        return len(self.failure_queue) >= self.auto_run_threshold

    def run_dpo_finetuning(self) -> bool:
        """Run the alignment cycle and preserve the previous bool-style contract."""
        return bool(self.run_alignment_cycle().get("executed"))
