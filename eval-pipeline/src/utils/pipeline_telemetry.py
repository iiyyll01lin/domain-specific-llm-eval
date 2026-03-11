from typing import Any, Dict, List, TypedDict, Optional, TypedDict, Optional
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


class TelemetryMetrics(TypedDict, total=False):
    execution_time_seconds: float
    documents_processed: int
    samples_generated: int
    failed_syntheses: int
    generation_success_rate: float


class PipelineTelemetry:
    """Automated telemetry for RAGAS pipeline generation"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.telemetry_dir = self.output_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics = {
            "session_id": self.session_id,
            "start_time": datetime.now().isoformat(),
            "status": "in_progress",
            "document_processing": {
                "total_files": 0,
                "processed_nodes": 0,
                "chunks": 0,
            },
            "generation_stats": {},
            "errors": [],
        }

    def log_document_stats(self, files: int, chunks: int):
        """Log document processing statistics."""
        self.metrics["document_processing"].update(
            {"total_files": files, "chunks": chunks}
        )
        self._save()

    def log_generation(self, generator_name: str, count: int, success: bool = True):
        """Track synthesizer generation distributions."""
        if "generation_stats" not in self.metrics:
            self.metrics["generation_stats"] = {}

        stats = self.metrics["generation_stats"].get(
            generator_name, {"success": 0, "failed": 0}
        )
        if success:
            stats["success"] += count
        else:
            stats["failed"] += count

        self.metrics["generation_stats"][generator_name] = stats
        self._save()

    def log_error(self, step: str, error_msg: str):
        """Log an error during pipeline execution."""
        self.metrics["errors"].append(
            {"timestamp": datetime.now().isoformat(), "step": step, "error": error_msg}
        )
        self._save()

    def finish(self, status: str = "completed"):
        """Mark telemetry as finished and save final state."""
        self.metrics["end_time"] = datetime.now().isoformat()
        self.metrics["status"] = status

        # Calculate totals
        gen_stats = self.metrics.get("generation_stats", {})
        total_success = sum(v.get("success", 0) for v in gen_stats.values())
        total_failed = sum(v.get("failed", 0) for v in gen_stats.values())

        self.metrics["summary"] = {
            "total_generated": total_success,
            "total_failed": total_failed,
            "success_rate": (
                total_success / (total_success + total_failed)
                if (total_success + total_failed) > 0
                else 0
            ),
        }

        self._save()
        logger.info(f"📊 Telemetry saved: {total_success} generated samples")

    def _save(self):
        """Save metrics to JSON."""
        file_path = self.telemetry_dir / f"pipeline_run_{self.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
