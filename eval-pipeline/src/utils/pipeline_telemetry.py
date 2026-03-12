import io
import json
import os

import boto3


def upload_to_s3(file_path, s3_key):
    try:
        s3 = boto3.client(
            "s3",
            endpoint_url="http://localhost:9000",
            aws_access_key_id=os.environ.get("MINIO_ROOT_USER", "minioadmin"),
            aws_secret_access_key=os.environ.get(
                "MINIO_ROOT_PASSWORD", "minioadmin123"
            ),
        )
        bucket = os.environ.get("MINIO_BUCKET", "rag-eval-dev")

        # Check if bucket exists, if not create it
        try:
            s3.head_bucket(Bucket=bucket)
        except:
            try:
                s3.create_bucket(Bucket=bucket)
            except:
                pass

        s3.upload_file(file_path, bucket, s3_key)
        print(f"Uploaded {file_path} to S3 bucket {bucket} at {s3_key}")
    except Exception as e:
        print(f"S3 Upload failed: {e}")


import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

workspace_root = Path(__file__).resolve().parents[3]
if str(workspace_root) not in sys.path:
    sys.path.insert(0, str(workspace_root))

try:
    from services.common.storage.object_store import ObjectStoreClient
except Exception:  # pragma: no cover - optional dependency for eval-pipeline runtime
    ObjectStoreClient = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


class TelemetryError(TypedDict):
    timestamp: str
    step: str
    error: str


class TelemetryStageEvent(TypedDict, total=False):
    timestamp: str
    stage: str
    status: str
    details: Dict[str, Any]


class TelemetryGenerationStats(TypedDict):
    success: int
    failed: int


class TelemetryDocumentProcessing(TypedDict):
    total_files: int
    processed_nodes: int
    chunks: int


class TelemetrySummary(TypedDict):
    total_generated: int
    total_failed: int
    success_rate: float


class TelemetryMetrics(TypedDict, total=False):
    session_id: str
    start_time: str
    end_time: str
    status: str
    document_processing: TelemetryDocumentProcessing
    generation_stats: Dict[str, TelemetryGenerationStats]
    errors: List[TelemetryError]
    stage_events: List[TelemetryStageEvent]
    summary: TelemetrySummary


class PipelineTelemetry:
    """Automated telemetry for RAGAS pipeline generation"""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.telemetry_dir = self.output_dir / "telemetry"
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        self.object_store_client = self._build_object_store_client()

        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics: TelemetryMetrics = {
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
            "stage_events": [],
        }

    def _build_object_store_client(self) -> Optional[ObjectStoreClient]:
        required = [
            os.environ.get("OBJECT_STORE_ENDPOINT"),
            os.environ.get("OBJECT_STORE_ACCESS_KEY"),
            os.environ.get("OBJECT_STORE_SECRET_KEY"),
            os.environ.get("OBJECT_STORE_BUCKET"),
        ]
        if not all(required) or ObjectStoreClient is None:
            return None
        try:
            return ObjectStoreClient()
        except Exception as exc:
            logger.warning("⚠️ Object store client unavailable for telemetry mirroring: %s", exc)
            return None

    def log_document_stats(self, files: int, chunks: int) -> None:
        """Log document processing statistics."""
        self.metrics["document_processing"].update(
            {"total_files": files, "chunks": chunks}
        )
        self._save()

    def log_generation(self, generator_name: str, count: int, success: bool = True) -> None:
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

    def log_error(self, step: str, error_msg: str) -> None:
        """Log an error during pipeline execution."""
        self.metrics["errors"].append(
            {"timestamp": datetime.now().isoformat(), "step": step, "error": error_msg}
        )
        self._save()

    def log_stage_event(
        self,
        stage: str,
        status: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.metrics["stage_events"].append(
            {
                "timestamp": datetime.now().isoformat(),
                "stage": stage,
                "status": status,
                "details": details or {},
            }
        )
        self._save()

    def finish(self, status: str = "completed") -> None:
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

    def _save(self) -> None:
        """Save metrics to JSON."""
        self.telemetry_dir.mkdir(parents=True, exist_ok=True)
        file_path = self.telemetry_dir / f"pipeline_run_{self.session_id}.json"
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.metrics, f, indent=2)
        self._mirror_to_object_store(file_path)

    def _mirror_to_object_store(self, file_path: Path) -> None:
        if self.object_store_client is None:
            return
        object_key = f"eval-pipeline/{self.output_dir.name}/telemetry/{file_path.name}"
        try:
            self.object_store_client.upload_file(bucket=None, key=object_key, file_path=str(file_path))
        except Exception as exc:
            logger.warning("⚠️ Failed to mirror telemetry artifact to object store: %s", exc)


__all__ = ["PipelineTelemetry", "TelemetryMetrics"]
