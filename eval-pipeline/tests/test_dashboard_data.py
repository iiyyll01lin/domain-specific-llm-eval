import json
from pathlib import Path

from src.ui.dashboard_data import (build_observability_retention_index,
                                   build_observability_views,
                                   load_telemetry_data)


def test_load_telemetry_data_reads_pipeline_run_artifacts_and_metadata(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "pure_ragas_run_demo"
    telemetry_dir = run_dir / "telemetry"
    metadata_dir = run_dir / "metadata"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    (telemetry_dir / "pipeline_run_demo.json").write_text(
        json.dumps(
            {
                "session_id": "demo",
                "status": "completed",
                "document_processing": {"total_files": 2, "chunks": 4},
                "summary": {"total_generated": 3, "total_failed": 1},
            }
        ),
        encoding="utf-8",
    )
    (metadata_dir / "evaluation_metadata_demo.json").write_text(
        json.dumps(
            {
                "hardware_acceleration_telemetry": {
                    "gpu_saturation": {"saturation_level": "moderate"}
                }
            }
        ),
        encoding="utf-8",
    )

    telemetry_payloads = load_telemetry_data(tmp_path)

    assert len(telemetry_payloads) == 1
    assert telemetry_payloads[0]["run_id"] == "pure_ragas_run_demo"
    assert telemetry_payloads[0]["evaluation_metadata"]["hardware_acceleration_telemetry"]["gpu_saturation"]["saturation_level"] == "moderate"


def test_build_observability_views_summarizes_latency_fallback_and_error_modes() -> None:
    views = build_observability_views(
        [
            {
                "run_id": "run-a",
                "evaluation_metadata": {
                    "hardware_acceleration_telemetry": {
                        "benchmarks": [
                            {
                                "latency_samples_seconds": [0.1, 0.2, 0.4],
                            }
                        ],
                        "request_distribution": {"total_requests": 4},
                        "fallback_paths": {"direct_vllm": 3, "simulated_response": 1},
                        "gpu_saturation": {
                            "current_utilization": 0.78,
                            "kv_cache_utilization": 0.44,
                            "saturation_level": "moderate",
                        },
                        "error_modes": {"none": 3, "backend_unreachable": 1},
                    }
                },
            }
        ]
    )

    assert views["latency_trends"][0]["p50_latency_seconds"] == 0.2
    assert views["fallback_trends"][0]["fallback_ratio"] == 0.25
    assert views["saturation_trends"][0]["saturation_level"] == "moderate"
    assert {row["error_mode"] for row in views["error_mode_trends"]} == {"none", "backend_unreachable"}


def test_build_observability_retention_index_persists_windowed_summary(tmp_path: Path) -> None:
    run_dir = tmp_path / "outputs" / "pure_ragas_run_demo"
    telemetry_dir = run_dir / "telemetry"
    metadata_dir = run_dir / "metadata"
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (telemetry_dir / "pipeline_run_demo.json").write_text(
        json.dumps({"session_id": "demo", "status": "completed"}),
        encoding="utf-8",
    )
    (metadata_dir / "evaluation_metadata_demo.json").write_text(
        json.dumps(
            {
                "hardware_acceleration_telemetry": {
                    "benchmarks": [{"latency_samples_seconds": [0.1, 0.2]}],
                    "request_distribution": {"total_requests": 2},
                    "fallback_paths": {"simulated_response": 1, "direct_vllm": 1},
                    "gpu_saturation": {
                        "current_utilization": 0.7,
                        "kv_cache_utilization": 0.3,
                        "saturation_level": "moderate",
                    },
                    "error_modes": {"backend_unreachable": 1},
                },
                "hardware_observability_artifact": "artifact.json",
            }
        ),
        encoding="utf-8",
    )

    retention = build_observability_retention_index(tmp_path, retention_limit=10, comparison_windows=[1])

    assert retention["retained_runs"] == 1
    assert retention["window_comparisons"]["recent_1_runs"]["avg_fallback_ratio"] == 0.5
    assert retention["artifacts"][0]["hardware_observability_artifact"] == "artifact.json"
    assert Path(retention["retention_index_path"]).exists()
    assert retention["per_run_diffs"][0]["p95_latency_delta"] == 0.0
    assert retention["error_mode_artifacts"]["backend_unreachable"][0]["run_id"] == "pure_ragas_run_demo"
    assert retention["searchable_artifacts"][0]["artifact_search_key"]


def test_build_observability_retention_index_flags_anomalies(tmp_path: Path) -> None:
    for idx, latency in enumerate(([0.1, 0.2], [0.2, 0.3], [1.2, 1.4]), start=1):
        run_dir = tmp_path / "outputs" / f"run_{idx}"
        telemetry_dir = run_dir / "telemetry"
        metadata_dir = run_dir / "metadata"
        telemetry_dir.mkdir(parents=True, exist_ok=True)
        metadata_dir.mkdir(parents=True, exist_ok=True)
        (telemetry_dir / f"pipeline_run_{idx}.json").write_text(
            json.dumps({"status": "completed"}), encoding="utf-8"
        )
        (metadata_dir / f"evaluation_metadata_{idx}.json").write_text(
            json.dumps(
                {
                    "hardware_acceleration_telemetry": {
                        "benchmarks": [{"latency_samples_seconds": list(latency)}],
                        "request_distribution": {"total_requests": 2},
                        "fallback_paths": {"simulated_response": 1 if idx == 3 else 0, "direct_vllm": 1},
                        "gpu_saturation": {"current_utilization": 0.4, "kv_cache_utilization": 0.2, "saturation_level": "low"},
                        "error_modes": {"backend_unreachable": 1 if idx == 3 else 0},
                    }
                }
            ),
            encoding="utf-8",
        )

    retention = build_observability_retention_index(tmp_path, retention_limit=10, comparison_windows=[3])

    anomaly_row = next(item for item in retention["anomaly_flags"] if item["run_id"] == "run_3")
    assert anomaly_row["is_anomalous"] is True
    assert "high_latency" in anomaly_row["flags"]
    assert anomaly_row["severity"] == "critical"
    regression_row = next(item for item in retention["regression_labels"] if item["run_id"] == "run_3")
    assert "latency_regression" in regression_row["labels"]
    assert retention["issue_clusters"]