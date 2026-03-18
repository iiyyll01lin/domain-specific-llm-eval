from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional


def _cluster_key(severity: str, labels: List[str]) -> str:
    label_part = "+".join(sorted(labels)) if labels else "stable"
    return f"{severity}:{label_part}"


def load_telemetry_data(base_dir: Optional[Path] = None) -> List[Dict[str, Any]]:
    resolved_base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    outputs_dir = resolved_base_dir / "outputs"
    if not outputs_dir.exists():
        return []

    results: List[Dict[str, Any]] = []
    for telemetry_file in sorted(outputs_dir.glob("*/telemetry/pipeline_run_*.json")):
        try:
            payload = json.loads(telemetry_file.read_text(encoding="utf-8"))
        except Exception:
            continue

        run_dir = telemetry_file.parents[1]
        payload["run_path"] = str(run_dir)
        payload.setdefault("run_id", run_dir.name)

        metadata_dir = run_dir / "metadata"
        metadata_files = sorted(metadata_dir.glob("evaluation_metadata_*.json"))
        if metadata_files:
            try:
                payload["evaluation_metadata"] = json.loads(
                    metadata_files[-1].read_text(encoding="utf-8")
                )
            except Exception:
                payload["evaluation_metadata"] = {}

        results.append(payload)

    return results


def _percentile(values: List[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    if lower_index == upper_index:
        return ordered[lower_index]
    weight = position - lower_index
    return (ordered[lower_index] * (1.0 - weight)) + (ordered[upper_index] * weight)


def build_observability_views(runs: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    latency_trends: List[Dict[str, Any]] = []
    fallback_trends: List[Dict[str, Any]] = []
    saturation_trends: List[Dict[str, Any]] = []
    error_mode_trends: List[Dict[str, Any]] = []

    for run in runs:
        hardware = run.get("evaluation_metadata", {}).get(
            "hardware_acceleration_telemetry", {}
        )
        benchmarks = hardware.get("benchmarks", []) if isinstance(hardware, dict) else []
        latencies: List[float] = []
        for benchmark in benchmarks:
            samples = benchmark.get("latency_samples_seconds")
            if isinstance(samples, list) and samples:
                latencies.extend(float(value) for value in samples)
            else:
                for key in (
                    "min_latency_seconds",
                    "median_latency_seconds",
                    "max_latency_seconds",
                ):
                    value = benchmark.get(key)
                    if value is not None:
                        latencies.append(float(value))

        request_distribution = hardware.get("request_distribution", {}) if isinstance(hardware, dict) else {}
        fallback_paths = hardware.get("fallback_paths", {}) if isinstance(hardware, dict) else {}
        gpu_saturation = hardware.get("gpu_saturation", {}) if isinstance(hardware, dict) else {}
        total_requests = int(request_distribution.get("total_requests", 0) or 0)
        fallback_total = sum(
            int(count)
            for path, count in fallback_paths.items()
            if path not in {"none", "direct_vllm"}
        )

        latency_trends.append(
            {
                "run_id": run.get("run_id", "unknown"),
                "p50_latency_seconds": round(_percentile(latencies, 0.5), 6),
                "p95_latency_seconds": round(_percentile(latencies, 0.95), 6),
                "mean_latency_seconds": round(statistics.mean(latencies), 6) if latencies else 0.0,
            }
        )
        fallback_trends.append(
            {
                "run_id": run.get("run_id", "unknown"),
                "total_requests": total_requests,
                "fallback_requests": fallback_total,
                "fallback_ratio": round((fallback_total / total_requests), 6) if total_requests else 0.0,
            }
        )
        saturation_trends.append(
            {
                "run_id": run.get("run_id", "unknown"),
                "gpu_utilization": float(gpu_saturation.get("current_utilization", 0.0) or 0.0),
                "kv_cache_utilization": float(gpu_saturation.get("kv_cache_utilization", 0.0) or 0.0),
                "saturation_level": gpu_saturation.get("saturation_level", "unknown"),
            }
        )
        for mode, count in (hardware.get("error_modes", {}) or {}).items():
            error_mode_trends.append(
                {
                    "run_id": run.get("run_id", "unknown"),
                    "error_mode": mode,
                    "count": int(count),
                }
            )

    return {
        "latency_trends": latency_trends,
        "fallback_trends": fallback_trends,
        "saturation_trends": saturation_trends,
        "error_mode_trends": error_mode_trends,
    }


def build_observability_retention_index(
    base_dir: Optional[Path] = None,
    *,
    retention_limit: int = 50,
    comparison_windows: Optional[List[int]] = None,
) -> Dict[str, Any]:
    resolved_base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    runs = load_telemetry_data(resolved_base_dir)
    trimmed_runs = runs[-retention_limit:]
    views = build_observability_views(trimmed_runs)
    windows = comparison_windows or [3, 5, 10]

    comparisons: Dict[str, Dict[str, float]] = {}
    for window in windows:
        recent_latency = views["latency_trends"][-window:]
        recent_fallback = views["fallback_trends"][-window:]
        recent_saturation = views["saturation_trends"][-window:]
        comparisons[f"recent_{window}_runs"] = {
            "avg_p95_latency_seconds": round(
                statistics.mean(item["p95_latency_seconds"] for item in recent_latency), 6
            )
            if recent_latency
            else 0.0,
            "avg_fallback_ratio": round(
                statistics.mean(item["fallback_ratio"] for item in recent_fallback), 6
            )
            if recent_fallback
            else 0.0,
            "avg_gpu_utilization": round(
                statistics.mean(item["gpu_utilization"] for item in recent_saturation), 6
            )
            if recent_saturation
            else 0.0,
        }

    per_run_diffs: List[Dict[str, Any]] = []
    previous_latency = None
    previous_fallback = None
    for latency_item, fallback_item in zip(
        views["latency_trends"], views["fallback_trends"]
    ):
        diff_entry = {
            "run_id": latency_item["run_id"],
            "p95_latency_delta": round(
                latency_item["p95_latency_seconds"] - previous_latency, 6
            )
            if previous_latency is not None
            else 0.0,
            "fallback_ratio_delta": round(
                fallback_item["fallback_ratio"] - previous_fallback, 6
            )
            if previous_fallback is not None
            else 0.0,
        }
        per_run_diffs.append(diff_entry)
        previous_latency = latency_item["p95_latency_seconds"]
        previous_fallback = fallback_item["fallback_ratio"]

    run_artifacts: List[Dict[str, Any]] = []
    for run in trimmed_runs:
        evaluation_metadata = run.get("evaluation_metadata", {})
        run_artifacts.append(
            {
                "run_id": run.get("run_id", "unknown"),
                "run_path": run.get("run_path", ""),
                "telemetry_artifact": str(
                    Path(run.get("run_path", "")) / "telemetry"
                )
                if run.get("run_path")
                else "",
                "hardware_observability_artifact": evaluation_metadata.get(
                    "hardware_observability_artifact", ""
                ),
                "evaluation_metadata_file": evaluation_metadata.get("evaluation_file", ""),
            }
        )

    error_drilldown: Dict[str, int] = {}
    error_mode_artifacts: Dict[str, List[Dict[str, str]]] = {}
    for row in views["error_mode_trends"]:
        mode = str(row.get("error_mode", "unknown"))
        error_drilldown[mode] = error_drilldown.get(mode, 0) + int(row.get("count", 0))
        matching_artifact = next(
            (
                artifact
                for artifact in run_artifacts
                if artifact["run_id"] == row.get("run_id")
            ),
            None,
        )
        if matching_artifact is not None:
            error_mode_artifacts.setdefault(mode, []).append(
                {
                    "run_id": matching_artifact["run_id"],
                    "hardware_observability_artifact": matching_artifact[
                        "hardware_observability_artifact"
                    ],
                    "telemetry_artifact": matching_artifact["telemetry_artifact"],
                }
            )

    anomaly_flags: List[Dict[str, Any]] = []
    latency_values = [item["p95_latency_seconds"] for item in views["latency_trends"]]
    fallback_values = [item["fallback_ratio"] for item in views["fallback_trends"]]
    latency_threshold = (
        statistics.mean(latency_values) + statistics.pstdev(latency_values)
        if len(latency_values) > 1
        else (latency_values[0] if latency_values else 0.0)
    )
    fallback_threshold = (
        statistics.mean(fallback_values) + statistics.pstdev(fallback_values)
        if len(fallback_values) > 1
        else (fallback_values[0] if fallback_values else 0.0)
    )
    for latency_item, fallback_item in zip(
        views["latency_trends"], views["fallback_trends"]
    ):
        flags: List[str] = []
        severity = "info"
        if latency_item["p95_latency_seconds"] > latency_threshold and latency_threshold > 0:
            flags.append("high_latency")
        if fallback_item["fallback_ratio"] > fallback_threshold and fallback_threshold > 0:
            flags.append("high_fallback_ratio")
        if len(flags) >= 2:
            severity = "critical"
        elif flags:
            severity = "warning"
        anomaly_flags.append(
            {
                "run_id": latency_item["run_id"],
                "flags": flags,
                "severity": severity,
                "is_anomalous": bool(flags),
            }
        )

    regression_labels: List[Dict[str, Any]] = []
    for diff_entry in per_run_diffs:
        labels: List[str] = []
        if diff_entry["p95_latency_delta"] > 0.2:
            labels.append("latency_regression")
        elif diff_entry["p95_latency_delta"] < -0.2:
            labels.append("latency_improvement")
        if diff_entry["fallback_ratio_delta"] > 0.1:
            labels.append("fallback_regression")
        elif diff_entry["fallback_ratio_delta"] < -0.1:
            labels.append("fallback_improvement")
        regression_labels.append(
            {
                "run_id": diff_entry["run_id"],
                "labels": labels or ["stable"],
            }
        )

    searchable_artifacts: List[Dict[str, Any]] = []
    issue_clusters: Dict[str, Dict[str, Any]] = {}
    for artifact in run_artifacts:
        anomaly = next(
            (row for row in anomaly_flags if row["run_id"] == artifact["run_id"]),
            {"severity": "info", "flags": [], "is_anomalous": False},
        )
        labels = next(
            (row["labels"] for row in regression_labels if row["run_id"] == artifact["run_id"]),
            ["stable"],
        )
        search_key = "|".join(
            [
                str(artifact["run_id"]),
                str(anomaly.get("severity", "info")),
                ",".join(labels),
                str(artifact.get("hardware_observability_artifact", "")),
                str(artifact.get("telemetry_artifact", "")),
            ]
        )
        searchable_entry = {
            **artifact,
            "severity": anomaly.get("severity", "info"),
            "regression_labels": labels,
            "artifact_search_key": search_key,
        }
        searchable_artifacts.append(searchable_entry)

        cluster_name = _cluster_key(str(anomaly.get("severity", "info")), list(labels))
        cluster = issue_clusters.setdefault(
            cluster_name,
            {"severity": anomaly.get("severity", "info"), "labels": labels, "run_ids": [], "count": 0},
        )
        cluster["run_ids"].append(artifact["run_id"])
        cluster["count"] += 1

    retention_index = {
        "retained_runs": len(trimmed_runs),
        "retention_limit": retention_limit,
        "window_comparisons": comparisons,
        "per_run_diffs": per_run_diffs,
        "error_drilldown": error_drilldown,
        "error_mode_artifacts": error_mode_artifacts,
        "anomaly_flags": anomaly_flags,
        "regression_labels": regression_labels,
        "searchable_artifacts": searchable_artifacts,
        "issue_clusters": issue_clusters,
        "artifacts": run_artifacts,
    }

    output_dir = resolved_base_dir / "outputs" / "observability"
    output_dir.mkdir(parents=True, exist_ok=True)
    retention_index_path = output_dir / "retention_index.json"
    retention_index_path.write_text(
        json.dumps(retention_index, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    retention_index["retention_index_path"] = str(retention_index_path)
    return retention_index


def load_observability_retention_index(base_dir: Optional[Path] = None) -> Dict[str, Any]:
    resolved_base_dir = Path(base_dir or Path(__file__).resolve().parents[2])
    retention_index_path = (
        resolved_base_dir / "outputs" / "observability" / "retention_index.json"
    )
    if retention_index_path.exists():
        return json.loads(retention_index_path.read_text(encoding="utf-8"))
    return build_observability_retention_index(resolved_base_dir)


def search_observability_artifacts(
    base_dir: Optional[Path] = None,
    *,
    query: str = "",
    severity: Optional[str] = None,
    regression_label: Optional[str] = None,
    run_id: Optional[str] = None,
    limit: int = 50,
) -> Dict[str, Any]:
    retention = load_observability_retention_index(base_dir)
    query_lower = query.lower().strip()
    matched: List[Dict[str, Any]] = []
    for artifact in retention.get("searchable_artifacts", []):
        if severity and str(artifact.get("severity", "")).lower() != severity.lower():
            continue
        labels = [str(label) for label in artifact.get("regression_labels", [])]
        if regression_label and regression_label not in labels:
            continue
        if run_id and str(artifact.get("run_id")) != str(run_id):
            continue
        haystack = " ".join(
            [
                str(artifact.get("run_id", "")),
                str(artifact.get("severity", "")),
                str(artifact.get("artifact_search_key", "")),
                " ".join(labels),
            ]
        ).lower()
        if query_lower and query_lower not in haystack:
            continue
        matched.append(artifact)
    return {
        "total": len(matched),
        "results": matched[: max(1, int(limit))],
        "query": query,
        "filters": {
            "severity": severity,
            "regression_label": regression_label,
            "run_id": run_id,
        },
    }


def build_observability_triage_queue(
    base_dir: Optional[Path] = None,
    *,
    limit: int = 50,
) -> Dict[str, Any]:
    retention = load_observability_retention_index(base_dir)
    severity_rank = {"critical": 0, "warning": 1, "info": 2}
    searchable = list(retention.get("searchable_artifacts", []))
    searchable.sort(
        key=lambda item: (
            severity_rank.get(str(item.get("severity", "info")), 99),
            str(item.get("run_id", "")),
        )
    )
    queue = [
        {
            "run_id": item.get("run_id"),
            "severity": item.get("severity"),
            "regression_labels": item.get("regression_labels", []),
            "artifact_search_key": item.get("artifact_search_key"),
            "telemetry_artifact": item.get("telemetry_artifact"),
            "hardware_observability_artifact": item.get("hardware_observability_artifact"),
        }
        for item in searchable[: max(1, int(limit))]
    ]
    return {"queue_size": len(queue), "queue": queue}


def build_artifact_diff_view(
    base_dir: Optional[Path] = None,
    *,
    run_id: str,
    compare_to_run_id: Optional[str] = None,
) -> Dict[str, Any]:
    retention = load_observability_retention_index(base_dir)
    artifacts = retention.get("artifacts", [])
    artifact_by_run = {str(item.get("run_id")): item for item in artifacts}
    current = artifact_by_run.get(str(run_id))
    if current is None:
        raise KeyError(f"Unknown run_id: {run_id}")

    if compare_to_run_id:
        baseline = artifact_by_run.get(str(compare_to_run_id))
        if baseline is None:
            raise KeyError(f"Unknown compare_to_run_id: {compare_to_run_id}")
    else:
        ordered = [item for item in artifacts if str(item.get("run_id")) != str(run_id)]
        baseline = ordered[-1] if ordered else None

    current_anomaly = next(
        (row for row in retention.get("anomaly_flags", []) if row.get("run_id") == str(run_id)),
        {},
    )
    baseline_anomaly = (
        next(
            (
                row
                for row in retention.get("anomaly_flags", [])
                if baseline and row.get("run_id") == baseline.get("run_id")
            ),
            {},
        )
        if baseline
        else {}
    )
    diff = {
        "run_id": str(run_id),
        "compare_to_run_id": baseline.get("run_id") if baseline else None,
        "changed_fields": {
            "severity": {
                "current": current_anomaly.get("severity", "info"),
                "baseline": baseline_anomaly.get("severity", "info") if baseline else None,
            },
            "flags": {
                "current": current_anomaly.get("flags", []),
                "baseline": baseline_anomaly.get("flags", []) if baseline else [],
            },
            "telemetry_artifact": {
                "current": current.get("telemetry_artifact", ""),
                "baseline": baseline.get("telemetry_artifact", "") if baseline else None,
            },
            "hardware_observability_artifact": {
                "current": current.get("hardware_observability_artifact", ""),
                "baseline": baseline.get("hardware_observability_artifact", "") if baseline else None,
            },
        },
    }
    return diff


def get_issue_cluster_drilldown(
    base_dir: Optional[Path] = None,
    *,
    cluster_id: str,
) -> Dict[str, Any]:
    retention = load_observability_retention_index(base_dir)
    cluster = retention.get("issue_clusters", {}).get(cluster_id)
    if cluster is None:
        raise KeyError(f"Unknown cluster_id: {cluster_id}")
    artifacts = [
        item
        for item in retention.get("searchable_artifacts", [])
        if item.get("run_id") in set(cluster.get("run_ids", []))
    ]
    return {"cluster_id": cluster_id, "cluster": cluster, "artifacts": artifacts}