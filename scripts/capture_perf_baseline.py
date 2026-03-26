#!/usr/bin/env python3
"""TASK-102: Performance Baseline Capture.

Benchmarks the ingestion→eval latency on a small synthetic dataset and
stores results in benchmarks/baseline.json.

Usage:
    python3 scripts/capture_perf_baseline.py [--docs 10] [--output benchmarks/baseline.json]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Synthetic benchmark (no live services needed — measures pure service logic)
# ---------------------------------------------------------------------------


def _benchmark_extraction(n_docs: int = 10) -> list[float]:
    """Measure extraction latency per document using the KG extract module."""
    from services.kg.extract import extract_all

    texts = [
        f"Document {i}: This is sample content for performance benchmarking. "
        f"Entity: ProductX, Company: AcmeCorp, Feature: reliability and scalability."
        for i in range(n_docs)
    ]
    latencies: list[float] = []
    for text in texts:
        t0 = time.perf_counter()
        extract_all(text)
        latencies.append(time.perf_counter() - t0)
    return latencies


def _benchmark_relationship_building(n_nodes: int = 10) -> list[float]:
    """Measure relationship building latency for n_nodes synthetic nodes."""
    from services.kg.relationships import build_jaccard_relationships

    nodes = [
        {
            "node_id": f"node-{i}",
            "entities": [f"ent{i}", f"shared{i % 3}"],
            "keyphrases": [f"kp{i}"],
            "sentences": [],
        }
        for i in range(n_nodes)
    ]
    latencies: list[float] = []
    for _ in range(5):
        t0 = time.perf_counter()
        build_jaccard_relationships(nodes, threshold=0.05)
        latencies.append(time.perf_counter() - t0)
    return latencies


def run_benchmarks(n_docs: int) -> dict:
    extract_latencies = _benchmark_extraction(n_docs)
    rel_latencies = _benchmark_relationship_building(n_docs)

    return {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config": {"n_docs": n_docs},
        "extraction": {
            "n_samples": len(extract_latencies),
            "mean_ms": round(statistics.mean(extract_latencies) * 1000, 2),
            "median_ms": round(statistics.median(extract_latencies) * 1000, 2),
            "p95_ms": round(sorted(extract_latencies)[int(len(extract_latencies) * 0.95)] * 1000, 2),
            "max_ms": round(max(extract_latencies) * 1000, 2),
        },
        "relationship_building": {
            "n_samples": len(rel_latencies),
            "mean_ms": round(statistics.mean(rel_latencies) * 1000, 2),
            "median_ms": round(statistics.median(rel_latencies) * 1000, 2),
            "p95_ms": round(sorted(rel_latencies)[int(len(rel_latencies) * 0.95)] * 1000, 2),
            "max_ms": round(max(rel_latencies) * 1000, 2),
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Capture performance baseline")
    parser.add_argument("--docs", type=int, default=10, help="Number of synthetic docs")
    parser.add_argument("--output", default="benchmarks/baseline.json", help="Output path")
    args = parser.parse_args()

    print(f"Running performance baseline with {args.docs} docs...")
    results = run_benchmarks(args.docs)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2))

    print(f"\nResults written to {output_path}")
    print(f"  Extraction  p95: {results['extraction']['p95_ms']} ms")
    print(f"  Rel. build  p95: {results['relationship_building']['p95_ms']} ms")
