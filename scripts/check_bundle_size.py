#!/usr/bin/env python3
"""
TASK-082: Bundle size budget enforcement script.

Checks that the built KG panel JavaScript chunk does not exceed 300 KB gzipped.
Designed to run in CI after `npm run build`.

Usage:
    python3 scripts/check_bundle_size.py [--build-dir <path>] [--budget-kb 300]

Exit codes:
    0  All chunks within budget
    1  One or more chunks exceed budget
"""
import argparse
import gzip
import json
import os
import pathlib
import sys


KG_CHUNK_PATTERN = "KgPanel"   # match any chunk whose name contains this
DEFAULT_BUDGET_KB = 300
DEFAULT_BUILD_DIR = "insights-portal/dist/assets"


def gzip_size(path: pathlib.Path) -> int:
    """Return gzipped byte size of *path* without writing to disk."""
    with open(path, "rb") as fh:
        data = fh.read()
    return len(gzip.compress(data, compresslevel=9))


def check_chunks(build_dir: str, budget_kb: int, pattern: str) -> list[dict]:
    """Return a list of {name, raw_kb, gz_kb, ok} dicts for every JS chunk."""
    results = []
    base = pathlib.Path(build_dir)
    if not base.exists():
        print(f"[bundle-check] Build directory not found: {base}", file=sys.stderr)
        sys.exit(1)

    js_files = sorted(base.glob("**/*.js"))
    if not js_files:
        print(f"[bundle-check] No .js files found in {base}", file=sys.stderr)
        sys.exit(1)

    for js_file in js_files:
        is_kg = pattern.lower() in js_file.name.lower()
        raw_kb = js_file.stat().st_size / 1024
        gz_kb = gzip_size(js_file) / 1024
        ok = (gz_kb <= budget_kb) if is_kg else True
        results.append(
            {
                "name": js_file.name,
                "raw_kb": round(raw_kb, 1),
                "gz_kb": round(gz_kb, 1),
                "is_kg_chunk": is_kg,
                "budget_kb": budget_kb if is_kg else None,
                "ok": ok,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Bundle size budget check")
    parser.add_argument("--build-dir", default=DEFAULT_BUILD_DIR)
    parser.add_argument("--budget-kb", type=int, default=DEFAULT_BUDGET_KB)
    parser.add_argument("--pattern", default=KG_CHUNK_PATTERN,
                        help="Substring to match KG panel chunks (case-insensitive)")
    parser.add_argument("--report", default=None,
                        help="Write JSON diff report to this path (optional)")
    args = parser.parse_args()

    results = check_chunks(args.build_dir, args.budget_kb, args.pattern)

    print(f"\n{'Chunk':<50} {'Raw KB':>8} {'GZ KB':>8} {'Budget':>8} {'Status':>8}")
    print("-" * 86)
    failed = []
    for r in results:
        budget_str = f"{r['budget_kb']} KB" if r["budget_kb"] else "—"
        status = "✓" if r["ok"] else "✗ OVER"
        if not r["ok"]:
            failed.append(r)
        print(f"{r['name']:<50} {r['raw_kb']:>8.1f} {r['gz_kb']:>8.1f} {budget_str:>8} {status:>8}")

    if args.report:
        report_path = pathlib.Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as fh:
            json.dump({"budget_kb": args.budget_kb, "chunks": results}, fh, indent=2)
        print(f"\n[bundle-check] Report written to {args.report}")

    if failed:
        print(f"\n[bundle-check] FAIL: {len(failed)} chunk(s) exceed the {args.budget_kb} KB gz budget:")
        for r in failed:
            print(f"  {r['name']}: {r['gz_kb']:.1f} KB gz (budget {r['budget_kb']} KB)")
        return 1

    print(f"\n[bundle-check] PASS: all KG chunks within {args.budget_kb} KB gz budget.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
