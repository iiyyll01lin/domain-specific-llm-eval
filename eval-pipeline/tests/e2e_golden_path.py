"""E2E 'Golden Path' smoke test.

Calls ``scripts/golden_path_runner.py`` via ``subprocess.run`` (black-box
style) — no monkeypatching, no internal imports, no mocks.  The runner is
treated as an opaque CLI command, exactly as a user would invoke it.

Test sequence
─────────────
1. Spin up the runner in a subprocess with a fresh ``tmp_path`` output dir.
2. Assert exit code == 0.
3. Load ``evaluation_result.json`` from the output dir and verify:
   a. ``score`` is a non-zero float in (0.0, 1.0].
   b. ``contract`` dict is present with all required keys.
   c. ``contract["backend"]`` == ``"graph_context_relevance"``.
4. Assert the CSV artifact exists under ``<output_dir>/testsets/``.
5. Assert the Excel artifact (``golden_path_report.xlsx``) exists.

Design notes
────────────
* Uses ``scope="module"`` so the subprocess runs exactly once per pytest
  session module — cheap enough that parallelism (``-n auto``) is fine.
* The runner is fully offline: SQLiteGraphStore + NetworkX only; no LLM,
  no network, no model downloads.
* Skips automatically when the runner script has not been installed into the
  container image (pre-Phase-2 Docker builds).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Path constants
# ---------------------------------------------------------------------------

#: Root of the eval-pipeline package (two levels above this test file).
PIPELINE_DIR = Path(__file__).resolve().parents[1]

#: The CLI entry point exercised by this test.
RUNNER = PIPELINE_DIR / "scripts" / "golden_path_runner.py"

#: Fixtures co-located with this test for easy volume-mount in Docker.
FIXTURES_DIR = PIPELINE_DIR / "tests" / "fixtures"
CORPUS = FIXTURES_DIR / "golden_corpus.csv"
GOLDEN_CONFIG = FIXTURES_DIR / "golden_path_config.yaml"

#: All keys that a valid GraphContextRelevanceEvaluator contract must contain.
CONTRACT_KEYS = {
    "backend",
    "entity_overlap",
    "structural_connectivity",
    "hub_noise_penalty",
    "hub_nodes",
    "largest_component_size",
    "retrieved_count",
    "alpha",
    "beta",
    "gamma",
}


# ---------------------------------------------------------------------------
# Module-scoped fixture: run the CLI once and share across all test functions.
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def golden_run(tmp_path_factory: pytest.TempPathFactory):
    """Invoke the golden-path runner as a subprocess and return (proc, output_dir).

    Skips the entire module if the runner script is not present — this happens
    on pre-Phase-2 Docker images where ``eval-pipeline/scripts/`` has not yet
    been COPY'd into the image.
    """
    if not RUNNER.exists():
        pytest.skip(
            f"Runner not found: {RUNNER}\n"
            "Phase-2 Docker integration is required to run this test "
            "(scripts/ must be added to Dockerfile.test COPY and docker-compose.test.yml volume)."
        )

    output_dir = tmp_path_factory.mktemp("golden_e2e")

    proc = subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--corpus", str(CORPUS),
            "--config", str(GOLDEN_CONFIG),
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    return proc, output_dir


# ---------------------------------------------------------------------------
# Test 1 — Exit code
# ---------------------------------------------------------------------------

def test_runner_exits_zero(golden_run):
    """The runner must exit 0; any non-zero code is a golden-path failure."""
    proc, _ = golden_run
    assert proc.returncode == 0, (
        f"Runner exited with code {proc.returncode}.\n"
        f"STDOUT:\n{proc.stdout}\n"
        f"STDERR:\n{proc.stderr}"
    )


# ---------------------------------------------------------------------------
# Test 2 — JSON artifact: existence + non-zero score
# ---------------------------------------------------------------------------

def test_json_artifact_exists_and_has_nonzero_score(golden_run):
    """evaluation_result.json must exist and contain a score strictly > 0."""
    _, output_dir = golden_run

    result_path = output_dir / "evaluation_result.json"
    assert result_path.exists(), (
        f"evaluation_result.json not found under {output_dir}. "
        "Check that the runner's _write_artifacts() call completed."
    )

    data = json.loads(result_path.read_text(encoding="utf-8"))
    assert "score" in data, "evaluation_result.json is missing the 'score' key"
    assert isinstance(data["score"], float), (
        f"'score' must be a float, got {type(data['score']).__name__}"
    )
    assert data["score"] > 0.0, (
        f"Expected a non-zero GCR score for the golden corpus but got {data['score']}.\n"
        "The corpus nodes should share tokens with the query/answer."
    )


# ---------------------------------------------------------------------------
# Test 3 — Contract completeness
# ---------------------------------------------------------------------------

def test_contract_keys_are_complete(golden_run):
    """The ``contract`` sub-dict must contain every required diagnostic key."""
    _, output_dir = golden_run

    data = json.loads((output_dir / "evaluation_result.json").read_text(encoding="utf-8"))
    assert "contract" in data, "evaluation_result.json is missing the 'contract' key"

    missing = CONTRACT_KEYS - data["contract"].keys()
    assert not missing, (
        f"Contract is missing required keys: {sorted(missing)}\n"
        f"Present keys: {sorted(data['contract'].keys())}"
    )


# ---------------------------------------------------------------------------
# Test 4 — Contract backend tag
# ---------------------------------------------------------------------------

def test_contract_backend_tag(golden_run):
    """The backend tag must identify the graph evaluator unambiguously."""
    _, output_dir = golden_run

    data = json.loads((output_dir / "evaluation_result.json").read_text(encoding="utf-8"))
    assert data["contract"]["backend"] == "graph_context_relevance", (
        f"Unexpected backend: {data['contract']['backend']!r}"
    )


# ---------------------------------------------------------------------------
# Test 5 — CSV artifact written via PipelineFileSaver
# ---------------------------------------------------------------------------

def test_csv_artifact_written_to_outputs(golden_run):
    """PipelineFileSaver must produce a dated CSV under <output_dir>/testsets/."""
    _, output_dir = golden_run

    csv_files = list((output_dir / "testsets").glob("golden_path_eval_*.csv"))
    assert csv_files, (
        f"No golden_path_eval_*.csv found under {output_dir / 'testsets'}.\n"
        "Ensure PipelineFileSaver.save_testset_csv() was called in the runner."
    )


# ---------------------------------------------------------------------------
# Test 6 — Excel artifact written directly by runner
# ---------------------------------------------------------------------------

def test_excel_artifact_written_to_outputs(golden_run):
    """An XLSX report must be written to the output directory root."""
    _, output_dir = golden_run

    xlsx_path = output_dir / "golden_path_report.xlsx"
    assert xlsx_path.exists(), (
        f"golden_path_report.xlsx not found under {output_dir}.\n"
        "Ensure the runner's pd.DataFrame.to_excel() call completed."
    )
