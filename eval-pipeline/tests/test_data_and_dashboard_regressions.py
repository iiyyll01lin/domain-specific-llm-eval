from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd

from src.data.csv_data_processor import CSVDataProcessor
from src.ui.dashboard_job_runner import DashboardJobRunner


# Resolve relative to this test file so the path works in any environment
# (local checkout, Docker container, GitHub Actions, …).
DOCUMENT_LOADER_PATH = Path(__file__).resolve().parent.parent / "document_loader.py"
DOCUMENT_LOADER_SPEC = importlib.util.spec_from_file_location(
    "legacy_eval_pipeline_document_loader",
    DOCUMENT_LOADER_PATH,
)
assert DOCUMENT_LOADER_SPEC is not None and DOCUMENT_LOADER_SPEC.loader is not None
DOCUMENT_LOADER_MODULE = importlib.util.module_from_spec(DOCUMENT_LOADER_SPEC)
DOCUMENT_LOADER_SPEC.loader.exec_module(DOCUMENT_LOADER_MODULE)
DocumentLoader = DOCUMENT_LOADER_MODULE.DocumentLoader


def test_csv_data_processor_skips_non_string_content_rows(tmp_path: Path) -> None:
    processor = CSVDataProcessor(
        config={
            "csv": {
                "format": {
                    "column_mapping": {"content": "content"},
                }
            }
        },
        output_dir=tmp_path,
    )

    row = pd.Series({"content": 123.45})
    document = processor._process_csv_row(
        row=row,
        idx=0,
        csv_path=tmp_path / "sample.csv",
        column_mapping={"content": "content"},
        content_json_fields={},
        json_text_template="",
    )

    assert document is None


def test_dashboard_job_runner_handles_stage_event_without_stage_key(tmp_path: Path) -> None:
    runner = DashboardJobRunner(tmp_path)

    progress = runner._build_progress(
        [
            {"status": "completed"},
        ]
    )

    assert progress["completed_stages"] == 1
    assert progress["latest_stage"] is None


def test_legacy_csv_simple_basic_mapping_is_covered(tmp_path: Path) -> None:
    csv_path = tmp_path / "test_content.csv"
    pd.DataFrame(
        {
            "id": [1, 2, 3],
            "content": [
                json.dumps({"text": "First CSV row content for mapping validation.", "title": "Doc 1"}),
                json.dumps({"text": "Second CSV row content for mapping validation.", "title": "Doc 2"}),
                json.dumps({"text": "Third CSV row content for mapping validation.", "title": "Doc 3"}),
            ],
        }
    ).to_csv(csv_path, index=False)

    config = {
        "data_sources": {
            "input_type": "csv",
            "csv": {
                "csv_files": [str(csv_path)],
            },
        }
    }

    loader = DocumentLoader(config)
    documents, metadata = loader.load_all_documents()

    assert len(documents) == 3
    assert len(metadata) == 3
    assert metadata[0].get("csv_id") is not None