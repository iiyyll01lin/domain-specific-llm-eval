from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.data.csv_data_processor import CSVDataProcessor
from src.ui.dashboard_job_runner import DashboardJobRunner


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