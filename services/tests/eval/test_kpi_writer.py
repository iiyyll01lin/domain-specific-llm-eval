from __future__ import annotations

import json
from pathlib import Path

from services.eval.kpi_writer import KPIWriter


def test_kpi_writer_creates_file_atomically(tmp_path: Path) -> None:
    target = tmp_path / "kpis.json"
    writer = KPIWriter(target)
    data = {"metrics": {"faithfulness": {"average": 0.9}}, "counts": {"records": 3}}

    path = writer.write(data)

    assert path == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload == data


def test_kpi_writer_overwrites_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "kpis.json"
    writer = KPIWriter(target)
    writer.write({"metrics": {"m1": {}}, "counts": {"records": 1}})
    writer.write({"metrics": {"m1": {"average": 0.5}}, "counts": {"records": 2}})

    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["counts"]["records"] == 2
