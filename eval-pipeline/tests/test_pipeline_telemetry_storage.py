import json

from src.utils.pipeline_telemetry import PipelineTelemetry


class _FakeObjectStoreClient:
    def __init__(self):
        self.uploads = []

    def upload_file(self, bucket, key, file_path):
        self.uploads.append((bucket, key, file_path))
        return "checksum"


def test_pipeline_telemetry_records_stage_events_and_mirrors(tmp_path, monkeypatch):
    fake_client = _FakeObjectStoreClient()
    monkeypatch.setattr(PipelineTelemetry, "_build_object_store_client", lambda self: fake_client)

    telemetry = PipelineTelemetry(tmp_path)
    telemetry.log_stage_event("knowledge_graph", "completed", {"nodes": 3})
    telemetry.log_generation("single_hop_specific", 2, success=True)
    telemetry.finish()

    telemetry_files = list((tmp_path / "telemetry").glob("pipeline_run_*.json"))
    assert len(telemetry_files) == 1

    payload = json.loads(telemetry_files[0].read_text(encoding="utf-8"))
    assert payload["stage_events"][0]["stage"] == "knowledge_graph"
    assert payload["summary"]["total_generated"] == 2
    assert fake_client.uploads
