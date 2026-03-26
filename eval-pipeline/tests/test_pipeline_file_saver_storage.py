import json

from src.utils.pipeline_file_saver import PipelineFileSaver


class _FakeObjectStoreClient:
    def __init__(self):
        self.uploads = []

    def upload_file(self, bucket, key, file_path):
        self.uploads.append((bucket, key, file_path))
        return "checksum"


def test_pipeline_file_saver_mirrors_saved_artifacts(tmp_path, monkeypatch):
    fake_client = _FakeObjectStoreClient()
    monkeypatch.setattr(PipelineFileSaver, "_build_object_store_client", lambda self: fake_client)

    saver = PipelineFileSaver(tmp_path)
    personas_path = saver.save_personas_json([{"name": "Operator", "role_description": "Tests"}])
    metadata_path = saver.save_pipeline_metadata({"pipeline_type": "pure_ragas"})

    metadata = json.loads(open(metadata_path, "r", encoding="utf-8").read())
    assert personas_path in metadata["mirrored_files"]
    assert fake_client.uploads
