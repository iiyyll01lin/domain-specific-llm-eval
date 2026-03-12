import sys
from types import ModuleType, SimpleNamespace

from langchain_core.documents import Document

import run_pure_ragas_pipeline as pipeline


def test_main_completes_with_configured_generation_settings(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = {
        "data_sources": {
            "documents": {"document_files": ["doc.txt"]},
        },
        "testset_generation": {
            "max_documents_for_generation": 1,
            "max_total_samples": 1,
            "prompt_config": {"profile": "default"},
        },
    }

    monkeypatch.setattr(pipeline, "load_config", lambda *_args, **_kwargs: config)
    monkeypatch.setattr(
        pipeline,
        "load_txt_documents",
        lambda document_files, max_docs: [Document(page_content="Inspection procedure", metadata={"title": "Doc"})],
    )
    monkeypatch.setattr(
        pipeline,
        "setup_ragas_components",
        lambda cfg: (object(), object()),
    )

    async def fake_create_kg(documents, embeddings_model, async_settings=None):
        return SimpleNamespace(nodes=[SimpleNamespace(properties={"content": "x", "title": "Doc"})], relationships=[])

    monkeypatch.setattr(pipeline, "create_knowledge_graph_from_documents", fake_create_kg)
    monkeypatch.setattr(pipeline, "save_knowledge_graph", lambda kg, output_dir: str(output_dir / "kg.json"))
    monkeypatch.setattr(
        pipeline,
        "build_generation_settings",
        lambda config, llm, kg: pipeline.GenerationSettings(
            run_config=pipeline.RunConfig(max_workers=4),
            batch_size=2,
            personas=[pipeline.Persona(name="User", role_description="Role")],
            persona_records=[{"name": "User", "role_description": "Role"}],
            query_distribution=[(SimpleNamespace(name="single_hop_specific"), 1.0)],
            query_distribution_records=[{"synthesizer": "single_hop_specific", "weight": 1.0}],
            prompt_profile="default",
            fallback_templates={},
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "generate_ragas_testset",
        lambda kg, generator_llm, generator_embeddings, generation_settings, num_samples: [
            {
                "question": "Q1",
                "contexts": ["ctx"],
                "ground_truth": "A1",
                "synthesizer_name": "single_hop_specific",
                "generation_method": "pure_ragas",
                "generation_timestamp": "ts",
                "source_type": "txt",
            }
        ],
    )
    monkeypatch.setattr(pipeline, "save_testset", lambda test_samples, output_dir: str(output_dir / "testset.csv"))

    fake_utils = ModuleType("utils.pipeline_file_saver")

    class FakePipelineFileSaver:
        def __init__(self, output_dir):
            self.output_dir = output_dir

        def save_personas_json(self, data):
            return str(self.output_dir / "personas.json")

        def save_scenarios_json(self, data):
            return str(self.output_dir / "scenarios.json")

        def save_pipeline_metadata(self, data):
            assert data["async_generation"]["max_workers"] == 4
            assert data["query_distribution"][0]["synthesizer"] == "single_hop_specific"
            return str(self.output_dir / "metadata.json")

    fake_utils.PipelineFileSaver = FakePipelineFileSaver
    monkeypatch.setitem(sys.modules, "utils.pipeline_file_saver", fake_utils)

    assert pipeline.main(["--docs", "4", "--samples", "6"]) is True