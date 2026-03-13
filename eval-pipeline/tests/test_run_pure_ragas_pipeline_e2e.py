import json
import sys
import uuid
from types import ModuleType, SimpleNamespace

from langchain_core.documents import Document
from ragas.testset.graph import Node, NodeType

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
        "sync_knowledge_graph_to_neo4j",
        lambda kg, config: {
            "enabled": True,
            "backend": "memory",
            "synced_nodes": 1,
            "synced_relationships": 0,
            "retrieval_preview_count": 1,
            "retrieval_preview": [{"hop_1": "A", "hop_2": "B"}],
        },
    )
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

    telemetry_files = list(tmp_path.glob("outputs/pure_ragas_run_*/telemetry/pipeline_run_*.json"))
    assert len(telemetry_files) == 1
    telemetry_payload = json.loads(telemetry_files[0].read_text(encoding="utf-8"))
    assert telemetry_payload["stage_events"]
    assert {event["stage"] for event in telemetry_payload["stage_events"]} >= {
        "configuration",
        "document_loading",
        "knowledge_graph",
        "neo4j_sync",
        "testset_generation",
        "artifact_save",
    }


def test_sync_knowledge_graph_to_neo4j_uses_manager_backend(monkeypatch):
    node = SimpleNamespace(id="A", properties={"title": "Alpha"})
    relationship = SimpleNamespace(
        source=SimpleNamespace(id="A"),
        target=SimpleNamespace(id="B"),
        type="RELATED_TO",
        properties={"score": 0.9},
    )
    kg = SimpleNamespace(nodes=[node], relationships=[relationship])

    class _FakeManager:
        def __init__(self, *args, **kwargs):
            self.backend = "memory"

        def connect(self):
            return None

        def add_node(self, *args, **kwargs):
            return None

        def add_relationship(self, *args, **kwargs):
            return None

        def execute_cypher(self, query):
            return [{"hop_1": "A", "hop_2": "B", "relation": "RELATED_TO"}]

    monkeypatch.setattr(pipeline, "Neo4jGraphManager", _FakeManager)

    result = pipeline.sync_knowledge_graph_to_neo4j(
        kg,
        {
            "testset_generation": {
                "knowledge_graph_config": {"neo4j": {"enabled": True}}
            }
        },
    )

    assert result["enabled"] is True
    assert result["retrieval_preview_count"] == 1


class DummyGraph:
    def __init__(self, nodes):
        self.nodes = nodes


def test_verify_multihop_semantic_relationships_builds_verified_links():
    node_a = Node(
        id=uuid.uuid4(),
        type=NodeType.DOCUMENT,
        properties={
            "entities": ["steel", "inspection"],
            "keyphrases": ["steel plate inspection"],
            "summary": "steel inspection checklist",
        },
    )
    node_b = Node(
        id=uuid.uuid4(),
        type=NodeType.DOCUMENT,
        properties={
            "entities": ["steel", "surface"],
            "keyphrases": ["steel plate inspection"],
            "summary": "surface inspection checklist",
        },
    )

    relationships = pipeline.asyncio.run(
        pipeline.verify_multihop_semantic_relationships(DummyGraph([node_a, node_b]))
    )

    assert len(relationships) == 1
    assert relationships[0].properties["verified_for_multihop"] is True
    assert relationships[0].properties["semantic_correlation_score"] >= 0.15