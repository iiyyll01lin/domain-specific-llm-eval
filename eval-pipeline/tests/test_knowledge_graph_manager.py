from types import SimpleNamespace

from src.utils.knowledge_graph_manager import KnowledgeGraphManager, find_and_use_latest_kg


def test_save_and_list_knowledge_graph_artifacts(tmp_path):
    manager = KnowledgeGraphManager(tmp_path)
    kg = SimpleNamespace(
        nodes=[SimpleNamespace(id="n1", label="Node 1", type="entity", properties={"tenant": "public"})],
        relationships=[],
    )

    json_path = manager.save_knowledge_graph(kg, metadata={"source_documents": 1}, run_id="run001")
    artifacts = manager.list_available_knowledge_graphs()

    assert json_path.endswith("knowledge_graph_run001.json")
    assert len(artifacts) == 1
    assert artifacts[0]["run_id"] == "run001"
    assert artifacts[0]["nodes_count"] == 1


def test_find_and_use_latest_kg_updates_config(tmp_path):
    manager = KnowledgeGraphManager(tmp_path)
    kg = SimpleNamespace(nodes=[SimpleNamespace(id="n1", label="Node 1", type="entity", properties={})], relationships=[])
    manager.save_knowledge_graph(kg, run_id="run002")

    updated = find_and_use_latest_kg({"testset_generation": {}}, tmp_path)

    assert updated["testset_generation"]["ragas_config"]["knowledge_graph_config"]["existing_kg_file"].endswith("knowledge_graph_run002.json")


def test_apply_tenant_isolation_filters_nodes_and_relationships(tmp_path):
    manager = KnowledgeGraphManager(tmp_path)
    public_node = SimpleNamespace(id="public", properties={"tenant": "public", "min_role": "viewer"})
    private_node = SimpleNamespace(id="private", properties={"tenant": "tenant-a", "min_role": "admin"})
    relationship = SimpleNamespace(source=public_node, target=private_node, type="related_to", properties={})
    manager.kg = SimpleNamespace(nodes=[public_node, private_node], relationships=[relationship])

    manager.apply_tenant_isolation("tenant-a", "viewer")

    assert [node.id for node in manager.kg.nodes] == ["public"]
    assert manager.kg.relationships == []