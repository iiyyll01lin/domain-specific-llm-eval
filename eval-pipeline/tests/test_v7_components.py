import pytest

from src.distributed.ray_runner import RayBatchProcessor
from src.utils.neo4j_manager import Neo4jGraphManager


def test_ray_processor() -> None:
    processor = RayBatchProcessor(num_cpus=2)
    items = [{"data": 1}, {"data": 2}]
    result = processor.process_items_distributed(items)
    assert len(result) == 2
    assert result[0]["processed_by"] == "ray_worker"


def test_neo4j_manager() -> None:
    manager = Neo4jGraphManager()
    manager.connect()
    assert manager.connected is True
    manager.add_relationship("NodeA", "NodeB", "KNOWS")
    res = manager.execute_cypher("MATCH (n) RETURN n")
    assert res[0]["n"]["id"] == "NodeA"
