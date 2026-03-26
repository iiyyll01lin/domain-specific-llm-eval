from __future__ import annotations

from src.utils.neo4j_manager import Neo4jGraphManager


class _FakeRelationship(dict):
    def type(self) -> str:
        return "RELATED_TO"


class _FakeRecord:
    def __init__(self, payload):
        self._payload = payload

    def data(self):
        return self._payload


class _FakeSession:
    def __init__(self, records):
        self._records = records
        self.runs = []

    def run(self, query, parameters=None):
        self.runs.append((query, parameters))
        return self._records

    def close(self):
        return None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return None


class _FakeDriver:
    def __init__(self, records):
        self.records = records
        self.sessions = []

    def verify_connectivity(self):
        return None

    def session(self, database=None):
        session = _FakeSession(self.records)
        self.sessions.append((database, session))
        return session

    def close(self):
        return None


def test_neo4j_manager_returns_added_relationships() -> None:
    manager = Neo4jGraphManager()
    manager.connect()
    manager.add_node("A", title="Alpha")
    manager.add_node("B", title="Beta")
    manager.add_relationship("A", "B", "RELATED_TO", score=0.8)

    result = manager.execute_cypher("MATCH (a)-[r]->(b) RETURN a, r, b")

    assert result[0]["hop_1"] == "A"
    assert result[0]["hop_2"] == "B"
    assert result[0]["relation"] == "RELATED_TO"
    assert result[0]["properties"]["score"] == 0.8


def test_neo4j_manager_supports_shortest_path_queries() -> None:
    manager = Neo4jGraphManager()
    manager.connect()
    manager.add_relationship("A", "B", "LINKS")
    manager.add_relationship("B", "C", "LINKS")

    result = manager.execute_cypher(
        "MATCH p = shortestPath((a {id: 'A'})-[*]-(b {id: 'C'})) RETURN p"
    )

    assert result[0]["path"] == ["A", "B", "C"]
    assert result[0]["length"] == 2


def test_neo4j_manager_uses_real_driver_when_available() -> None:
    records = [
        _FakeRecord(
            {
                "a": {"id": "A", "title": "Alpha"},
                "b": {"id": "B", "title": "Beta"},
                "r": _FakeRelationship(score=0.9, relation="RELATED_TO"),
            }
        )
    ]
    driver = _FakeDriver(records)
    manager = Neo4jGraphManager(driver=driver, database="neo4j")

    manager.connect()
    result = manager.execute_cypher("MATCH (a)-[r]->(b) RETURN a, r, b")

    assert manager.backend == "neo4j"
    assert driver.sessions[0][0] == "neo4j"
    assert result[0]["hop_1"] == "A"
    assert result[0]["hop_2"] == "B"
    assert result[0]["properties"]["score"] == 0.9