"""Tests for services/kg/repository.py (TASK-060)."""
import os
import tempfile
import pytest
from services.kg.repository import KgRepository


@pytest.fixture
def repo(tmp_path):
    db_path = str(tmp_path / "test_kg.db")
    return KgRepository(db_path)


def test_create_returns_job(repo):
    job = repo.create(doc_count=5)
    assert job.kg_id
    assert job.status == "queued"
    assert job.doc_count == 5


def test_get_returns_created_job(repo):
    job = repo.create(doc_count=3)
    fetched = repo.get(job.kg_id)
    assert fetched is not None
    assert fetched.kg_id == job.kg_id
    assert fetched.doc_count == 3


def test_get_nonexistent_returns_none(repo):
    assert repo.get("nonexistent-id") is None


def test_update_status(repo):
    job = repo.create(doc_count=2)
    repo.update_status(job.kg_id, "running")
    updated = repo.get(job.kg_id)
    assert updated.status == "running"


def test_update_completed(repo):
    job = repo.create(doc_count=2)
    repo.update_completed(
        job.kg_id,
        node_count=10,
        edge_count=20,
        artifacts={"nodes": "/tmp/nodes.json"},
    )
    updated = repo.get(job.kg_id)
    assert updated.status == "completed"
    assert updated.node_count == 10
    assert updated.edge_count == 20
    assert updated.artifacts["nodes"] == "/tmp/nodes.json"


def test_update_failed(repo):
    job = repo.create(doc_count=2)
    repo.update_failed(job.kg_id, "extraction failed")
    updated = repo.get(job.kg_id)
    assert updated.status == "failed"
    assert updated.error_message == "extraction failed"


def test_list_all_returns_all_jobs(repo):
    repo.create(doc_count=1)
    repo.create(doc_count=2)
    jobs = repo.list_all()
    assert len(jobs) == 2


def test_list_all_empty(repo):
    assert repo.list_all() == []


def test_creates_parent_dir(tmp_path):
    nested = str(tmp_path / "deep" / "nested" / "kg.db")
    r = KgRepository(nested)
    assert os.path.exists(nested)
