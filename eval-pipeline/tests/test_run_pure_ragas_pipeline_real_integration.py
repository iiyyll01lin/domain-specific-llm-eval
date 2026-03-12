import json
import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document
from ragas.testset.graph import KnowledgeGraph, Node

from run_pure_ragas_pipeline import (build_query_distribution,
                                     create_knowledge_graph_from_documents,
                                     save_testset)


def test_embedding_failure_handling():
    import asyncio
    asyncio.run(run_test_embedding_failure_handling())

async def run_test_embedding_failure_handling():
    """Test that embedding model failure doesn't crash KG creation"""
    doc = Document(page_content="Test valid content for chunking. " * 20, metadata={"title": "test"})
    # Pass a MagicMock that raises an exception when encode/embed_documents is called
    mock_embeddings = MagicMock()
    mock_embeddings.embed_documents.side_effect = Exception("Mock Embedding Failure")
    mock_embeddings.embed_text.side_effect = Exception("Mock Embedding Failure")
    
    kg = await create_knowledge_graph_from_documents([doc], embeddings_model=mock_embeddings)
    assert kg is not None
    assert len(kg.nodes) > 0
    # The nodes should just be missing embeddings instead of crashing
    assert "embedding" not in kg.nodes[0].properties or kg.nodes[0].properties.get("embedding") is None

def test_json_contexts_round_trip(tmp_path):
    """Test that generated samples with complex JSON contexts can be saved and reloaded properly."""
    samples = [{
        "question": "test question?",
        "ground_truth": "test answer",
        "contexts": ["Context 1 with JSON-like structure: {\"key\":\"value\"}", "Context 2"],
        "synthesizer_name": "test_synthesizer"
    }]
    
    save_testset(samples, output_dir=tmp_path)
    
    testset_dir = tmp_path / "testsets"
    files = list(testset_dir.glob("pure_ragas_testset_*.csv"))
    assert len(files) == 1
    
    import pandas as pd
    df = pd.read_csv(files[0])
    assert len(df) == 1
    # Check if contexts have been properly dumped as JSON strings
    contexts_str = df.iloc[0]["contexts"]
    assert isinstance(contexts_str, str)
    parsed_contexts = json.loads(contexts_str)
    assert len(parsed_contexts) == 2
    assert "Context 1" in parsed_contexts[0]

def test_query_distribution_fallback():
    """Test query distribution graceful degradation when a synthesizer is unavailable"""
    from run_pure_ragas_pipeline import build_query_distribution
    
    config = {
        "testset_generation": {
            "generation": {
                "query_distribution": [
                    {"synthesizer": "unavailable_synth", "weight": 1.0}
                ]
            }
        }
    }
    mock_llm = MagicMock()
    mock_kg = KnowledgeGraph()
    mock_library = {}
    
    dist, records = build_query_distribution(config, mock_llm, mock_kg, mock_library, "default")
    
    # It should fall back to the default distribution
    assert len(dist) > 0
    assert any(weight > 0 for _, weight in dist)
    # The fallback should NOT contain unavailable_synth since it's not a known builder or it was skipped
    assert all("unavailable_synth" not in r.get("synthesizer", "") for r in records)

