# Domain-Specific LLM Evaluation Pipeline - AI Agent Guide

## Architecture Overview

This is a **domain-specific LLM evaluation framework** with **RAGAS integration** for professional testset generation and evaluation. The project consists of two main components:

- **Modified RAGAS Library** (`/ragas/`): Custom RAGAS implementation with relationship builders
- **Evaluation Pipeline** (`/eval-pipeline/`): Main orchestration system for end-to-end evaluation

### Key Data Flow
```
CSV Documents → Knowledge Graph (with relationships) → RAGAS Testset → RAG Evaluation → Metrics
```

## Critical Files & Patterns

### Core Entry Points
- **`eval-pipeline/run_pure_ragas_pipeline.py`** - Main RAGAS testset generation with relationship building
- **`eval-pipeline/src/utils/knowledge_graph_manager.py`** - KG persistence for pipeline management  
- **`config/pipeline_config.yaml`** - Central configuration (max_docs, max_samples, LLM endpoints)

### Configuration Structure
```yaml
testset_generation:
  ragas_config:
    knowledge_graph_config:
      existing_kg_file: "outputs/run_*/testsets/knowledge_graphs/*.json"
    custom_llm:
      endpoint: "http://localhost:8000/v1/chat/completions"  # Your custom LLM
```

### Data Processing Patterns
- **CSV to Documents**: `load_csv_documents()` handles Chinese/English mixed content with JSON parsing
- **Knowledge Graph Creation**: `create_knowledge_graph_from_documents()` builds nodes AND relationships
- **Relationship Building**: Uses Jaccard similarity, overlap scores, and cosine similarity (if embeddings available)
- **Output Structure**: Timestamped runs in `outputs/run_YYYYMMDD_HHMMSS_*/`

## Knowledge Graph System (CRITICAL)

### Two Parallel KG Storage Systems
1. **RAGAS Testset KG** (`/testsets/knowledge_graphs/`) - Direct JSON export for RAGAS
2. **Pipeline Management KG** (`/metadata/knowledge_graphs/`) - Pickled KG objects for reuse

### Relationship Building (Recently Fixed)
- **Previous Issue**: KGs were created with nodes only, no relationships
- **Fixed Implementation**: Uses RAGAS relationship builders:
  - `JaccardSimilarityBuilder` (entity-based, threshold=0.1)
  - `OverlapScoreBuilder` (keyphrase-based, threshold=0.05)  
  - `CosineSimilarityBuilder` (embedding-based, threshold=0.7)
  - `SummaryCosineSimilarityBuilder` (summary embedding-based, threshold=0.5)

### Node Properties for Relationships
```python
node.properties = {
    'entities': ['檢查', '鋼板', '表面'],  # For Jaccard similarity
    'keyphrases': ['檢查鋼板表面狀況'],    # For overlap scores
    'embedding': [0.1, 0.2, ...],        # For cosine similarity
    'summary_embedding': [0.3, 0.4, ...]  # For summary similarity
}
```

## Development Workflows

### Testing KG Relationships
```bash
cd eval-pipeline
python3 test_kg_quick.py  # Quick relationship test with 5 docs
python3 test_relationship_building.py  # Unit test for relationship builders
```

### Full Pipeline Execution
```bash
cd eval-pipeline
python3 run_pure_ragas_pipeline.py  # Generate testsets with relationships
```

### Config Updates
- Always use `apply_pipeline_fixes.py` to sync existing KG file references
- Check both config locations: root `/config/` and `/eval-pipeline/config/`

## Common Issues & Debugging

### Knowledge Graph Problems
- **Empty relationships**: Check if `build_relationships()` is called in `create_knowledge_graph_from_documents()`
- **UUID vs String IDs**: Nodes use `uuid.UUID` objects, serialization converts to strings
- **Import errors**: `SummaryCosineSimilarityBuilder` requires direct import from `.cosine` module

### Integration Points
- **RAGAS modifications**: Custom patches in `/ragas/ragas/src/` override standard RAGAS
- **LLM endpoints**: Custom LLM wrappers support local/API models (test with `setup_ragas_components()`)
- **Embedding models**: HuggingFace sentence-transformers integration for relationship building

## Project-Specific Conventions

- **Logging**: Emoji prefixes (🧠 KG, 📄 docs, 🔗 relationships, ✅ success, ❌ error)
- **Timestamps**: `YYYYMMDD_HHMMSS` format throughout output directories
- **Error Handling**: Graceful fallbacks with detailed logging, continues on relationship build failures
- **Bilingual Support**: Chinese/English mixed content handling in CSV processing

## Key Recent Fixes

1. **Relationship Building**: Added async `build_relationships()` function with multiple similarity builders
2. **Node Properties**: Enhanced nodes with entities, keyphrases, sentences for relationship building  
3. **UUID Handling**: Fixed Node ID creation (UUID objects) and serialization (string conversion)
4. **Import Structure**: Fixed `SummaryCosineSimilarityBuilder` import from `.cosine` module

## Testing & Validation

- Test relationship building with `test_kg_quick.py` - should show > 0 relationships
- Verify KG JSON files contain `relationships` array with similarity scores
- Check logs for relationship building progress: "🔗 Building X relationships..."
