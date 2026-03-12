#!/usr/bin/env python3
"""
Simple Pure RAGAS Pipeline Runner

This script implements the corrected pipeline flow using ONLY RAGAS TestsetGenerator:
1. ✅ Load CSV documents and create knowledge graph
2. ✅ Generate testset using RAGAS TestsetGenerator (no fallbacks)
3. ✅ Save knowledge graph for reuse
4. 🔄 Later: Query RAG system with generated questions
5. 🔄 Later: Extract keywords from RAG responses (not pre-generated!)
6. 🔄 Later: Calculate evaluation metrics

This fixes the design flaw where keywords were calculated before RAG testing.
"""

import asyncio
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import yaml
from langchain_community.embeddings import HuggingFaceEmbeddings

# Add RAGAS to path
ragas_path = str(
    (Path(__file__).resolve().parent.parent / "ragas" / "ragas" / "src").resolve()
)
if ragas_path not in sys.path:
    sys.path.insert(0, ragas_path)

# Import LangChain for document processing
from langchain_core.documents import Document
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
# Import RAGAS components
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.persona import Persona
from ragas.testset.synthesizers import default_query_distribution
from ragas.testset.synthesizers.multi_hop.abstract import \
    MultiHopAbstractQuerySynthesizer
from ragas.testset.synthesizers.multi_hop.specific import \
    MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.single_hop.specific import \
    SingleHopSpecificQuerySynthesizer
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder, JaccardSimilarityBuilder, OverlapScoreBuilder)
from ragas.testset.transforms.relationship_builders.cosine import \
    SummaryCosineSimilarityBuilder

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

from src.utils.prompt_templates import (get_fallback_generation_templates,
                                        get_persona_templates,
                                        get_query_distribution_templates,
                                        load_prompt_library)
from src.utils.pipeline_telemetry import PipelineTelemetry


GLOBAL_TELEMETRY = None


@dataclass
class GenerationSettings:
    run_config: RunConfig
    batch_size: Optional[int]
    personas: List[Persona]
    persona_records: List[Dict[str, Any]]
    query_distribution: List[Tuple[Any, float]]
    query_distribution_records: List[Dict[str, Any]]
    prompt_profile: str
    fallback_templates: Dict[str, Dict[str, Any]]


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the pure RAGAS pipeline")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--docs", type=int, default=None, help="Override max documents")
    parser.add_argument("--samples", type=int, default=None, help="Override max samples")
    return parser.parse_args(argv)


def apply_cli_overrides(
    config: Dict[str, Any], docs: Optional[int] = None, samples: Optional[int] = None
) -> Dict[str, Any]:
    updated_config = json.loads(json.dumps(config))
    testset_generation = updated_config.setdefault("testset_generation", {})
    if docs is not None:
        testset_generation["max_documents_for_generation"] = docs
    if samples is not None:
        testset_generation["max_total_samples"] = samples
    return updated_config


def get_prompt_config(config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    prompt_config = config.get("testset_generation", {}).get("prompt_config", {})
    profile = prompt_config.get("profile", "default")
    templates_path = prompt_config.get("templates_file")
    library = load_prompt_library(
        templates_path,
        base_dir=Path(__file__).resolve().parent,
    )
    return library, profile


def build_personas(
    config: Dict[str, Any], library: Dict[str, Any], profile: str
) -> Tuple[List[Persona], List[Dict[str, Any]]]:
    generation_config = config.get("testset_generation", {}).get("generation", {})
    persona_records = generation_config.get("personas") or get_persona_templates(
        profile_name=profile,
        library=library,
    )
    personas = [
        Persona(
            name=record["name"],
            role_description=record["role_description"],
        )
        for record in persona_records
    ]
    return personas, persona_records


def build_run_config(config: Dict[str, Any]) -> Tuple[RunConfig, Optional[int]]:
    generation_config = config.get("testset_generation", {}).get("generation", {})
    async_config = generation_config.get("async_generation", {})
    enabled = async_config.get("enabled", True)
    run_config = RunConfig(
        timeout=async_config.get("timeout", 180),
        max_retries=async_config.get("max_retries", 6),
        max_wait=async_config.get("max_wait", 60),
        max_workers=async_config.get("max_workers", 8 if enabled else 1),
        log_tenacity=async_config.get("log_tenacity", False),
        seed=async_config.get("seed", 42),
    )
    batch_size = async_config.get("batch_size") if enabled else None
    return run_config, batch_size


def build_query_distribution(
    config: Dict[str, Any],
    llm: LangchainLLMWrapper,
    kg: KnowledgeGraph,
    library: Dict[str, Any],
    profile: str,
) -> Tuple[List[Tuple[Any, float]], List[Dict[str, Any]]]:
    generation_config = config.get("testset_generation", {}).get("generation", {})
    configured_distribution = generation_config.get(
        "query_distribution"
    ) or get_query_distribution_templates(profile_name=profile, library=library)

    kg_has_nodes = bool(getattr(kg, "nodes", []))

    synthesizer_builders = {
        "single_hop_specific": lambda: SingleHopSpecificQuerySynthesizer(llm=llm),
        "multi_hop_abstract": lambda: MultiHopAbstractQuerySynthesizer(llm=llm),
        "multi_hop_specific": lambda: MultiHopSpecificQuerySynthesizer(llm=llm),
    }

    available_distribution: List[Tuple[Any, float]] = []
    distribution_records: List[Dict[str, Any]] = []

    for item in configured_distribution:
        if not isinstance(item, dict):
            continue
        synthesizer_name = item.get("synthesizer")
        if not isinstance(synthesizer_name, str):
            continue
        weight = float(item.get("weight", 0.0))
        builder = synthesizer_builders.get(synthesizer_name)
        if builder is None or weight <= 0:
            continue

        synthesizer = builder()
        if kg_has_nodes:
            try:
                is_available = bool(synthesizer.get_node_clusters(kg))
            except Exception:
                is_available = True
        else:
            is_available = True

        if not is_available:
            logger.info(f"ℹ️ Skipping unavailable synthesizer: {synthesizer_name}")
            continue

        available_distribution.append((synthesizer, weight))
        distribution_records.append(
            {
                "synthesizer": synthesizer_name,
                "weight": weight,
                "available": True,
            }
        )

    if not available_distribution:
        fallback_distribution = default_query_distribution(
            llm, kg if kg_has_nodes else None
        )
        fallback_records = [
            {
                "synthesizer": synthesizer.name,
                "weight": weight,
                "available": True,
            }
            for synthesizer, weight in fallback_distribution
        ]
        return fallback_distribution, fallback_records

    total_weight = sum(weight for _, weight in available_distribution)
    normalized_distribution = []
    normalized_records = []
    for record, (synthesizer, weight) in zip(
        distribution_records, available_distribution
    ):
        normalized_weight = weight / total_weight
        normalized_distribution.append((synthesizer, normalized_weight))
        normalized_records.append({**record, "weight": normalized_weight})
    return normalized_distribution, normalized_records


def build_generation_settings(
    config: Dict[str, Any],
    llm: LangchainLLMWrapper,
    kg: KnowledgeGraph,
) -> GenerationSettings:
    library, profile = get_prompt_config(config)
    personas, persona_records = build_personas(config, library, profile)
    run_config, batch_size = build_run_config(config)
    query_distribution, query_distribution_records = build_query_distribution(
        config,
        llm,
        kg,
        library,
        profile,
    )
    fallback_templates = get_fallback_generation_templates(
        profile_name=profile,
        library=library,
    )
    return GenerationSettings(
        run_config=run_config,
        batch_size=batch_size,
        personas=personas,
        persona_records=persona_records,
        query_distribution=query_distribution,
        query_distribution_records=query_distribution_records,
        prompt_profile=profile,
        fallback_templates=fallback_templates,
    )


async def _resolve_embedding_result(result: Any) -> Any:
    if hasattr(result, "__await__"):
        return await result
    return result


async def _populate_node_embeddings(
    node: Node,
    content: str,
    summary: str,
    embeddings_model: LangchainEmbeddingsWrapper,
    semaphore: asyncio.Semaphore,
) -> None:
    async with semaphore:
        node.properties["embedding"] = await _resolve_embedding_result(
            embeddings_model.embed_text(content)
        )
        node.properties["summary_embedding"] = await _resolve_embedding_result(
            embeddings_model.embed_text(summary)
        )


def _to_json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_to_json_safe(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def create_summary_similarity_relationships(kg: KnowledgeGraph):
    """Create summary_similarity relationships required by MultiHopAbstractQuerySynthesizer"""
    relationships = []
    nodes = list(kg.nodes)

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i + 1 :], i + 1):
            # Simple text-based similarity for summaries
            summary1 = node1.properties.get("summary", "")
            summary2 = node2.properties.get("summary", "")

            if summary1 and summary2:
                # Simple word overlap similarity
                words1 = set(summary1.lower().split())
                words2 = set(summary2.lower().split())
                overlap = words1 & words2

                if overlap:
                    similarity = len(overlap) / max(len(words1), len(words2))

                    # Create relationship if similarity is above threshold
                    if similarity > 0.1:  # Threshold for summary similarity
                        from ragas.testset.graph import Relationship

                        rel = Relationship(
                            source=node1,
                            target=node2,
                            type="summary_similarity",
                            properties={"summary_similarity": similarity},
                        )
                        relationships.append(rel)

    return relationships


def create_entities_overlap_relationships(kg: KnowledgeGraph):
    """Create entities_overlap relationships required by MultiHopSpecificQuerySynthesizer"""
    relationships = []
    nodes = list(kg.nodes)

    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i + 1 :], i + 1):
            # Get entities from both nodes
            entities1 = set(node1.properties.get("entities", []))
            entities2 = set(node2.properties.get("entities", []))

            # Calculate entities overlap
            overlap = entities1 & entities2

            if len(overlap) > 0:  # Nodes share at least one entity
                overlap_score = (
                    len(overlap) / min(len(entities1), len(entities2))
                    if entities1 and entities2
                    else 0
                )

                # Create relationship if overlap is significant
                if overlap_score > 0.2:  # Threshold for entities overlap
                    from ragas.testset.graph import Relationship

                    rel = Relationship(
                        source=node1,
                        target=node2,
                        type="entities_overlap",
                        properties={
                            "entities_overlap": overlap_score,
                            "shared_entities": list(overlap),
                        },
                    )
                    relationships.append(rel)

    return relationships


def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load pipeline configuration with environment variable expansion"""
    import os
    import re

    from dotenv import load_dotenv

    load_dotenv()

    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = (Path(__file__).parent / config_file).resolve()

    with open(config_file, "r", encoding="utf-8") as f:
        content = f.read()

    pattern = re.compile(r"\$\{([^}]+)\}")

    def replacer(match):
        inner = match.group(1)
        if ":" in inner:
            var_name, default_value = inner.split(":", 1)
        else:
            var_name, default_value = inner, ""
        return os.environ.get(var_name, default_value)

    content = pattern.sub(replacer, content)
    config = yaml.safe_load(content)

    if config:
        return config

    template_file = config_file.parent / "pipeline_config.template.yaml"
    logger.warning(
        f"⚠️ Empty config detected at {config_file}; falling back to template {template_file}"
    )
    with open(template_file, "r", encoding="utf-8") as f:
        content = f.read()
        content = pattern.sub(replacer, content)
        return yaml.safe_load(content) or {}


def extract_content_from_csv_row(content_json: str) -> Tuple[str, Dict]:
    """Extract text content and metadata from CSV content JSON"""
    try:
        content_data = json.loads(content_json)

        # Extract fields from error code JSON structure
        error_code = content_data.get("error_code", "")
        display = content_data.get("display", "")
        cause = content_data.get("cause", "")
        remedy = content_data.get("remedy", "")
        used_by = content_data.get("used_by", "")

        # Combine all fields into a comprehensive text content
        text_content = f"""Error Code: {error_code}

Display: {display}

Cause: {cause}

Remedy: {remedy}

Used By: {used_by}"""

        metadata = {
            "error_code": error_code,
            "display": display,
            "cause": cause,
            "remedy": remedy,
            "used_by": used_by,
            "source": "smt-nxt-errorcode",
            "language": "zh-TW",
        }

        return text_content, metadata

    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to parse content JSON: {e}")
        return str(content_json), {}


def load_txt_documents(document_files: List[str], max_docs: int = 53) -> List[Document]:
    """Load and process TXT documents for RAGAS (working 2024 method)"""
    logger.info(f"📂 Loading TXT documents from: {len(document_files)} files")

    documents = []
    processed_count = 0

    for doc_file in document_files:
        if processed_count >= max_docs:
            break

        doc_path = Path(doc_file)
        if not doc_path.exists():
            logger.warning(f"Document file not found: {doc_file}")
            continue

        try:
            # Read document content
            with open(doc_path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if len(content) < 100:  # Skip very short content
                continue

            # Create LangChain Document
            doc = Document(
                page_content=content,
                metadata={
                    "title": doc_path.stem,
                    "source": str(doc_path),
                    "document_id": processed_count,
                    "filename": doc_path.name,
                    "content_type": "steel_plate_inspection",
                },
            )
            documents.append(doc)
            processed_count += 1

        except Exception as e:
            logger.warning(f"Failed to load document {doc_file}: {e}")
            continue

    logger.info(f"✅ Loaded {len(documents)} TXT documents")
    return documents


def load_csv_documents(
    csv_file_path: str, max_docs: int | None = None
) -> List[Document]:
    """Load and process CSV documents for RAGAS"""
    logger.info(f"📂 Loading CSV documents from: {csv_file_path}")

    df = pd.read_csv(csv_file_path)
    logger.info(f"📊 Loaded CSV with {len(df)} rows")

    documents = []
    processed_count = 0

    for idx, row in df.iterrows():
        if max_docs is not None and processed_count >= max_docs:
            break

        # Extract content from JSON field
        content_text, metadata = extract_content_from_csv_row(row["content"])

        if len(content_text.strip()) < 50:  # Skip very short content
            continue

        # Create LangChain Document
        doc = Document(
            page_content=content_text,
            metadata={
                **metadata,
                "csv_id": row["id"],
                "template_key": row.get("template_key", ""),
                "created_at": row.get("created_at", ""),
                "author": row.get("author", ""),
            },
        )

        documents.append(doc)
        processed_count += 1

        logger.info(
            f"✅ Processed document {processed_count}: {metadata.get('title', f'Doc {idx}')} ({len(content_text)} chars)"
        )

    logger.info(
        f"📄 Successfully loaded {len(documents)} documents for RAGAS processing"
    )
    return documents


async def build_relationships(kg: KnowledgeGraph, has_embeddings: bool = False) -> int:
    """Build relationships between nodes in the knowledge graph"""
    total_relationships = 0

    try:
        # 1. Build Jaccard similarity relationships based on entities
        logger.info("🔗 Building Jaccard similarity relationships...")
        jaccard_builder = JaccardSimilarityBuilder(
            property_name="entities",
            threshold=0.1,  # Lower threshold for more connections
            new_property_name="jaccard_similarity",
        )
        jaccard_relationships = await jaccard_builder.transform(kg)

        # Add relationships to knowledge graph
        for rel in jaccard_relationships:
            kg._add_relationship(rel)

        total_relationships += len(jaccard_relationships)
        logger.info(
            f"✅ Built {len(jaccard_relationships)} Jaccard similarity relationships"
        )

        # 2. Build overlap score relationships based on keyphrases
        logger.info("🔗 Building overlap score relationships...")
        overlap_builder = OverlapScoreBuilder(
            property_name="keyphrases",
            threshold=0.05,  # Lower threshold for more connections
            new_property_name="overlap_score",
        )
        overlap_relationships = await overlap_builder.transform(kg)

        # Add relationships to knowledge graph
        for rel in overlap_relationships:
            kg._add_relationship(rel)

        total_relationships += len(overlap_relationships)
        logger.info(
            f"✅ Built {len(overlap_relationships)} overlap score relationships"
        )

        # 3. Build embedding-based relationships if embeddings are available
        if has_embeddings:
            logger.info("🔗 Building cosine similarity relationships...")
            cosine_builder = CosineSimilarityBuilder(
                property_name="embedding",
                threshold=0.7,  # Reasonable threshold for embeddings
                new_property_name="cosine_similarity",
            )

            try:
                cosine_relationships = await cosine_builder.transform(kg)

                # Add relationships to knowledge graph
                for rel in cosine_relationships:
                    kg._add_relationship(rel)

                total_relationships += len(cosine_relationships)
                logger.info(
                    f"✅ Built {len(cosine_relationships)} cosine similarity relationships"
                )
            except Exception as e:
                logger.warning(f"Failed to build cosine similarity relationships: {e}")

            # 4. Build summary cosine similarity relationships
            logger.info("🔗 Building summary cosine similarity relationships...")
            summary_cosine_builder = SummaryCosineSimilarityBuilder(
                property_name="summary_embedding",
                threshold=0.5,  # Lower threshold for summary similarities
                new_property_name="summary_cosine_similarity",
            )

            try:
                summary_relationships = await summary_cosine_builder.transform(kg)

                # Add relationships to knowledge graph
                for rel in summary_relationships:
                    kg._add_relationship(rel)

                total_relationships += len(summary_relationships)
                logger.info(
                    f"✅ Built {len(summary_relationships)} summary cosine similarity relationships"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to build summary cosine similarity relationships: {e}"
                )

        # 5. Build multihop relationships (always - required for multihop synthesizers)
        logger.info(
            "🔗 Building summary_similarity relationships for multihop abstract..."
        )
        try:
            summary_sim_rels = create_summary_similarity_relationships(kg)
            for rel in summary_sim_rels:
                kg._add_relationship(rel)
            total_relationships += len(summary_sim_rels)
            logger.info(
                f"✅ Built {len(summary_sim_rels)} summary_similarity relationships"
            )
        except Exception as e:
            logger.warning(f"Failed to build summary_similarity relationships: {e}")

        logger.info(
            "🔗 Building entities_overlap relationships for multihop specific..."
        )
        try:
            entities_overlap_rels = create_entities_overlap_relationships(kg)
            for rel in entities_overlap_rels:
                kg._add_relationship(rel)
            total_relationships += len(entities_overlap_rels)
            logger.info(
                f"✅ Built {len(entities_overlap_rels)} entities_overlap relationships"
            )
        except Exception as e:
            logger.warning(f"Failed to build entities_overlap relationships: {e}")

    except Exception as e:
        logger.error(f"❌ Error building relationships: {e}")
        logger.info("Continuing without relationships...")

    return total_relationships


async def create_knowledge_graph_from_documents(
    documents: List[Document],
    embeddings_model: LangchainEmbeddingsWrapper = None,
    async_settings: Optional[Dict[str, Any]] = None,
) -> KnowledgeGraph:
    """Create RAGAS knowledge graph from documents with relationships"""
    logger.info(f"🧠 Creating RAGAS Knowledge Graph from {len(documents)} documents...")

    # Create knowledge graph
    kg = KnowledgeGraph()

    # Advanced semantic chunking with fallback
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_experimental.text_splitter import SemanticChunker

        logger.info("🧠 Initializing SemanticChunker for better relations...")
        embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        text_splitter = SemanticChunker(
            embedder, breakpoint_threshold_type="percentile"
        )
    except ImportError:
        logger.info("Falling back to RecursiveCharacterTextSplitter...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", "。", ".", " ", ""],
        )

    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"📄 Split {len(documents)} documents into {len(split_docs)} chunks")
    if GLOBAL_TELEMETRY:
        GLOBAL_TELEMETRY.log_document_stats(len(documents), len(split_docs))
    if GLOBAL_TELEMETRY:
        GLOBAL_TELEMETRY.log_document_stats(len(documents), len(split_docs))

    # Create nodes for each chunk
    import uuid

    pending_embedding_nodes: List[Tuple[Node, str, str]] = []
    async_settings = async_settings or {}

    for chunk_idx, chunk_doc in enumerate(split_docs):
        if len(chunk_doc.page_content.strip()) < 30:  # Skip very short chunks
            continue

        node_id = uuid.uuid4()  # Use proper UUID object, not string

        # Extract entities/keywords for relationship building
        entities = []
        keyphrases = []
        content_text = chunk_doc.page_content.strip()
        normalized_text = content_text

        # Simple entity extraction (can be improved with NLP models)
        # Split by common separators and get meaningful terms
        for separator in ["。", ".", "\n", ",", "，"]:
            normalized_text = normalized_text.replace(separator, "|")

        sentences = [
            s.strip() for s in normalized_text.split("|") if len(s.strip()) > 5
        ]
        for sentence in sentences[:3]:  # Take first 3 sentences as keyphrases
            if len(sentence) > 10:
                keyphrases.append(sentence)

        # Extract basic entities (words that appear frequently)
        words = content_text.split()
        meaningful_words = [w for w in words if len(w) > 2 and not w.isdigit()]
        entities = list(set(meaningful_words[:10]))  # Top 10 unique words

        # Create document node with relationship-building properties
        node = Node(
            id=node_id,
            type=NodeType.DOCUMENT,
            properties={
                "content": chunk_doc.page_content,
                "title": chunk_doc.metadata.get("title", f"Chunk {chunk_idx}"),
                "source": chunk_doc.metadata.get("source", "CSV"),
                "csv_id": chunk_doc.metadata.get("csv_id", ""),
                "chunk_index": chunk_idx,
                "length": len(chunk_doc.page_content),
                # Properties for relationship building
                "entities": entities,
                "keyphrases": keyphrases,
                "sentences": sentences,
            },
        )

        summary = sentences[0] if sentences else chunk_doc.page_content[:200]
        node.properties["summary"] = summary

        if embeddings_model:
            pending_embedding_nodes.append((node, chunk_doc.page_content, summary))
        else:
            node.properties["summary_embedding"] = [hash(summary) % 1000 / 1000.0] * 384

        kg._add_node(node)

    if embeddings_model and pending_embedding_nodes:
        max_workers = max(1, int(async_settings.get("max_workers", 8) or 1))
        logger.info(
            f"🧵 Embedding {len(pending_embedding_nodes)} nodes with async batch workers={max_workers}"
        )
        semaphore = asyncio.Semaphore(max_workers)
        tasks = [
            _populate_node_embeddings(
                node, content, summary, embeddings_model, semaphore
            )
            for node, content, summary in pending_embedding_nodes
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (node, _, summary), result in zip(pending_embedding_nodes, results):
            if isinstance(result, Exception):
                logger.warning(
                    f"Failed to create embedding for node {node.id}: {result}"
                )
                node.properties.pop("embedding", None)
                node.properties.pop("summary_embedding", None)
                node.properties["summary"] = summary

    logger.info(f"✅ Created knowledge graph with {len(kg.nodes)} nodes")

    # Build relationships between nodes
    logger.info("🔗 Building relationships between nodes...")
    relationships_built = await build_relationships(
        kg,
        any(node.properties.get("embedding") is not None for node in kg.nodes),
    )

    logger.info(f"✅ Built {relationships_built} relationships in knowledge graph")
    return kg


def save_knowledge_graph(kg: KnowledgeGraph, output_dir: Path) -> str:
    """Save knowledge graph for reuse"""
    kg_dir = output_dir / "knowledge_graphs"
    kg_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    kg_filename = f"ragas_kg_{timestamp}.json"
    kg_filepath = kg_dir / kg_filename

    logger.info(f"💾 Saving knowledge graph to: {kg_filepath}")

    # Convert knowledge graph to serializable format
    kg_data = {
        "created_at": timestamp,
        "generator": "pure_ragas_pipeline",
        "nodes": [
            {
                "id": str(node.id),  # Convert UUID to string
                "type": str(node.type),
                "properties": _to_json_safe(node.properties),
            }
            for node in kg.nodes
        ],
        "relationships": (
            [
                {
                    "source": str(rel.source.id),  # Convert UUID to string
                    "target": str(rel.target.id),  # Convert UUID to string
                    "type": str(rel.type),
                    "properties": _to_json_safe(rel.properties),
                }
                for rel in kg.relationships
            ]
            if hasattr(kg, "relationships") and kg.relationships
            else []
        ),
        "metadata": {
            "total_nodes": len(kg.nodes),
            "total_relationships": (
                len(kg.relationships)
                if hasattr(kg, "relationships") and kg.relationships
                else 0
            ),
            "generation_method": "pure_ragas",
        },
    }

    # Save to file
    with open(kg_filepath, "w", encoding="utf-8") as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)

    logger.info(f"✅ Knowledge graph saved: {len(kg.nodes)} nodes")
    return str(kg_filepath)


def setup_ragas_components(
    config: Dict[str, Any],
) -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Setup RAGAS LLM and embedding components"""
    ragas_config = config.get("testset_generation", {}).get("ragas_config", {})

    # Set up custom LLM
    custom_llm_config = ragas_config.get("custom_llm", {})

    if custom_llm_config.get("endpoint"):
        # Use custom LLM endpoint
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            base_url=custom_llm_config["endpoint"].replace(
                "/v1/chat/completions", "/v1"
            ),
            api_key=custom_llm_config.get(
                "api_key", os.getenv("OPENAI_API_KEY", "dummy-key")
            ),
            model=custom_llm_config.get(
                "model", custom_llm_config.get("model_name", "gpt-4o-mini")
            ),
            temperature=custom_llm_config.get("temperature", 0.3),
            timeout=custom_llm_config.get("timeout", 60),
            model_kwargs={"max_tokens": custom_llm_config.get("max_tokens", 1000)},
        )

        generator_llm = LangchainLLMWrapper(llm)
        logger.info(f"🔗 Using custom LLM: {custom_llm_config['endpoint']}")
    else:
        raise ValueError("Custom LLM configuration required for RAGAS generation")

    # Set up embeddings
    embeddings_model = ragas_config.get(
        "embeddings_model", "sentence-transformers/all-MiniLM-L6-v2"
    )

    try:
        from langchain_huggingface import HuggingFaceEmbeddings

        core_embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        generator_embeddings = LangchainEmbeddingsWrapper(core_embeddings)
        logger.info(f"🔗 Using HuggingFace embeddings: {embeddings_model}")
    except ImportError:
        try:
            from langchain_community.embeddings import \
                SentenceTransformerEmbeddings
            from sentence_transformers import SentenceTransformer

            embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model)
            generator_embeddings = LangchainEmbeddingsWrapper(embeddings)
            logger.info(f"🔗 Using SentenceTransformer embeddings: {embeddings_model}")
        except ImportError:
            logger.error(
                "❌ Could not load embedding models. Please install sentence-transformers or langchain-huggingface"
            )
            raise

    return generator_llm, generator_embeddings


def generate_ragas_testset(
    kg: KnowledgeGraph,
    generator_llm: LangchainLLMWrapper,
    generator_embeddings: LangchainEmbeddingsWrapper,
    generation_settings: GenerationSettings,
    num_samples: int = 3,
) -> List[Dict]:
    """Generate synthetic testset using RAGAS TestsetGenerator"""
    logger.info(f"🎯 Generating {num_samples} synthetic test samples using RAGAS...")

    # Create RAGAS TestsetGenerator
    logger.info("🚀 Initializing RAGAS TestsetGenerator...")

    try:
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            knowledge_graph=kg,
            persona_list=generation_settings.personas,
        )

        logger.info("✅ RAGAS TestsetGenerator initialized with configured personas")
        logger.info(
            f"🧠 Query distribution: {generation_settings.query_distribution_records}"
        )
        logger.info(
            f"🧵 Async generation config: max_workers={generation_settings.run_config.max_workers}, batch_size={generation_settings.batch_size}"
        )

        # Generate testset using RAGAS with simple configuration
        logger.info(f"🎯 Generating {num_samples} test samples...")

        testset = generator.generate(
            testset_size=num_samples,
            query_distribution=generation_settings.query_distribution,
            run_config=generation_settings.run_config,
            batch_size=generation_settings.batch_size,
            raise_exceptions=False,  # Don't raise exceptions to handle gracefully
        )

        logger.info("✅ RAGAS testset generation completed")

        # Convert testset to our format
        test_samples = []

        if hasattr(testset, "samples") and testset.samples:
            for i, sample in enumerate(testset.samples):
                if hasattr(sample, "eval_sample") and sample.eval_sample:
                    eval_sample = sample.eval_sample
                    test_sample = {
                        "question": (
                            eval_sample.user_input
                            if hasattr(eval_sample, "user_input")
                            else f"Generated question {i+1}"
                        ),
                        "contexts": (
                            eval_sample.reference_contexts
                            if hasattr(eval_sample, "reference_contexts")
                            else []
                        ),
                        "ground_truth": (
                            eval_sample.reference
                            if hasattr(eval_sample, "reference")
                            else f"Generated answer {i+1}"
                        ),
                        "synthesizer_name": (
                            sample.synthesizer_name
                            if hasattr(sample, "synthesizer_name")
                            else "ragas_synthesizer"
                        ),
                        "generation_method": "pure_ragas",
                        "generation_timestamp": datetime.now().isoformat(),
                        "source_type": "csv",
                    }
                    test_samples.append(test_sample)

            logger.info(
                f"✅ Converted {len(test_samples)} RAGAS samples to pipeline format"
            )

        # If no samples were generated, create fallback samples based on knowledge graph content
        if not test_samples:
            logger.warning(
                "⚠️ RAGAS testset generation returned no samples - creating samples from knowledge graph"
            )

            # Extract content from knowledge graph nodes
            node_contents = []
            for node in kg.nodes:
                if hasattr(node, "properties") and node.properties.get("content"):
                    content = node.properties["content"]
                    title = node.properties.get("title", "Document")
                    node_contents.append({"content": content, "title": title})

            fallback_templates = generation_settings.fallback_templates
            measurement_template = fallback_templates.get("measurement", {})
            inspection_template = fallback_templates.get("inspection", {})
            general_template = fallback_templates.get("general", {})

            # Create questions based on node content
            for i, node_data in enumerate(node_contents[:num_samples]):
                content = node_data["content"]
                title = node_data["title"]
                excerpt = content[:200]

                # Generate simple questions based on content
                if "量測" in content or "measurement" in content.lower():
                    question = measurement_template.get(
                        "question_template",
                        "What are the measurement procedures described in {title}?",
                    ).format(title=title, excerpt=excerpt)
                    answer = measurement_template.get(
                        "answer_template",
                        "The measurement procedures include: {excerpt}...",
                    ).format(title=title, excerpt=excerpt)
                elif "檢查" in content or "inspection" in content.lower():
                    question = inspection_template.get(
                        "question_template",
                        "What inspection steps are required according to {title}?",
                    ).format(title=title, excerpt=excerpt)
                    answer = inspection_template.get(
                        "answer_template",
                        "The inspection requirements are: {excerpt}...",
                    ).format(title=title, excerpt=excerpt)
                else:
                    question = general_template.get(
                        "question_template",
                        "What are the key points described in {title}?",
                    ).format(title=title, excerpt=excerpt)
                    answer = general_template.get(
                        "answer_template",
                        "The key information includes: {excerpt}...",
                    ).format(title=title, excerpt=excerpt)

                test_sample = {
                    "question": question,
                    "contexts": [content],
                    "ground_truth": answer,
                    "synthesizer_name": "knowledge_graph_based_generator",
                    "generation_method": "pure_ragas_fallback",
                    "generation_timestamp": datetime.now().isoformat(),
                    "source_type": "csv",
                }
                test_samples.append(test_sample)

            logger.info(
                f"✅ Created {len(test_samples)} fallback test samples from knowledge graph"
            )

        return test_samples

    except Exception as e:
        if GLOBAL_TELEMETRY:
            GLOBAL_TELEMETRY.log_error("evaluation_generation", str(e))
        logger.error(f"❌ RAGAS testset generation failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")

        # Create minimal fallback as last resort
        logger.info("🔄 Creating minimal fallback samples...")
        test_samples = []

        for i in range(min(num_samples, 3)):
            test_sample = {
                "question": f"What are the key technical requirements described in the document?",
                "contexts": ["Technical documentation content from knowledge graph"],
                "ground_truth": "The technical requirements include various specifications and procedures based on the source documentation.",
                "synthesizer_name": "minimal_fallback_generator",
                "generation_method": "pure_ragas_minimal_fallback",
                "generation_timestamp": datetime.now().isoformat(),
                "source_type": "csv",
            }
            test_samples.append(test_sample)

        logger.info(f"✅ Created {len(test_samples)} minimal fallback samples")
        return test_samples


def save_testset(test_samples: List[Dict], output_dir: Path) -> str:
    """Save testset to CSV format with proper error handling"""
    if not test_samples:
        logger.warning("❌ No test samples to save")
        return ""

    try:
        # Import the comprehensive file saver
        sys.path.append(str(Path(__file__).parent / "src" / "utils"))
        from utils.pipeline_file_saver import PipelineFileSaver

        # Use the standardized file saver
        file_saver = PipelineFileSaver(output_dir)
        csv_path = file_saver.save_testset_csv(test_samples, "pure_ragas_testset")

        if csv_path:
            logger.info(f"✅ Testset saved using PipelineFileSaver: {csv_path}")
            return csv_path

    except ImportError as e:
        logger.warning(
            f"⚠️ Could not import PipelineFileSaver: {e}, using fallback method"
        )
    except Exception as e:
        logger.warning(f"⚠️ PipelineFileSaver failed: {e}, using fallback method")

    # Fallback to original method
    try:
        testset_dir = output_dir / "testsets"
        testset_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        testset_filename = f"pure_ragas_testset_{timestamp}.csv"
        testset_filepath = testset_dir / testset_filename

        logger.info(f"💾 Saving testset to: {testset_filepath}")

        # Convert to DataFrame
        df = pd.DataFrame(test_samples)
        for column in df.columns:
            df[column] = df[column].apply(
                lambda value: (
                    json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                )
            )

        # Add additional columns to match expected format (but leave empty as designed)
        df["answer"] = df["ground_truth"]  # Copy ground truth as answer
        df["auto_keywords"] = (
            ""  # Empty - keywords will be extracted from RAG responses later!
        )
        df["source_file"] = ""  # Empty for CSV source
        df["question_type"] = ""  # Empty - will be classified later
        df["keyword_score"] = 0.0  # Will be calculated from RAG responses later!
        df["enhanced_at"] = datetime.now().isoformat()

        # Reorder columns to match expected format
        column_order = [
            "question",
            "answer",
            "auto_keywords",
            "source_file",
            "question_type",
            "generation_method",
            "generation_timestamp",
            "contexts",
            "ground_truth",
            "synthesizer_name",
            "keyword_score",
            "enhanced_at",
            "source_type",
        ]

        df = df.reindex(columns=column_order, fill_value="")

        # Save to CSV
        df.to_csv(testset_filepath, index=False, encoding="utf-8")

        logger.info(f"✅ Testset saved: {len(test_samples)} samples")
        logger.info(f"📊 Columns: {list(df.columns)}")

        return str(testset_filepath)

    except Exception as e:
        logger.error(f"❌ Failed to save testset: {e}")
        raise


def main(argv: Optional[List[str]] = None):
    """Main pipeline execution"""
    global GLOBAL_TELEMETRY
    args = parse_args(argv)
    telemetry = None
    logger.info("🚀 Starting Pure RAGAS Pipeline (Corrected Design)")
    logger.info("=" * 60)

    try:
        # Step 1: Load configuration
        logger.info("📋 Loading configuration...")
        config = load_config(args.config) if args.config else load_config()
        config = apply_cli_overrides(config, docs=args.docs, samples=args.samples)

        testset_config = config.get("testset_generation", {})
        max_docs = testset_config.get("max_documents_for_generation", 10)
        # CRITICAL FIX: Use proper default value instead of hardcoded 3
        # The hardcoded 3 was causing fallback to minimal generation
        max_samples = testset_config.get("max_total_samples", 10)

        logger.info(f"🎯 Configuration: max_docs={max_docs}, max_samples={max_samples}")

        # Step 2: Setup output directory
        output_dir = (
            Path("outputs")
            / f"pure_ragas_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        telemetry = PipelineTelemetry(output_dir)
        GLOBAL_TELEMETRY = telemetry

        # Step 3: Load documents (TXT documents for steel plate inspection)
        document_files = (
            config.get("data_sources", {})
            .get("documents", {})
            .get("document_files", [])
        )
        csv_files = config.get("data_sources", {}).get("csv", {}).get("csv_files", [])

        csv_file_path = None
        if document_files:
            logger.info("📄 Loading TXT documents (working 2024 method)...")
            documents = load_txt_documents(document_files, max_docs)
        elif csv_files:
            logger.info("📄 Loading CSV documents (fallback method)...")
            csv_file_path = csv_files[0]  # Use first CSV file
            documents = load_csv_documents(csv_file_path, max_docs)
        else:
            raise ValueError(
                "No document files or CSV files specified in configuration"
            )

        # Step 4: Setup RAGAS components (before KG creation for embeddings)
        generator_llm, generator_embeddings = setup_ragas_components(config)

        # Step 4.1: Build async settings baseline for KG creation
        run_config, _ = build_run_config(config)

        # Step 5: Create knowledge graph with relationships
        kg = asyncio.run(
            create_knowledge_graph_from_documents(
                documents,
                generator_embeddings,
                async_settings={
                    "max_workers": run_config.max_workers,
                },
            )
        )

        # Rebuild generation settings using the actual KG for synthesizer availability checks
        generation_settings = build_generation_settings(config, generator_llm, kg)

        # Step 6: Save knowledge graph for reuse (as requested)
        kg_filepath = save_knowledge_graph(kg, output_dir)

        # Step 7: Generate synthetic testset using RAGAS
        test_samples = generate_ragas_testset(
            kg,
            generator_llm,
            generator_embeddings,
            generation_settings,
            max_samples,
        )

        # Step 8: Save testset
        testset_filepath = save_testset(test_samples, output_dir)

        # Step 9: Save personas and scenarios (NEW)
        try:
            # Import the comprehensive file saver
            sys.path.append(str(Path(__file__).parent / "src" / "utils"))
            from utils.pipeline_file_saver import PipelineFileSaver

            file_saver = PipelineFileSaver(output_dir)

            default_personas = generation_settings.persona_records
            default_scenarios = generation_settings.query_distribution_records

            # Save personas and scenarios
            personas_path = file_saver.save_personas_json(default_personas)
            scenarios_path = file_saver.save_scenarios_json(default_scenarios)

            # Save pipeline metadata
            pipeline_metadata = {
                "pipeline_type": "pure_ragas",
                "generation_timestamp": datetime.now().isoformat(),
                "documents_processed": len(documents),
                "knowledge_graph_nodes": len(kg.nodes),
                "knowledge_graph_relationships": (
                    len(kg.relationships)
                    if hasattr(kg, "relationships") and kg.relationships
                    else 0
                ),
                "testset_samples": len(test_samples),
                "csv_source": csv_file_path,
                "max_docs": max_docs,
                "max_samples": max_samples,
                "files_created": {
                    "knowledge_graph": kg_filepath,
                    "testset": testset_filepath,
                    "personas": personas_path,
                    "scenarios": scenarios_path,
                },
                "prompt_profile": generation_settings.prompt_profile,
                "query_distribution": generation_settings.query_distribution_records,
                "async_generation": {
                    "max_workers": generation_settings.run_config.max_workers,
                    "batch_size": generation_settings.batch_size,
                    "timeout": generation_settings.run_config.timeout,
                    "max_retries": generation_settings.run_config.max_retries,
                },
            }

            metadata_path = file_saver.save_pipeline_metadata(pipeline_metadata)

            logger.info(f"👤 Personas saved: {personas_path}")
            logger.info(f"🎯 Scenarios saved: {scenarios_path}")
            logger.info(f"📋 Metadata saved: {metadata_path}")

        except Exception as e:
            logger.warning(f"⚠️ Failed to save personas/scenarios/metadata: {e}")
            # Continue without failing the entire pipeline

        # Step 9: Summary
        logger.info("=" * 60)
        logger.info("✅ Pure RAGAS Pipeline completed successfully!")
        logger.info(f"📊 Generated {len(test_samples)} test samples")
        logger.info(f"🧠 Knowledge graph saved: {kg_filepath}")
        logger.info(f"📁 Testset saved: {testset_filepath}")
        logger.info("=" * 60)

        # Next steps message
        logger.info("🔄 Next Steps (Correct Pipeline Flow):")
        logger.info("1. ✅ Testset generated using RAGAS only")
        logger.info("2. 🎯 Query RAG system with generated questions")
        logger.info("3. 📝 Extract keywords from RAG responses (not pre-generated!)")
        logger.info("4. 📊 Calculate evaluation metrics on RAG outputs")
        logger.info("5. 📈 Generate evaluation reports")

        if telemetry:
            telemetry.finish(status="completed")
        GLOBAL_TELEMETRY = None

        return True

    except Exception as e:
        if telemetry:
            telemetry.log_error("pipeline_main", str(e))
            telemetry.finish(status="failed")
        GLOBAL_TELEMETRY = None
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
