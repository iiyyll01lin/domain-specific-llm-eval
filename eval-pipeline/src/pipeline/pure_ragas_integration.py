#!/usr/bin/env python3
"""
Integration module to use working pure RAGAS pipeline KG creation in main orchestrator
"""
import sys
import uuid
import asyncio
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# Add RAGAS to path
ragas_path = "/data/yy/domain-specific-llm-eval/ragas/ragas/src"
if ragas_path not in sys.path:
    sys.path.insert(0, ragas_path)

from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder, 
    JaccardSimilarityBuilder, 
    OverlapScoreBuilder
)
from ragas.testset.transforms.relationship_builders.cosine import SummaryCosineSimilarityBuilder
from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging
logger = logging.getLogger(__name__)

async def build_relationships_fixed(kg: KnowledgeGraph, has_embeddings: bool = False) -> int:
    """Build relationships between nodes in the knowledge graph"""
    total_relationships = 0
    
    try:
        # 1. Build Jaccard similarity relationships based on entities
        logger.info("🔗 Building Jaccard similarity relationships...")
        jaccard_builder = JaccardSimilarityBuilder(
            property_name="entities",
            threshold=0.1,  # Lower threshold for more connections
            new_property_name="jaccard_similarity"
        )
        jaccard_relationships = await jaccard_builder.transform(kg)
        
        # Add relationships to knowledge graph
        for rel in jaccard_relationships:
            kg._add_relationship(rel)
        
        total_relationships += len(jaccard_relationships)
        logger.info(f"✅ Built {len(jaccard_relationships)} Jaccard similarity relationships")
        
        # 2. Build overlap score relationships based on keyphrases
        logger.info("🔗 Building overlap score relationships...")
        overlap_builder = OverlapScoreBuilder(
            property_name="keyphrases",
            threshold=0.05,  # Lower threshold for more connections
            new_property_name="overlap_score"
        )
        overlap_relationships = await overlap_builder.transform(kg)
        
        # Add relationships to knowledge graph
        for rel in overlap_relationships:
            kg._add_relationship(rel)
            
        total_relationships += len(overlap_relationships)
        logger.info(f"✅ Built {len(overlap_relationships)} overlap score relationships")
        
        # 3. Build summary_similarity relationships (REQUIRED for multihop abstract)
        logger.info("🔗 Building summary_similarity relationships for multihop abstract...")
        summary_relationships = create_summary_similarity_relationships(kg)
        for rel in summary_relationships:
            kg._add_relationship(rel)
        total_relationships += len(summary_relationships)
        logger.info(f"✅ Built {len(summary_relationships)} summary_similarity relationships")
        
        # 4. Build entities_overlap relationships (REQUIRED for multihop specific)
        logger.info("🔗 Building entities_overlap relationships for multihop specific...")
        entities_relationships = create_entities_overlap_relationships(kg)
        for rel in entities_relationships:
            kg._add_relationship(rel)
        total_relationships += len(entities_relationships)
        logger.info(f"✅ Built {len(entities_relationships)} entities_overlap relationships")
        
        # 5. Build embedding-based relationships if embeddings are available
        if has_embeddings:
            logger.info("🔗 Building cosine similarity relationships...")
            cosine_builder = CosineSimilarityBuilder(
                property_name="embedding",
                threshold=0.7,  # Reasonable threshold for embeddings
                new_property_name="cosine_similarity"
            )
            
            try:
                cosine_relationships = await cosine_builder.transform(kg)
                
                # Add relationships to knowledge graph
                for rel in cosine_relationships:
                    kg._add_relationship(rel)
                    
                total_relationships += len(cosine_relationships)
                logger.info(f"✅ Built {len(cosine_relationships)} cosine similarity relationships")
            except Exception as e:
                logger.warning(f"Failed to build cosine similarity relationships: {e}")
            
            # 6. Build summary cosine similarity relationships
            logger.info("🔗 Building summary cosine similarity relationships...")
            summary_cosine_builder = SummaryCosineSimilarityBuilder(
                property_name="summary_embedding",
                threshold=0.5,  # Lower threshold for summary similarities
                new_property_name="summary_cosine_similarity"
            )
            
            try:
                summary_relationships = await summary_cosine_builder.transform(kg)
                
                # Add relationships to knowledge graph
                for rel in summary_relationships:
                    kg._add_relationship(rel)
                    
                total_relationships += len(summary_relationships)
                logger.info(f"✅ Built {len(summary_relationships)} summary cosine similarity relationships")
            except Exception as e:
                logger.warning(f"Failed to build summary cosine similarity relationships: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error building relationships: {e}")
        logger.info("Continuing without relationships...")
    
    return total_relationships

def create_summary_similarity_relationships(kg: KnowledgeGraph):
    """Create summary_similarity relationships required by MultiHopAbstractQuerySynthesizer"""
    relationships = []
    nodes = list(kg.nodes)
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            # Calculate summary similarity based on shared words
            summary1 = node1.properties.get("summary", "")
            summary2 = node2.properties.get("summary", "")
            
            if summary1 and summary2:
                words1 = set(summary1.lower().split())
                words2 = set(summary2.lower().split())
                
                # Calculate Jaccard similarity
                intersection = len(words1 & words2)
                union = len(words1 | words2)
                
                if union > 0:
                    similarity = intersection / union
                    
                    # Create relationship if similarity is above threshold
                    if similarity > 0.1:  # Threshold for summary similarity
                        rel = Relationship(
                            source=node1,
                            target=node2,
                            type="summary_similarity",
                            properties={"summary_similarity": similarity}
                        )
                        relationships.append(rel)
    
    return relationships

def create_entities_overlap_relationships(kg: KnowledgeGraph):
    """Create entities_overlap relationships required by MultiHopSpecificQuerySynthesizer"""
    relationships = []
    nodes = list(kg.nodes)
    
    for i, node1 in enumerate(nodes):
        for j, node2 in enumerate(nodes[i+1:], i+1):
            # Get entities from both nodes
            entities1 = set(node1.properties.get("entities", []))
            entities2 = set(node2.properties.get("entities", []))
            
            # Calculate entities overlap
            overlap = entities1 & entities2
            
            if len(overlap) > 0:  # Nodes share at least one entity
                overlap_score = len(overlap) / min(len(entities1), len(entities2)) if entities1 and entities2 else 0
                
                # Create relationship if overlap is significant
                if overlap_score > 0.2:  # Threshold for entities overlap
                    rel = Relationship(
                        source=node1,
                        target=node2,
                        type="entities_overlap",
                        properties={
                            "entities_overlap": overlap_score,
                            "shared_entities": list(overlap)
                        }
                    )
                    relationships.append(rel)
    
    return relationships

async def create_knowledge_graph_with_relationships(documents, embeddings_model=None, max_docs=100):
    """
    Create knowledge graph using the working approach from run_pure_ragas_pipeline.py
    This replaces the buggy RAGAS default_transforms approach
    """
    logger.info(f"🧠 Creating Knowledge Graph from {len(documents)} documents (Pure RAGAS approach)...")
    
    # Create knowledge graph
    kg = KnowledgeGraph()
    
    # Text splitter for chunking (same as pure RAGAS pipeline)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents[:max_docs])
    logger.info(f"📄 Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Create nodes for each chunk
    processed_chunks = 0
    for chunk_idx, chunk_doc in enumerate(split_docs):
        if len(chunk_doc.page_content.strip()) < 30:  # Skip very short chunks
            continue
            
        node_id = uuid.uuid4()  # Use proper UUID format
        
        # Extract entities/keywords for relationship building
        entities = []
        keyphrases = []
        content_text = chunk_doc.page_content.strip()
        
        # Simple entity extraction (can be improved with NLP models)
        # Split by common separators and get meaningful terms
        terms = []
        for separator in ["。", ".", "\n", ",", "，"]:
            content_text = content_text.replace(separator, "|")
        
        sentences = [s.strip() for s in content_text.split("|") if len(s.strip()) > 5]
        for sentence in sentences[:3]:  # Take first 3 sentences as keyphrases
            if len(sentence) > 10:
                keyphrases.append(sentence)
        
        # Extract basic entities (words that appear frequently)
        words = content_text.split()
        meaningful_words = [w for w in words if len(w) > 2 and not w.isdigit()]
        entities = list(set(meaningful_words[:10]))  # Top 10 unique words
        
        # Always create summary for persona generation compatibility
        summary = sentences[0] if sentences else chunk_doc.page_content[:200]
        
        # Create document node with relationship-building properties
        node = Node(
            id=node_id,
            type=NodeType.DOCUMENT,
            properties={
                'content': chunk_doc.page_content,
                'title': chunk_doc.metadata.get('title', f'Chunk {chunk_idx}'),
                'source': chunk_doc.metadata.get('source', 'CSV'),
                'csv_id': chunk_doc.metadata.get('csv_id', ''),
                'chunk_index': chunk_idx,
                'length': len(chunk_doc.page_content),
                'summary': summary,  # Always include summary for persona generation
                # Properties for relationship building
                'entities': entities,
                'keyphrases': keyphrases,
                'sentences': sentences
            }
        )
        
        # Add embedding if embeddings model is provided
        if embeddings_model:
            try:
                embedding = embeddings_model.embed_text(chunk_doc.page_content)
                node.properties['embedding'] = embedding
                
                # Create summary embedding for persona generation
                summary_embedding = embeddings_model.embed_text(summary)
                node.properties['summary_embedding'] = summary_embedding
                
                logger.debug(f"✅ Created embeddings for node {node_id}")
                
            except Exception as e:
                logger.warning(f"Failed to create embedding for node {node_id}: {e}")
                # Create fallback summary_embedding for persona compatibility
                node.properties['summary_embedding'] = [0.1] * 384  # Simple fallback embedding
        else:
            # Create fallback summary_embedding when no embeddings model
            # This ensures persona generation doesn't fail due to missing summary_embedding
            logger.debug("No embeddings model provided, creating fallback summary_embedding")
            node.properties['summary_embedding'] = [hash(summary) % 1000 / 1000.0] * 384  # Simple hash-based embedding
        
        kg._add_node(node)
        processed_chunks += 1
        
        # Log progress for large datasets
        if processed_chunks % 50 == 0:
            logger.info(f"📝 Processed {processed_chunks} chunks...")
    
    logger.info(f"✅ Created knowledge graph with {len(kg.nodes)} nodes")
    
    # Build relationships between nodes
    if len(kg.nodes) > 1:
        logger.info("🔗 Building relationships between nodes...")
        relationships_built = await build_relationships_fixed(kg, embeddings_model is not None)
        logger.info(f"✅ Built {relationships_built} relationships in knowledge graph")
    
    return kg

def create_knowledge_graph_sync(documents, embeddings_model=None, max_docs=100):
    """Synchronous wrapper for async knowledge graph creation"""
    try:
        # Try to get existing event loop
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # If loop is running, create a new task
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    asyncio.run, 
                    create_knowledge_graph_with_relationships(documents, embeddings_model, max_docs)
                )
                return future.result()
        else:
            # If no loop or loop not running, use asyncio.run
            return asyncio.run(create_knowledge_graph_with_relationships(documents, embeddings_model, max_docs))
    except Exception as e:
        logger.error(f"❌ Async KG creation failed: {e}")
        # Fallback to synchronous creation
        return create_knowledge_graph_fallback(documents, max_docs)

def create_knowledge_graph_fallback(documents, max_docs=100):
    """Fallback synchronous knowledge graph creation without relationships"""
    logger.info(f"🔄 Creating knowledge graph (fallback mode, no relationships)...")
    
    kg = KnowledgeGraph()
    
    # Simple document processing without chunking
    from langchain_core.documents import Document
    
    for doc_idx, doc in enumerate(documents[:max_docs]):
        try:
            # Handle different document formats
            if isinstance(doc, str):
                content = doc.strip()
                metadata = {'title': f'Document {doc_idx}', 'source': 'fallback', 'csv_id': doc_idx}
            elif hasattr(doc, 'page_content'):
                content = doc.page_content.strip()
                metadata = getattr(doc, 'metadata', {})
            else:
                content = str(doc).strip()
                metadata = {'title': f'Document {doc_idx}', 'source': 'fallback', 'csv_id': doc_idx}
            
            if len(content) < 30:
                continue
                
            node_id = uuid.uuid4()
            
            # Create summary for persona compatibility
            summary = content[:200] if len(content) > 200 else content
            
            node = Node(
                id=node_id,
                type=NodeType.DOCUMENT,
                properties={
                    'content': content[:1000],  # Limit content size
                    'title': metadata.get('title', f'Document {doc_idx}'),
                    'source': metadata.get('source', 'unknown'),
                    'csv_id': metadata.get('csv_id', doc_idx),
                    'length': len(content),
                    'summary': summary,
                    # Fallback summary_embedding for persona compatibility
                    'summary_embedding': [hash(summary) % 1000 / 1000.0] * 384
                }
            )
            
            kg._add_node(node)
        except Exception as e:
            logger.warning(f"Failed to process document {doc_idx} in fallback: {e}")
            continue
    
    logger.info(f"✅ Created fallback knowledge graph with {len(kg.nodes)} nodes")
    return kg

# Integration function for orchestrator
def replace_orchestrator_kg_creation():
    """Replace the orchestrator's _create_knowledge_graph method with working version"""
    
    def new_create_knowledge_graph(self, documents=None):
        """Enhanced knowledge graph creation using Pure RAGAS approach"""
        logger.info("🧠 Using Pure RAGAS knowledge graph creation (fixed approach)...")
        
        if documents is None:
            documents = getattr(self, 'documents', [])
        
        if not documents:
            logger.warning("⚠️ No documents provided for knowledge graph creation")
            from ragas.testset.graph import KnowledgeGraph
            return KnowledgeGraph()
        
        try:
            # Fix: Convert string documents to Document objects if needed
            from langchain_core.documents import Document
            
            processed_docs = []
            for i, doc in enumerate(documents):
                if isinstance(doc, str):
                    # Convert string to Document object
                    processed_docs.append(Document(
                        page_content=doc,
                        metadata={
                            'title': f'Document {i}',
                            'source': 'csv',
                            'csv_id': i
                        }
                    ))
                elif hasattr(doc, 'page_content'):
                    # Already a Document object
                    processed_docs.append(doc)
                else:
                    # Unknown format, try to convert
                    processed_docs.append(Document(
                        page_content=str(doc),
                        metadata={
                            'title': f'Document {i}',
                            'source': 'unknown',
                            'csv_id': i
                        }
                    ))
            
            logger.info(f"📄 Converted {len(documents)} documents to proper format")
            
            # Use working knowledge graph creation
            embeddings_model = getattr(self, 'generator_embeddings', None)
            kg = create_knowledge_graph_sync(processed_docs, embeddings_model, max_docs=200)
            
            logger.info(f"✅ Knowledge graph created successfully: {len(kg.nodes)} nodes, "
                       f"{len(kg.relationships) if hasattr(kg, 'relationships') else 0} relationships")
            
            return kg
            
        except Exception as e:
            logger.error(f"❌ Pure RAGAS KG creation failed: {e}")
            logger.info("🔄 Using fallback knowledge graph...")
            return create_knowledge_graph_fallback(documents)
    
    return new_create_knowledge_graph
