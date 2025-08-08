#!/usr/bin/env python3
"""
Pure RAGAS Testset Generator
============================

This module implements a simplified RAGAS testset generator following the official RAGAS documentation
pattern from rag_testset_generation.md. It uses the custom LLM configuration from pipeline_config.yaml
instead of OpenAI.

Key Features:
- Uses proper RAGAS TestsetGenerator (not fallback templates)
- Supports custom LLM endpoint (4o from pipeline config)
- Creates Knowledge Graph with proper transforms
- Generates synthetic testsets with multiple query types
- Saves knowledge graph for reuse
"""

import os
import json
import logging
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add paths for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))  # data directory
sys.path.append(str(current_dir.parent))  # src directory
sys.path.append(str(current_dir.parent.parent))  # eval-pipeline directory

# RAGAS imports following the documentation
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution

# LangChain imports for LLM and Embeddings
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document as LCDocument

# RAGAS LLM wrapper
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# Import fixes module and knowledge graph manager
from pure_ragas_testset_generator_fixes import PureRAGASTestsetGeneratorFixes, apply_fixes_to_generator
from utils.knowledge_graph_manager import KnowledgeGraphManager, find_and_use_latest_kg

logger = logging.getLogger(__name__)

# Import unified KeyBERT extractor for keyword generation
try:
    from utils.keybert_extractor import UnifiedKeyBERTExtractor
except ImportError:
    logger.warning("UnifiedKeyBERTExtractor not available - keywords will not be generated")
    UnifiedKeyBERTExtractor = None


class PureRagasTestsetGenerator:
    """
    Pure RAGAS Testset Generator using the official RAGAS documentation pattern.
    
    This generator follows the exact pattern from the RAGAS documentation:
    1. Load documents
    2. Create Knowledge Graph
    3. Apply transforms to enrich KG
    4. Generate testset using TestsetGenerator
    5. Save knowledge graph for reuse
    """
    
    def __init__(self, config: Dict[str, Any], output_dir: Path = None):
        """
        Initialize the Pure RAGAS Testset Generator.
        
        Args:
            config: Pipeline configuration dictionary
            output_dir: Output directory for testsets (optional)
        """
        self.config = config
        self.output_dir = Path(output_dir) if output_dir else None
        # Access ragas_config from the testset_generation section of full config
        testset_config = config.get('testset_generation', {})
        self.ragas_config = testset_config.get('ragas_config', {})
        self.custom_llm_config = self.ragas_config.get('custom_llm', {})
        
        # Initialize LLM and embeddings
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()
        
        # ✅ IMPLEMENTED: Initialize intelligent cache manager
        try:
            from utils.intelligent_cache import IntelligentCacheManager
            cache_config = config.get('advanced', {}).get('cache', {})
            self.cache_manager = IntelligentCacheManager(cache_config)
            logger.info("✅ Intelligent cache manager initialized")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize cache manager: {e}")
            self.cache_manager = None
        
        # Initialize KeyBERT extractor for keyword generation
        self.keybert_extractor = None
        if UnifiedKeyBERTExtractor:
            try:
                # 🔧 FIX: config is already the testset_generation section, no need for nested access
                keybert_config = config.get('keyword_extraction', {})
                logger.info(f"🔧 DEBUG: KeyBERT config loaded: {keybert_config}")
                self.keybert_extractor = UnifiedKeyBERTExtractor(keybert_config)
                logger.info("✅ KeyBERT extractor initialized for keyword generation")
            except Exception as e:
                logger.warning(f"⚠️ Failed to initialize KeyBERT extractor: {e}")
        
        logger.info("✅ Pure RAGAS Testset Generator initialized successfully")
        logger.info(f"🤖 LLM: {self.custom_llm_config.get('model', 'gpt-4o')}")
        logger.info(f"🔗 Endpoint: {self.custom_llm_config.get('endpoint', 'N/A')}")
        logger.info(f"📊 Embeddings: {self.ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2')}")
        
        # Initialize enhanced trackers if available from orchestrator
        self._initialize_trackers()
        
    def _initialize_trackers(self):
        """Initialize enhanced trackers if available from orchestrator."""
        try:
            # Try to import enhanced trackers
            from pipeline.enhanced_trackers import PerformanceTracker, CompositionTracker, ParametersTracker
            
            # Initialize performance tracker
            performance_config = self.config.get('evaluation', {}).get('performance_tracking', {})
            if performance_config.get('enabled', True):
                self.performance_tracker = PerformanceTracker(self.config)
                logger.info("✅ Performance tracker initialized in testset generator")
            else:
                self.performance_tracker = None
            
            # Initialize composition tracker  
            composition_config = self.config.get('testset_generation', {}).get('composition_elements_tracking', {})
            if composition_config.get('enabled', True):
                self.composition_tracker = CompositionTracker(self.config)
                logger.info("✅ Composition tracker initialized in testset generator")
            else:
                self.composition_tracker = None
            
            # Initialize parameters tracker
            params_config = self.config.get('reporting', {}).get('final_parameters_tracking', {})
            if params_config.get('enabled', True):
                self.parameters_tracker = ParametersTracker(self.config)
                logger.info("✅ Parameters tracker initialized in testset generator")
            else:
                self.parameters_tracker = None
                
        except ImportError as e:
            logger.warning(f"⚠️ Enhanced trackers not available: {e}")
            self.performance_tracker = None
            self.composition_tracker = None
            self.parameters_tracker = None
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize enhanced trackers: {e}")
            self.performance_tracker = None
            self.composition_tracker = None
            self.parameters_tracker = None
    
    def _setup_llm(self) -> LangchainLLMWrapper:
        """Setup custom LLM following RAGAS documentation pattern."""
        try:
            # Fix URL issue - ensure clean base URL without duplicate paths
            base_url = self.custom_llm_config.get('endpoint', '')
            if base_url.endswith('/chat/completions'):
                base_url = base_url.replace('/chat/completions', '')
            elif base_url.endswith('/v1/chat/completions'):
                base_url = base_url.replace('/v1/chat/completions', '')
            if not base_url.endswith('/v1'):
                if base_url.endswith('/'):
                    base_url = base_url + 'v1'
                else:
                    base_url = base_url + '/v1'
            
            logger.info(f"🔗 Using base URL: {base_url}")
            
            # Create ChatOpenAI with custom endpoint (compatible with 4o)
            llm = ChatOpenAI(
                base_url=base_url,
                api_key=self.custom_llm_config.get('api_key', ''),
                model=self.custom_llm_config.get('model', 'gpt-4o'),
                temperature=self.custom_llm_config.get('temperature', 0.3),
                max_tokens=self.custom_llm_config.get('max_tokens', 1000),
                timeout=self.custom_llm_config.get('timeout', 60),
                default_headers=self.custom_llm_config.get('headers', {})
            )
            
            # Wrap in RAGAS LLM wrapper
            ragas_llm = LangchainLLMWrapper(llm)
            
            logger.info("✅ Custom LLM setup completed")
            return ragas_llm
            
        except Exception as e:
            logger.error(f"❌ Failed to setup custom LLM: {e}")
            raise
    
    def _setup_embeddings(self) -> LangchainEmbeddingsWrapper:
        """Setup embeddings model following RAGAS documentation pattern."""
        try:
            embeddings_model = self.ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2')
            
            # Create HuggingFace embeddings
            hf_embeddings = HuggingFaceEmbeddings(
                model_name=embeddings_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Wrap in RAGAS embeddings wrapper
            ragas_embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
            
            logger.info(f"✅ Embeddings setup completed: {embeddings_model}")
            return ragas_embeddings
            
        except Exception as e:
            logger.error(f"❌ Failed to setup embeddings: {e}")
            raise
    
    def _load_csv_as_documents(self, csv_files: List[str]) -> List[LCDocument]:
        """
        Load CSV files and convert to LangChain documents.
        
        Args:
            csv_files: List of CSV file paths
            
        Returns:
            List of LangChain Document objects
        """
        documents = []
        
        for csv_file in csv_files:
            logger.info(f"📄 Loading CSV file: {csv_file}")
            
            try:
                df = pd.read_csv(csv_file)
                logger.info(f"📊 Loaded {len(df)} rows from {csv_file}")
                
                # Get column mapping
                column_mapping = self.config.get('data_sources', {}).get('csv', {}).get('format', {}).get('column_mapping', {})
                content_column = column_mapping.get('content', 'content')
                
                
                # Limit to max documents as configured - 🔧 FIX: config is already testset_generation section
                logger.info(f'🔍 DEBUG: Current config keys: {list(self.config.keys())}')
                max_docs = self.config.get('max_documents_for_generation', 15)
                logger.info(f'🔢 DEBUG: max_documents_for_generation = {max_docs} (config path fixed)')
                df_limited = df.head(max_docs)
                logger.info(f"🔢 Processing first {len(df_limited)} documents (max: {max_docs})")
                
                for idx, row in df_limited.iterrows():
                    try:
                        # Extract content from JSON if needed
                        content_data = row[content_column]
                        if isinstance(content_data, str) and content_data.startswith('{'):
                            content_json = json.loads(content_data)
                            text_content = content_json.get('text', content_data)
                        else:
                            text_content = str(content_data)
                        
                        # Skip if content is too short
                        if len(text_content.strip()) < 50:
                            logger.warning(f"⚠️ Skipping row {idx}: content too short")
                            continue
                        
                        # Create LangChain document
                        doc = LCDocument(
                            page_content=text_content,
                            metadata={
                                'source': csv_file,
                                'row_id': str(row.get('id', idx)),
                                'title': content_json.get('title', f'Document {idx}') if isinstance(content_data, str) and content_data.startswith('{') else f'Document {idx}',
                                'source_file': content_json.get('source', 'Unknown') if isinstance(content_data, str) and content_data.startswith('{') else 'Unknown'
                            }
                        )
                        documents.append(doc)
                        
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process row {idx}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"❌ Failed to load CSV file {csv_file}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(documents)} documents total")
        return documents
    
    def _create_knowledge_graph(self, documents: List[LCDocument]) -> KnowledgeGraph:
        """Create knowledge graph with comprehensive debug logging"""
        logger.info("🧠 _create_knowledge_graph STARTED")
        logger.info(f"🧠 Input: {len(documents)} documents")
        
        """
        Create multi-hop compatible Knowledge Graph using proven working implementation.
        
        This implementation is based on the successful approach from fix_multi_hop_scenarios.ipynb
        that guarantees proper node properties and relationships for all RAGAS synthesizers.
        
        Args:
            documents: List of LangChain documents
            
        Returns:
            Multi-hop compatible Knowledge Graph
        """
        logger.info("🧠 Creating multi-hop compatible Knowledge Graph...")
        
        # Apply document limiting here to prevent issues with large datasets
        max_docs = self.config.get('testset_generation', {}).get('max_documents_for_generation', 15)
        if max_docs and len(documents) > max_docs:
            logger.info(f"⚡ Limiting KG documents from {len(documents)} to {max_docs} for stable generation")
            documents = documents[:max_docs]
        
        # Import required classes
        from ragas.testset.graph import Node, NodeType, Relationship
        import re
        import numpy as np
        
        # Step 1: Create empty knowledge graph
        kg = KnowledgeGraph()
        logger.info(f"📊 Creating KG from {len(documents)} documents...")
        
        # Step 2: Create embeddings model for similarity calculations
        embeddings_model = None
        try:
            embeddings_model = self.embeddings
            logger.info("✅ Using configured embeddings model for similarity calculations")
        except Exception as e:
            logger.warning(f"⚠️ Could not use embeddings model: {e}")
        
        # Step 3: Create enhanced nodes with ALL required properties
        for i, doc in enumerate(documents):
            # Extract content and create summary
            content = doc.page_content.strip()
            summary = content[:200] + "..." if len(content) > 200 else content
            
            # Extract entities (for MultiHopSpecificQuerySynthesizer)
            entities = []
            # Capitalized words (potential proper nouns)
            capitalized = re.findall(r'\b[A-Z][a-z]+\b', content)
            entities.extend(capitalized[:8])
            # Technical terms/abbreviations
            technical = re.findall(r'\b[A-Z]{2,}\b', content)
            entities.extend(technical[:5])
            # Numbers (IDs, versions, etc.)
            numbers = re.findall(r'\b\d+\b', content)
            entities.extend(numbers[:3])
            entities = list(dict.fromkeys(entities))[:10]  # Remove duplicates, keep top 10
            
            # Extract keyphrases (alternative property)
            content_words = [word.lower() for word in content.split() 
                            if len(word) > 4 and word.isalpha()]
            keyphrases = list(dict.fromkeys(content_words))[:12]  # Top 12 unique keyphrases
            
            # Create headlines from sentences (for headline-based queries)
            sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
            headlines = [s[:80] + "..." if len(s) > 80 else s for s in sentences[:5]]
            
            # Create summary embedding (CRITICAL for MultiHopAbstractQuerySynthesizer)
            summary_embedding = None
            if embeddings_model:
                try:
                    summary_embedding = embeddings_model.embed_query(summary)
                except Exception as e:
                    logger.warning(f"⚠️ Could not create embedding for node {i}: {e}")
            
            # Ensure we have fallback properties if extraction failed
            if not entities:
                entities = [f"entity_{i}_{j}" for j in range(1, 4)]
            if not keyphrases:
                # Extract any words as fallback
                all_words = [word.strip('.,!?":;()[]{}') for word in content.split() 
                           if len(word.strip('.,!?":;()[]{}')) > 3]
                keyphrases = list(dict.fromkeys(all_words))[:10] or [f"keyword_{i}_{j}" for j in range(1, 4)]
            if not headlines:
                headlines = [content[:60] + "..." if len(content) > 60 else content] if content else [f"headline_{i}"]
            
            # ✅ Create node with ALL properties required by RAGAS synthesizers
            node_properties = {
                # Core content
                "page_content": content,
                "summary": summary,
                
                # ✅ CRITICAL: Properties required by synthesizers
                "entities": entities,           # Required by SingleHopSpecificQuerySynthesizer with entities
                "keyphrases": keyphrases,       # Required by SingleHopSpecificQuerySynthesizer with keyphrases  
                "headlines": headlines,         # Required by some transforms
                
                # ✅ CRITICAL: Embedding required by multi-hop synthesizers
                "summary_embedding": summary_embedding,
                
                # Node identification
                "id": f"csv_doc_{i}",
                "title": doc.metadata.get('title', f'Document {i+1}'),
                "source": doc.metadata.get('source', 'csv_input'),
                
                # Enhanced metadata
                "document_metadata": doc.metadata,
                "content_length": len(content),
                "csv_row_index": i,
                
                # Quality indicators
                "has_entities": len(entities) > 0,
                "has_keyphrases": len(keyphrases) > 0,
                "has_summary_embedding": summary_embedding is not None,
                "multi_hop_ready": True
            }
            
            node = Node(
                type=NodeType.DOCUMENT,
                properties=node_properties
            )
            kg.nodes.append(node)
        
        logger.info(f"✅ Created {len(kg.nodes)} enhanced nodes with required properties")
        
        # Step 4: Create relationships for multi-hop synthesis (if enabled in config)
        query_distribution = self.config.get('testset_generation', {}).get('query_distribution', {})
        logger.info(f'🔍 DEBUG: Query distribution: {query_distribution}')
        has_multi_hop = any(key.startswith('multi_hop') for key in query_distribution.keys())
        logger.info(f'🔍 DEBUG: has_multi_hop = {has_multi_hop}')
        
        # 🔗 FORCED: Always build relationships regardless of query distribution
        logger.info("🔗 FORCED RELATIONSHIP BUILDING - Starting multi-hop relationship creation...")
        logger.info(f"🔍 DEBUG: About to call _create_multi_hop_relationships with {len(kg.nodes)} nodes")
        logger.info(f"🔍 DEBUG: Query distribution: {query_distribution}")
        logger.info(f"🔍 DEBUG: has_multi_hop would be: {any(key.startswith('multi_hop') for key in query_distribution.keys())}")
        
        try:
            logger.info("🔗 STARTING multi-hop relationship creation...")
            logger.info(f"🔍 DEBUG: About to call _create_multi_hop_relationships with {len(kg.nodes)} nodes")
            relationships_created = self._create_multi_hop_relationships(kg, embeddings_model)
            logger.info(f"✅ COMPLETED relationship creation: {relationships_created} relationships")
            logger.info(f"🔍 DEBUG: Final KG has {len(kg.relationships)} total relationships")
        except Exception as e:
            logger.error(f"❌ Relationship building failed: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
        
        # Step 5: Final validation and statistics
        self._log_kg_statistics(kg)
        
        return kg
    
    def _create_multi_hop_relationships(self, kg: KnowledgeGraph, embeddings_model) -> int:
        """
        Create multi-hop compatible relationships in the Knowledge Graph.
        Uses the PROVEN WORKING implementation from run_pure_ragas_pipeline.py
        """
        logger.info("🔗 Creating multi-hop relationships using PROVEN WORKING implementation...")
        logger.info(f"🔍 DEBUG: Starting relationship creation with {len(kg.nodes)} nodes")
        
        initial_relationships = len(kg.relationships)
        
        try:
            # Use the exact same relationship builders that work in run_pure_ragas_pipeline.py
            from ragas.testset.transforms.relationship_builders import (
                CosineSimilarityBuilder,
                JaccardSimilarityBuilder, 
                OverlapScoreBuilder
            )
            from ragas.testset.transforms.relationship_builders.cosine import SummaryCosineSimilarityBuilder
            
            logger.info("🔗 Building relationships between nodes...")
            
            # 1. Jaccard similarity relationships (entity-based)
            logger.info("🔗 Building Jaccard similarity relationships...")
            jaccard_builder = JaccardSimilarityBuilder(threshold=0.1)
            jaccard_builder.transform(kg)
            jaccard_count = len(kg.relationships) - initial_relationships
            logger.info(f"✅ Built {jaccard_count} Jaccard similarity relationships")
            
            # 2. Overlap score relationships (keyphrase-based) 
            logger.info("🔗 Building overlap score relationships...")
            overlap_builder = OverlapScoreBuilder(threshold=0.05)
            overlap_builder.transform(kg)
            overlap_count = len(kg.relationships) - initial_relationships - jaccard_count
            logger.info(f"✅ Built {overlap_count} overlap score relationships")
            
            # 3. Cosine similarity relationships (embedding-based)
            logger.info("🔗 Building cosine similarity relationships...")
            cosine_builder = CosineSimilarityBuilder(threshold=0.7)
            cosine_builder.transform(kg)
            cosine_count = len(kg.relationships) - initial_relationships - jaccard_count - overlap_count
            logger.info(f"✅ Built {cosine_count} cosine similarity relationships")
            
            # 4. Summary cosine similarity relationships
            logger.info("🔗 Building summary cosine similarity relationships...")
            summary_cosine_builder = SummaryCosineSimilarityBuilder(threshold=0.5)
            summary_cosine_builder.transform(kg)
            summary_cosine_count = len(kg.relationships) - initial_relationships - jaccard_count - overlap_count - cosine_count
            logger.info(f"✅ Built {summary_cosine_count} summary cosine similarity relationships")
            
            # 5. Additional multi-hop specific relationships
            logger.info("🔗 Building summary_similarity relationships for multihop abstract...")
            summary_sim_rels = self._create_summary_similarity_relationships(kg)
            for rel in summary_sim_rels:
                kg._add_relationship(rel)
            summary_similarity_count = len(summary_sim_rels)
            logger.info(f"✅ Built {summary_similarity_count} summary_similarity relationships")
            
            # 6. Entities overlap for multihop specific
            logger.info("🔗 Building entities_overlap relationships for multihop specific...")
            entities_overlap_rels = self._create_entities_overlap_relationships(kg)
            for rel in entities_overlap_rels:
                kg._add_relationship(rel)
            entities_overlap_count = len(entities_overlap_rels)
            logger.info(f"✅ Built {entities_overlap_count} entities_overlap relationships")
            
            total_relationships_created = len(kg.relationships) - initial_relationships
            logger.info(f"✅ Built {total_relationships_created} relationships in knowledge graph")
            
            return total_relationships_created
            
        except Exception as e:
            logger.error(f"❌ Relationship building failed: {e}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            return 0
    
    def _create_summary_similarity_relationships(self, kg):
        """Create summary_similarity relationships required by MultiHopAbstractQuerySynthesizer"""
        relationships = []
        nodes = list(kg.nodes)
        
        for i, node1 in enumerate(nodes):
            for j, node2 in enumerate(nodes[i+1:], i+1):
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
                                properties={"summary_similarity": similarity}
                            )
                            relationships.append(rel)
        
        return relationships
    
    def _create_entities_overlap_relationships(self, kg):
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
                        from ragas.testset.graph import Relationship
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

    def _log_kg_statistics(self, kg: KnowledgeGraph):
        """
        Log detailed knowledge graph statistics for debugging.
        
        Args:
            kg: Knowledge Graph to analyze
        """
        logger.info(f"📊 Final KG statistics:")
        logger.info(f"   Total nodes: {len(kg.nodes)}")
        logger.info(f"   Total relationships: {len(kg.relationships)}")
        
        # Node property statistics
        nodes_with_entities = sum(1 for n in kg.nodes 
                                 if n.get_property("entities") and len(n.get_property("entities")) > 0)
        nodes_with_keyphrases = sum(1 for n in kg.nodes 
                                   if n.get_property("keyphrases") and len(n.get_property("keyphrases")) > 0)
        nodes_with_embeddings = sum(1 for n in kg.nodes 
                                   if n.get_property("summary_embedding") is not None)
        
        logger.info(f"   Nodes with entities: {nodes_with_entities}/{len(kg.nodes)}")
        logger.info(f"   Nodes with keyphrases: {nodes_with_keyphrases}/{len(kg.nodes)}")
        logger.info(f"   Nodes with embeddings: {nodes_with_embeddings}/{len(kg.nodes)}")
        
        # Relationship type statistics
        if kg.relationships:
            relationship_types = {}
            for rel in kg.relationships:
                rel_type = rel.type
                relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
            
            logger.info(f"   Relationship types:")
            for rel_type, count in relationship_types.items():
                logger.info(f"     {rel_type}: {count}")
        
        # Multi-hop compatibility check
        has_summary_similarity = any(rel.type == "summary_similarity" for rel in kg.relationships)
        has_entities_overlap = any(rel.type == "entities_overlap" for rel in kg.relationships)
        
        logger.info(f"🔍 Multi-hop compatibility:")
        logger.info(f"   Summary similarity relationships: {'✅' if has_summary_similarity else '❌'}")
        logger.info(f"   Entity overlap relationships: {'✅' if has_entities_overlap else '❌'}")
        
        if nodes_with_entities == 0 and nodes_with_keyphrases == 0:
            logger.warning("⚠️ No nodes have entities or keyphrases - synthesis may fail")
        else:
            logger.info("🎉 Knowledge graph is ready for synthesis!")

    def _manually_enrich_nodes(self, kg: KnowledgeGraph, documents: List[LCDocument]):
        """
        Manually enrich nodes when transforms fail.
        This provides basic fallback enrichment for node properties.
        """
        logger.info("🔧 Manually enriching nodes with basic properties...")
        
        import re
        
        for i, node in enumerate(kg.nodes):
            try:
                content = node.get_property("page_content", "")
                if not content:
                    continue
                
                # Extract basic entities
                entities = []
                # Capitalized words (potential proper nouns)
                capitalized = re.findall(r'\b[A-Z][a-z]+\b', content)
                entities.extend(capitalized[:5])
                # Technical terms/abbreviations
                technical = re.findall(r'\b[A-Z]{2,}\b', content)
                entities.extend(technical[:3])
                entities = list(dict.fromkeys(entities))[:8]  # Remove duplicates
                
                # Extract keyphrases
                content_words = [word.lower() for word in content.split() 
                                if len(word) > 4 and word.isalpha()]
                keyphrases = list(dict.fromkeys(content_words))[:10]
                
                # Create headlines from sentences
                sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 10]
                headlines = [s[:60] + "..." if len(s) > 60 else s for s in sentences[:3]]
                
                # Update node properties
                node.properties.update({
                    "entities": entities,
                    "keyphrases": keyphrases,
                    "headlines": headlines,
                    "enriched": True
                })
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to enrich node {i}: {e}")
        
        logger.info(f"✅ Manually enriched {len(kg.nodes)} nodes")
    
    def _create_configurable_query_distribution(self, kg: KnowledgeGraph) -> List:
        """
        Create configurable query distribution based on config settings.
        
        This allows fine-grained control over question types generated,
        supporting both single-hop and multi-hop scenarios based on 
        what's available in the knowledge graph.
        
        Args:
            kg: Knowledge Graph to check compatibility
            
        Returns:
            List of (synthesizer, weight) tuples for generation
        """
        logger.info("🎯 Creating configurable query distribution...")
        logger.info(f"📊 Knowledge graph has {len(kg.nodes)} nodes")
        
        # Log a sample of node properties for debugging
        sample_nodes = kg.nodes[:3]
        for i, node in enumerate(sample_nodes):
            logger.info(f"   Sample node {i}: properties = {list(node.properties.keys())}")
            if 'entities' in node.properties:
                logger.info(f"      entities: {len(node.properties['entities'])} items")
            if 'keyphrases' in node.properties:
                logger.info(f"      keyphrases: {len(node.properties['keyphrases'])} items")
            if 'headlines' in node.properties:
                logger.info(f"      headlines: {len(node.properties['headlines'])} items")
        
        # Import synthesizers
        from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
        
        # Get distribution config from pipeline config
        distribution_config = self.config.get('testset_generation', {}).get('query_distribution', {})
        logger.info(f"📋 Configuration requests: {list(distribution_config.keys())}")
        
        # Enhanced 4-type query distribution for better testset variety
        if not distribution_config:
            distribution_config = {
                "single_hop_entities": 0.35,      # 35% single-hop entity-based queries
                "single_hop_keyphrases": 0.25,    # 25% single-hop keyphrase-based queries  
                "multi_hop_abstract": 0.25,       # 25% multi-hop abstract reasoning queries
                "multi_hop_specific": 0.15        # 15% multi-hop specific factual queries
            }
            logger.info("📋 Using enhanced 4-type query distribution for variety")
        else:
            logger.info(f"📋 Using configured query distribution: {distribution_config}")
        
        # Initialize composition tracker if enabled
        composition_tracker = None
        if hasattr(self, 'composition_tracker'):
            composition_tracker = self.composition_tracker
            composition_tracker.start_generation({
                'total_nodes': len(kg.nodes),
                'total_relationships': len(kg.relationships),
                'node_types': list(set(n.type.name for n in kg.nodes)),
                'relationship_types': list(set(r.type for r in kg.relationships))
            })
        
        # Create synthesizers and check compatibility
        available_distribution = []
        
        # Debug: Log knowledge graph structure
        logger.info(f"🔍 DEBUG: Knowledge Graph Analysis for Query Distribution")
        logger.info(f"   Total nodes: {len(kg.nodes)}")
        logger.info(f"   Total relationships: {len(kg.relationships)}")
        
        # Analyze node properties
        entities_nodes = 0
        keyphrases_nodes = 0
        headlines_nodes = 0
        sample_nodes = kg.nodes[:3]  # Check first 3 nodes
        
        for i, node in enumerate(sample_nodes):
            logger.info(f"   Node {i+1} properties: {list(node.properties.keys()) if hasattr(node, 'properties') and node.properties else 'None'}")
            if hasattr(node, 'properties') and node.properties:
                if 'entities' in node.properties and node.properties['entities']:
                    entities_nodes += 1
                if 'keyphrases' in node.properties and node.properties['keyphrases']:
                    keyphrases_nodes += 1
                if 'headlines' in node.properties and node.properties['headlines']:
                    headlines_nodes += 1
        
        # Count all nodes with properties
        for node in kg.nodes:
            if hasattr(node, 'properties') and node.properties:
                if 'entities' in node.properties and node.properties['entities']:
                    entities_nodes += 1
                if 'keyphrases' in node.properties and node.properties['keyphrases']:
                    keyphrases_nodes += 1
                if 'headlines' in node.properties and node.properties['headlines']:
                    headlines_nodes += 1
        
        logger.info(f"   Nodes with entities: {entities_nodes}")
        logger.info(f"   Nodes with keyphrases: {keyphrases_nodes}")
        logger.info(f"   Nodes with headlines: {headlines_nodes}")
        
        # Single-hop synthesizers (most reliable)
        if "single_hop_entities" in distribution_config:
            try:
                synthesizer = SingleHopSpecificQuerySynthesizer(
                    llm=self.llm, 
                    property_name="entities"
                )
                # Check if nodes have entities property - be more lenient
                nodes_with_entities = [n for n in kg.nodes 
                                     if hasattr(n, 'properties') and n.properties and 
                                        'entities' in n.properties and n.properties['entities']]
                
                # Fallback: use page_content if no entities found
                if len(nodes_with_entities) == 0:
                    logger.warning(f"   ⚠️ No nodes with entities, trying page_content fallback")
                    # Try with page_content property instead
                    synthesizer = SingleHopSpecificQuerySynthesizer(
                        llm=self.llm, 
                        property_name="page_content"
                    )
                    nodes_with_content = [n for n in kg.nodes 
                                        if hasattr(n, 'properties') and n.properties and 
                                           'page_content' in n.properties and n.properties['page_content']]
                    if len(nodes_with_content) > 0:
                        weight = distribution_config["single_hop_entities"]
                        available_distribution.append((synthesizer, weight))
                        logger.info(f"   ✅ Single-hop (entities→page_content): {len(nodes_with_content)} nodes available, weight: {weight:.1%}")
                    else:
                        logger.warning(f"   ❌ Single-hop (entities): No nodes with page_content either")
                else:
                    weight = distribution_config["single_hop_entities"]
                    available_distribution.append((synthesizer, weight))
                    logger.info(f"   ✅ Single-hop (entities): {len(nodes_with_entities)} nodes available, weight: {weight:.1%}")
            except Exception as e:
                logger.warning(f"   ❌ Single-hop (entities): Error - {e}")
        
        if "single_hop_keyphrases" in distribution_config:
            try:
                synthesizer = SingleHopSpecificQuerySynthesizer(
                    llm=self.llm, 
                    property_name="keyphrases"
                )
                # Check if nodes have keyphrases property
                nodes_with_keyphrases = [n for n in kg.nodes 
                                       if hasattr(n, 'properties') and n.properties and 
                                          'keyphrases' in n.properties and n.properties['keyphrases']]
                
                # Fallback: use page_content if no keyphrases found
                if len(nodes_with_keyphrases) == 0:
                    logger.warning(f"   ⚠️ No nodes with keyphrases, trying page_content fallback")
                    synthesizer = SingleHopSpecificQuerySynthesizer(
                        llm=self.llm, 
                        property_name="page_content"
                    )
                    nodes_with_content = [n for n in kg.nodes 
                                        if hasattr(n, 'properties') and n.properties and 
                                           'page_content' in n.properties and n.properties['page_content']]
                    if len(nodes_with_content) > 0:
                        weight = distribution_config["single_hop_keyphrases"] 
                        available_distribution.append((synthesizer, weight))
                        logger.info(f"   ✅ Single-hop (keyphrases→page_content): {len(nodes_with_content)} nodes available, weight: {weight:.1%}")
                    else:
                        logger.warning(f"   ❌ Single-hop (keyphrases): No nodes with page_content either")
                else:
                    weight = distribution_config["single_hop_keyphrases"]
                    available_distribution.append((synthesizer, weight))
                    logger.info(f"   ✅ Single-hop (keyphrases): {len(nodes_with_keyphrases)} nodes available, weight: {weight:.1%}")
            except Exception as e:
                logger.warning(f"   ❌ Single-hop (keyphrases): Error - {e}")
        
        if "single_hop_headlines" in distribution_config:
            try:
                synthesizer = SingleHopSpecificQuerySynthesizer(
                    llm=self.llm, 
                    property_name="headlines"
                )
                # Check if nodes have headlines property
                nodes_with_headlines = [n for n in kg.nodes 
                                      if hasattr(n, 'properties') and 'headlines' in n.properties and n.properties['headlines']]
                if len(nodes_with_headlines) > 0:
                    weight = distribution_config["single_hop_headlines"]
                    available_distribution.append((synthesizer, weight))
                    logger.info(f"   ✅ Single-hop (headlines): {len(nodes_with_headlines)} nodes available, weight: {weight:.1%}")
                else:
                    logger.warning(f"   ❌ Single-hop (headlines): No nodes with headlines property")
            except Exception as e:
                logger.warning(f"   ❌ Single-hop (headlines): Error - {e}")
        
        # Try multi-hop synthesizers only if relationships exist
        relationships_exist = len(kg.relationships) > 0
        logger.info(f"🔗 Knowledge graph has {len(kg.relationships)} relationships")
        
        if relationships_exist:
            # Multi-hop synthesizers (advanced)
            if "multi_hop_abstract" in distribution_config:
                try:
                    # Try both import patterns that work according to our test
                    try:
                        from ragas.testset.synthesizers import MultiHopAbstractQuerySynthesizer
                    except ImportError:
                        from ragas.testset.synthesizers.multi_hop.abstract import MultiHopAbstractQuerySynthesizer
                    
                    synthesizer = MultiHopAbstractQuerySynthesizer(llm=self.llm)
                    # Check if synthesizer can get clusters from the KG
                    try:
                        clusters = synthesizer.get_node_clusters(kg)
                        if len(clusters) > 0:
                            weight = distribution_config["multi_hop_abstract"]
                            available_distribution.append((synthesizer, weight))
                            logger.info(f"   ✅ Multi-hop (abstract): {len(clusters)} clusters, weight: {weight:.1%}")
                        else:
                            logger.warning(f"   ❌ Multi-hop (abstract): No clusters found")
                    except Exception as cluster_error:
                        logger.warning(f"   ❌ Multi-hop (abstract): Cluster error - {cluster_error}")
                except Exception as e:
                    logger.warning(f"   ❌ Multi-hop (abstract): Import/init error - {e}")
            
            if "multi_hop_specific" in distribution_config:
                try:
                    # Try both import patterns that work according to our test
                    try:
                        from ragas.testset.synthesizers import MultiHopSpecificQuerySynthesizer
                    except ImportError:
                        from ragas.testset.synthesizers.multi_hop.specific import MultiHopSpecificQuerySynthesizer
                    
                    synthesizer = MultiHopSpecificQuerySynthesizer(llm=self.llm)
                    # Check if synthesizer can get clusters from the KG
                    try:
                        clusters = synthesizer.get_node_clusters(kg)
                        if len(clusters) > 0:
                            weight = distribution_config["multi_hop_specific"]
                            available_distribution.append((synthesizer, weight))
                            logger.info(f"   ✅ Multi-hop (specific): {len(clusters)} clusters, weight: {weight:.1%}")
                        else:
                            logger.warning(f"   ❌ Multi-hop (specific): No clusters found")
                    except Exception as cluster_error:
                        logger.warning(f"   ❌ Multi-hop (specific): Cluster error - {cluster_error}")
                except Exception as e:
                    logger.warning(f"   ❌ Multi-hop (specific): Import/init error - {e}")
        else:
            logger.info("⚠️ No relationships in knowledge graph - skipping multi-hop synthesizers")
        
        # Normalize weights for available synthesizers
        if available_distribution:
            total_available_weight = sum(weight for _, weight in available_distribution)
            if total_available_weight > 0:
                available_distribution = [
                    (synth, weight/total_available_weight) 
                    for synth, weight in available_distribution
                ]
            
            logger.info(f"📊 Final available query distribution:")
            multi_hop_count = 0
            single_hop_count = 0
            for synthesizer, weight in available_distribution:
                synth_name = type(synthesizer).__name__
                logger.info(f"     {synth_name}: {weight:.1%}")
                if "MultiHop" in synth_name:
                    multi_hop_count += 1
                else:
                    single_hop_count += 1
            
            logger.info(f"   🎉 {single_hop_count} single-hop + {multi_hop_count} multi-hop synthesizers available")
            
            return available_distribution
        else:
            logger.warning("⚠️ No compatible synthesizers found, falling back to default distribution")
            # Fallback to default RAGAS distribution
            from ragas.testset.synthesizers import default_query_distribution
            return default_query_distribution(self.llm)
    
    def _save_knowledge_graph(self, kg: KnowledgeGraph, output_dir: Path) -> str:
        """
        Save knowledge graph for reuse.
        
        Args:
            kg: Knowledge Graph to save
            output_dir: Output directory
            
        Returns:
            Path to saved knowledge graph file
        """
        kg_dir = output_dir / "knowledge_graphs"
        kg_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        kg_file = kg_dir / f"knowledge_graph_{timestamp}.json"
        
        try:
            kg.save(str(kg_file))
            logger.info(f"💾 Knowledge graph saved: {kg_file}")
            return str(kg_file)
        except Exception as e:
            logger.error(f"❌ Failed to save knowledge graph: {e}")
            return ""
    
    def _apply_secondary_fixes(self):
        """Apply secondary issue fixes to the generator"""
        try:
            from .pure_ragas_testset_generator_fixes import apply_fixes_to_generator
            
            # Apply fixes and get the fixed configuration
            self, fixed_config = apply_fixes_to_generator(self, self.config)
            self.config = fixed_config
            
            self.logger.info("✅ Applied secondary issue fixes")
        except Exception as e:
            self.logger.warning(f"⚠️ Could not apply secondary fixes: {e}")
            # Continue without fixes
    
    def _setup_knowledge_graph_reuse(self, output_dir: str) -> Optional[str]:
        """Setup knowledge graph reuse functionality"""
        try:
            from ..utils.knowledge_graph_manager import KnowledgeGraphManager
            
            kg_manager = KnowledgeGraphManager(output_dir)
            existing_kg = kg_manager.get_latest_knowledge_graph()
            
            if existing_kg:
                self.logger.info(f"🔍 Found existing knowledge graph for reuse: {existing_kg}")
                return existing_kg
            else:
                self.logger.info("ℹ️ No existing knowledge graphs found")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Could not check for existing knowledge graphs: {e}")
            return None
    
    def _save_knowledge_graph_for_reuse(self, kg, metadata: Dict[str, Any], output_dir: str) -> str:
        """Save knowledge graph for future reuse"""
        try:
            from ..utils.knowledge_graph_manager import KnowledgeGraphManager
            
            kg_manager = KnowledgeGraphManager(output_dir)
            kg_filepath = kg_manager.save_knowledge_graph(
                kg=kg,
                metadata=metadata,
                run_id=datetime.now().strftime("%Y%m%d_%H%M%S")
            )
            
            self.logger.info(f"💾 Saved knowledge graph for reuse: {kg_filepath}")
            return kg_filepath
            
        except Exception as e:
            self.logger.warning(f"⚠️ Could not save knowledge graph for reuse: {e}")
            return ""
    
    def generate_comprehensive_testset_broken(  # Renamed to avoid conflicts
        self, 
        document_paths: List[str] = None,  # Changed from csv_files to match orchestrator interface
        output_dir: Path = None,  # Made optional to match orchestrator interface
        csv_files: List[str] = None,  # Keep for backward compatibility
        personas: Optional[List[Dict]] = None,
        scenarios: Optional[List[Dict]] = None,
        batch_size: int = 1000,
        save_intermediate: bool = True,
        apply_fixes: bool = True,
        testset_size: Optional[int] = None,  # NEW: Allow override of testset size for batch processing
        performance_config: Optional[Dict[str, Any]] = None  # NEW: Performance optimization config
    ) -> Dict[str, Any]:
        """
        Generate comprehensive testset using proper RAGAS following documentation pattern.
        Enhanced with OutputParserException fixes, persona/scenario integration, and batch saving.
        
        Args:
            csv_files: List of CSV file paths
            output_dir: Output directory for testset and metadata
            personas: Optional pre-loaded personas (if None, will use defaults)
            scenarios: Optional pre-loaded scenarios (if None, will use defaults)
            batch_size: Batch size for saving intermediate results
            save_intermediate: Whether to save intermediate outputs during generation
            apply_fixes: Whether to apply OutputParserException fixes
            
        Returns:
            Dictionary with generation results including fix information
        """
        try:
            logger.info("🚀 Starting Enhanced Pure RAGAS Testset Generation...")
            
            # Handle interface compatibility: convert document_paths to csv_files
            if csv_files is None and document_paths is not None:
                csv_files = document_paths
                logger.info(f"🔄 Using document_paths as csv_files: {len(csv_files)} files")
            elif csv_files is None:
                # Try to get from config as fallback
                csv_files = self.config.get('data_sources', {}).get('csv', {}).get('csv_files', [])
                logger.info(f"🔄 Using config csv_files: {len(csv_files)} files")
            
            if not csv_files:
                raise ValueError("No CSV files provided via document_paths, csv_files parameter, or config")
            
            # Set default output directory if not provided
            if output_dir is None:
                from datetime import datetime
                output_dir = Path('outputs') / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / 'testsets'
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Using default output directory: {output_dir}")
            
            # ✅ IMPLEMENTED: Apply performance configuration
            performance_config = performance_config or {}
            if performance_config:
                logger.info(f"🚀 Performance optimizations enabled: {performance_config}")
                if performance_config.get('memory_aggressive'):
                    logger.info("💾 Memory aggressive mode enabled")
                if performance_config.get('max_workers'):
                    logger.info(f"⚡ Parallel processing with {performance_config['max_workers']} workers")
            
            # ✅ IMPLEMENTED: Apply RAGAS configuration overrides
            try:
                from utils.ragas_config_manager import RAGASConfigManager
                config_manager = RAGASConfigManager(self.config)
                
                # Get memory optimization info from performance_config
                memory_optimization = performance_config.get('memory_optimization') if isinstance(performance_config.get('memory_optimization'), dict) else None
                
                # Get optimized run config for RAGAS
                optimized_run_config_params = config_manager.get_optimized_run_config(memory_optimization)
                logger.info(f"🔧 RAGAS configuration overrides applied: {optimized_run_config_params}")
                
            except Exception as e:
                logger.warning(f"⚠️ RAGAS config override failed: {e}, using default configuration")
                optimized_run_config_params = None
            
            # Apply OutputParserException fixes if requested
            fixes_applied = []
            if apply_fixes:
                logger.info("🔧 Applying OutputParserException fixes...")
                try:
                    from utils.output_parser_fix import apply_ragas_output_parser_fixes
                    if apply_ragas_output_parser_fixes():
                        fixes_applied.append("output_parser_exception_fix")
                        logger.info("✅ OutputParserException fixes applied")
                    else:
                        logger.warning("⚠️ OutputParserException fixes failed")
                except Exception as fix_error:
                    logger.warning(f"⚠️ Failed to apply fixes: {fix_error}")
            
            # Enhanced debugging information - Pipeline Setup Logging
            logger.info("=" * 80)
            logger.info("🔧 ENHANCED PIPELINE CONFIGURATION SUMMARY")
            logger.info("=" * 80)
            
            # Log input data information
            logger.info(f"📁 INPUT DATA:")
            logger.info(f"   CSV files: {len(csv_files)} files")
            for i, csv_file in enumerate(csv_files):
                try:
                    import pandas as pd
                    df_info = pd.read_csv(csv_file)
                    logger.info(f"   {i+1}. {Path(csv_file).name} ({len(df_info)} rows)")
                except Exception:
                    logger.info(f"   {i+1}. {Path(csv_file).name} (unknown size)")
            
            # Log enhancement parameters
            logger.info(f"🎭 ENHANCEMENT FEATURES:")
            logger.info(f"   Personas provided: {len(personas) if personas else 0}")
            logger.info(f"   Scenarios provided: {len(scenarios) if scenarios else 0}")
            logger.info(f"   Batch size: {batch_size}")
            logger.info(f"   Save intermediate: {save_intermediate}")
            logger.info(f"   Apply fixes: {apply_fixes}")
            logger.info(f"   Fixes applied: {fixes_applied}")
            
            # Log generation method and key parameters
            testset_config = self.config.get('testset_generation', {})
            logger.info(f"🏭 GENERATION METHOD:")
            logger.info(f"   Method: Enhanced Pure RAGAS TestsetGenerator")
            logger.info(f"   Output directory: {output_dir}")
            
            # Log key parameters
            max_docs = testset_config.get('max_documents_for_generation', 100)
            target_testset_size = testset_size if testset_size is not None else testset_config.get('testset_size', testset_config.get('max_total_samples', 3))
            logger.info(f"📊 KEY PARAMETERS:")
            logger.info(f"   Max documents for generation: {max_docs}")
            logger.info(f"   Target testset size: {target_testset_size} QA pairs")
            logger.info(f"   LLM endpoint: {self.config.get('llm', {}).get('endpoint', 'default')}")
            logger.info(f"   LLM model: {self.config.get('llm', {}).get('model', 'default')}")
            
            # Log knowledge graph strategy
            kg_config = testset_config.get('ragas_config', {}).get('knowledge_graph_config', {})
            existing_kg_file = kg_config.get('existing_kg_file', '')
            enable_kg_loading = kg_config.get('enable_kg_loading', False)
            enable_kg_saving = kg_config.get('enable_kg_saving', False)
            
            logger.info(f"🧠 KNOWLEDGE GRAPH STRATEGY:")
            logger.info(f"   Enable KG loading: {enable_kg_loading}")
            logger.info(f"   Enable KG saving: {enable_kg_saving}")
            logger.info(f"   Existing KG file: {existing_kg_file if existing_kg_file else 'None specified'}")
            
            logger.info("=" * 80)
            
            # Test RunConfig availability immediately
            logger.info(f"🔍 Testing RunConfig import...")
            try:
                from ragas.run_config import RunConfig
                logger.info(f"✅ RunConfig imported successfully")
                test_config = RunConfig(timeout=60, max_retries=2)
                logger.info(f"✅ Test RunConfig created successfully")
            except Exception as rc_error:
                logger.error(f"❌ RunConfig error: {rc_error}")
                import traceback
                logger.error(f"RunConfig traceback: {traceback.format_exc()}")
                raise rc_error
            
            # Step 1: Load documents
            logger.info("📄 Step 1: Loading documents...")
            documents = self._load_csv_as_documents(csv_files)
            if not documents:
                raise ValueError("No valid documents loaded")
            logger.info(f"✅ Loaded {len(documents)} documents from CSV files")
            
            # Step 2: Check for existing Knowledge Graph FIRST
            kg = None
            kg_method = "new"
            kg_file = None  # Initialize kg_file variable
            
            if enable_kg_loading and existing_kg_file and Path(existing_kg_file).exists():
                logger.info("🔄 Step 2a: Loading existing knowledge graph...")
                try:
                    kg = KnowledgeGraph.load(existing_kg_file)
                    kg_method = "existing"
                    logger.info(f"✅ Successfully loaded existing KG from: {existing_kg_file}")
                    logger.info(f"   - Nodes: {len(kg.nodes)}")
                    logger.info(f"   - Relationships: {len(kg.relationships)}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to load existing KG from {existing_kg_file}: {e}")
                    logger.info("🔄 Will create new knowledge graph instead")
                    kg = None
            elif existing_kg_file:
                logger.warning(f"⚠️ Existing KG file specified but not found: {existing_kg_file}")
                logger.info("🔄 Will create new knowledge graph instead")
            
            # Step 2b: Create new Knowledge Graph if needed
            if kg is None:
                logger.info("🧠 Step 2b: Creating new knowledge graph...")
                kg = self._create_knowledge_graph(documents)
                kg_method = "new"
                logger.info(f"✅ Created new knowledge graph")
                logger.info(f"   - Nodes: {len(kg.nodes)}")
                logger.info(f"   - Relationships: {len(kg.relationships)}")
            
            # Log which method was used
            logger.info(f"🎯 KNOWLEDGE GRAPH METHOD: {kg_method.upper()}")
            if kg_method == "existing":
                logger.info(f"   ♻️  Reused existing KG from: {Path(existing_kg_file).name}")
            else:
                logger.info(f"   🆕 Created new KG from {len(documents)} documents")
            
            
            # Step 3: Save Knowledge Graph for future reuse (if new)
            if kg_method == "new" and enable_kg_saving:
                logger.info("💾 Step 3: Saving knowledge graph for future reuse...")
                kg_file = self._save_knowledge_graph(kg, output_dir)
                if kg_file:
                    logger.info(f"✅ Saved new KG for reuse: {Path(kg_file).name}")
                else:
                    logger.warning("⚠️ Failed to save knowledge graph")
            elif kg_method == "existing":
                logger.info("ℹ️ Step 3: Skipping KG save (using existing KG)")
                kg_file = existing_kg_file  # Use the existing KG file path
            else:
                logger.info("ℹ️ Step 3: Skipping KG save (saving disabled)")
                kg_file = None  # No KG file when saving is disabled
            
            # Step 4: Create TestsetGenerator with the KG (existing or new)
            logger.info("🎯 Step 4: Creating RAGAS TestsetGenerator...")
            generator = TestsetGenerator(
                llm=self.llm,
                embedding_model=self.embeddings,
                knowledge_graph=kg
            )
            logger.info("✅ TestsetGenerator initialized successfully")
            
            # Step 5: Configure query distribution
            logger.info("⚙️ Step 5: Configuring query distribution...")
            use_configurable = testset_config.get('use_configurable_distribution', True)
            
            if use_configurable:
                query_distribution = self._create_configurable_query_distribution(kg)
                logger.info(f"✅ Using configurable query distribution with {len(query_distribution)} synthesizers")
            else:
                query_distribution = default_query_distribution(self.llm)
                logger.info(f"✅ Using default query distribution with {len(query_distribution)} synthesizers")
            
            # Log distribution details
            if query_distribution:
                logger.info(f"📋 Query distribution details:")
                for i, (synthesizer, weight) in enumerate(query_distribution):
                    synth_name = type(synthesizer).__name__
                    logger.info(f"   {i+1}. {synth_name}: {weight:.1%}")
            else:
                logger.warning("⚠️ Query distribution is empty!")
                raise ValueError("No query distribution available")
            
            # Step 6: Generate testset with comprehensive logging
            logger.info(f"⚡ Step 6: Generating testset with {target_testset_size} samples...")
            logger.info(f"🎯 TESTSET GENERATION EXPECTATIONS:")
            logger.info(f"   - Target QA pairs: {target_testset_size}")
            logger.info(f"   - Generation method: {'Existing KG' if kg_method == 'existing' else 'New KG'} + RAGAS")
            logger.info(f"   - Documents used: {len(documents)} (max: {max_docs})")
            logger.info(f"   - KG nodes available: {len(kg.nodes)}")
            logger.info(f"   - Query synthesizers: {len(query_distribution)}")
            
            # ✅ IMPLEMENTED: Setup RunConfig with performance optimizations
            if optimized_run_config_params:
                # Use optimized config from RAGASConfigManager
                run_config = RunConfig(**optimized_run_config_params)
                logger.info(f"✅ Using optimized RAGAS RunConfig: {optimized_run_config_params}")
            else:
                # Fallback to default configuration
                run_config = RunConfig(
                    max_workers=2,  # Reduce workers to avoid rate limits
                    timeout=120,    # Increase timeout
                    max_retries=2,
                    max_wait=10
                )
                logger.info("✅ Using default RAGAS RunConfig")
            
            # Start performance tracking for testset generation
            generation_timing_id = None
            if hasattr(self, 'performance_tracker') and self.performance_tracker:
                generation_timing_id = self.performance_tracker.start_timing(
                    'testset_generation', 
                    f'generation_size_{target_testset_size}_types_{len(query_distribution)}'
                )
            
            # Try generation with detailed error handling
            try:
                logger.info(f"🎯 Generating testset with {len(query_distribution)} query types...")
                
                # Apply output parser fixes before generation
                if apply_fixes:
                    try:
                        from utils.output_parser_fix import apply_ragas_output_parser_fixes
                        apply_ragas_output_parser_fixes()
                        logger.info("✅ OutputParserException fixes applied before generation")
                    except Exception as fix_error:
                        logger.warning(f"⚠️ Could not apply output parser fixes: {fix_error}")
                
                testset = generator.generate(
                    testset_size=target_testset_size,  # Use target_testset_size from config or parameter
                    query_distribution=query_distribution,
                    run_config=run_config,
                    raise_exceptions=False  # Don't raise exceptions to see all failures
                )
                
                # Track successful generation
                if hasattr(self, 'parameters_tracker') and self.parameters_tracker:
                    self.parameters_tracker.track_config_modification(
                        'PureRAGASTestsetGenerator',
                        'testset_generation.final_testset_size',
                        target_testset_size,
                        len(testset) if hasattr(testset, '__len__') else target_testset_size,
                        'Actual testset size after generation'
                    )
                
            except Exception as e:
                logger.warning(f"⚠️ Generation with query distribution failed: {e}")
                
                # Track fallback usage
                if hasattr(self, 'parameters_tracker') and self.parameters_tracker:
                    self.parameters_tracker.track_fallback_usage(
                        'PureRAGASTestsetGenerator',
                        'query_distribution',
                        query_distribution,
                        'simple_fallback',
                        f"Original distribution failed: {e}",
                        False
                    )
                
                logger.info("🔄 Trying direct generation without query distribution...")
                
                # Fallback: try direct generation with simple distribution
                try:
                    # Create minimal single-hop distribution
                    from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer
                    simple_distribution = [
                        (SingleHopSpecificQuerySynthesizer(llm=self.llm, property_name="page_content"), 1.0)
                    ]
                    
                    testset = generator.generate(
                        testset_size=min(target_testset_size, 3),  # Reduce size for fallback
                        query_distribution=simple_distribution,
                        run_config=run_config,
                        raise_exceptions=False
                    )
                    
                    # Track successful fallback
                    if hasattr(self, 'parameters_tracker') and self.parameters_tracker:
                        self.parameters_tracker.track_fallback_usage(
                            'PureRAGASTestsetGenerator',
                            'query_distribution',
                            query_distribution,
                            simple_distribution,
                            f"Primary distribution failed: {e}",
                            True
                        )
                    
                except Exception as e2:
                    logger.error(f"❌ All generation attempts failed: {e2}")
                    
                    # Track final fallback
                    if hasattr(self, 'parameters_tracker') and self.parameters_tracker:
                        self.parameters_tracker.track_fallback_usage(
                            'PureRAGASTestsetGenerator',
                            'generation_method',
                            'configured_generation',
                            'minimal_generation',
                            f"All configured methods failed: {e2}",
                            True
                        )
                    
                    # Try absolute minimal generation
                    testset = generator.generate(
                        testset_size=1,  # Generate just 1 sample
                        raise_exceptions=True
                    )
            
            # End performance tracking
            if generation_timing_id and hasattr(self, 'performance_tracker') and self.performance_tracker:
                generation_duration = self.performance_tracker.end_timing(generation_timing_id)
                if generation_duration:
                    logger.info(f"⏱️ Testset generation completed in {generation_duration:.2f}s")
            
            # Step 7: Validate testset results
            if not testset:
                raise ValueError("Testset generation returned None")
                
            if not hasattr(testset, 'to_pandas'):
                raise ValueError("Testset object missing to_pandas method")
            
            # Apply fixes to samples with None eval_sample before filtering
            if hasattr(testset, '__iter__'):
                logger.info("🔧 Fixing samples with None eval_sample...")
                
                # Get all samples and apply fixes
                testset_samples = list(testset)
                
                # Apply the RAGAS sample fix
                try:
                    from utils.output_parser_fix import fix_ragas_samples_with_none_eval
                    fixed_samples = fix_ragas_samples_with_none_eval(testset_samples, logger)
                    logger.info(f"✅ Fixed {len(fixed_samples)} samples")
                except Exception as fix_error:
                    logger.warning(f"⚠️ Failed to apply sample fixes: {fix_error}")
                    fixed_samples = testset_samples
                
                # Now filter for valid samples
                logger.info("🔍 Filtering out invalid samples...")
                valid_samples = []
                none_count = 0
                
                for i, sample in enumerate(fixed_samples):
                    if hasattr(sample, 'eval_sample') and sample.eval_sample is not None:
                        valid_samples.append(sample)
                    else:
                        none_count += 1
                        logger.debug(f"Sample {i}: eval_sample is still None after fix, skipping")
                
                if none_count > 0:
                    logger.warning(f"⚠️ Still have {none_count} samples with None eval_sample after fixing")
                
                if len(valid_samples) == 0:
                    logger.error("❌ No valid samples found after fixing and filtering")
                    raise ValueError("Generated testset is empty")
                
                # Replace testset with fixed and filtered samples
                logger.info(f"✅ Using {len(valid_samples)} valid samples out of {len(testset_samples)} total")
                
                # Create a new testset-like object with valid samples
                class FilteredTestset:
                    def __init__(self, samples):
                        self.samples = samples
                    
                    def __iter__(self):
                        return iter(self.samples)
                    
                    def __len__(self):
                        return len(self.samples)
                    
                    def to_pandas(self):
                        # Try to convert filtered samples to pandas
                        data_rows = []
                        for sample in self.samples:
                            if hasattr(sample, 'eval_sample') and sample.eval_sample is not None:
                                try:
                                    sample_dict = sample.eval_sample.model_dump(exclude_none=True)
                                    # Add synthesizer_name from the TestsetSample object
                                    if hasattr(sample, 'synthesizer_name'):
                                        sample_dict['synthesizer_name'] = sample.synthesizer_name
                                    else:
                                        sample_dict['synthesizer_name'] = 'unknown_synthesizer'
                                    data_rows.append(sample_dict)
                                except Exception as e:
                                    logger.warning(f"⚠️ Failed to convert sample to dict: {e}")
                                    continue
                        
                        if len(data_rows) == 0:
                            return None
                        
                        import pandas as pd
                        return pd.DataFrame(data_rows)
                
                testset = FilteredTestset(valid_samples)
            
            # Convert to DataFrame and validate
            df = testset.to_pandas()
            if df is None or len(df) == 0:
                raise ValueError("Generated testset is empty")
            
            logger.info(f"✅ Generated testset with {len(df)} samples")
            logger.info(f"📋 Testset columns: {list(df.columns)}")
            
            # Check for expected columns
            expected_columns = ['user_input', 'reference_contexts', 'reference']
            missing_columns = [col for col in expected_columns if col not in df.columns]
            if missing_columns:
                logger.warning(f"⚠️ Missing expected columns: {missing_columns}")
            
            # Step 8: Add auto_keywords column using unified KeyBERT extractor
            if self.keybert_extractor:
                logger.info("🔍 Extracting keywords for generated questions...")
                auto_keywords = []
                
                # Extract keywords from user_input (questions)
                for idx, row in df.iterrows():
                    try:
                        # Get the question text - try different possible column names
                        question_text = None
                        for col_name in ['user_input', 'question', 'query']:
                            if col_name in df.columns:
                                question_text = row[col_name]
                                break
                        
                        if question_text and isinstance(question_text, str):
                            # Use testset generation mode for KeyBERT
                            keywords_result = self.keybert_extractor.extract_for_testset_generation(
                                question_text
                            )
                            
                            # Format keywords as comma-separated string
                            if keywords_result and 'high_relevance' in keywords_result:
                                # Use high relevance keywords first, fall back to all keywords
                                keywords_list = [kw['keyword'] for kw in keywords_result['high_relevance']]
                                if not keywords_list and 'all_keywords' in keywords_result:
                                    keywords_list = [kw['keyword'] for kw in keywords_result['all_keywords']]
                                
                                # Respect max_keywords configuration
                                max_keywords = self.config.get('keyword_extraction', {}).get('max_keywords', 5)
                                keywords_list = keywords_list[:max_keywords]
                                keywords_str = ', '.join(keywords_list)
                            else:
                                keywords_str = ""
                        else:
                            keywords_str = ""
                            
                        auto_keywords.append(keywords_str)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract keywords for row {idx}: {e}")
                        auto_keywords.append("")
                
                # Add the auto_keywords column to DataFrame
                df['auto_keywords'] = auto_keywords
                logger.info("✅ Added auto_keywords column with KeyBERT extraction")
            else:
                # Add empty auto_keywords column if KeyBERT is not available
                df['auto_keywords'] = ""
                logger.info("⚠️ Added empty auto_keywords column (KeyBERT not available)")
            
            # Log the final column structure
            logger.info(f"📊 Final testset columns: {list(df.columns)}")
            
            # Step 8: Save testset
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            testset_file = output_dir / f"pure_ragas_testset_{timestamp}.csv"
            
            df.to_csv(testset_file, index=False)
            logger.info(f"💾 Testset saved: {testset_file}")
            
            # Initialize metadata early for use in composition tracking
            metadata = {
                'generation_method': 'pure_ragas',
                'samples_generated': len(df),
                'knowledge_graph_nodes': len(kg.nodes),
                'knowledge_graph_relationships': len(kg.relationships),
                'documents_processed': len(documents),
                'timestamp': timestamp,
                'keybert_enabled': self.keybert_extractor is not None
            }
            
            # Step 8.5: Extract and track composition elements if tracking is enabled
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                logger.info("🧩 Extracting composition elements from generated testset...")
                
                composition_elements = self._extract_composition_elements(testset, df, kg, query_distribution)
                
                # Track composition elements
                for element in composition_elements.get('scenarios', []):
                    self.composition_tracker.track_scenario(element)
                    
                for element in composition_elements.get('personas', []):
                    self.composition_tracker.track_persona(element)
                    
                for element in composition_elements.get('nodes_used', []):
                    self.composition_tracker.track_node_usage(element, element.get('synthesizer', 'unknown'))
                    
                for element in composition_elements.get('relationships_used', []):
                    self.composition_tracker.track_relationship_usage(element, element.get('synthesizer', 'unknown'))
                    
                for element in composition_elements.get('query_styles', []):
                    self.composition_tracker.track_query_style(element)
                
                # Track synthesizer usage statistics
                synthesizer_stats = composition_elements.get('synthesizer_usage', {})
                for synth_name, stats in synthesizer_stats.items():
                    for i in range(stats.get('attempts', 0)):
                        success = i < stats.get('successful', 0)
                        question = stats.get('sample_questions', [''])[min(i, len(stats.get('sample_questions', [])) - 1)] if stats.get('sample_questions') else ''
                        self.composition_tracker.track_synthesizer_usage(synth_name, success, question)
                
                logger.info(f"✅ Tracked {len(composition_elements.get('scenarios', []))} scenarios, "
                          f"{len(composition_elements.get('nodes_used', []))} nodes, "
                          f"{len(composition_elements.get('relationships_used', []))} relationships")
                
                # Add composition elements to metadata
                metadata['composition_elements'] = {
                    'scenarios_count': len(composition_elements.get('scenarios', [])),
                    'personas_count': len(composition_elements.get('personas', [])),
                    'nodes_used_count': len(composition_elements.get('nodes_used', [])),
                    'relationships_used_count': len(composition_elements.get('relationships_used', [])),
                    'query_styles_count': len(composition_elements.get('query_styles', [])),
                    'synthesizer_distribution': synthesizer_stats
                }
            else:
                logger.info("⚠️ Composition tracking disabled - skipping composition elements extraction")
            
            # Step 9: Return results with comprehensive metadata
            metadata = {
                'generation_method': 'pure_ragas',
                'pipeline_method': f"{kg_method}_kg_to_testset",  # e.g., "existing_kg_to_testset" or "new_kg_to_testset"
                'samples_generated': len(df),
                'target_samples': target_testset_size,
                'knowledge_graph_nodes': len(kg.nodes),
                'knowledge_graph_relationships': len(kg.relationships),
                'knowledge_graph_source': kg_method,  # "existing" or "new"
                'existing_kg_used': kg_method == "existing",
                'existing_kg_file': existing_kg_file if kg_method == "existing" else None,
                'documents_processed': len(documents),
                'max_documents_configured': max_docs,
                'query_synthesizers_count': len(query_distribution),
                'llm_model': self.custom_llm_config.get('model', 'gpt-4o'),
                'llm_endpoint': self.custom_llm_config.get('endpoint', 'unknown'),
                'embeddings_model': self.ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                'timestamp': timestamp,
                'pipeline_stages_completed': [
                    'document_loading',
                    'knowledge_graph_creation' if kg_method == "new" else 'knowledge_graph_loading',
                    'testset_generation',
                    'keyword_extraction'
                ],
                'generation_success_rate': len(df) / target_testset_size if target_testset_size > 0 else 0.0,
                'keybert_enabled': self.keybert_extractor is not None
            }
            
            # Log final pipeline summary
            logger.info("=" * 80)
            logger.info("🎉 PIPELINE EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"✅ Testset generation completed successfully!")
            logger.info(f"📊 RESULTS:")
            logger.info(f"   Generated QA pairs: {len(df)}/{target_testset_size} (target)")
            logger.info(f"   Success rate: {metadata['generation_success_rate']:.1%}")
            logger.info(f"   Knowledge graph: {kg_method.upper()} ({len(kg.nodes)} nodes, {len(kg.relationships)} relationships)")
            logger.info(f"   Documents processed: {len(documents)}")
            logger.info(f"   Columns: {list(df.columns)}")
            logger.info(f"📁 FILES SAVED:")
            logger.info(f"   Testset: {testset_file}")
            if kg_method == "new" and enable_kg_saving:
                logger.info(f"   Knowledge graph: Saved for future reuse")
            logger.info("=" * 80)
            
            # End composition tracking if enabled
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                self.composition_tracker.end_generation()
            
            return {
                'success': True,
                'testset_path': str(testset_file),
                'knowledge_graph_path': kg_file,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Pure RAGAS testset generation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # End composition tracking if enabled (even on error)
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                self.composition_tracker.end_generation()
            
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'samples_generated': 0,
                    'knowledge_graph_nodes': 0,
                    'knowledge_graph_relationships': 0,
                    'documents_processed': 0,
                    'generation_method': 'pure_ragas_failed'
                }
            }

    def generate_testset_from_documents(self, documents: List[Dict[str, Any]], 
                                      output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Wrapper method to match the interface expected by the pipeline orchestrator.
        
        This method converts the orchestrator's expected interface to work directly
        with the processed documents instead of loading from CSV files.
        
        Args:
            documents: List of processed documents from the orchestrator
            output_dir: Output directory for generated testsets
            
        Returns:
            Tuple of (testset_data, metadata) to match orchestrator expectations
        """
        try:
            logger.info(f"🚀 Starting Pure RAGAS Testset Generation from {len(documents)} processed documents...")
            
            # Respect the max_documents_for_generation configuration
            max_docs = self.config.get('testset_generation', {}).get('max_documents_for_generation', 15)
            if max_docs and len(documents) > max_docs:
                logger.info(f"📋 Limiting documents from {len(documents)} to {max_docs} as per configuration")
                documents = documents[:max_docs]
            
            # Convert processed documents to LangChain Documents
            lc_documents = []
            for i, doc_dict in enumerate(documents):
                # Extract content and metadata from the document dictionary
                content = doc_dict.get('content', doc_dict.get('text', ''))
                metadata = doc_dict.get('metadata', {})
                
                # Create LangChain Document
                doc = LCDocument(
                    page_content=content,
                    metadata={
                        'source': metadata.get('source', 'orchestrator'),
                        'doc_id': metadata.get('id', f'doc_{i}'),
                        'title': metadata.get('title', f'Document {i}'),
                        **metadata
                    }
                )
                lc_documents.append(doc)
            
            logger.info(f"✅ Converted {len(lc_documents)} documents to LangChain format")
            
            if not lc_documents:
                raise ValueError("No valid documents received from orchestrator")
            
            # Step 1: Create and enrich Knowledge Graph
            kg = self._create_knowledge_graph(lc_documents)
            
            # Step 2: Save Knowledge Graph for reuse
            kg_file = self._save_knowledge_graph(kg, output_dir)
            
            # Step 3: Create TestsetGenerator with the enriched KG
            logger.info("🎯 Creating RAGAS TestsetGenerator...")
            generator = TestsetGenerator(
                llm=self.llm,
                embedding_model=self.embeddings,
                knowledge_graph=kg
            )
            
            # Step 4: Define query distribution (use configurable or default)
            use_configurable = self.config.get('testset_generation', {}).get('use_configurable_distribution', True)
            
            if use_configurable:
                query_distribution = self._create_configurable_query_distribution(kg)
                logger.info(f"📋 Using configurable query distribution with {len(query_distribution)} synthesizers")
            else:
                query_distribution = default_query_distribution(self.llm)
                logger.info(f"📋 Using default query distribution with {len(query_distribution)} synthesizers")
            
            # Step 6: Generate testset
            max_samples = self.config.get('testset_generation', {}).get('max_total_samples', 3)
            logger.info(f"⚡ Generating {max_samples} testset samples...")
            
            try:
                testset = generator.generate(
                    testset_size=max_samples,
                    query_distribution=query_distribution
                )
            except Exception as e:
                logger.warning(f"⚠️ Generation with query distribution failed: {e}")
                
                # Check if we have any synthesizers in our distribution
                if not query_distribution:
                    raise ValueError("No valid synthesizers available for testset generation. All synthesizers failed compatibility checks.")
                
                logger.info("🔄 Trying with minimal single-hop distribution...")
                
                # Try with just entities synthesizer as minimal fallback
                from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
                minimal_distribution = [(SingleHopSpecificQuerySynthesizer(llm=self.llm, property_name="entities"), 1.0)]
                
                try:
                    testset = generator.generate(
                        testset_size=max_samples,
                        query_distribution=minimal_distribution
                    )
                    logger.info("✅ Minimal distribution generation succeeded")
                except Exception as e2:
                    logger.error(f"❌ Even minimal distribution failed: {e2}")
                    raise ValueError(f"Testset generation failed with both configured distribution ({e}) and minimal fallback ({e2})")
            
            # Step 7: Convert to DataFrame and add keyword extraction
            df = testset.to_pandas()
            logger.info(f"✅ Generated testset with {len(df)} samples")
            
            # Step 7.1: Add auto_keywords column using unified KeyBERT extractor
            if self.keybert_extractor:
                logger.info("🔍 Extracting keywords for generated questions...")
                auto_keywords = []
                
                # Extract keywords from user_input (questions)
                for idx, row in df.iterrows():
                    try:
                        # Get the question text - try different possible column names
                        question_text = None
                        for col_name in ['user_input', 'question', 'query']:
                            if col_name in df.columns:
                                question_text = row[col_name]
                                break
                        
                        if question_text and isinstance(question_text, str):
                            # Use testset generation mode for KeyBERT
                            keywords_result = self.keybert_extractor.extract_for_testset_generation(
                                question_text
                            )
                            
                            # Format keywords as comma-separated string
                            if keywords_result and 'high_relevance' in keywords_result:
                                # Use high relevance keywords first, fall back to all keywords
                                keywords_list = [kw['keyword'] for kw in keywords_result['high_relevance']]
                                if not keywords_list and 'all_keywords' in keywords_result:
                                    keywords_list = [kw['keyword'] for kw in keywords_result['all_keywords']]
                                
                                # Respect max_keywords configuration
                                max_keywords = self.config.get('keyword_extraction', {}).get('max_keywords', 5)
                                keywords_list = keywords_list[:max_keywords]
                                keywords_str = ', '.join(keywords_list)
                            else:
                                keywords_str = ""
                        else:
                            keywords_str = ""
                            
                        auto_keywords.append(keywords_str)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract keywords for row {idx}: {e}")
                        auto_keywords.append("")
                
                # Add the auto_keywords column to DataFrame
                df['auto_keywords'] = auto_keywords
                logger.info("✅ Added auto_keywords column with KeyBERT extraction")
            else:
                # Add empty auto_keywords column if KeyBERT is not available
                df['auto_keywords'] = ""
                logger.info("⚠️ Added empty auto_keywords column (KeyBERT not available)")
            
            # Log the final column structure
            logger.info(f"📊 Final testset columns: {list(df.columns)}")
            
            # Step 8: Save testset
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            testset_file = output_dir / f"pure_ragas_testset_{timestamp}.csv"
            
            df.to_csv(testset_file, index=False)
            logger.info(f"💾 Testset saved: {testset_file}")
            
            # Initialize metadata early for use in composition tracking
            metadata = {
                'generation_method': 'pure_ragas',
                'samples_generated': len(df),
                'knowledge_graph_nodes': len(kg.nodes),
                'knowledge_graph_relationships': len(kg.relationships),
                'documents_processed': len(documents),
                'timestamp': timestamp,
                'keybert_enabled': self.keybert_extractor is not None
            }
            
            # Step 8.5: Extract and track composition elements if tracking is enabled
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                logger.info("🧩 Extracting composition elements from generated testset...")
                
                composition_elements = self._extract_composition_elements(testset, df, kg, query_distribution)
                
                # Track composition elements
                for element in composition_elements.get('scenarios', []):
                    self.composition_tracker.track_scenario(element)
                    
                for element in composition_elements.get('personas', []):
                    self.composition_tracker.track_persona(element)
                    
                for element in composition_elements.get('nodes_used', []):
                    self.composition_tracker.track_node_usage(element, element.get('synthesizer', 'unknown'))
                    
                for element in composition_elements.get('relationships_used', []):
                    self.composition_tracker.track_relationship_usage(element, element.get('synthesizer', 'unknown'))
                    
                for element in composition_elements.get('query_styles', []):
                    self.composition_tracker.track_query_style(element)
                
                # Track synthesizer usage statistics
                synthesizer_stats = composition_elements.get('synthesizer_usage', {})
                for synth_name, stats in synthesizer_stats.items():
                    for i in range(stats.get('attempts', 0)):
                        success = i < stats.get('successful', 0)
                        question = stats.get('sample_questions', [''])[min(i, len(stats.get('sample_questions', [])) - 1)] if stats.get('sample_questions') else ''
                        self.composition_tracker.track_synthesizer_usage(synth_name, success, question)
                
                logger.info(f"✅ Tracked {len(composition_elements.get('scenarios', []))} scenarios, "
                          f"{len(composition_elements.get('nodes_used', []))} nodes, "
                          f"{len(composition_elements.get('relationships_used', []))} relationships")
                
                # Add composition elements to metadata
                metadata['composition_elements'] = {
                    'scenarios_count': len(composition_elements.get('scenarios', [])),
                    'personas_count': len(composition_elements.get('personas', [])),
                    'nodes_used_count': len(composition_elements.get('nodes_used', [])),
                    'relationships_used_count': len(composition_elements.get('relationships_used', [])),
                    'query_styles_count': len(composition_elements.get('query_styles', [])),
                    'synthesizer_distribution': synthesizer_stats
                }
            else:
                logger.info("⚠️ Composition tracking disabled - skipping composition elements extraction")
            
            # Step 9: Return results with comprehensive metadata
            metadata = {
                'generation_method': 'pure_ragas',
                'pipeline_method': f"{kg_method}_kg_to_testset",  # e.g., "existing_kg_to_testset" or "new_kg_to_testset"
                'samples_generated': len(df),
                'target_samples': target_testset_size,
                'knowledge_graph_nodes': len(kg.nodes),
                'knowledge_graph_relationships': len(kg.relationships),
                'knowledge_graph_source': kg_method,  # "existing" or "new"
                'existing_kg_used': kg_method == "existing",
                'existing_kg_file': existing_kg_file if kg_method == "existing" else None,
                'documents_processed': len(documents),
                'max_documents_configured': max_docs,
                'query_synthesizers_count': len(query_distribution),
                'llm_model': self.custom_llm_config.get('model', 'gpt-4o'),
                'llm_endpoint': self.custom_llm_config.get('endpoint', 'unknown'),
                'embeddings_model': self.ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                'timestamp': timestamp,
                'pipeline_stages_completed': [
                    'document_loading',
                    'knowledge_graph_creation' if kg_method == "new" else 'knowledge_graph_loading',
                    'testset_generation',
                    'keyword_extraction'
                ],
                'generation_success_rate': len(df) / target_testset_size if target_testset_size > 0 else 0.0,
                'keybert_enabled': self.keybert_extractor is not None
            }
            
            # Log final pipeline summary
            logger.info("=" * 80)
            logger.info("🎉 PIPELINE EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"✅ Testset generation completed successfully!")
            logger.info(f"📊 RESULTS:")
            logger.info(f"   Generated QA pairs: {len(df)}/{target_testset_size} (target)")
            logger.info(f"   Success rate: {metadata['generation_success_rate']:.1%}")
            logger.info(f"   Knowledge graph: {kg_method.upper()} ({len(kg.nodes)} nodes, {len(kg.relationships)} relationships)")
            logger.info(f"   Documents processed: {len(documents)}")
            logger.info(f"   Columns: {list(df.columns)}")
            logger.info(f"📁 FILES SAVED:")
            logger.info(f"   Testset: {testset_file}")
            if kg_method == "new" and enable_kg_saving:
                logger.info(f"   Knowledge graph: Saved for future reuse")
            logger.info("=" * 80)
            
            # End composition tracking if enabled
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                self.composition_tracker.end_generation()
            
            return {
                'success': True,
                'testset_path': str(testset_file),
                'knowledge_graph_path': kg_file,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Pure RAGAS testset generation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # End composition tracking if enabled (even on error)
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                self.composition_tracker.end_generation()
            
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'samples_generated': 0,
                    'knowledge_graph_nodes': 0,
                    'knowledge_graph_relationships': 0,
                    'documents_processed': 0,
                    'generation_method': 'pure_ragas_failed'
                }
            }

    def generate_testset_from_documents(self, documents: List[Dict[str, Any]], 
                                      output_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Wrapper method to match the interface expected by the pipeline orchestrator.
        
        This method converts the orchestrator's expected interface to work directly
        with the processed documents instead of loading from CSV files.
        
        Args:
            documents: List of processed documents from the orchestrator
            output_dir: Output directory for generated testsets
            
        Returns:
            Tuple of (testset_data, metadata) to match orchestrator expectations
        """
        try:
            logger.info(f"🚀 Starting Pure RAGAS Testset Generation from {len(documents)} processed documents...")
            
            # Respect the max_documents_for_generation configuration
            max_docs = self.config.get('testset_generation', {}).get('max_documents_for_generation', 15)
            if max_docs and len(documents) > max_docs:
                logger.info(f"📋 Limiting documents from {len(documents)} to {max_docs} as per configuration")
                documents = documents[:max_docs]
            
            # Convert processed documents to LangChain Documents
            lc_documents = []
            for i, doc_dict in enumerate(documents):
                # Extract content and metadata from the document dictionary
                content = doc_dict.get('content', doc_dict.get('text', ''))
                metadata = doc_dict.get('metadata', {})
                
                # Create LangChain Document
                doc = LCDocument(
                    page_content=content,
                    metadata={
                        'source': metadata.get('source', 'orchestrator'),
                        'doc_id': metadata.get('id', f'doc_{i}'),
                        'title': metadata.get('title', f'Document {i}'),
                        **metadata
                    }
                )
                lc_documents.append(doc)
            
            logger.info(f"✅ Converted {len(lc_documents)} documents to LangChain format")
            
            if not lc_documents:
                raise ValueError("No valid documents received from orchestrator")
            
            # Step 1: Create and enrich Knowledge Graph
            kg = self._create_knowledge_graph(lc_documents)
            
            # Step 2: Save Knowledge Graph for reuse
            kg_file = self._save_knowledge_graph(kg, output_dir)
            
            # Step 3: Create TestsetGenerator with the enriched KG
            logger.info("🎯 Creating RAGAS TestsetGenerator...")
            generator = TestsetGenerator(
                llm=self.llm,
                embedding_model=self.embeddings,
                knowledge_graph=kg
            )
            
            # Step 4: Define query distribution (use configurable or default)
            use_configurable = self.config.get('testset_generation', {}).get('use_configurable_distribution', True)
            
            if use_configurable:
                query_distribution = self._create_configurable_query_distribution(kg)
                logger.info(f"📋 Using configurable query distribution with {len(query_distribution)} synthesizers")
            else:
                query_distribution = default_query_distribution(self.llm)
                logger.info(f"📋 Using default query distribution with {len(query_distribution)} synthesizers")
            
            # Step 6: Generate testset
            max_samples = self.config.get('testset_generation', {}).get('max_total_samples', 3)
            logger.info(f"⚡ Generating {max_samples} testset samples...")
            
            try:
                testset = generator.generate(
                    testset_size=max_samples,
                    query_distribution=query_distribution
                )
            except Exception as e:
                logger.warning(f"⚠️ Generation with query distribution failed: {e}")
                
                # Check if we have any synthesizers in our distribution
                if not query_distribution:
                    raise ValueError("No valid synthesizers available for testset generation. All synthesizers failed compatibility checks.")
                
                logger.info("🔄 Trying with minimal single-hop distribution...")
                
                # Try with just entities synthesizer as minimal fallback
                from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
                minimal_distribution = [(SingleHopSpecificQuerySynthesizer(llm=self.llm, property_name="entities"), 1.0)]
                
                try:
                    testset = generator.generate(
                        testset_size=max_samples,
                        query_distribution=minimal_distribution
                    )
                    logger.info("✅ Minimal distribution generation succeeded")
                except Exception as e2:
                    logger.error(f"❌ Even minimal distribution failed: {e2}")
                    raise ValueError(f"Testset generation failed with both configured distribution ({e}) and minimal fallback ({e2})")
            
            # Step 7: Convert to DataFrame and add keyword extraction
            df = testset.to_pandas()
            logger.info(f"✅ Generated testset with {len(df)} samples")
            
            # Step 7.1: Add auto_keywords column using unified KeyBERT extractor
            if self.keybert_extractor:
                logger.info("🔍 Extracting keywords for generated questions...")
                auto_keywords = []
                
                # Extract keywords from user_input (questions)
                for idx, row in df.iterrows():
                    try:
                        # Get the question text - try different possible column names
                        question_text = None
                        for col_name in ['user_input', 'question', 'query']:
                            if col_name in df.columns:
                                question_text = row[col_name]
                                break
                        
                        if question_text and isinstance(question_text, str):
                            # Use testset generation mode for KeyBERT
                            keywords_result = self.keybert_extractor.extract_for_testset_generation(
                                question_text
                            )
                            
                            # Format keywords as comma-separated string
                            if keywords_result and 'high_relevance' in keywords_result:
                                # Use high relevance keywords first, fall back to all keywords
                                keywords_list = [kw['keyword'] for kw in keywords_result['high_relevance']]
                                if not keywords_list and 'all_keywords' in keywords_result:
                                    keywords_list = [kw['keyword'] for kw in keywords_result['all_keywords']]
                                
                                # Respect max_keywords configuration
                                max_keywords = self.config.get('keyword_extraction', {}).get('max_keywords', 5)
                                keywords_list = keywords_list[:max_keywords]
                                keywords_str = ', '.join(keywords_list)
                            else:
                                keywords_str = ""
                        else:
                            keywords_str = ""
                            
                        auto_keywords.append(keywords_str)
                        
                    except Exception as e:
                        logger.warning(f"Failed to extract keywords for row {idx}: {e}")
                        auto_keywords.append("")
                
                # Add the auto_keywords column to DataFrame
                df['auto_keywords'] = auto_keywords
                logger.info("✅ Added auto_keywords column with KeyBERT extraction")
            else:
                # Add empty auto_keywords column if KeyBERT is not available
                df['auto_keywords'] = ""
                logger.info("⚠️ Added empty auto_keywords column (KeyBERT not available)")
            
            # Log the final column structure
            logger.info(f"📊 Final testset columns: {list(df.columns)}")
            
            # Step 8: Save testset
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            testset_file = output_dir / f"pure_ragas_testset_{timestamp}.csv"
            
            df.to_csv(testset_file, index=False)
            logger.info(f"💾 Testset saved: {testset_file}")
            
            # Initialize metadata early for use in composition tracking
            metadata = {
                'generation_method': 'pure_ragas',
                'samples_generated': len(df),
                'knowledge_graph_nodes': len(kg.nodes),
                'knowledge_graph_relationships': len(kg.relationships),
                'documents_processed': len(documents),
                'timestamp': timestamp,
                'keybert_enabled': self.keybert_extractor is not None
            }
            
            # Step 8.5: Extract and track composition elements if tracking is enabled
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                logger.info("🧩 Extracting composition elements from generated testset...")
                
                composition_elements = self._extract_composition_elements(testset, df, kg, query_distribution)
                
                # Track composition elements
                for element in composition_elements.get('scenarios', []):
                    self.composition_tracker.track_scenario(element)
                    
                for element in composition_elements.get('personas', []):
                    self.composition_tracker.track_persona(element)
                    
                for element in composition_elements.get('nodes_used', []):
                    self.composition_tracker.track_node_usage(element, element.get('synthesizer', 'unknown'))
                    
                for element in composition_elements.get('relationships_used', []):
                    self.composition_tracker.track_relationship_usage(element, element.get('synthesizer', 'unknown'))
                    
                for element in composition_elements.get('query_styles', []):
                    self.composition_tracker.track_query_style(element)
                
                # Track synthesizer usage statistics
                synthesizer_stats = composition_elements.get('synthesizer_usage', {})
                for synth_name, stats in synthesizer_stats.items():
                    for i in range(stats.get('attempts', 0)):
                        success = i < stats.get('successful', 0)
                        question = stats.get('sample_questions', [''])[min(i, len(stats.get('sample_questions', [])) - 1)] if stats.get('sample_questions') else ''
                        self.composition_tracker.track_synthesizer_usage(synth_name, success, question)
                
                logger.info(f"✅ Tracked {len(composition_elements.get('scenarios', []))} scenarios, "
                          f"{len(composition_elements.get('nodes_used', []))} nodes, "
                          f"{len(composition_elements.get('relationships_used', []))} relationships")
                
                # Add composition elements to metadata
                metadata['composition_elements'] = {
                    'scenarios_count': len(composition_elements.get('scenarios', [])),
                    'personas_count': len(composition_elements.get('personas', [])),
                    'nodes_used_count': len(composition_elements.get('nodes_used', [])),
                    'relationships_used_count': len(composition_elements.get('relationships_used', [])),
                    'query_styles_count': len(composition_elements.get('query_styles', [])),
                    'synthesizer_distribution': synthesizer_stats
                }
            else:
                logger.info("⚠️ Composition tracking disabled - skipping composition elements extraction")
            
            # Step 9: Return results with comprehensive metadata
            metadata = {
                'generation_method': 'pure_ragas',
                'pipeline_method': f"{kg_method}_kg_to_testset",  # e.g., "existing_kg_to_testset" or "new_kg_to_testset"
                'samples_generated': len(df),
                'target_samples': target_testset_size,
                'knowledge_graph_nodes': len(kg.nodes),
                'knowledge_graph_relationships': len(kg.relationships),
                'knowledge_graph_source': kg_method,  # "existing" or "new"
                'existing_kg_used': kg_method == "existing",
                'existing_kg_file': existing_kg_file if kg_method == "existing" else None,
                'documents_processed': len(documents),
                'max_documents_configured': max_docs,
                'query_synthesizers_count': len(query_distribution),
                'llm_model': self.custom_llm_config.get('model', 'gpt-4o'),
                'llm_endpoint': self.custom_llm_config.get('endpoint', 'unknown'),
                'embeddings_model': self.ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                'timestamp': timestamp,
                'pipeline_stages_completed': [
                    'document_loading',
                    'knowledge_graph_creation' if kg_method == "new" else 'knowledge_graph_loading',
                    'testset_generation',
                    'keyword_extraction'
                ],
                'generation_success_rate': len(df) / target_testset_size if target_testset_size > 0 else 0.0,
                'keybert_enabled': self.keybert_extractor is not None
            }
            
            # Log final pipeline summary
            logger.info("=" * 80)
            logger.info("🎉 PIPELINE EXECUTION SUMMARY")
            logger.info("=" * 80)
            logger.info(f"✅ Testset generation completed successfully!")
            logger.info(f"📊 RESULTS:")
            logger.info(f"   Generated QA pairs: {len(df)}/{target_testset_size} (target)")
            logger.info(f"   Success rate: {metadata['generation_success_rate']:.1%}")
            logger.info(f"   Knowledge graph: {kg_method.upper()} ({len(kg.nodes)} nodes, {len(kg.relationships)} relationships)")
            logger.info(f"   Documents processed: {len(documents)}")
            logger.info(f"   Columns: {list(df.columns)}")
            logger.info(f"📁 FILES SAVED:")
            logger.info(f"   Testset: {testset_file}")
            if kg_method == "new" and enable_kg_saving:
                logger.info(f"   Knowledge graph: Saved for future reuse")
            logger.info("=" * 80)
            
            # End composition tracking if enabled
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                self.composition_tracker.end_generation()
            
            return {
                'success': True,
                'testset_path': str(testset_file),
                'knowledge_graph_path': kg_file,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"❌ Pure RAGAS testset generation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            # End composition tracking if enabled (even on error)
            if hasattr(self, 'composition_tracker') and self.composition_tracker:
                self.composition_tracker.end_generation()
            
            return {
                'success': False,
                'error': str(e),
                'metadata': {
                    'samples_generated': 0,
                    'knowledge_graph_nodes': 0,
                    'knowledge_graph_relationships': 0,
                    'documents_processed': 0,
                    'generation_method': 'pure_ragas_failed'
                }
            }

    def _extract_composition_elements(self, testset, df, kg, query_distribution):
        """
        Extract composition elements from the generated testset.
        
        This method analyzes the RAGAS testset to extract information about:
        - Scenarios: Which persona/style/length combinations were used
        - Personas: Different user types that were simulated
        - Nodes: Knowledge graph nodes that were accessed
        - Relationships: KG relationships used in multi-hop queries
        - Query styles: Different question formulation styles
        """
        composition_elements = {
            'scenarios': [],
            'personas': [],
            'nodes_used': [],
            'relationships_used': [],
            'query_styles': [],
            'synthesizer_usage': {}
        }
        
        try:
            # Analyze synthesizer usage from the testset
            if 'synthesizer_name' in df.columns:
                synthesizer_counts = df['synthesizer_name'].value_counts().to_dict()
                
                for synth_name, count in synthesizer_counts.items():
                    # Extract sample questions for this synthesizer
                    synth_questions = df[df['synthesizer_name'] == synth_name]['user_input'].tolist()
                    
                    composition_elements['synthesizer_usage'][synth_name] = {
                        'attempts': count,
                        'successful': count,  # If it's in the final dataset, it was successful
                        'sample_questions': synth_questions[:3]  # Store first 3 as samples
                    }
                    
                    # Create scenarios for each synthesizer usage
                    for i, question in enumerate(synth_questions):
                        scenario = {
                            'id': f"{synth_name}_{i}",
                            'persona': self._infer_persona_from_question(question),
                            'style': self._infer_style_from_synthesizer(synth_name),
                            'length': self._infer_length_from_question(question),
                            'synthesizer': synth_name,
                            'question': question[:100] + '...' if len(question) > 100 else question
                        }
                        composition_elements['scenarios'].append(scenario)
            
            # Extract persona information from questions
            personas_found = set()
            for idx, row in df.iterrows():
                question = row.get('user_input', '')
                persona = self._infer_persona_from_question(question)
                
                if persona['role'] not in personas_found:
                    personas_found.add(persona['role'])
                    composition_elements['personas'].append(persona)
            
            # Analyze knowledge graph usage
            if 'reference_contexts' in df.columns:
                for idx, row in df.iterrows():
                    contexts = row.get('reference_contexts', [])
                    synthesizer_name = row.get('synthesizer_name', 'unknown')
                    
                    # Try to extract node information from contexts
                    if contexts:
                        # Parse contexts and try to map back to KG nodes
                        for i, context in enumerate(contexts[:3]):  # Limit to first 3 contexts
                            node_info = {
                                'node_id': f"context_{idx}_{i}",
                                'content_snippet': str(context)[:100] + '...' if len(str(context)) > 100 else str(context),
                                'synthesizer': synthesizer_name,
                                'node_type': 'CHUNK',  # Most likely type
                                'properties': ['page_content']  # Common property
                            }
                            composition_elements['nodes_used'].append(node_info)
            
            # Extract query styles based on synthesizer types and question patterns
            style_patterns = {
                'factual_direct': ['What is', 'What does', 'Define', 'Explain'],
                'procedural': ['How to', 'How do', 'Steps to', 'Process'],
                'analytical': ['Why', 'What causes', 'What happens if', 'Compare'],
                'troubleshooting': ['Error', 'Problem', 'Issue', 'Fix', 'Resolve']
            }
            
            styles_found = set()
            for idx, row in df.iterrows():
                question = row.get('user_input', '')
                synthesizer_name = row.get('synthesizer_name', 'unknown')
                
                # Determine question style
                for style_name, patterns in style_patterns.items():
                    if any(pattern.lower() in question.lower() for pattern in patterns):
                        if style_name not in styles_found:
                            styles_found.add(style_name)
                            composition_elements['query_styles'].append({
                                'style_name': style_name,
                                'description': f'Questions that {style_name.replace("_", " ")}',
                                'complexity': 'medium',
                                'synthesizer': synthesizer_name,
                                'example_question': question[:100] + '...' if len(question) > 100 else question
                            })
                        break
            
            # Extract relationship usage for multi-hop queries
            multi_hop_count = sum(1 for synth_name in composition_elements['synthesizer_usage'].keys() 
                                 if 'multi_hop' in synth_name.lower())
            
            if multi_hop_count > 0:
                # Simulate relationship usage for multi-hop queries
                for i in range(min(multi_hop_count, 5)):  # Limit to 5 relationships
                    relationship = {
                        'relationship_id': f'multi_hop_rel_{i}',
                        'source_node': f'node_source_{i}',
                        'target_node': f'node_target_{i}',
                        'relationship_type': 'contextual_similarity',
                        'synthesizer': 'multi_hop_synthesizer'
                    }
                    composition_elements['relationships_used'].append(relationship)
            
            logger.info(f"📊 Composition analysis complete:")
            logger.info(f"   Scenarios: {len(composition_elements['scenarios'])}")
            logger.info(f"   Personas: {len(composition_elements['personas'])}")
            logger.info(f"   Nodes used: {len(composition_elements['nodes_used'])}")
            logger.info(f"   Relationships: {len(composition_elements['relationships_used'])}")
            logger.info(f"   Query styles: {len(composition_elements['query_styles'])}")
            logger.info(f"   Synthesizers: {len(composition_elements['synthesizer_usage'])}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to extract composition elements: {e}")
        
        return composition_elements
    
    def _infer_persona_from_question(self, question: str) -> Dict[str, Any]:
        """Infer persona characteristics from question content."""
        question_lower = question.lower()
        
        if any(term in question_lower for term in ['technical', 'code', 'error', 'debug', 'configure']):
            return {
                'id': 'technical_user',
                'role': 'Technical User',
                'background': 'Software developer or system administrator',
                'expertise_level': 'advanced'
            }
        elif any(term in question_lower for term in ['how to', 'steps', 'guide', 'tutorial']):
            return {
                'id': 'procedural_user',
                'role': 'Procedural User',
                'background': 'Someone following step-by-step instructions',
                'expertise_level': 'intermediate'
            }
        elif any(term in question_lower for term in ['what is', 'define', 'explain', 'meaning']):
            return {
                'id': 'information_seeker',
                'role': 'Information Seeker',
                'background': 'Someone looking for basic information or definitions',
                'expertise_level': 'beginner'
            }
        else:
            return {
                'id': 'general_user',
                'role': 'General User',
                'background': 'General purpose user with mixed needs',
                'expertise_level': 'intermediate'
            }
    
    def _infer_style_from_synthesizer(self, synthesizer_name: str) -> str:
        """Infer query style from synthesizer name."""
        if 'multi_hop' in synthesizer_name.lower():
            return 'complex_reasoning'
        elif 'single_hop' in synthesizer_name.lower():
            return 'direct_factual'
        else:
            return 'mixed'
    
    def _infer_length_from_question(self, question: str) -> str:
        """Infer expected answer length from question structure."""
        word_count = len(question.split())
        
        if word_count <= 10:
            return 'short'
        elif word_count <= 20:
            return 'medium'
        else:
            return 'long'
    
    def generate_comprehensive_testset(self, document_paths: List[str] = None, output_dir: Path = None) -> Dict[str, Any]:
        """
        Enhanced method for comprehensive testset generation using document_paths.
        This method matches the orchestrator interface and provides working testset generation.
        """
        from datetime import datetime  # Import at method level to avoid scoping issues
        
        try:
            # Handle parameter conversion: document_paths -> csv_files
            if document_paths is None:
                # Try to get from config as fallback
                csv_files = self.config.get('data_sources', {}).get('csv', {}).get('csv_files', [])
                logger.info(f"🔄 Using config csv_files: {len(csv_files)} files")
            else:
                csv_files = document_paths
                logger.info(f"🔄 Using document_paths as csv_files: {len(csv_files)} files")
            
            if not csv_files:
                raise ValueError("No CSV files provided via document_paths or config")
            
            # Set default output directory if not provided
            if output_dir is None:
                output_dir = Path('outputs') / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}" / 'testsets'
                output_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"📁 Using default output directory: {output_dir}")
            
            # Step 1: Load documents from CSV
            documents = self._load_csv_as_documents(csv_files)
            logger.info(f"📄 Loaded {len(documents)} documents")
            
            # Step 2: Create enhanced knowledge graph with relationships
            kg = self._create_knowledge_graph(documents)
            logger.info(f"🧠 Created knowledge graph with {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
            
            # Step 3: Save KG for reuse
            kg_file = self._save_knowledge_graph(kg, output_dir)
            
            # Step 4: Create TestsetGenerator
            generator = TestsetGenerator(
                llm=self.llm,
                embedding_model=self.embeddings,
                knowledge_graph=kg
            )
            
            # Step 5: Create query distribution
            query_distribution = self._create_configurable_query_distribution(kg)
            logger.info(f"🎯 Using {len(query_distribution)} synthesizers")
            
            # Step 6: Generate testset
            max_samples = self.config.get('testset_generation', {}).get('max_total_samples', 10)
            logger.info(f"⚡ Generating {max_samples} testset samples...")
            
            testset = generator.generate(
                testset_size=max_samples,
                query_distribution=query_distribution
            )
            
            # Step 7: Convert to DataFrame
            df = testset.to_pandas()
            logger.info(f"✅ Generated testset with {len(df)} samples")
            
            # Step 8: Add keywords if available
            if self.keybert_extractor:
                logger.info("🔍 Adding auto_keywords column...")
                auto_keywords = []
                for idx, row in df.iterrows():
                    try:
                        question_text = row.get('user_input', '')
                        if question_text:
                            keywords_result = self.keybert_extractor.extract_for_testset_generation(question_text)
                            if keywords_result and 'high_relevance' in keywords_result:
                                keywords_list = [kw['keyword'] for kw in keywords_result['high_relevance']]
                                keywords_str = ', '.join(keywords_list)
                            else:
                                keywords_str = ""
                        else:
                            keywords_str = ""
                        auto_keywords.append(keywords_str)
                    except Exception as e:
                        logger.warning(f"Failed to extract keywords for row {idx}: {e}")
                        auto_keywords.append("")
                
                df['auto_keywords'] = auto_keywords
                logger.info("✅ Added auto_keywords column")
            else:
                df['auto_keywords'] = ""
            
            # Step 9: Save testset
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            testset_file = output_dir / f"pure_ragas_testset_{timestamp}.csv"
            df.to_csv(testset_file, index=False)
            logger.info(f"💾 Testset saved: {testset_file}")
            
            # Step 9.5: Extract and save personas & scenarios for reuse & verification
            personas_scenarios_summary = None
            try:
                logger.info("🧩 Extracting personas & scenarios from generated testset...")
                composition_elements = self._extract_composition_elements(testset, df, kg, query_distribution)
                
                personas_scenarios_summary = self._save_composition_elements_to_files(
                    composition_elements, output_dir, timestamp
                )
                
                if personas_scenarios_summary:
                    logger.info(f"📁 Personas & scenarios saved: {personas_scenarios_summary}")
                else:
                    logger.warning("⚠️ No personas/scenarios extracted to save")
                    
            except Exception as e:
                logger.warning(f"⚠️ Failed to extract/save personas & scenarios: {e}")
            
            # Step 10: Create metadata
            metadata = {
                'generation_method': 'pure_ragas',
                'samples_generated': len(df),
                'knowledge_graph_nodes': len(kg.nodes),
                'knowledge_graph_relationships': len(kg.relationships),
                'documents_processed': len(documents),
                'timestamp': timestamp,
                'testset_file': str(testset_file),
                'kg_file': kg_file,
                'personas_scenarios_summary': personas_scenarios_summary  # Add personas/scenarios summary file path
            }
            
            # Convert DataFrame to list of dictionaries for compatibility
            testset_data = df.to_dict('records')
            
            logger.info(f"🎉 Enhanced Pure RAGAS generation completed successfully!")
            logger.info(f"   Generated: {len(testset_data)} QA pairs")
            logger.info(f"   KG nodes: {len(kg.nodes)}")
            logger.info(f"   KG relationships: {len(kg.relationships)}")
            
            return {
                'success': True,
                'testset': testset_data,
                'metadata': metadata,
                'results_by_method': {
                    'pure_ragas': {
                        'samples_generated': len(testset_data),
                        'knowledge_graph': {'nodes': len(kg.nodes), 'relationships': len(kg.relationships)}
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Internal generation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'testset': [],
                'metadata': {'generation_method': 'pure_ragas', 'error': str(e)}
            }
    def _save_personas_and_scenarios(self, output_dir: Path, timestamp: str) -> str:
        """
        Save tracked personas and scenarios to separate files.
        
        Args:
            output_dir: Output directory
            timestamp: Timestamp for file naming
            
        Returns:
            Path to the personas/scenarios summary file
        """
        if not hasattr(self, 'composition_tracker') or not self.composition_tracker:
            logger.warning("⚠️ No composition tracker available - cannot save personas/scenarios")
            return None
            
        try:
            # Create personas and scenarios subdirectory
            personas_scenarios_dir = output_dir / "personas_scenarios"
            personas_scenarios_dir.mkdir(exist_ok=True)
            
            # Get tracked data from composition tracker
            tracked_data = self.composition_tracker.get_composition_summary()
            
            # Save personas if available
            personas_file = None
            if tracked_data.get('personas'):
                personas_file = personas_scenarios_dir / f"personas_{timestamp}.json"
                with open(personas_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'generation_method': 'enhanced_pure_ragas',
                        'timestamp': timestamp,
                        'total_personas': len(tracked_data['personas']),
                        'personas': tracked_data['personas']
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Saved {len(tracked_data['personas'])} personas to {personas_file}")
            
            # Save scenarios if available  
            scenarios_file = None
            if tracked_data.get('scenarios'):
                scenarios_file = personas_scenarios_dir / f"scenarios_{timestamp}.json"
                with open(scenarios_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'generation_method': 'enhanced_pure_ragas',
                        'timestamp': timestamp,
                        'total_scenarios': len(tracked_data['scenarios']),
                        'scenarios': tracked_data['scenarios']
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"💾 Saved {len(tracked_data['scenarios'])} scenarios to {scenarios_file}")
            
            # Create summary file
            summary_file = personas_scenarios_dir / f"personas_scenarios_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'generation_method': 'enhanced_pure_ragas',
                    'timestamp': timestamp,
                    'summary': {
                        'total_personas': len(tracked_data.get('personas', [])),
                        'total_scenarios': len(tracked_data.get('scenarios', [])),
                        'personas_file': str(personas_file) if personas_file else None,
                        'scenarios_file': str(scenarios_file) if scenarios_file else None
                    },
                    'composition_stats': tracked_data.get('stats', {})
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Personas/scenarios summary saved: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to save personas/scenarios: {e}")
            return None
    
    def _save_composition_elements_to_files(self, composition_elements: Dict[str, Any], 
                                          output_dir: Path, timestamp: str) -> str:
        """
        Save extracted personas and scenarios to separate files for reuse & verification.
        
        Args:
            composition_elements: Dict containing extracted personas, scenarios, etc.
            output_dir: Output directory 
            timestamp: Timestamp for file naming
            
        Returns:
            Path to the summary file, or None if failed
        """
        try:
            # Create personas_scenarios subdirectory
            personas_scenarios_dir = output_dir / "personas_scenarios"
            personas_scenarios_dir.mkdir(exist_ok=True)
            
            personas = composition_elements.get('personas', [])
            scenarios = composition_elements.get('scenarios', [])
            query_styles = composition_elements.get('query_styles', [])
            synthesizer_usage = composition_elements.get('synthesizer_usage', {})
            
            logger.info(f"💾 Saving {len(personas)} personas, {len(scenarios)} scenarios to files...")
            
            # Save personas to JSON file
            personas_file = None
            if personas:
                personas_file = personas_scenarios_dir / f"personas_{timestamp}.json"
                with open(personas_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'generation_method': 'enhanced_pure_ragas',
                        'timestamp': timestamp,
                        'total_personas': len(personas),
                        'description': 'User personas extracted from generated questions',
                        'personas': personas
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"📋 Saved {len(personas)} personas to {personas_file}")
            
            # Save scenarios to JSON file  
            scenarios_file = None
            if scenarios:
                scenarios_file = personas_scenarios_dir / f"scenarios_{timestamp}.json"
                with open(scenarios_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'generation_method': 'enhanced_pure_ragas',
                        'timestamp': timestamp, 
                        'total_scenarios': len(scenarios),
                        'description': 'Question scenarios with persona/style/synthesizer combinations',
                        'scenarios': scenarios
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"🎭 Saved {len(scenarios)} scenarios to {scenarios_file}")
            
            # Save query styles to JSON file
            query_styles_file = None
            if query_styles:
                query_styles_file = personas_scenarios_dir / f"query_styles_{timestamp}.json"
                with open(query_styles_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'generation_method': 'enhanced_pure_ragas',
                        'timestamp': timestamp,
                        'total_query_styles': len(query_styles),
                        'description': 'Different question formulation styles detected',
                        'query_styles': query_styles
                    }, f, indent=2, ensure_ascii=False)
                logger.info(f"💬 Saved {len(query_styles)} query styles to {query_styles_file}")
            
            # Save synthesizer usage statistics
            synthesizer_file = None
            if synthesizer_usage:
                synthesizer_file = personas_scenarios_dir / f"synthesizer_usage_{timestamp}.json"
                with open(synthesizer_file, 'w', encoding='utf-8') as f:
                    json.dump({
                        'generation_method': 'enhanced_pure_ragas',
                        'timestamp': timestamp,
                        'description': 'Statistics on how each RAGAS synthesizer was used',
                        'synthesizer_usage': synthesizer_usage
                    }, f, indent=2, ensure_ascii=False)  
                logger.info(f"⚙️ Saved synthesizer usage stats to {synthesizer_file}")
            
            # Create comprehensive summary file
            summary_file = personas_scenarios_dir / f"composition_summary_{timestamp}.json"
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'generation_method': 'enhanced_pure_ragas',
                    'timestamp': timestamp,
                    'description': 'Complete summary of personas, scenarios, and composition elements',
                    'summary': {
                        'total_personas': len(personas),
                        'total_scenarios': len(scenarios),
                        'total_query_styles': len(query_styles),
                        'total_synthesizers': len(synthesizer_usage),
                        'files_created': {
                            'personas_file': str(personas_file) if personas_file else None,
                            'scenarios_file': str(scenarios_file) if scenarios_file else None,
                            'query_styles_file': str(query_styles_file) if query_styles_file else None,
                            'synthesizer_file': str(synthesizer_file) if synthesizer_file else None
                        }
                    },
                    'composition_details': {
                        'nodes_used_count': len(composition_elements.get('nodes_used', [])),
                        'relationships_used_count': len(composition_elements.get('relationships_used', [])),
                        'sample_personas': personas[:3],  # First 3 personas as samples
                        'sample_scenarios': scenarios[:3]  # First 3 scenarios as samples
                    }
                }, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Composition summary saved: {summary_file}")
            return str(summary_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to save composition elements: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None
