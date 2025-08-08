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
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

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

logger = logging.getLogger(__name__)


class PureRAGASTestsetGenerator:
    """
    Pure RAGAS Testset Generator using the official RAGAS documentation pattern.
    
    This generator follows the exact pattern from the RAGAS documentation:
    1. Load documents
    2. Create Knowledge Graph
    3. Apply transforms to enrich KG
    4. Generate testset using TestsetGenerator
    5. Save knowledge graph for reuse
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the Pure RAGAS Testset Generator.
        
        Args:
            config: Pipeline configuration dictionary
        """
        self.config = config
        self.ragas_config = config.get('testset_generation', {}).get('ragas_config', {})
        self.custom_llm_config = self.ragas_config.get('custom_llm', {})
        
        # Initialize LLM and embeddings
        self.llm = self._setup_llm()
        self.embeddings = self._setup_embeddings()
        
        logger.info("✅ Pure RAGAS Testset Generator initialized successfully")
        logger.info(f"🤖 LLM: {self.custom_llm_config.get('model', 'gpt-4o')}")
        logger.info(f"🔗 Endpoint: {self.custom_llm_config.get('endpoint', 'N/A')}")
        logger.info(f"📊 Embeddings: {self.ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2')}")
    
    def _setup_llm(self) -> LangchainLLMWrapper:
        """Setup custom LLM following RAGAS documentation pattern."""
        try:
            # Create ChatOpenAI with custom endpoint (compatible with 4o)
            llm = ChatOpenAI(
                base_url=self.custom_llm_config.get('endpoint', ''),
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
                
                # Limit to max documents as configured
                max_docs = self.config.get('testset_generation', {}).get('max_documents_for_generation', 3)
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
        """
        Create Knowledge Graph following RAGAS documentation pattern.
        
        Args:
            documents: List of LangChain documents
            
        Returns:
            Enriched Knowledge Graph
        """
        logger.info("🧠 Creating Knowledge Graph...")
        
        # Step 1: Create empty knowledge graph
        kg = KnowledgeGraph()
        logger.info(f"📊 Initial KG: {kg}")
        
        # Step 2: Add documents to knowledge graph as nodes
        for i, doc in enumerate(documents):
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata
                }
            )
            kg.nodes.append(node)
        
        logger.info(f"📊 KG after adding documents: {kg}")
        
        # Step 3: Apply default transforms to enrich the knowledge graph
        logger.info("🔧 Applying transforms to enrich Knowledge Graph...")
        
        try:
            trans = default_transforms(
                documents=documents,
                llm=self.llm,
                embedding_model=self.embeddings
            )
            
            apply_transforms(kg, trans)
            logger.info(f"✅ KG after transforms: {kg}")
            
        except Exception as e:
            logger.warning(f"⚠️ Transform failed, using basic KG: {e}")
            # Continue with basic KG if transforms fail
        
        return kg
    
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
    
    def generate_comprehensive_testset(self, csv_files: List[str], output_dir: Path) -> Dict[str, Any]:
        """
        Generate comprehensive testset using proper RAGAS following documentation pattern.
        
        Args:
            csv_files: List of CSV file paths
            output_dir: Output directory for testset and metadata
            
        Returns:
            Dictionary with generation results
        """
        try:
            logger.info("🚀 Starting Pure RAGAS Testset Generation...")
            
            # Step 1: Load documents
            documents = self._load_csv_as_documents(csv_files)
            if not documents:
                raise ValueError("No valid documents loaded")
            
            # Step 2: Create and enrich Knowledge Graph
            kg = self._create_knowledge_graph(documents)
            
            # Step 3: Save Knowledge Graph for reuse
            kg_file = self._save_knowledge_graph(kg, output_dir)
            
            # Step 4: Create TestsetGenerator with the enriched KG
            logger.info("🎯 Creating RAGAS TestsetGenerator...")
            generator = TestsetGenerator(
                llm=self.llm,
                embedding_model=self.embeddings,
                knowledge_graph=kg
            )
            
            # Step 5: Define query distribution
            query_distribution = default_query_distribution(self.llm)
            logger.info(f"📋 Query distribution: {len(query_distribution)} synthesizers")
            
            # Step 6: Generate testset
            max_samples = self.config.get('testset_generation', {}).get('max_total_samples', 3)
            logger.info(f"⚡ Generating {max_samples} testset samples...")
            
            testset = generator.generate(
                testset_size=max_samples,
                query_distribution=query_distribution
            )
            
            # Step 7: Convert to DataFrame and save
            df = testset.to_pandas()
            logger.info(f"✅ Generated testset with {len(df)} samples")
            
            # Step 8: Save testset
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            testset_file = output_dir / f"pure_ragas_testset_{timestamp}.csv"
            
            df.to_csv(testset_file, index=False)
            logger.info(f"💾 Testset saved: {testset_file}")
            
            # Step 9: Return results
            return {
                'success': True,
                'testset_path': str(testset_file),
                'knowledge_graph_path': kg_file,
                'metadata': {
                    'samples_generated': len(df),
                    'knowledge_graph_nodes': len(kg.nodes),
                    'knowledge_graph_relationships': len(kg.relationships),
                    'documents_processed': len(documents),
                    'generation_method': 'pure_ragas',
                    'llm_model': self.custom_llm_config.get('model', 'gpt-4o'),
                    'embeddings_model': self.ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2'),
                    'timestamp': timestamp
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Pure RAGAS testset generation failed: {e}")
            import traceback
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
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


def main():
    """Test the Pure RAGAS generator directly."""
    import sys
    sys.path.append('/data/yy/domain-specific-llm-eval/eval-pipeline')
    
    import yaml
    from pathlib import Path
    
    # Load config
    config_path = Path('/data/yy/domain-specific-llm-eval/eval-pipeline/config/pipeline_config.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize generator
    generator = PureRAGASTestsetGenerator(config)
    
    # Get CSV files
    csv_files = config.get('data_sources', {}).get('csv', {}).get('csv_files', [])
    output_dir = Path('/data/yy/domain-specific-llm-eval/eval-pipeline/outputs/testsets')
    
    # Generate testset
    results = generator.generate_comprehensive_testset(csv_files, output_dir)
    
    if results['success']:
        print(f"✅ Success! Generated {results['metadata']['samples_generated']} samples")
        print(f"📄 Testset: {results['testset_path']}")
        print(f"🧠 Knowledge Graph: {results['knowledge_graph_path']}")
    else:
        print(f"❌ Failed: {results['error']}")


if __name__ == "__main__":
    main()
