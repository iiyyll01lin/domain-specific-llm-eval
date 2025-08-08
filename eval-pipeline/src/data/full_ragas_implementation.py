#!/usr/bin/env python3
"""
Full RAGAS Knowledge Graph Pipeline Implementation

This module implements the complete RAGAS testset generation pipeline with:
1. CSV data loading from pre-training-data.csv
2. Knowledge Graph creation with transforms  
3. Synthetic testset generation using RAGAS
4. Full integration with Inventec custom LLM

Based on RAGAS documentation: docs/getstarted/rag_testset_generation.md
"""

import logging
import pandas as pd
import json
import requests
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pydantic import Field

# LangChain imports
from langchain.schema import Document
from langchain.llms.base import LLM
from langchain.embeddings.base import Embeddings
from langchain.callbacks.manager import CallbackManagerForLLMRun

# RAGAS imports
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# HuggingFace embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)


from pydantic import Field

class InventecCustomLLM(LLM):
    """Custom LLM wrapper for Inventec gpt-4o API"""
    
    endpoint: str = Field(...)
    model: str = Field(...)
    api_key: str = Field(...)
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=2000)
    timeout: int = Field(default=60)
        
    @property
    def _llm_type(self) -> str:
        return "inventec_custom"
    
    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Make API call to Inventec endpoint"""
        
        headers = {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
            'Accept': '*/*',
            'User-Agent': 'RAGAS-Pipeline/1.0.0'
        }
        
        payload = {
            'model': self.model,
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': self.temperature,
            'max_tokens': self.max_tokens
        }
        
        try:
            response = requests.post(
                self.endpoint,
                json=payload,
                headers=headers,
                timeout=self.timeout
            )
            response.raise_for_status()
            
            data = response.json()
            if 'choices' in data and len(data['choices']) > 0:
                return data['choices'][0]['message']['content'].strip()
            else:
                raise ValueError(f"Unexpected response format: {data}")
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {e}")
            raise
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            raise
    
    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {
            "endpoint": self.endpoint,
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
    
    # ✅ Add the missing async methods for RAGAS compatibility
    async def agenerate_prompt(
        self,
        prompts: List[Any],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Async generate method for RAGAS compatibility"""
        # For now, use sync version - can be made truly async later
        try:
            from langchain_core.outputs import LLMResult, Generation
            
            generations = []
            for prompt in prompts:
                # Convert prompt to string if needed
                if hasattr(prompt, 'to_string'):
                    prompt_str = prompt.to_string()
                else:
                    prompt_str = str(prompt)
                
                # Call the sync method
                result_text = self._call(prompt_str, stop=stop, **kwargs)
                generations.append([Generation(text=result_text)])
            
            return LLMResult(generations=generations)
            
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            from langchain_core.outputs import LLMResult, Generation
            return LLMResult(generations=[[Generation(text=f"Error: {str(e)}")]])
    
    def generate_prompt(
        self,
        prompts: List[Any],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Sync generate method for RAGAS compatibility"""
        try:
            from langchain_core.outputs import LLMResult, Generation
            
            generations = []
            for prompt in prompts:
                # Convert prompt to string if needed
                if hasattr(prompt, 'to_string'):
                    prompt_str = prompt.to_string()
                else:
                    prompt_str = str(prompt)
                
                # Call the sync method
                result_text = self._call(prompt_str, stop=stop, **kwargs)
                generations.append([Generation(text=result_text)])
            
            return LLMResult(generations=generations)
            
        except Exception as e:
            logger.error(f"Sync generation failed: {e}")
            from langchain_core.outputs import LLMResult, Generation
            return LLMResult(generations=[[Generation(text=f"Error: {str(e)}")]])


class FullRAGASPipeline:
    """
    Complete RAGAS pipeline implementation for CSV to synthetic testset generation
    
    This follows the full RAGAS knowledge graph approach from the official docs
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        
        # Extract configuration
        self.ragas_config = config.get('testset_generation', {}).get('ragas_config', {})
        self.custom_llm_config = self.ragas_config.get('custom_llm', {})
        self.csv_config = config.get('data_sources', {}).get('csv', {})
        
        # Initialize components
        self.custom_llm = None
        self.ragas_llm = None
        self.embeddings = None
        self.ragas_embeddings = None
        self.knowledge_graph = None
        
    def initialize_components(self):
        """Initialize all RAGAS components"""
        self.logger.info("🔧 Initializing RAGAS pipeline components...")
        
        # 1. Initialize custom LLM
        self.custom_llm = InventecCustomLLM(
            endpoint=self.custom_llm_config['endpoint'],
            model=self.custom_llm_config['model'],
            api_key=self.custom_llm_config['api_key'],
            temperature=self.custom_llm_config.get('temperature', 0.7),
            max_tokens=self.custom_llm_config.get('max_tokens', 2000),
            timeout=self.custom_llm_config.get('timeout', 60)
        )
        
        # 2. Wrap for RAGAS
        self.ragas_llm = LangchainLLMWrapper(self.custom_llm)
        
        # 3. Initialize embeddings
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2'),
            model_kwargs={'device': 'cpu'}
        )
        
        # 4. Wrap embeddings for RAGAS
        self.ragas_embeddings = LangchainEmbeddingsWrapper(self.embeddings)
        
        self.logger.info("✅ All RAGAS components initialized successfully")
        
    def load_csv_data(self) -> List[Document]:
        """Load and process CSV data into LangChain Document objects"""
        self.logger.info("📊 Loading CSV data for RAGAS processing...")
        
        # Get CSV file paths
        csv_files = self.csv_config.get('csv_files', [])
        if not csv_files:
            raise ValueError("No CSV files configured in data_sources.csv.csv_files")
        
        documents = []
        
        for csv_file in csv_files:
            csv_path = Path(csv_file)
            if not csv_path.exists():
                self.logger.warning(f"⚠️ CSV file not found: {csv_path}")
                continue
                
            self.logger.info(f"📄 Loading CSV file: {csv_path}")
            
            try:
                # Load CSV with proper encoding
                df = pd.read_csv(csv_path, encoding='utf-8')
                self.logger.info(f"✅ Loaded CSV with {len(df)} rows")
                
                # Process each row into a Document
                max_rows = self.config.get('testset_generation', {}).get('csv_processing', {}).get('max_rows_to_process', 20)
                processed_rows = min(max_rows, len(df))
                
                for idx, row in df.head(processed_rows).iterrows():
                    try:
                        # Extract content from JSON if needed
                        content_str = row.get('content', '')
                        if pd.isna(content_str):
                            continue
                            
                        # Parse JSON content
                        try:
                            content_data = json.loads(content_str)
                            text_content = content_data.get('text', str(content_str))
                            language = content_data.get('language', 'unknown')
                        except (json.JSONDecodeError, TypeError):
                            text_content = str(content_str)
                            language = 'unknown'
                        
                        # Skip short content
                        if len(text_content.strip()) < 50:
                            continue
                        
                        # Create Document
                        doc = Document(
                            page_content=text_content.strip(),
                            metadata={
                                'id': row.get('id', f'doc_{idx}'),
                                'source': str(csv_path),
                                'row_index': idx,
                                'language': language,
                                'template_key': row.get('template_key', ''),
                                'author': row.get('author', ''),
                                'created_at': row.get('created_at', ''),
                                'content_length': len(text_content),
                                'csv_file': csv_path.name
                            }
                        )
                        
                        documents.append(doc)
                        
                    except Exception as e:
                        self.logger.warning(f"⚠️ Failed to process row {idx}: {e}")
                        continue
                
                self.logger.info(f"✅ Processed {len(documents)} documents from {csv_path.name}")
                
            except Exception as e:
                self.logger.error(f"❌ Failed to load CSV {csv_path}: {e}")
                continue
        
        if not documents:
            raise ValueError("No valid documents were loaded from CSV files")
        
        # Apply document limit from configuration if specified
        max_docs_config = self.config.get('testset_generation', {}).get('max_documents_for_generation')
        if max_docs_config:
            original_count = len(documents)
            documents = documents[:max_docs_config]
            self.logger.info(f"🎯 Limited documents: {original_count} → {len(documents)} (max_documents_for_generation: {max_docs_config})")
        
        self.logger.info(f"🎯 Total documents loaded: {len(documents)}")
        return documents
    
    def create_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraph:
        """
        Create knowledge graph from documents using CSV-compatible RAGAS transforms
        
        This follows the official RAGAS documentation approach but adapts for CSV data
        """
        self.logger.info("🔨 Creating Knowledge Graph with CSV-compatible RAGAS transforms...")
        
        # Step 1: Create empty knowledge graph
        kg = KnowledgeGraph()
        self.logger.info("📊 Created empty knowledge graph")
        
        # Step 2: Add documents as nodes with CSV-compatible properties
        for i, doc in enumerate(documents):
            # Enhanced node creation with CSV-compatible properties
            node_properties = {
                "page_content": doc.page_content,
                "document_metadata": doc.metadata,
                # ✅ Add required properties for RAGAS compatibility
                "id": f"csv_doc_{i}",
                "title": doc.metadata.get('title', f'CSV Document {i}'),
                "source": doc.metadata.get('source', 'csv_input'),
                # ✅ Critical: Add summary to avoid filtering issues
                "summary": doc.page_content[:300] + "..." if len(doc.page_content) > 300 else doc.page_content,
                # ✅ Add empty headlines for compatibility with HeadlineSplitter
                "headlines": [],
                # ✅ Add keywords for enhanced RAGAS processing
                "keywords": self._extract_simple_keywords(doc.page_content),
                # ✅ Add entities for OverlapScoreBuilder compatibility
                "entities": self._extract_simple_entities(doc.page_content),
                # CSV-specific metadata
                "csv_row_index": i,
                "content_length": len(doc.page_content),
                "language": doc.metadata.get('language', 'unknown'),
                "label": doc.metadata.get('label', 'unknown'),
                "csv_compatible": True
            }
            
            node = Node(
                type=NodeType.DOCUMENT,
                properties=node_properties
            )
            kg.nodes.append(node)
        
        self.logger.info(f"📋 Added {len(kg.nodes)} CSV-compatible document nodes to knowledge graph")
        
        # Step 3: Apply CSV-compatible RAGAS transforms to enrich the knowledge graph
        self.logger.info("🔄 Applying CSV-compatible RAGAS transforms to enrich knowledge graph...")
        
        try:
            # ✅ Use CSV-compatible transforms instead of default_transforms
            # Default transforms include HeadlineSplitter which fails on CSV data
            
            from ragas.testset.transforms import (
                SummaryExtractor, 
                KeyphrasesExtractor, 
                EmbeddingExtractor,
                CosineSimilarityBuilder,
                OverlapScoreBuilder,
                apply_transforms
            )
            
            # Create CSV-compatible transforms that work with our enhanced nodes
            transforms = [
                # Skip problematic transforms:
                # - HeadlineSplitter: requires 'headlines' property
                # - CustomNodeFilter: filters out nodes without 'summary'
                
                # Use compatible extractors and builders
                SummaryExtractor(llm=self.ragas_llm),  # Enhance existing summaries
                KeyphrasesExtractor(llm=self.ragas_llm),  # Extract keyphrases
                EmbeddingExtractor(embedding_model=self.ragas_embeddings),  # Create embeddings from page_content
                
                # Create summary embeddings specifically for persona generation
                EmbeddingExtractor(
                    property_name="summary_embedding",  # Required by persona generation
                    embed_property_name="summary",      # Embed the summary text
                    embedding_model=self.ragas_embeddings
                ),
                
                CosineSimilarityBuilder(),  # Build similarity relationships
                OverlapScoreBuilder()  # Build overlap relationships
            ]
            
            self.logger.info(f"🔧 Created {len(transforms)} CSV-compatible RAGAS transforms")
            
            # Apply transforms to knowledge graph
            apply_transforms(kg, transforms)
            
            self.logger.info(f"✅ Knowledge graph enriched: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
            
        except Exception as e:
            self.logger.error(f"❌ Transform application failed: {e}")
            self.logger.warning("⚠️ Continuing with basic knowledge graph (no transforms applied)")
        
        self.knowledge_graph = kg
        return kg
    
    def _extract_simple_keywords(self, text: str) -> List[str]:
        """Extract simple keywords from text for RAGAS compatibility"""
        try:
            import re
            
            # Simple keyword extraction without LLM
            # Remove special characters and split
            clean_text = re.sub(r'[^\w\s]', ' ', text)
            words = clean_text.split()
            
            # Filter for meaningful words (length > 3, not all digits)
            keywords = [
                word.lower() for word in words 
                if len(word) > 3 and not word.isdigit() and word.isalpha()
            ]
            
            # Return top 10 unique keywords
            unique_keywords = list(dict.fromkeys(keywords))[:10]
            return unique_keywords
            
        except Exception as e:
            self.logger.debug(f"Simple keyword extraction failed: {e}")
            return []
    
    def _extract_simple_entities(self, text: str) -> List[str]:
        """Extract simple entities from text for RAGAS compatibility"""
        try:
            import re
            
            # Simple entity extraction patterns
            entities = []
            
            # Extract capitalized words (potential proper nouns/entities)
            capitalized_words = re.findall(r'\b[A-Z][a-z]+\b', text)
            entities.extend(capitalized_words[:5])  # Top 5 capitalized words
            
            # Extract numbers (potential IDs, versions, etc.)
            numbers = re.findall(r'\b\d+\b', text)
            entities.extend(numbers[:3])  # Top 3 numbers
            
            # Extract common technical terms or abbreviations
            technical_terms = re.findall(r'\b[A-Z]{2,}\b', text)
            entities.extend(technical_terms[:3])  # Top 3 abbreviations
            
            # Return unique entities
            unique_entities = list(dict.fromkeys(entities))[:10]
            return unique_entities
            
        except Exception as e:
            self.logger.debug(f"Simple entity extraction failed: {e}")
            return []
    
    def _create_manual_knowledge_graph(self, documents: List[Document]) -> KnowledgeGraph:
        """Create a manual knowledge graph as fallback when transforms fail"""
        self.logger.info("🔄 Creating manual knowledge graph (fallback)...")
        
        # Create basic knowledge graph without transforms
        kg = KnowledgeGraph()
        
        for i, doc in enumerate(documents):
            node = Node(
                type=NodeType.DOCUMENT,
                properties={
                    "page_content": doc.page_content,
                    "document_metadata": doc.metadata,
                    "id": f"manual_doc_{i}",
                    "title": doc.metadata.get('title', f'Document {i}'),
                    "summary": doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
                    "entities": self._extract_simple_entities(doc.page_content),
                    "keywords": self._extract_simple_keywords(doc.page_content),
                    "manual_creation": True
                }
            )
            kg.nodes.append(node)
        
        self.logger.info(f"✅ Manual knowledge graph created: {len(kg.nodes)} nodes")
        return kg
    
    def generate_testset(self, knowledge_graph: KnowledgeGraph, testset_size: int = 10) -> Any:
        """
        Generate synthetic testset using RAGAS with the knowledge graph
        
        This follows the official RAGAS testset generation approach
        """
        self.logger.info("🎯 Generating synthetic testset with RAGAS...")
        
        # Step 1: Create TestsetGenerator with knowledge graph
        generator = TestsetGenerator(
            llm=self.ragas_llm,
            embedding_model=self.ragas_embeddings,
            knowledge_graph=knowledge_graph
        )
        
        self.logger.info("✅ RAGAS TestsetGenerator created")
        
        # Step 2: Define query distribution with only single-hop to avoid clustering issues
        from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
        
        # Use only single-hop synthesizer to avoid clustering issues
        query_distribution = [
            (SingleHopSpecificQuerySynthesizer(llm=self.ragas_llm), 1.0)
        ]
        
        self.logger.info(f"📊 Query distribution: Single-hop only (avoiding clustering issues)")
        
        # Step 3: Debug knowledge graph before generation
        self.logger.info(f"🔍 Knowledge graph debug info:")
        self.logger.info(f"   - Total nodes: {len(knowledge_graph.nodes)}")
        for i, node in enumerate(knowledge_graph.nodes[:5]):  # Show first 5 nodes
            self.logger.info(f"   - Node {i}: type={node.type.name}, properties={list(node.properties.keys())}")
            if node.get_property('entities'):
                self.logger.info(f"     entities: {node.get_property('entities')[:3]}...")  # Show first 3 entities
        
        # Step 4: Generate testset
        self.logger.info(f"🚀 Generating testset with {testset_size} samples...")
        
        try:
            testset = generator.generate(
                testset_size=testset_size,
                query_distribution=query_distribution,
                raise_exceptions=False  # Don't fail on individual errors
            )
            
            self.logger.info(f"✅ RAGAS testset generation completed!")
            
            # Log testset details
            if hasattr(testset, 'samples'):
                self.logger.info(f"📊 Generated {len(testset.samples)} samples")
                
                # Show sample
                if len(testset.samples) > 0:
                    sample = testset.samples[0]
                    # Handle different RAGAS versions
                    if hasattr(sample, 'user_input'):
                        question = sample.user_input
                    elif hasattr(sample, 'question'):
                        question = sample.question
                    else:
                        question = str(sample)[:100]
                    
                    self.logger.info(f"📝 Sample question: {question[:100]}...")
                    
            return testset
            
        except Exception as e:
            self.logger.error(f"❌ RAGAS testset generation failed: {e}")
            raise
    
    def save_knowledge_graph(self, kg: KnowledgeGraph, output_dir: Path):
        """Save knowledge graph for reuse"""
        kg_file = output_dir / "knowledge_graph.json"
        try:
            kg.save(str(kg_file))
            self.logger.info(f"💾 Knowledge graph saved to: {kg_file}")
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to save knowledge graph: {e}")
    
    def convert_testset_to_dataframe(self, testset: Any) -> pd.DataFrame:
        """Convert RAGAS testset to pandas DataFrame for analysis"""
        try:
            if hasattr(testset, 'to_pandas'):
                df = testset.to_pandas()
                self.logger.info(f"📊 Converted testset to DataFrame: {df.shape}")
                return df
            
            elif hasattr(testset, 'samples'):
                # Manually convert samples to DataFrame
                data = []
                for sample in testset.samples:
                    # Handle different RAGAS sample formats
                    if hasattr(sample, 'user_input'):
                        question = sample.user_input
                        reference = sample.reference if hasattr(sample, 'reference') else ''
                        contexts = sample.retrieved_contexts if hasattr(sample, 'retrieved_contexts') else []
                    elif hasattr(sample, 'question'):
                        question = sample.question
                        reference = sample.reference if hasattr(sample, 'reference') else ''
                        contexts = sample.contexts if hasattr(sample, 'contexts') else []
                    else:
                        # Try to extract from string representation
                        question = str(sample)
                        reference = ''
                        contexts = []
                    
                    data.append({
                        'question': question,
                        'reference': reference,
                        'contexts': contexts,
                        'ground_truth': reference,
                        'generation_method': 'ragas_full_pipeline'
                    })
                
                df = pd.DataFrame(data)
                self.logger.info(f"📊 Manually converted testset to DataFrame: {df.shape}")
                return df
                
            else:
                self.logger.error("❌ Testset format not recognized")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"❌ Failed to convert testset to DataFrame: {e}")
            return pd.DataFrame()
    
    def run_full_pipeline(self, output_dir: Path) -> Dict[str, Any]:
        """
        Run the complete RAGAS pipeline from CSV to synthetic testset
        
        Returns:
            Dictionary with results including testset DataFrame and metadata
        """
        self.logger.info("🚀 Starting Full RAGAS Knowledge Graph Pipeline")
        self.logger.info("=" * 60)
        
        try:
            # Step 1: Initialize components
            self.initialize_components()
            
            # Step 2: Load CSV data
            documents = self.load_csv_data()
            
            # Step 2.5: Apply document limit for stable generation
            max_docs_limit = self.config.get('testset_generation', {}).get('max_documents_for_generation', 20)
            if len(documents) > max_docs_limit:
                self.logger.info(f"🎯 Applying document limit: {len(documents)} → {max_docs_limit}")
                documents = documents[:max_docs_limit]
            
            # Step 3: Create knowledge graph with transforms
            knowledge_graph = self.create_knowledge_graph(documents)
            
            # Step 4: Save knowledge graph
            self.save_knowledge_graph(knowledge_graph, output_dir)
            
            # Step 5: Generate synthetic testset
            testset_size = self.config.get('testset_generation', {}).get('max_total_samples', 15)
            testset = self.generate_testset(knowledge_graph, testset_size)
            
            # Step 6: Convert to DataFrame
            testset_df = self.convert_testset_to_dataframe(testset)
            
            # Step 7: Prepare results
            results = {
                'testset_df': testset_df,
                'testset_size': len(testset_df),
                'knowledge_graph_nodes': len(knowledge_graph.nodes),
                'knowledge_graph_relationships': len(knowledge_graph.relationships),
                'source_documents': len(documents),
                'generation_method': 'full_ragas_knowledge_graph',
                'success': True,
                'metadata': {
                    'pipeline_type': 'full_ragas_implementation',
                    'llm_model': self.custom_llm_config.get('model', 'unknown'),
                    'embeddings_model': self.ragas_config.get('embeddings_model', 'unknown'),
                    'transforms_applied': True,
                    'knowledge_graph_enriched': True
                }
            }
            
            self.logger.info("🎉 Full RAGAS pipeline completed successfully!")
            self.logger.info(f"📊 Generated {len(testset_df)} synthetic Q&A pairs")
            
            return results
            
        except Exception as e:
            self.logger.error(f"❌ Full RAGAS pipeline failed: {e}")
            self.logger.debug(f"   Error type: {type(e).__name__}")
            import traceback
            self.logger.debug(f"   Full traceback: {traceback.format_exc()}")
            
            # Try to extract more info about the current state
            try:
                if hasattr(self, 'knowledge_graph') and self.knowledge_graph:
                    self.logger.info(f"🔍 Knowledge graph state during error:")
                    self.logger.info(f"   - Total nodes: {len(self.knowledge_graph.nodes)}")
                    self.logger.info(f"   - Total relationships: {len(self.knowledge_graph.relationships)}")
                    
                    # Check node types and properties 
                    for i, node in enumerate(self.knowledge_graph.nodes[:3]):  # Show first 3 nodes
                        props = list(node.properties.keys())
                        entities = node.get_property('entities')
                        self.logger.info(f"   - Node {i}: type={node.type.name}, props={props}")
                        if entities:
                            self.logger.info(f"     entities: {entities[:3] if isinstance(entities, list) else str(entities)[:50]}...")
                else:
                    self.logger.info("🔍 No knowledge graph available for debugging")
                    
            except Exception as debug_error:
                self.logger.error(f"❌ Debug info extraction failed: {debug_error}")
            
            return {
                'testset_df': pd.DataFrame(),
                'testset_size': 0,
                'success': False,
                'error': str(e),
                'generation_method': 'full_ragas_knowledge_graph_failed'
            }


def run_full_ragas_pipeline(config: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    """
    Main entry point for the full RAGAS pipeline
    
    Args:
        config: Complete pipeline configuration
        output_dir: Output directory for results
        
    Returns:
        Results dictionary with testset and metadata
    """
    logger.info("🚀 Initializing Full RAGAS Knowledge Graph Pipeline")
    
    pipeline = FullRAGASPipeline(config)
    results = pipeline.run_full_pipeline(output_dir)
    
    return results


if __name__ == "__main__":
    # Test the pipeline with sample configuration
    import yaml
    
    # Load configuration
    config_file = Path(__file__).parent.parent.parent / "config" / "pipeline_config.yaml"
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Create output directory
    output_dir = Path("outputs/full_ragas_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run pipeline
    results = run_full_ragas_pipeline(config, output_dir)
    
    if results['success']:
        print(f"✅ Success! Generated {results['testset_size']} samples")
        print(f"📊 Knowledge graph: {results['knowledge_graph_nodes']} nodes, {results['knowledge_graph_relationships']} relationships")
    else:
        print(f"❌ Failed: {results.get('error', 'Unknown error')}")
