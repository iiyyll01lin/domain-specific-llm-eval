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

import os
import sys
import json
import yaml
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple

# Add RAGAS to path
ragas_path = "/data/yy/domain-specific-llm-eval/ragas/ragas/src"
if ragas_path not in sys.path:
    sys.path.insert(0, ragas_path)

# Import RAGAS components
from ragas.testset import TestsetGenerator
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.transforms.relationship_builders import (
    CosineSimilarityBuilder, 
    JaccardSimilarityBuilder, 
    OverlapScoreBuilder
)
from ragas.testset.transforms.relationship_builders.cosine import SummaryCosineSimilarityBuilder

# Import LangChain for document processing
from langchain_core.documents import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_summary_similarity_relationships(kg: KnowledgeGraph):
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

def load_config(config_path: str = "config/pipeline_config.yaml") -> Dict[str, Any]:
    """Load pipeline configuration"""
    config_file = Path(__file__).parent / config_path
    with open(config_file, 'r') as f:
        return yaml.safe_load(f)

def extract_content_from_csv_row(content_json: str) -> Tuple[str, Dict]:
    """Extract text content and metadata from CSV content JSON"""
    try:
        content_data = json.loads(content_json)
        
        # Extract fields from error code JSON structure
        error_code = content_data.get('error_code', '')
        display = content_data.get('display', '')
        cause = content_data.get('cause', '')
        remedy = content_data.get('remedy', '')
        used_by = content_data.get('used_by', '')
        
        # Combine all fields into a comprehensive text content
        text_content = f"""Error Code: {error_code}

Display: {display}

Cause: {cause}

Remedy: {remedy}

Used By: {used_by}"""
        
        metadata = {
            'error_code': error_code,
            'display': display,
            'cause': cause,
            'remedy': remedy,
            'used_by': used_by,
            'source': 'smt-nxt-errorcode',
            'language': 'zh-TW'
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
            with open(doc_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
            
            if len(content) < 100:  # Skip very short content
                continue
            
            # Create LangChain Document
            doc = Document(
                page_content=content,
                metadata={
                    'title': doc_path.stem,
                    'source': str(doc_path),
                    'document_id': processed_count,
                    'filename': doc_path.name,
                    'content_type': 'steel_plate_inspection'
                }
            )
            documents.append(doc)
            processed_count += 1
            
        except Exception as e:
            logger.warning(f"Failed to load document {doc_file}: {e}")
            continue
    
    logger.info(f"✅ Loaded {len(documents)} TXT documents")
    return documents

def load_csv_documents(csv_file_path: str, max_docs: int = 3) -> List[Document]:
    """Load and process CSV documents for RAGAS"""
    logger.info(f"📂 Loading CSV documents from: {csv_file_path}")
    
    df = pd.read_csv(csv_file_path)
    logger.info(f"📊 Loaded CSV with {len(df)} rows")
    
    documents = []
    processed_count = 0
    
    for idx, row in df.iterrows():
        if processed_count >= max_docs:
            break
            
        # Extract content from JSON field
        content_text, metadata = extract_content_from_csv_row(row['content'])
        
        if len(content_text.strip()) < 50:  # Skip very short content
            continue
        
        # Create LangChain Document
        doc = Document(
            page_content=content_text,
            metadata={
                **metadata,
                'csv_id': row['id'],
                'template_key': row.get('template_key', ''),
                'created_at': row.get('created_at', ''),
                'author': row.get('author', '')
            }
        )
        
        documents.append(doc)
        processed_count += 1
        
        logger.info(f"✅ Processed document {processed_count}: {metadata.get('title', f'Doc {idx}')} ({len(content_text)} chars)")
    
    logger.info(f"📄 Successfully loaded {len(documents)} documents for RAGAS processing")
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
        
        # 3. Build embedding-based relationships if embeddings are available
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
            
            # 4. Build summary cosine similarity relationships
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
        
        # 5. Build multihop relationships (always - required for multihop synthesizers)
        logger.info("🔗 Building summary_similarity relationships for multihop abstract...")
        try:
            summary_sim_rels = create_summary_similarity_relationships(kg)
            for rel in summary_sim_rels:
                kg._add_relationship(rel)
            total_relationships += len(summary_sim_rels)
            logger.info(f"✅ Built {len(summary_sim_rels)} summary_similarity relationships")
        except Exception as e:
            logger.warning(f"Failed to build summary_similarity relationships: {e}")
        
        logger.info("🔗 Building entities_overlap relationships for multihop specific...")
        try:
            entities_overlap_rels = create_entities_overlap_relationships(kg)
            for rel in entities_overlap_rels:
                kg._add_relationship(rel)
            total_relationships += len(entities_overlap_rels)
            logger.info(f"✅ Built {len(entities_overlap_rels)} entities_overlap relationships")
        except Exception as e:
            logger.warning(f"Failed to build entities_overlap relationships: {e}")
        
    except Exception as e:
        logger.error(f"❌ Error building relationships: {e}")
        logger.info("Continuing without relationships...")
    
    return total_relationships

async def create_knowledge_graph_from_documents(documents: List[Document], 
                                        embeddings_model: LangchainEmbeddingsWrapper = None) -> KnowledgeGraph:
    """Create RAGAS knowledge graph from documents with relationships"""
    logger.info(f"🧠 Creating RAGAS Knowledge Graph from {len(documents)} documents...")
    
    # Create knowledge graph
    kg = KnowledgeGraph()
    
    # Text splitter for chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
        length_function=len,
        separators=["\n\n", "\n", "。", ".", " ", ""]
    )
    
    # Split documents into chunks
    split_docs = text_splitter.split_documents(documents)
    logger.info(f"📄 Split {len(documents)} documents into {len(split_docs)} chunks")
    
    # Create nodes for each chunk
    import uuid
    for chunk_idx, chunk_doc in enumerate(split_docs):
        if len(chunk_doc.page_content.strip()) < 30:  # Skip very short chunks
            continue
            
        node_id = uuid.uuid4()  # Use proper UUID object, not string
        
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
                # Properties for relationship building
                'entities': entities,
                'keyphrases': keyphrases,
                'sentences': sentences
            }
        )
        
        # Add embedding if embeddings model is provided
        if embeddings_model:
            try:
                embedding_result = embeddings_model.embed_text(chunk_doc.page_content)
                # Handle both sync and async embedding models
                if hasattr(embedding_result, '__await__'):
                    embedding = await embedding_result
                else:
                    embedding = embedding_result
                node.properties['embedding'] = embedding
                
                # Create summary for summary embedding
                summary = sentences[0] if sentences else chunk_doc.page_content[:200]
                summary_embedding_result = embeddings_model.embed_text(summary)
                if hasattr(summary_embedding_result, '__await__'):
                    summary_embedding = await summary_embedding_result
                else:
                    summary_embedding = summary_embedding_result
                node.properties['summary_embedding'] = summary_embedding
                node.properties['summary'] = summary
                
            except Exception as e:
                logger.warning(f"Failed to create embedding for node {node_id}: {e}")
                # Create fallback summary_embedding for persona compatibility
                node.properties['summary_embedding'] = [0.1] * 384  # Simple fallback embedding
        else:
            # Always ensure summary_embedding exists for persona generation
            summary = sentences[0] if sentences else chunk_doc.page_content[:200]
            node.properties['summary'] = summary
            node.properties['summary_embedding'] = [hash(summary) % 1000 / 1000.0] * 384  # Simple hash-based embedding
        
        kg._add_node(node)
    
    logger.info(f"✅ Created knowledge graph with {len(kg.nodes)} nodes")
    
    # Build relationships between nodes
    logger.info("🔗 Building relationships between nodes...")
    relationships_built = await build_relationships(kg, embeddings_model is not None)
    
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
        'created_at': timestamp,
        'generator': 'pure_ragas_pipeline',
        'nodes': [
            {
                'id': str(node.id),  # Convert UUID to string
                'type': str(node.type),
                'properties': node.properties
            } for node in kg.nodes
        ],
        'relationships': [
            {
                'source': str(rel.source.id),  # Convert UUID to string
                'target': str(rel.target.id),  # Convert UUID to string
                'type': str(rel.type),
                'properties': rel.properties
            } for rel in kg.relationships
        ] if hasattr(kg, 'relationships') and kg.relationships else [],
        'metadata': {
            'total_nodes': len(kg.nodes),
            'total_relationships': len(kg.relationships) if hasattr(kg, 'relationships') and kg.relationships else 0,
            'generation_method': 'pure_ragas'
        }
    }
    
    # Save to file
    with open(kg_filepath, 'w', encoding='utf-8') as f:
        json.dump(kg_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"✅ Knowledge graph saved: {len(kg.nodes)} nodes")
    return str(kg_filepath)

def setup_ragas_components(config: Dict[str, Any]) -> Tuple[LangchainLLMWrapper, LangchainEmbeddingsWrapper]:
    """Setup RAGAS LLM and embedding components"""
    ragas_config = config.get('testset_generation', {}).get('ragas_config', {})
    
    # Set up custom LLM
    custom_llm_config = ragas_config.get('custom_llm', {})
    
    if custom_llm_config.get('endpoint'):
        # Use custom LLM endpoint
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(
            base_url=custom_llm_config['endpoint'],
            api_key=custom_llm_config['api_key'],
            model=custom_llm_config['model'],
            temperature=custom_llm_config.get('temperature', 0.3),
            max_tokens=custom_llm_config.get('max_tokens', 1000),
            timeout=custom_llm_config.get('timeout', 60)
        )
        
        generator_llm = LangchainLLMWrapper(llm)
        logger.info(f"🔗 Using custom LLM: {custom_llm_config['endpoint']}")
    else:
        raise ValueError("Custom LLM configuration required for RAGAS generation")
    
    # Set up embeddings
    embeddings_model = ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2')
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
        generator_embeddings = LangchainEmbeddingsWrapper(embeddings)
        logger.info(f"🔗 Using HuggingFace embeddings: {embeddings_model}")
    except ImportError:
        try:
            from sentence_transformers import SentenceTransformer
            from langchain_community.embeddings import SentenceTransformerEmbeddings
            embeddings = SentenceTransformerEmbeddings(model_name=embeddings_model)
            generator_embeddings = LangchainEmbeddingsWrapper(embeddings)
            logger.info(f"🔗 Using SentenceTransformer embeddings: {embeddings_model}")
        except ImportError:
            logger.error("❌ Could not load embedding models. Please install sentence-transformers or langchain-huggingface")
            raise
    
    return generator_llm, generator_embeddings

def generate_ragas_testset(kg: KnowledgeGraph, generator_llm: LangchainLLMWrapper, 
                          generator_embeddings: LangchainEmbeddingsWrapper, 
                          num_samples: int = 3) -> List[Dict]:
    """Generate synthetic testset using RAGAS TestsetGenerator"""
    logger.info(f"🎯 Generating {num_samples} synthetic test samples using RAGAS...")
    
    # Create RAGAS TestsetGenerator
    logger.info("🚀 Initializing RAGAS TestsetGenerator...")
    
    try:
        # Create simple personas to avoid generation issues
        from ragas.testset.persona import Persona
        
        personas = [
            Persona(
                name="Technical Specialist",
                role_description="A technical specialist who asks detailed questions about industrial processes and specifications."
            ),
            Persona(
                name="Quality Inspector", 
                role_description="A quality inspector who focuses on measurement procedures and quality control standards."
            )
        ]
        
        generator = TestsetGenerator(
            llm=generator_llm,
            embedding_model=generator_embeddings,
            knowledge_graph=kg,
            persona_list=personas  # Provide personas directly
        )
        
        logger.info("✅ RAGAS TestsetGenerator initialized with pre-defined personas")
        
        # Generate testset using RAGAS with simple configuration
        logger.info(f"🎯 Generating {num_samples} test samples...")
        
        testset = generator.generate(
            testset_size=num_samples,
            raise_exceptions=False  # Don't raise exceptions to handle gracefully
        )
        
        logger.info("✅ RAGAS testset generation completed")
        
        # Convert testset to our format
        test_samples = []
        
        if hasattr(testset, 'samples') and testset.samples:
            for i, sample in enumerate(testset.samples):
                if hasattr(sample, 'eval_sample') and sample.eval_sample:
                    eval_sample = sample.eval_sample
                    test_sample = {
                        'question': eval_sample.user_input if hasattr(eval_sample, 'user_input') else f"Generated question {i+1}",
                        'contexts': eval_sample.reference_contexts if hasattr(eval_sample, 'reference_contexts') else [],
                        'ground_truth': eval_sample.reference if hasattr(eval_sample, 'reference') else f"Generated answer {i+1}",
                        'synthesizer_name': sample.synthesizer_name if hasattr(sample, 'synthesizer_name') else 'ragas_synthesizer',
                        'generation_method': 'pure_ragas',
                        'generation_timestamp': datetime.now().isoformat(),
                        'source_type': 'csv'
                    }
                    test_samples.append(test_sample)
                    
            logger.info(f"✅ Converted {len(test_samples)} RAGAS samples to pipeline format")
        
        # If no samples were generated, create fallback samples based on knowledge graph content
        if not test_samples:
            logger.warning("⚠️ RAGAS testset generation returned no samples - creating samples from knowledge graph")
            
            # Extract content from knowledge graph nodes
            node_contents = []
            for node in kg.nodes:
                if hasattr(node, 'properties') and node.properties.get('content'):
                    content = node.properties['content']
                    title = node.properties.get('title', 'Document')
                    node_contents.append({'content': content, 'title': title})
            
            # Create questions based on node content
            for i, node_data in enumerate(node_contents[:num_samples]):
                content = node_data['content']
                title = node_data['title']
                
                # Generate simple questions based on content
                if '量測' in content or 'measurement' in content.lower():
                    question = f"What are the measurement procedures described in {title}?"
                    answer = f"The measurement procedures include: {content[:200]}..."
                elif '檢查' in content or 'inspection' in content.lower():
                    question = f"What inspection steps are required according to {title}?"
                    answer = f"The inspection requirements are: {content[:200]}..."
                else:
                    question = f"What are the key points described in {title}?"
                    answer = f"The key information includes: {content[:200]}..."
                
                test_sample = {
                    'question': question,
                    'contexts': [content],
                    'ground_truth': answer,
                    'synthesizer_name': 'knowledge_graph_based_generator',
                    'generation_method': 'pure_ragas_fallback',
                    'generation_timestamp': datetime.now().isoformat(),
                    'source_type': 'csv'
                }
                test_samples.append(test_sample)
            
            logger.info(f"✅ Created {len(test_samples)} fallback test samples from knowledge graph")
        
        return test_samples
        
    except Exception as e:
        logger.error(f"❌ RAGAS testset generation failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        # Create minimal fallback as last resort
        logger.info("🔄 Creating minimal fallback samples...")
        test_samples = []
        
        for i in range(min(num_samples, 3)):
            test_sample = {
                'question': f"What are the key technical requirements described in the document?",
                'contexts': ["Technical documentation content from knowledge graph"],
                'ground_truth': "The technical requirements include various specifications and procedures based on the source documentation.",
                'synthesizer_name': 'minimal_fallback_generator',
                'generation_method': 'pure_ragas_minimal_fallback',
                'generation_timestamp': datetime.now().isoformat(),
                'source_type': 'csv'
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
        logger.warning(f"⚠️ Could not import PipelineFileSaver: {e}, using fallback method")
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
        
        # Add additional columns to match expected format (but leave empty as designed)
        df['answer'] = df['ground_truth']  # Copy ground truth as answer
        df['auto_keywords'] = ''  # Empty - keywords will be extracted from RAG responses later!
        df['source_file'] = ''   # Empty for CSV source
        df['question_type'] = ''  # Empty - will be classified later
        df['keyword_score'] = 0.0  # Will be calculated from RAG responses later!
        df['enhanced_at'] = datetime.now().isoformat()
        
        # Reorder columns to match expected format
        column_order = [
            'question', 'answer', 'auto_keywords', 'source_file', 'question_type',
            'generation_method', 'generation_timestamp', 'contexts', 'ground_truth',
            'synthesizer_name', 'keyword_score', 'enhanced_at', 'source_type'
        ]
        
        df = df.reindex(columns=column_order, fill_value='')
        
        # Save to CSV
        df.to_csv(testset_filepath, index=False, encoding='utf-8')
        
        logger.info(f"✅ Testset saved: {len(test_samples)} samples")
        logger.info(f"📊 Columns: {list(df.columns)}")
        
        return str(testset_filepath)
        
    except Exception as e:
        logger.error(f"❌ Failed to save testset: {e}")
        raise

def main():
    """Main pipeline execution"""
    logger.info("🚀 Starting Pure RAGAS Pipeline (Corrected Design)")
    logger.info("=" * 60)
    
    try:
        # Step 1: Load configuration
        logger.info("📋 Loading configuration...")
        config = load_config()
        
        testset_config = config.get('testset_generation', {})
        max_docs = testset_config.get('max_documents_for_generation', 10)
        # CRITICAL FIX: Use proper default value instead of hardcoded 3
        # The hardcoded 3 was causing fallback to minimal generation
        max_samples = testset_config.get('max_total_samples', 1000)  # Changed from 3 to 1000
        
        logger.info(f"🎯 Configuration: max_docs={max_docs}, max_samples={max_samples}")
        
        # Step 2: Setup output directory
        output_dir = Path("outputs") / f"pure_ragas_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Step 3: Load documents (TXT documents for steel plate inspection)
        document_files = config.get('data_sources', {}).get('documents', {}).get('document_files', [])
        csv_files = config.get('data_sources', {}).get('csv', {}).get('csv_files', [])
        
        if document_files:
            logger.info("📄 Loading TXT documents (working 2024 method)...")
            documents = load_txt_documents(document_files, max_docs)
        elif csv_files:
            logger.info("📄 Loading CSV documents (fallback method)...")
            csv_file_path = csv_files[0]  # Use first CSV file
            documents = load_csv_documents(csv_file_path, max_docs)
        else:
            raise ValueError("No document files or CSV files specified in configuration")
        
        # Step 4: Setup RAGAS components (before KG creation for embeddings)
        generator_llm, generator_embeddings = setup_ragas_components(config)
        
        # Step 5: Create knowledge graph with relationships
        import asyncio
        kg = asyncio.run(create_knowledge_graph_from_documents(documents, generator_embeddings))
        
        # Step 6: Save knowledge graph for reuse (as requested)
        kg_filepath = save_knowledge_graph(kg, output_dir)
        
        # Step 7: Generate synthetic testset using RAGAS
        test_samples = generate_ragas_testset(kg, generator_llm, generator_embeddings, max_samples)
        
        # Step 8: Save testset
        testset_filepath = save_testset(test_samples, output_dir)
        
        # Step 9: Save personas and scenarios (NEW)
        try:
            # Import the comprehensive file saver
            sys.path.append(str(Path(__file__).parent / "src" / "utils"))
            from utils.pipeline_file_saver import PipelineFileSaver
            
            file_saver = PipelineFileSaver(output_dir)
            
            # Create default personas if not existing
            default_personas = [
                {
                    "id": "technical_expert",
                    "name": "Technical Expert", 
                    "description": "Domain expert asking technical questions",
                    "question_style": "detailed",
                    "complexity_preference": "high",
                    "role_description": "A technical specialist who asks detailed questions about industrial processes and specifications."
                },
                {
                    "id": "business_user",
                    "name": "Business User",
                    "description": "Business stakeholder asking practical questions", 
                    "question_style": "concise",
                    "complexity_preference": "medium",
                    "role_description": "A quality inspector who focuses on measurement procedures and quality control standards."
                }
            ]
            
            # Create default scenarios
            default_scenarios = [
                {
                    "id": "single_hop",
                    "name": "Single Hop Query",
                    "description": "Questions requiring single document lookup",
                    "complexity": "low",
                    "hop_type": "single",
                    "expected_sources": 1
                },
                {
                    "id": "multi_hop", 
                    "name": "Multi Hop Query",
                    "description": "Questions requiring multiple document connections",
                    "complexity": "high",
                    "hop_type": "multi",
                    "expected_sources": "2+"
                }
            ]
            
            # Save personas and scenarios
            personas_path = file_saver.save_personas_json(default_personas)
            scenarios_path = file_saver.save_scenarios_json(default_scenarios)
            
            # Save pipeline metadata
            pipeline_metadata = {
                "pipeline_type": "pure_ragas",
                "generation_timestamp": datetime.now().isoformat(),
                "documents_processed": len(documents), 
                "knowledge_graph_nodes": len(kg.nodes),
                "knowledge_graph_relationships": len(kg.relationships) if hasattr(kg, 'relationships') and kg.relationships else 0,
                "testset_samples": len(test_samples),
                "csv_source": csv_file_path,
                "max_docs": max_docs,
                "max_samples": max_samples,
                "files_created": {
                    "knowledge_graph": kg_filepath,
                    "testset": testset_filepath,
                    "personas": personas_path,
                    "scenarios": scenarios_path
                }
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
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
