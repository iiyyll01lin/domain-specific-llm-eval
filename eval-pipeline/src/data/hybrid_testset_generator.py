#!/usr/bin/env python3
"""
Hybrid Testset Generator - Combines Your Existing Methods with RAGAS

This module creates a unified testset generation approach that leverages:
1. Your existing configurable dataset generator with auto-keyword extraction
2. RAGAS TestsetGenerator for advanced question types
3. Document-adaptive keyword extraction from your contextual system
4. Multi-format output with comprehensive metadata
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime
import json

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ragas"))

# Import your existing systems
from generate_dataset_configurable import ConfigurableDatasetGenerator
from document_loader import DocumentLoader

# Import RAGAS components
try:
    from ragas.testset.generator import TestsetGenerator
    from ragas.testset.evolutions import simple, reasoning, multi_context, conditional
    from langchain.embeddings import HuggingFaceEmbeddings
    from langchain.llms import OpenAI
    RAGAS_AVAILABLE = True
except ImportError as e:
    RAGAS_AVAILABLE = False
    print(f"Warning: RAGAS not available: {e}")

class HybridTestsetGenerator:
    """
    Hybrid testset generator that combines multiple approaches:
    - Your existing configurable generator with auto-keyword extraction
    - RAGAS TestsetGenerator for sophisticated question types
    - Document-adaptive processing
    - Comprehensive metadata tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize generation settings
        self.generation_config = config.get('testset_generation', {})
        self.method = self.generation_config.get('method', 'hybrid')
        self.samples_per_doc = self.generation_config.get('samples_per_document', 50)
        self.max_total_samples = self.generation_config.get('max_total_samples', 1000)
        
        # Initialize your existing components
        self.configurable_generator = None
        self.ragas_generator = None
        
        # Document processing
        self.document_loader = DocumentLoader()
        self.processed_documents = []
        
        # Results tracking
        self.generation_results = []
        self.metadata = {
            'generation_start': datetime.now(),
            'total_generated': 0,
            'source_documents': [],
            'generation_method': self.method
        }
        
        self.logger.info(f"HybridTestsetGenerator initialized with method: {self.method}")
    
    def initialize_generators(self):
        """Initialize the specific generators based on configuration"""
        try:
            # Initialize your existing configurable generator
            if self.method in ['configurable', 'hybrid']:
                self.logger.info("Initializing configurable dataset generator...")
                # Create temporary config for your existing system
                temp_config_path = self._create_temp_config()
                self.configurable_generator = ConfigurableDatasetGenerator(temp_config_path)
                self.logger.info("✅ Configurable generator initialized")
            
            # Initialize RAGAS generator if available and needed
            if self.method in ['ragas', 'hybrid'] and RAGAS_AVAILABLE:
                self.logger.info("Initializing RAGAS testset generator...")
                self._initialize_ragas_generator()
                self.logger.info("✅ RAGAS generator initialized")
            elif self.method in ['ragas', 'hybrid'] and not RAGAS_AVAILABLE:
                self.logger.warning("⚠️ RAGAS not available, falling back to configurable method")
                self.method = 'configurable'
                
        except Exception as e:
            self.logger.error(f"❌ Generator initialization failed: {e}")
            raise
    
    def _initialize_ragas_generator(self):
        """Initialize RAGAS TestsetGenerator with custom LLM support"""
        try:
            # Get RAGAS configuration
            ragas_config = self.generation_config.get('ragas_config', {})
            
            # Initialize embeddings (always local for privacy)
            embeddings_model = ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2')
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
            
            # Initialize LLMs based on configuration
            generator_llm = None
            critic_llm = None
            
            # Check for custom LLM configuration first
            if ragas_config.get('use_custom_llm', False):
                custom_llm_config = ragas_config.get('custom_llm', {})
                generator_llm = self._create_custom_llm(custom_llm_config, temperature=0.3)
                critic_llm = self._create_custom_llm(custom_llm_config, temperature=0)
                if generator_llm and critic_llm:
                    self.logger.info("✅ Using custom LLM for RAGAS generation")
            
            # Fallback to OpenAI if configured and custom LLM failed
            elif ragas_config.get('use_openai', False):
                api_key = ragas_config.get('openai_api_key')
                if api_key:
                    generator_llm = OpenAI(openai_api_key=api_key, temperature=0.3)
                    critic_llm = OpenAI(openai_api_key=api_key, temperature=0)
                    self.logger.info("✅ Using OpenAI for RAGAS generation")
            
            # Create RAGAS generator
            if generator_llm and critic_llm:
                self.ragas_generator = TestsetGenerator.from_langchain(
                    generator_llm=generator_llm,
                    critic_llm=critic_llm,
                    embeddings=embeddings
                )
                self.logger.info("RAGAS generator created with LLM support")
            else:
                self.logger.warning("⚠️ RAGAS generator created without LLM support (limited functionality)")
                # Create basic generator without LLMs (embeddings only)
                self.ragas_generator = TestsetGenerator.from_langchain(
                    generator_llm=None,
                    critic_llm=None,
                    embeddings=embeddings
                )
                
        except Exception as e:
            self.logger.error(f"❌ RAGAS initialization failed: {e}")
            self.ragas_generator = None

    def _create_custom_llm(self, custom_llm_config: Dict[str, Any], temperature: float = 0.3):
        """Create a custom LLM wrapper for RAGAS"""
        try:
            # Import required libraries for custom LLM
            import requests
            import yaml
            from typing import List, Dict, Any, Optional
            from langchain.llms.base import LLM
            from langchain.callbacks.manager import CallbackManagerForLLMRun
            
            class CustomLLMWrapper(LLM):
                """Custom LLM wrapper for internal/private LLM endpoints"""
                
                endpoint: str
                api_key: str
                model: str
                temperature: float
                max_tokens: int
                headers: Dict[str, str]
                
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                
                @property
                def _llm_type(self) -> str:
                    return "custom_llm"
                
                def _call(
                    self,
                    prompt: str,
                    stop: Optional[List[str]] = None,
                    run_manager: Optional[CallbackManagerForLLMRun] = None,
                    **kwargs: Any,
                ) -> str:
                    """Call the custom LLM endpoint"""
                    try:
                        # Prepare request payload
                        payload = {
                            "model": self.model,
                            "messages": [
                                {"role": "user", "content": prompt}
                            ],
                            "temperature": self.temperature,
                            "max_tokens": self.max_tokens,
                            "stream": False
                        }
                        
                        # Prepare headers
                        headers = self.headers.copy()
                        if self.api_key:
                            headers["Authorization"] = f"Bearer {self.api_key}"
                        
                        # Make API request
                        response = requests.post(
                            self.endpoint,
                            json=payload,
                            headers=headers,
                            timeout=30
                        )
                        response.raise_for_status()
                        
                        # Parse response
                        result = response.json()
                        
                        # Extract content from response
                        if "choices" in result and len(result["choices"]) > 0:
                            content = result["choices"][0].get("message", {}).get("content", "")
                            return content.strip()
                        else:
                            raise ValueError(f"Unexpected response format: {result}")
                            
                    except Exception as e:
                        self.logger.error(f"Custom LLM call failed: {e}")
                        return f"Error: Failed to generate response - {str(e)}"
            
            # Load API key from secrets file
            api_key = self._load_api_key_from_secrets(custom_llm_config)
            
            # Create custom LLM instance
            custom_llm = CustomLLMWrapper(
                endpoint=custom_llm_config.get('endpoint', ''),
                api_key=api_key,
                model=custom_llm_config.get('model', 'default'),
                temperature=temperature,
                max_tokens=custom_llm_config.get('max_tokens', 1000),
                headers=custom_llm_config.get('headers', {})
            )
            
            return custom_llm
            
        except Exception as e:
            self.logger.error(f"❌ Failed to create custom LLM: {e}")
            return None
    
    def _load_api_key_from_secrets(self, custom_llm_config: Dict[str, Any]) -> str:
        """Load API key from secrets configuration file"""
        try:
            import yaml
            
            # Get secrets file path
            secrets_file = custom_llm_config.get('api_key_file', 'config/secrets.yaml')
            api_key_path = custom_llm_config.get('api_key_path', 'api_key')
            
            # Convert relative path to absolute
            if not Path(secrets_file).is_absolute():
                secrets_file = Path(__file__).parent.parent.parent / secrets_file
            
            # Load secrets file
            if not Path(secrets_file).exists():
                self.logger.warning(f"⚠️ Secrets file not found: {secrets_file}")
                return ""
            
            with open(secrets_file, 'r', encoding='utf-8') as f:
                secrets = yaml.safe_load(f)
            
            # Navigate to API key using dot notation
            api_key = secrets
            for key_part in api_key_path.split('.'):
                if isinstance(api_key, dict) and key_part in api_key:
                    api_key = api_key[key_part]
                else:
                    self.logger.warning(f"⚠️ API key path not found: {api_key_path}")
                    return ""
            
            if isinstance(api_key, str) and api_key:
                self.logger.info("✅ API key loaded from secrets file")
                return api_key
            else:
                self.logger.warning(f"⚠️ Invalid API key in secrets file")
                return ""
                
        except Exception as e:
            self.logger.error(f"❌ Failed to load API key from secrets: {e}")
            return ""
    
    def generate_comprehensive_testset(self, 
                                     document_paths: List[str],
                                     output_dir: Path) -> Dict[str, Any]:
        """
        Generate comprehensive testset using hybrid approach
        
        Args:
            document_paths: List of document file paths
            output_dir: Directory to save generated testsets
        
        Returns:
            Dictionary containing generation results and metadata
        """
        self.logger.info(f"🚀 Starting hybrid testset generation for {len(document_paths)} documents")
        
        # Initialize generators
        self.initialize_generators()
        
        # Process documents
        processed_docs = self._process_documents(document_paths)
        
        # Generate testsets based on method
        if self.method == 'configurable':
            results = self._generate_with_configurable(processed_docs, output_dir)
        elif self.method == 'ragas':
            results = self._generate_with_ragas(processed_docs, output_dir)
        elif self.method == 'hybrid':
            results = self._generate_hybrid(processed_docs, output_dir)
        else:
            raise ValueError(f"Unknown generation method: {self.method}")
        
        # Generate final combined testset
        combined_testset = self._combine_and_format_results(results, output_dir)
        
        # Update metadata
        self.metadata.update({
            'generation_end': datetime.now(),
            'total_generated': len(combined_testset),
            'source_documents': [str(Path(p).name) for p in document_paths],
            'generation_results': results
        })
        
        self.logger.info(f"✅ Hybrid testset generation completed. Generated {len(combined_testset)} samples")
        
        return {
            'testset': combined_testset,
            'metadata': self.metadata,
            'results_by_method': results
        }
    
    def _process_documents(self, document_paths: List[str]) -> List[Dict[str, Any]]:
        """Process documents for testset generation"""
        self.logger.info("📄 Processing documents...")
        
        processed_docs = []
        
        for doc_path in document_paths:
            try:
                self.logger.info(f"Processing: {Path(doc_path).name}")
                
                # Load document using your existing loader
                document_data = self.document_loader.load_document(doc_path)
                
                if document_data:
                    processed_doc = {
                        'path': doc_path,
                        'name': Path(doc_path).name,
                        'content': document_data.get('content', ''),
                        'metadata': document_data.get('metadata', {}),
                        'pages': document_data.get('pages', []),
                        'word_count': len(document_data.get('content', '').split()),
                        'processing_timestamp': datetime.now().isoformat()
                    }
                    
                    processed_docs.append(processed_doc)
                    self.logger.info(f"✅ Processed {Path(doc_path).name}: {processed_doc['word_count']} words")
                else:
                    self.logger.warning(f"⚠️ Failed to process document: {doc_path}")
                    
            except Exception as e:
                self.logger.error(f"❌ Error processing {doc_path}: {e}")
        
        self.logger.info(f"📄 Document processing completed. {len(processed_docs)} documents ready")
        return processed_docs
    
    def _generate_with_configurable(self, 
                                  processed_docs: List[Dict[str, Any]], 
                                  output_dir: Path) -> Dict[str, Any]:
        """Generate testset using your existing configurable method"""
        self.logger.info("🔄 Generating testset with configurable method...")
        
        try:
            # Update document paths in temporary config
            self._update_temp_config_with_docs(processed_docs)
            
            # Generate using your existing system
            testset_df = self.configurable_generator.generate_dataset()
            
            # Add source tracking and metadata
            enhanced_testset = self._enhance_testset_with_metadata(
                testset_df, processed_docs, method='configurable'
            )
            
            self.logger.info(f"✅ Configurable method generated {len(enhanced_testset)} samples")
            
            return {
                'method': 'configurable',
                'testset': enhanced_testset,
                'samples_generated': len(enhanced_testset),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ Configurable generation failed: {e}")
            return {
                'method': 'configurable',
                'testset': pd.DataFrame(),
                'samples_generated': 0,
                'success': False,
                'error': str(e)
            }
    
    def _generate_with_ragas(self, 
                           processed_docs: List[Dict[str, Any]], 
                           output_dir: Path) -> Dict[str, Any]:
        """Generate testset using RAGAS method"""
        self.logger.info("🔄 Generating testset with RAGAS method...")
        
        if not self.ragas_generator:
            self.logger.error("❌ RAGAS generator not available")
            return {
                'method': 'ragas',
                'testset': pd.DataFrame(),
                'samples_generated': 0,
                'success': False,
                'error': 'RAGAS generator not available'
            }
        
        try:
            # Convert documents to LangChain format
            langchain_docs = self._convert_to_langchain_docs(processed_docs)
            
            # Calculate samples per document
            total_samples = min(
                self.samples_per_doc * len(processed_docs),
                self.max_total_samples
            )
            
            # Define question type distributions
            distributions = self.generation_config.get('question_types', {
                'simple': 0.3,
                'multi_context': 0.3,
                'conditional': 0.2,
                'reasoning': 0.2
            })
            
            # Generate testset with RAGAS
            testset = self.ragas_generator.generate_with_langchain_docs(
                documents=langchain_docs,
                test_size=total_samples,
                distributions={
                    simple: distributions['simple'],
                    multi_context: distributions['multi_context'],
                    conditional: distributions['conditional'],
                    reasoning: distributions['reasoning']
                }
            )
            
            # Convert to DataFrame and add metadata
            testset_df = testset.to_pandas()
            enhanced_testset = self._enhance_testset_with_metadata(
                testset_df, processed_docs, method='ragas'
            )
            
            self.logger.info(f"✅ RAGAS method generated {len(enhanced_testset)} samples")
            
            return {
                'method': 'ragas',
                'testset': enhanced_testset,
                'samples_generated': len(enhanced_testset),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"❌ RAGAS generation failed: {e}")
            return {
                'method': 'ragas',
                'testset': pd.DataFrame(),
                'samples_generated': 0,
                'success': False,
                'error': str(e)
            }
    
    def _generate_hybrid(self, 
                        processed_docs: List[Dict[str, Any]], 
                        output_dir: Path) -> Dict[str, Any]:
        """Generate testset using hybrid approach (both methods)"""
        self.logger.info("🔄 Generating testset with hybrid method...")
        
        # Generate with both methods
        configurable_results = self._generate_with_configurable(processed_docs, output_dir)
        ragas_results = self._generate_with_ragas(processed_docs, output_dir)
        
        # Combine results
        combined_testset = pd.DataFrame()
        total_samples = 0
        
        if configurable_results['success']:
            combined_testset = pd.concat([combined_testset, configurable_results['testset']], ignore_index=True)
            total_samples += configurable_results['samples_generated']
            
        if ragas_results['success']:
            combined_testset = pd.concat([combined_testset, ragas_results['testset']], ignore_index=True)
            total_samples += ragas_results['samples_generated']
        
        # Remove duplicates and limit samples
        if len(combined_testset) > 0:
            # Simple deduplication based on question similarity
            combined_testset = self._deduplicate_questions(combined_testset)
            
            # Limit to max samples
            if len(combined_testset) > self.max_total_samples:
                combined_testset = combined_testset.sample(n=self.max_total_samples, random_state=42)
        
        self.logger.info(f"✅ Hybrid method generated {len(combined_testset)} samples")
        
        return {
            'method': 'hybrid',
            'testset': combined_testset,
            'samples_generated': len(combined_testset),
            'success': len(combined_testset) > 0,
            'configurable_results': configurable_results,
            'ragas_results': ragas_results
        }
    
    def _enhance_testset_with_metadata(self, 
                                     testset_df: pd.DataFrame,
                                     processed_docs: List[Dict[str, Any]],
                                     method: str) -> pd.DataFrame:
        """Enhance testset with comprehensive metadata"""
        
        enhanced_testset = testset_df.copy()
        
        # Add generation metadata
        enhanced_testset['generation_method'] = method
        enhanced_testset['generation_timestamp'] = datetime.now().isoformat()
        
        # Add source document mapping (simplified approach)
        if 'source_file' not in enhanced_testset.columns:
            # Map to documents based on content similarity or order
            doc_names = [doc['name'] for doc in processed_docs]
            if len(doc_names) == 1:
                enhanced_testset['source_file'] = doc_names[0]
            else:
                # Distribute questions across documents
                enhanced_testset['source_file'] = [
                    doc_names[i % len(doc_names)] 
                    for i in range(len(enhanced_testset))
                ]
        
        # Extract or enhance keywords
        if 'auto_keywords' not in enhanced_testset.columns:
            enhanced_testset['auto_keywords'] = enhanced_testset.apply(
                lambda row: self._extract_keywords_from_qa(row.get('question', ''), row.get('answer', '')),
                axis=1
            )
        
        # Add question classification
        enhanced_testset['question_type'] = enhanced_testset['question'].apply(
            self._classify_question_type
        )
        
        # Add word counts and other metrics
        enhanced_testset['question_word_count'] = enhanced_testset['question'].apply(
            lambda x: len(str(x).split())
        )
        enhanced_testset['answer_word_count'] = enhanced_testset['answer'].apply(
            lambda x: len(str(x).split())
        )
        
        return enhanced_testset
    
    def _extract_keywords_from_qa(self, question: str, answer: str) -> List[str]:
        """Extract keywords from Q&A pair using simple NLP"""
        try:
            # Use keybert if available
            from keybert import KeyBERT
            kw_model = KeyBERT()
            
            # Combine question and answer for keyword extraction
            combined_text = f"{question} {answer}"
            keywords = kw_model.extract_keywords(
                combined_text, 
                keyphrase_ngram_range=(1, 2), 
                stop_words='english',
                top_k=5
            )
            
            return [kw[0] for kw in keywords]
            
        except ImportError:
            # Fallback to simple keyword extraction
            import re
            from collections import Counter
            
            # Simple extraction based on word frequency
            combined_text = f"{question} {answer}".lower()
            words = re.findall(r'\b[a-zA-Z]{3,}\b', combined_text)
            
            # Filter out common stop words
            stop_words = {'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had'}
            words = [w for w in words if w not in stop_words]
            
            # Return top 5 most common words
            return [word for word, count in Counter(words).most_common(5)]
    
    def _classify_question_type(self, question: str) -> str:
        """Classify question type based on content"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what', 'who', 'where', 'when', 'which']):
            return 'factual'
        elif any(word in question_lower for word in ['how', 'why']):
            return 'explanatory'
        elif any(word in question_lower for word in ['compare', 'difference', 'similar']):
            return 'comparative'
        elif any(word in question_lower for word in ['if', 'when', 'suppose', 'assume']):
            return 'conditional'
        else:
            return 'general'
    
    def _deduplicate_questions(self, testset_df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate or very similar questions"""
        if len(testset_df) <= 1:
            return testset_df
        
        try:
            from sentence_transformers import SentenceTransformer, util
            
            # Use sentence transformer for similarity
            model = SentenceTransformer('all-MiniLM-L6-v2')
            questions = testset_df['question'].tolist()
            
            # Get embeddings
            embeddings = model.encode(questions)
            
            # Calculate similarity matrix
            similarities = util.pytorch_cos_sim(embeddings, embeddings)
            
            # Find duplicates (similarity > 0.8)
            duplicates_to_remove = set()
            for i in range(len(questions)):
                for j in range(i + 1, len(questions)):
                    if similarities[i][j] > 0.8:
                        duplicates_to_remove.add(j)  # Remove the later one
            
            # Remove duplicates
            indices_to_keep = [i for i in range(len(testset_df)) if i not in duplicates_to_remove]
            deduplicated_df = testset_df.iloc[indices_to_keep].reset_index(drop=True)
            
            self.logger.info(f"Removed {len(duplicates_to_remove)} duplicate questions")
            return deduplicated_df
            
        except ImportError:
            # Fallback to simple text-based deduplication
            deduplicated_df = testset_df.drop_duplicates(subset=['question'], keep='first')
            self.logger.info(f"Simple deduplication: kept {len(deduplicated_df)} unique questions")
            return deduplicated_df
    
    def _combine_and_format_results(self, results: Dict[str, Any], output_dir: Path) -> pd.DataFrame:
        """Combine and format final testset results"""
        
        if results['method'] == 'hybrid':
            final_testset = results['testset']
        else:
            final_testset = results['testset']
        
        if len(final_testset) == 0:
            self.logger.warning("⚠️ No testset samples generated")
            return final_testset
        
        # Ensure required columns exist
        required_columns = ['question', 'answer']
        for col in required_columns:
            if col not in final_testset.columns:
                final_testset[col] = ''
        
        # Reorder columns for better readability
        column_order = [
            'question', 'answer', 'auto_keywords', 'source_file', 
            'question_type', 'generation_method', 'generation_timestamp'
        ]
        
        # Add missing columns
        for col in column_order:
            if col not in final_testset.columns:
                final_testset[col] = ''
        
        # Reorder
        final_testset = final_testset[column_order + [col for col in final_testset.columns if col not in column_order]]
        
        # Save testset files
        self._save_testset_files(final_testset, output_dir)
        
        return final_testset
    
    def _save_testset_files(self, testset_df: pd.DataFrame, output_dir: Path):
        """Save testset in multiple formats"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save as Excel (primary format)
            excel_file = output_dir / f"hybrid_testset_{timestamp}.xlsx"
            testset_df.to_excel(excel_file, index=False, engine='openpyxl')
            self.logger.info(f"💾 Saved Excel testset: {excel_file}")
            
            # Save as CSV (backup format)
            csv_file = output_dir / f"hybrid_testset_{timestamp}.csv"
            testset_df.to_csv(csv_file, index=False, encoding='utf-8')
            self.logger.info(f"💾 Saved CSV testset: {csv_file}")
            
            # Save metadata
            metadata_file = output_dir / f"testset_metadata_{timestamp}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            self.logger.info(f"💾 Saved metadata: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving testset files: {e}")
    
    def _create_temp_config(self) -> str:
        """Create temporary configuration file for your existing generator"""
        temp_config = {
            'custom_data': {
                'enabled': True,
                'data_sources': {
                    'pdf_files': [],  # Will be updated with actual document paths
                }
            },
            'testset_generation': self.generation_config
        }
        
        temp_config_path = Path(__file__).parent / 'temp_config.yaml'
        
        import yaml
        with open(temp_config_path, 'w') as f:
            yaml.dump(temp_config, f)
        
        return str(temp_config_path)
    
    def _update_temp_config_with_docs(self, processed_docs: List[Dict[str, Any]]):
        """Update temporary config with processed document paths"""
        import yaml
        
        temp_config_path = Path(__file__).parent / 'temp_config.yaml'
        
        with open(temp_config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update with document paths
        pdf_files = [doc['path'] for doc in processed_docs if doc['path'].endswith('.pdf')]
        txt_files = [doc['path'] for doc in processed_docs if doc['path'].endswith(('.txt', '.md'))]
        
        config['custom_data']['data_sources']['pdf_files'] = pdf_files
        if txt_files:
            config['custom_data']['data_sources']['text_files'] = txt_files
        
        with open(temp_config_path, 'w') as f:
            yaml.dump(config, f)
    
    def _convert_to_langchain_docs(self, processed_docs: List[Dict[str, Any]]) -> List:
        """Convert processed documents to LangChain document format"""
        try:
            from langchain.schema import Document
            
            langchain_docs = []
            for doc in processed_docs:
                langchain_doc = Document(
                    page_content=doc['content'],
                    metadata={
                        'source': doc['path'],
                        'name': doc['name'],
                        **doc['metadata']
                    }
                )
                langchain_docs.append(langchain_doc)
            
            return langchain_docs
            
        except ImportError:
            self.logger.error("LangChain not available for document conversion")
            return []
    
    def generate_testsets(self, processed_documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compatibility method for legacy orchestrator API.
        
        Args:
            processed_documents: List of processed document dictionaries
            
        Returns:
            List of testset generation results (legacy format)
        """
        self.logger.info("🔄 Using compatibility API for testset generation")
        
        try:
            # Extract document paths from processed documents
            document_paths = []
            for doc in processed_documents:
                if 'source_file' in doc:
                    document_paths.append(doc['source_file'])
                elif 'path' in doc:
                    document_paths.append(doc['path'])
                elif 'filename' in doc:
                    # Assume it's in the parent directory
                    document_paths.append(f"../{doc['filename']}")
                else:
                    self.logger.warning(f"Could not extract path from document: {doc}")
            
            if not document_paths:
                self.logger.error("No valid document paths found in processed documents")
                return []
            
            # Use the comprehensive testset generation
            results = self.generate_comprehensive_testset(
                document_paths=document_paths,
                output_dir=Path("outputs/testsets")  # Default output dir
            )
            
            # Convert to legacy format
            testset_data = results.get('testset', [])
            metadata = results.get('metadata', {})
            
            # Create legacy-compatible result
            legacy_result = {
                'source_document': document_paths[0] if document_paths else 'unknown',
                'qa_pairs': testset_data if isinstance(testset_data, list) else testset_data.to_dict('records'),
                'output_file': 'hybrid_testset_output.xlsx',
                'method': metadata.get('generation_method', 'hybrid'),
                'samples_generated': len(testset_data),
                'generation_metadata': metadata
            }
            
            return [legacy_result]  # Return as list for compatibility
            
        except Exception as e:
            self.logger.error(f"Compatibility method failed: {e}")
            return []
