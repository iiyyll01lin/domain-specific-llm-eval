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
import traceback
import requests
import yaml
from datetime import datetime, timedelta
import json
import hashlib
import gzip
import os

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ragas"))

# Import your existing systems
from generate_dataset_configurable import ConfigurableDatasetGenerator
from document_loader import DocumentLoader

# Import DocumentProcessor for consistent document handling
from data.document_processor import DocumentProcessor

# Import CSV processing components - Use existing DocumentLoader instead
# from data.csv_document_processor import CSVDocumentProcessor
# from data.csv_testset_generator import CSVTestsetGenerator

# Import RAGAS components
try:
    from ragas.testset import TestsetGenerator
    # Note: evolutions module no longer exists in current RAGAS version
    # Evolution types are now handled within the TestsetGenerator
    from langchain.embeddings import HuggingFaceEmbeddings
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
        
        # Initialize generation settings - config is already the testset_generation section
        self.method = config.get('method', 'configurable')
        self.samples_per_doc = config.get('samples_per_document', 10)
        self.max_total_samples = config.get('max_total_samples', 50)
        
        # Initialize generation_config
        self.generation_config = config  # Use the full config as generation_config
        
        # Initialize your existing components
        self.configurable_generator = None
        self.ragas_generator = None
        
        # CSV processing components - Use DocumentLoader instead
        # self.csv_processor = None
        # self.csv_testset_generator = None
        
        # Document processing - use the config that was already loaded and passed to us
        # The 'config' parameter already contains the full configuration
        self.document_processor = DocumentProcessor(config, 'outputs/testsets')
        self.document_loader = DocumentLoader({'testset_generation': config.get('testset_generation', {})})  # Keep for compatibility
        self.processed_documents = []
        
        # Results tracking
        self.generation_results = []
        self.metadata = {
            'generation_start': datetime.now().isoformat(),
            'total_generated': 0,
            'source_documents': [],
            'generation_method': self.method
        }        
        self.logger.info(f"HybridTestsetGenerator initialized with method: {self.method}")
        
        # Debug: Log current working directory and paths
        import os
        current_dir = os.getcwd()
        file_dir = Path(__file__).parent
        self.logger.info(f"🔍 Constructor Debug:")
        self.logger.info(f"   Current working directory: {current_dir}")
        self.logger.info(f"   __file__ location: {__file__}")
        self.logger.info(f"   File parent directory: {file_dir}")
        self.logger.info(f"   sys.path preview: {sys.path[:3]}...")
        
        # Initialize generators after configuration is complete
        print("🔧 DEBUG: About to call initialize_generators() from constructor...")
        try:
            self.initialize_generators()
            print("✅ DEBUG: initialize_generators() completed successfully")
            print(f"✅ DEBUG: configurable_generator = {self.configurable_generator}")
        except Exception as e:
            print(f"❌ DEBUG: initialize_generators() failed in constructor: {e}")
            import traceback
            print(f"❌ DEBUG: Constructor traceback: {traceback.format_exc()}")
            # Don't re-raise here, let the pipeline continue and fail later with better context
    
    def initialize_generators(self):
        """Initialize the specific generators based on configuration"""
        print("🔧 DEBUG: initialize_generators() method called")
        print(f"🔧 DEBUG: self.method = '{self.method}'")
        print(f"🔧 DEBUG: Checking if method in ['configurable', 'hybrid']: {self.method in ['configurable', 'hybrid']}")
        try:
            # Initialize your existing configurable generator
            if self.method in ['configurable', 'hybrid']:
                print("🔧 DEBUG: Initializing configurable dataset generator...")
                self.logger.info("Initializing configurable dataset generator...")
                
                # Create a minimal config for ConfigurableDatasetGenerator (NO temp file needed)
                minimal_config = {
                    'mode': 'local',
                    'custom_data': {'enabled': False},  # Don't load documents - we'll inject them
                    'dataset': {
                        'num_samples': self.samples_per_doc,
                        'output_file': 'temp_output.xlsx'
                    },
                    'local': {
                        'sentence_model': 'all-MiniLM-L6-v2',
                        'keybert': {'enabled': True},
                        'yake': {'enabled': True},
                        'thresholds': {
                            'relevance_threshold': 0.3,
                            'similarity_min': 0.1,
                            'similarity_max': 1.0
                        }
                    },
                    'fallback': {
                        'score_ranges': {
                            'context_precision': [0.6, 0.9],
                            'context_recall': [0.6, 0.9], 
                            'faithfulness': [0.6, 0.9],
                            'answer_relevancy': [0.6, 0.9],
                            'kw_metric': [0.5, 0.9]
                        }
                    },
                    'logging': {'level': 'INFO', 'show_progress': True}
                }
                
                # Initialize ConfigurableDatasetGenerator with in-memory config
                print("🔧 DEBUG: Creating ConfigurableDatasetGenerator object...")
                self.configurable_generator = ConfigurableDatasetGenerator.__new__(ConfigurableDatasetGenerator)
                self.configurable_generator.config = minimal_config
                self.configurable_generator.mode = 'local'
                self.configurable_generator.custom_documents = []
                self.configurable_generator.custom_topics = []
                print(f"🔧 DEBUG: ConfigurableDatasetGenerator created: {self.configurable_generator}")
                
                # Initialize the internal generator with proper path handling
                print("🔧 DEBUG: About to start try block for LocalSyntheticDatasetGenerator import...")
                try:
                    # Add the correct path for import
                    import sys
                    from pathlib import Path
                    eval_pipeline_dir = str(Path(__file__).parent.parent.parent)
                    self.logger.info(f"🔍 Attempting import from: {eval_pipeline_dir}")
                    if eval_pipeline_dir not in sys.path:
                        sys.path.insert(0, eval_pipeline_dir)
                        self.logger.info(f"🔍 Added to sys.path: {eval_pipeline_dir}")
                    
                    # Check if file exists
                    import_file = Path(eval_pipeline_dir) / "local_dataset_generator.py"
                    self.logger.info(f"🔍 Looking for file: {import_file}")
                    self.logger.info(f"🔍 File exists: {import_file.exists()}")
                    
                    # Now import with proper path
                    self.logger.info("🔍 Attempting import of LocalSyntheticDatasetGenerator...")
                    from local_dataset_generator import LocalSyntheticDatasetGenerator
                    self.logger.info("✅ Import successful, creating generator instance...")
                    self.configurable_generator.generator = LocalSyntheticDatasetGenerator(minimal_config)
                    self.logger.info("✅ LocalSyntheticDatasetGenerator imported and initialized successfully")
                    
                except ImportError as e:
                    self.logger.error(f"❌ Failed to import LocalSyntheticDatasetGenerator: {e}")
                    self.logger.error(f"   Current sys.path: {sys.path}")
                    self.logger.error(f"   Looking for file at: {eval_pipeline_dir}/local_dataset_generator.py")
                    
                    # Fallback: set configurable_generator to None to prevent AttributeError
                    self.configurable_generator = None
                    raise ImportError(f"Cannot import LocalSyntheticDatasetGenerator: {e}")
                
                except Exception as e:
                    self.logger.error(f"❌ Unexpected error during generator creation: {e}")
                    self.logger.error(f"   Error type: {type(e)}")
                    import traceback
                    self.logger.error(f"   Full traceback: {traceback.format_exc()}")
                    self.configurable_generator = None
                    raise Exception(f"Generator creation failed: {e}")
                
                # Rest of the method continues...
                if self.configurable_generator and hasattr(self.configurable_generator, 'generator'):
                    # Inject already processed documents directly (NO document loading)
                    if self.processed_documents:
                        processed_content = [doc['content'] for doc in self.processed_documents if doc.get('content')]
                        if processed_content:
                            self.configurable_generator.custom_documents = processed_content
                            self.configurable_generator.generator.documents = processed_content
                            self.logger.info(f"✅ Injected {len(processed_content)} processed documents directly")
                    else:
                        # Use the default documents from LocalSyntheticDatasetGenerator if no processed documents
                        self.logger.info("⚠️ No processed documents found, using default corpus")
                        # The LocalSyntheticDatasetGenerator already has default documents in its constructor
                    
                    self.logger.info("✅ Configurable generator initialized (NO temp config used)")
            
            # Initialize RAGAS generator if available and needed
            print(f"🔧 DEBUG: method='{self.method}', RAGAS_AVAILABLE={RAGAS_AVAILABLE}")
            print(f"🔧 DEBUG: method in ['ragas', 'hybrid']: {self.method in ['ragas', 'hybrid']}")
            if self.method in ['ragas', 'hybrid'] and RAGAS_AVAILABLE:
                print("🔧 DEBUG: Initializing RAGAS generator...")
                self.logger.info("Initializing RAGAS testset generator...")
                self._initialize_ragas_generator()
                self.logger.info("✅ RAGAS generator initialized")
                print(f"🔧 DEBUG: RAGAS generator initialized: {self.ragas_generator is not None}")
            elif self.method in ['ragas', 'hybrid'] and not RAGAS_AVAILABLE:
                print("🔧 DEBUG: RAGAS not available, falling back...")
                self.logger.warning("⚠️ RAGAS not available, falling back to configurable method")
                self.method = 'configurable'
            else:
                print(f"🔧 DEBUG: Skipping RAGAS initialization - method='{self.method}', RAGAS_AVAILABLE={RAGAS_AVAILABLE}")
                
        except Exception as e:
            self.logger.error(f"❌ Generator initialization failed: {e}")
            # Log more details about the error
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise
    
    def _initialize_ragas_generator(self):
        """Initialize RAGAS TestsetGenerator with custom LLM support"""
        try:
            # Get RAGAS configuration
            ragas_config = self.generation_config.get('ragas_config', {})
              # Initialize embeddings (always local for privacy)
            embeddings_model = ragas_config.get('embeddings_model', 'sentence-transformers/all-MiniLM-L6-v2')
            embeddings = HuggingFaceEmbeddings(model_name=embeddings_model)
            
            # Initialize LLM based on configuration
            llm = None
              # Check for custom LLM configuration first
            if ragas_config.get('use_custom_llm', False):
                custom_llm_config = ragas_config.get('custom_llm', {})
                llm = self._create_custom_llm(custom_llm_config, temperature=0.3)
                if llm:
                    self.logger.info("✅ Using custom LLM for RAGAS generation")
                else:
                    self.logger.error("❌ Custom LLM configuration failed")
                    llm = None
            else:
                self.logger.warning("⚠️ No custom LLM configured - RAGAS generation not available")
                llm = None
            
            # Create RAGAS generator
            if llm:
                self.ragas_generator = TestsetGenerator.from_langchain(
                    llm=llm,
                    embedding_model=embeddings
                )
                self.logger.info("RAGAS generator created with LLM support")
            else:
                self.logger.warning("⚠️ No LLM available for RAGAS - cannot generate testsets")
                self.ragas_generator = None
                
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
            from ragas.llms import LangchainLLMWrapper  # ✅ Import RAGAS wrapper
            
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
                        # Log the prompt for debugging (first 200 chars)
                        prompt_preview = prompt[:200] + "..." if len(prompt) > 200 else prompt
                        # self.logger.debug(f"🔤 LLM Prompt: {prompt_preview}")
                        
                        # Prepare request payload - Updated to match your working API
                        payload = {
                            "model": self.model,
                            "messages": [
                                {"role": "system", "content": ""},
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
                            timeout=60  # Increased timeout for complex prompts
                        )
                        
                        # self.logger.debug(f"📡 API Response Status: {response.status_code}")
                        
                        if response.status_code == 200:
                            result = response.json()
                            # self.logger.debug(f"📄 API Response Keys: {list(result.keys())}")
                            
                            # Extract content from response
                            if "choices" in result and len(result["choices"]) > 0:
                                content = result["choices"][0].get("message", {}).get("content", "")
                                if content:
                                    # self.logger.debug(f"✅ Generated {len(content)} characters")
                                    return content.strip()
                                else:
                                    raise ValueError("Empty content in API response")
                            else:
                                raise ValueError(f"Unexpected response format: {result}")
                        else:
                            error_msg = f"API call failed with status {response.status_code}: {response.text}"
                            # self.logger.error(f"❌ {error_msg}")
                            raise ValueError(error_msg)
                            
                    except Exception as e:
                        error_msg = f"Custom LLM call failed: {str(e)}"
                        # self.logger.error(f"❌ {error_msg}")
                        # Return a generic error response instead of failing completely
                        return f"I apologize, but I cannot process this request due to a technical issue: {str(e)}"
            
            # Load API key from secrets file or use direct key
            api_key = custom_llm_config.get('api_key')
            if not api_key:
                api_key = self._load_api_key_from_secrets(custom_llm_config)
            
            # Create custom LLM instance
            langchain_llm = CustomLLMWrapper(
                endpoint=custom_llm_config.get('endpoint', ''),
                api_key=api_key,
                model=custom_llm_config.get('model', 'gpt-4o'),  # ✅ Updated default
                temperature=temperature,
                max_tokens=custom_llm_config.get('max_tokens', 1000),
                headers=custom_llm_config.get('headers', {})
            )
            
            # ✅ **KEY FIX**: Wrap in RAGAS LangchainLLMWrapper for proper integration
            ragas_llm = LangchainLLMWrapper(langchain_llm)
            
            self.logger.info("✅ Custom LLM wrapped for RAGAS integration")
            return ragas_llm
            
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
                                     output_dir) -> Dict[str, Any]:
        """
        Generate comprehensive testset using hybrid approach
        
        Args:
            document_paths: List of document file paths
            output_dir: Directory to save generated testsets (string or Path)
        
        Returns:
            Dictionary containing generation results and metadata
        """
        # Ensure output_dir is a Path object
        output_dir = Path(output_dir)
        
        self.logger.info(f"🚀 Starting hybrid testset generation for {len(document_paths)} documents")
        
        # Initialize generators
        self.initialize_generators()
        
        # Check if we're using CSV input
        if self.method == 'csv' or self._should_use_csv_input():
            self.logger.info("📊 Using CSV input method")
            results = self._generate_from_csv(output_dir)
        else:
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
        
        # Update metadata with processed document info
        try:
            if self.method == 'csv':
                self.metadata.update({
                    'total_generated': len(combined_testset),
                    'source_documents': ['CSV input'],
                    'documents_processed': 1
                })
            else:
                processed_docs = self.processed_documents if hasattr(self, 'processed_documents') else []
                self.metadata.update({
                    'total_generated': 0,  # Will be updated after generation
                    'source_documents': [doc.get('name', 'unknown') for doc in processed_docs],
                    'documents_processed': len(processed_docs)
                })
        except Exception as e:
            self.logger.error(f"❌ Error updating metadata: {e}")
        
        # Update metadata
        try:
            self.metadata.update({
                'generation_end': datetime.now().isoformat(),
                'total_generated': len(combined_testset),
                'source_documents': [str(Path(p).name) for p in document_paths],
                'generation_results': results
            })
        except Exception as e:
            print(f"❌ Error updating metadata: {e}")
            import traceback
            traceback.print_exc()
        
        self.logger.info(f"✅ Hybrid testset generation completed. Generated {len(combined_testset)} samples")
        
        try:
            final_result = {
                'testset': combined_testset,
                'metadata': self.metadata,
                'results_by_method': results
            }
            return final_result
        except Exception as e:
            print(f"❌ Error creating final result: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _process_documents(self, document_paths: List[str]) -> List[Dict[str, Any]]:
        """Process documents for testset generation using the fixed DocumentProcessor"""
        self.logger.info("📄 Processing documents using DocumentProcessor...")        
        
        # Use the DocumentProcessor which properly handles the Redfish PDF
        processed_docs = self.document_processor.process_documents()
        
        if not processed_docs:
            self.logger.warning("❌ No documents were processed by DocumentProcessor!")
            return []
        
        # Convert DocumentProcessor format to HybridTestsetGenerator format
        converted_docs = []
        for doc in processed_docs:
            converted_doc = {
                'path': doc.get('source_file', ''),
                'name': doc.get('filename', ''),
                'content': doc.get('content', ''),
                'metadata': {'source': doc.get('source_file', ''), 'type': doc.get('file_type', '')},
                'pages': [],  # DocumentProcessor doesn't provide pages
                'word_count': doc.get('word_count', 0),
                'processing_timestamp': datetime.now().isoformat()
            }
            converted_docs.append(converted_doc)
            self.logger.info(f"✅ Converted {doc.get('filename', 'unknown')}: {doc.get('word_count', 0)} words")
        
        self.logger.info(f"📄 Document processing completed. {len(converted_docs)} documents ready")
        
        # IMPORTANT: Store processed documents for generator access
        self.processed_documents = converted_docs
        self.logger.info(f"✅ Stored {len(self.processed_documents)} documents in self.processed_documents")
        
        # DEBUG: Check document content for RAGAS compatibility
        for i, doc in enumerate(converted_docs):
            content_length = len(doc['content'])
            word_count = doc['word_count']
            self.logger.info(f"Doc {i+1}: {doc['name']} - {content_length} chars, {word_count} words")
            
            # Check for Redfish content
            content = doc['content'].lower()
            redfish_keywords = ['redfish', 'api', 'schema', 'properties', 'json', 'resource', 'protocol', 'dmtf', 'bmc']
            found_keywords = [kw for kw in redfish_keywords if kw in content]
            self.logger.info(f"  Redfish keywords found: {found_keywords}")
            
            if content_length > 0:
                preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
                self.logger.info(f"  Content preview: {preview}")
        
        return converted_docs
    
    def _generate_with_configurable(self, 
                                  processed_docs: List[Dict[str, Any]], 
                                  output_dir: Path) -> Dict[str, Any]:
        """Generate testset using your existing configurable method"""
        self.logger.info("🔄 Generating testset with configurable method...")
        
        try:
            # Ensure processed documents are stored for generator access
            self.processed_documents = processed_docs
            
            # Initialize generators now that we have processed documents
            if not self.configurable_generator:
                self.logger.warning("⚠️ configurable_generator is None, attempting to initialize...")
                try:
                    self.initialize_generators()
                    if self.configurable_generator:
                        self.logger.info("✅ Successfully initialized configurable_generator")
                    else:
                        self.logger.error("❌ configurable_generator is still None after initialization")
                        raise RuntimeError("Failed to initialize configurable_generator")
                except Exception as e:
                    self.logger.error(f"❌ Failed to initialize generators: {e}")
                    import traceback
                    self.logger.error(f"Full traceback: {traceback.format_exc()}")
                    raise RuntimeError(f"Generator initialization failed: {e}")
            
            # Safety check: Ensure configurable_generator is not None before proceeding
            if not self.configurable_generator:
                self.logger.error("❌ configurable_generator is None - cannot proceed with generation")
                raise RuntimeError("configurable_generator is None after initialization attempt")
            
            # Ensure documents are properly injected into the configurable generator
            if self.processed_documents:
                processed_content = [doc['content'] for doc in self.processed_documents if doc.get('content')]
                if processed_content:
                    # Inject into configurable generator
                    self.configurable_generator.custom_documents = processed_content
                    
                    # Also inject into the internal generator
                    if hasattr(self.configurable_generator, 'generator'):
                        self.configurable_generator.generator.documents = processed_content
                        self.logger.info(f"✅ Injected {len(processed_content)} documents into generator.documents")
                    
                    # Force re-initialization of any internal state that depends on documents
                    if hasattr(self.configurable_generator.generator, 'initialize_with_documents'):
                        self.configurable_generator.generator.initialize_with_documents(processed_content)
                    
                    # Log success with document preview
                    self.logger.info(f"✅ Successfully injected {len(processed_content)} processed documents")
                    for i, content in enumerate(processed_content[:2]):  # Show first 2 docs
                        # Handle both string content and Document objects
                        if hasattr(content, 'page_content'):
                            text_content = content.page_content
                        elif hasattr(content, 'content'):
                            text_content = content.content
                        elif isinstance(content, str):
                            text_content = content
                        else:
                            text_content = str(content)
                        
                        preview = text_content[:200] + "..." if len(text_content) > 200 else text_content
                        self.logger.info(f"  Doc {i+1} preview: {preview}")
                else:
                    self.logger.warning("⚠️ No valid document content found in processed_docs")
            else:
                self.logger.warning("⚠️ No processed documents available")
            
            # Generate using your existing system
            testset_df = self.configurable_generator.generate_dataset()
            
            # Check if we got valid results
            if testset_df is None or len(testset_df) == 0:
                self.logger.warning("⚠️ Configurable generator returned empty result, using fallback")
                testset_df = self._generate_fallback_testset()
            
            # Add source tracking and metadata
            enhanced_testset = self._enhance_testset_with_metadata(
                testset_df, processed_docs, method='configurable'
            )
            
            self.logger.info(f"✅ Configurable method generated {len(enhanced_testset)} samples")
            
            result = {
                'method': 'configurable',
                'testset': enhanced_testset,
                'samples_generated': len(enhanced_testset),
                'success': True
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Configurable generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback result
            fallback_testset = self._generate_fallback_testset()
            return {
                'method': 'configurable',
                'testset': fallback_testset,
                'samples_generated': len(fallback_testset),
                'success': False,
                'error': str(e)
            }
    
    def _generate_with_ragas(self, 
                           processed_docs: List[Dict[str, Any]], 
                           output_dir: Path) -> Dict[str, Any]:
        """Enhanced RAGAS generation with multiple implementation strategies"""
        self.logger.info("🔄 Generating testset with enhanced RAGAS method...")
        
        # Check generation strategy configuration
        ragas_config = self.generation_config.get('ragas_config', {})
        generation_strategy = ragas_config.get('generation_strategy', '')
        csv_to_ragas_mode = self.generation_config.get('csv_to_ragas_mode', False)
        use_full_kg_pipeline = ragas_config.get('use_full_knowledge_graph_pipeline', False)
        
        print(f"🎯 DEBUG: RAGAS generation strategy: '{generation_strategy}'")
        print(f"🎯 DEBUG: Use full KG pipeline: {use_full_kg_pipeline}")
        print(f"🎯 DEBUG: CSV to RAGAS mode: {csv_to_ragas_mode}")
        print(f"🎯 DEBUG: Available config keys: {list(self.generation_config.keys())}")
        print(f"🎯 DEBUG: Available ragas_config keys: {list(ragas_config.keys())}")
        
        self.logger.info(f"🎯 RAGAS generation strategy: {generation_strategy}")
        self.logger.info(f"🎯 Use full KG pipeline: {use_full_kg_pipeline}")
        
        # Choose implementation based on configuration
        if generation_strategy == "full_ragas_with_knowledge_graph" or use_full_kg_pipeline:
            print("🔄 DEBUG: Using Full RAGAS Knowledge Graph Pipeline")
            self.logger.info("🔄 Using Full RAGAS Knowledge Graph Pipeline")
            return self._generate_with_full_ragas_implementation(processed_docs, output_dir)
        
        elif csv_to_ragas_mode:
            self.logger.info("🔄 Using CSV-to-RAGAS conversion mode")
            return self._generate_with_csv_ragas_integration(processed_docs, output_dir)
        
        else:
            self.logger.info("🔄 Using standard RAGAS document processing")
            return self._generate_with_standard_ragas(processed_docs, output_dir)
    
    def _aggregate_documents_for_better_clusters(self, processed_docs, target_token_count=None):
        """
        Aggregate short documents into larger chunks to improve multi-hop relationship creation
        
        Args:
            processed_docs: List of processed documents (dict format from CSV processing)
            target_token_count: Target token count per aggregated document (None = use config)
            
        Returns:
            List of aggregated documents with better clustering potential
        """
        # Get aggregation configuration from pipeline config
        csv_config = self.generation_config.get('csv_processing', {})
        aggregation_config = csv_config.get('document_aggregation', {})
        
        # Use configuration values or defaults
        if target_token_count is None:
            target_token_count = aggregation_config.get('target_token_count', 500)
        
        min_chunk_size = aggregation_config.get('min_chunk_size', 100)
        max_chunk_size = aggregation_config.get('max_chunk_size', 1000) 
        overlap_tokens = aggregation_config.get('overlap_tokens', 50)
        preserve_categories = aggregation_config.get('preserve_categories', True)
        enable_aggregation = aggregation_config.get('enable_aggregation', True)
        aggregation_strategy = aggregation_config.get('aggregation_strategy', 'balanced')
        force_aggregation_threshold = aggregation_config.get('force_aggregation_threshold', 100)
        
        # ✅ NEW: Multi-hop specific configuration
        multi_hop_threshold = aggregation_config.get('multi_hop_threshold', 1500)  # Minimum tokens for multi-hop
        multi_hop_overlap_ratio = aggregation_config.get('multi_hop_overlap_ratio', 0.15)  # 15% overlap for better relationships
        enable_cross_category_mixing = aggregation_config.get('enable_cross_category_mixing', True)  # Mix categories for better clustering
        
        self.logger.info(f"🔄 Document Aggregation Settings:")
        self.logger.info(f"   Target tokens: {target_token_count}")
        self.logger.info(f"   Min chunk size: {min_chunk_size}")
        self.logger.info(f"   Max chunk size: {max_chunk_size}")
        self.logger.info(f"   Overlap tokens: {overlap_tokens}")
        self.logger.info(f"   Preserve categories: {preserve_categories}")
        self.logger.info(f"   Aggregation enabled: {enable_aggregation}")
        self.logger.info(f"   Strategy: {aggregation_strategy}")
        self.logger.info(f"   🎯 Multi-hop threshold: {multi_hop_threshold} tokens")
        self.logger.info(f"   🎯 Multi-hop overlap ratio: {multi_hop_overlap_ratio:.1%}")
        self.logger.info(f"   🎯 Cross-category mixing: {enable_cross_category_mixing}")
        
        # Adjust target based on strategy
        if aggregation_strategy == "multi_hop_optimized":
            # Override target for multi-hop optimization
            target_token_count = max(target_token_count, multi_hop_threshold)
            self.logger.info(f"   🚀 Multi-hop optimization: Using {target_token_count} target tokens")
        
        # Skip aggregation if disabled
        if not enable_aggregation:
            self.logger.info("📋 Document aggregation disabled by configuration")
            return processed_docs
        
        self.logger.info(f"🔄 Aggregating documents for better clustering (target: {target_token_count} tokens)...")
        
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
        except ImportError:
            self.logger.warning("⚠️ tiktoken not available, using word count estimation")
            encoding = None
        
        def count_tokens(text):
            if encoding:
                return len(encoding.encode(text))
            else:
                # Rough estimation: 1 token ≈ 0.75 words
                return int(len(text.split()) * 1.33)
        
        # Group documents by label/category if available
        if preserve_categories and not enable_cross_category_mixing:
            # Standard category-based grouping
            grouped_docs = {}
            
            for doc in processed_docs:
                # Handle both dict and object formats
                if isinstance(doc, dict):
                    content = doc.get('page_content', doc.get('content', ''))
                    metadata = doc.get('metadata', {})
                else:
                    content = getattr(doc, 'page_content', getattr(doc, 'content', ''))
                    metadata = getattr(doc, 'metadata', {})
                
                # Extract label from metadata for grouping
                label = "general"
                if metadata:
                    if 'label' in metadata:
                        label = metadata['label']
                    elif 'template_key' in metadata:
                        label = metadata['template_key']
                    elif 'source' in metadata and isinstance(metadata['source'], str):
                        label = metadata['source'][:20]  # Use first 20 chars of source
                
                if label not in grouped_docs:
                    grouped_docs[label] = []
                grouped_docs[label].append(doc)
        
        elif aggregation_strategy == "multi_hop_optimized":
            # ✅ NEW: Multi-hop optimization strategy
            self.logger.info("🎯 Using multi-hop optimization strategy...")
            
            # Group documents more intelligently for multi-hop relationships
            grouped_docs = self._create_multi_hop_optimized_groups(
                processed_docs, multi_hop_threshold, count_tokens
            )
            
        else:
            # Don't group by category - treat all docs as one group for maximum mixing
            grouped_docs = {"mixed_documents": processed_docs}
            self.logger.info("📊 Processing all documents as mixed category for better clustering")
        
        self.logger.info(f"📊 Grouped documents into {len(grouped_docs)} categories: {list(grouped_docs.keys())}")
        
        aggregated_docs = []
        
        for label, docs in grouped_docs.items():
            self.logger.info(f"🔄 Processing category '{label}' with {len(docs)} documents...")
            
            # Apply different aggregation strategies
            if aggregation_strategy == "multi_hop_optimized":
                # Multi-hop strategy: Create larger, more diverse chunks
                chunk_docs = self._aggregate_for_multi_hop(
                    docs, label, multi_hop_threshold, count_tokens, 
                    multi_hop_overlap_ratio, max_chunk_size
                )
            elif aggregation_strategy == "max_size":
                # Max size strategy: Fill chunks to maximum capacity
                chunk_docs = self._aggregate_max_size(
                    docs, label, max_chunk_size, count_tokens, overlap_tokens
                )
            elif aggregation_strategy == "min_chunks":
                # Min chunks strategy: Create fewer, larger chunks
                chunk_docs = self._aggregate_min_chunks(
                    docs, label, target_token_count * 1.5, count_tokens, overlap_tokens
                )
            else:
                # Balanced strategy (default): Standard aggregation
                chunk_docs = self._aggregate_balanced(
                    docs, label, target_token_count, min_chunk_size, 
                    max_chunk_size, count_tokens, overlap_tokens
                )
            
            aggregated_docs.extend(chunk_docs)
            self.logger.info(f"✅ Category '{label}': {len(docs)} docs → {len(chunk_docs)} aggregated chunks")
        
        self.logger.info(f"🎯 Document aggregation complete: {len(processed_docs)} → {len(aggregated_docs)} documents")
        
        # Log token distribution
        token_counts = []
        for doc in aggregated_docs:
            content = doc.get('page_content', doc.get('content', ''))
            token_counts.append(count_tokens(content))
        
        if token_counts:
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)
            self.logger.info(f"📊 Token distribution: avg={avg_tokens:.0f}, min={min_tokens}, max={max_tokens}")
        
        return aggregated_docs

    def _create_multi_hop_optimized_groups(self, processed_docs, multi_hop_threshold, count_tokens):
        """
        Create document groups optimized for multi-hop question generation
        
        Strategy:
        1. Mix documents from different categories to increase diversity
        2. Ensure each group has enough tokens for meaningful relationships
        3. Create overlapping groups to enable cross-references
        
        Args:
            processed_docs: List of processed documents
            multi_hop_threshold: Minimum tokens required for multi-hop
            count_tokens: Function to count tokens in text
            
        Returns:
            Dict of grouped documents optimized for multi-hop clustering
        """
        
        # Step 1: Categorize documents by type
        categories = {}
        uncategorized = []
        
        for doc in processed_docs:
            if isinstance(doc, dict):
                metadata = doc.get('metadata', {})
            else:
                metadata = getattr(doc, 'metadata', {})
            
            category = None
            if metadata:
                if 'template_key' in metadata:
                    category = metadata['template_key']
                elif 'label' in metadata:
                    category = metadata['label']
                elif 'source' in metadata:
                    category = str(metadata['source'])[:15]
            
            if category:
                if category not in categories:
                    categories[category] = []
                categories[category].append(doc)
            else:
                uncategorized.append(doc)
        
        self.logger.info(f"📊 Multi-hop grouping: {len(categories)} categories, {len(uncategorized)} uncategorized")
        
        # Step 2: Create mixed groups for better multi-hop relationships
        mixed_groups = {}
        group_id = 1
        
        # Strategy A: Mix documents from different categories
        if len(categories) >= 2:
            category_lists = list(categories.values())
            max_docs_per_category = 3  # Limit to avoid huge groups
            
            # Create groups by taking documents from each category
            for i in range(0, max(len(cat) for cat in category_lists), max_docs_per_category):
                group_docs = []
                group_tokens = 0
                
                for cat_docs in category_lists:
                    # Take a few docs from this category
                    start_idx = i
                    end_idx = min(i + max_docs_per_category, len(cat_docs))
                    
                    for doc in cat_docs[start_idx:end_idx]:
                        content = doc.get('content', doc.get('page_content', '')) if isinstance(doc, dict) else getattr(doc, 'content', getattr(doc, 'page_content', ''))
                        doc_tokens = count_tokens(content)
                        
                        if group_tokens + doc_tokens <= multi_hop_threshold * 1.5:  # Allow some overflow
                            group_docs.append(doc)
                            group_tokens += doc_tokens
                        else:
                            break
                
                if group_docs and group_tokens >= multi_hop_threshold * 0.7:  # Minimum threshold
                    mixed_groups[f"multi_hop_group_{group_id}"] = group_docs
                    group_id += 1
                    self.logger.info(f"🎯 Created multi-hop group {group_id-1}: {len(group_docs)} docs, {group_tokens} tokens")
        
        # Strategy B: Handle remaining documents and uncategorized
        remaining_docs = uncategorized.copy()
        
        # Add any documents not yet grouped
        for category, cat_docs in categories.items():
            for doc in cat_docs:
                if not any(doc in group for group in mixed_groups.values()):
                    remaining_docs.append(doc)
        
        # Group remaining documents
        if remaining_docs:
            current_group = []
            current_tokens = 0
            
            for doc in remaining_docs:
                content = doc.get('content', doc.get('page_content', '')) if isinstance(doc, dict) else getattr(doc, 'content', getattr(doc, 'page_content', ''))
                doc_tokens = count_tokens(content)
                
                if current_tokens + doc_tokens <= multi_hop_threshold * 1.2:
                    current_group.append(doc)
                    current_tokens += doc_tokens
                else:
                    # Save current group if it meets minimum threshold
                    if current_tokens >= multi_hop_threshold * 0.6:
                        mixed_groups[f"remaining_group_{group_id}"] = current_group
                        group_id += 1
                    
                    # Start new group
                    current_group = [doc]
                    current_tokens = doc_tokens
            
            # Don't forget the last group
            if current_group and current_tokens >= multi_hop_threshold * 0.6:
                mixed_groups[f"final_group_{group_id}"] = current_group
        
        # Fallback: If no groups created, create one big group
        if not mixed_groups:
            mixed_groups["fallback_group"] = processed_docs[:20]  # Limit size
            self.logger.warning("⚠️ Multi-hop optimization failed, using fallback group")
        
        self.logger.info(f"🎉 Multi-hop optimization complete: {len(mixed_groups)} groups created")
        return mixed_groups

    def _aggregate_for_multi_hop(self, docs, label, multi_hop_threshold, count_tokens, overlap_ratio, max_chunk_size):
        """Aggregate documents specifically optimized for multi-hop generation"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_count = 0
        
        # Calculate dynamic overlap based on ratio
        overlap_tokens = int(multi_hop_threshold * overlap_ratio)
        
        for doc in docs:
            content = doc.get('content', doc.get('page_content', '')) if isinstance(doc, dict) else getattr(doc, 'content', getattr(doc, 'page_content', ''))
            doc_tokens = count_tokens(content)
            
            # Check if we should save current chunk (prioritize reaching multi-hop threshold)
            should_save = False
            
            if current_tokens >= multi_hop_threshold and (current_tokens + doc_tokens) > max_chunk_size:
                should_save = True
            elif current_tokens >= multi_hop_threshold * 1.5:  # Large enough for good multi-hop
                should_save = True
            
            if should_save and current_chunk:
                chunk_count += 1
                chunk_doc = self._create_chunk_document(
                    current_chunk, label, chunk_count, current_tokens, "multi_hop_optimized"
                )
                chunks.append(chunk_doc)
                
                # Start new chunk with significant overlap for multi-hop relationships
                if overlap_tokens > 0:
                    overlap_content = current_chunk[-overlap_tokens*4:]  # Rough character estimation
                    current_chunk = overlap_content
                    current_tokens = count_tokens(current_chunk)
                else:
                    current_chunk = ""
                    current_tokens = 0
            
            # Add document to current chunk
            if current_chunk:
                current_chunk += "\n\n--- Document ---\n\n"
            current_chunk += content
            current_tokens += doc_tokens
        
        # Save the final chunk if it meets minimum multi-hop threshold
        if current_chunk and current_tokens >= multi_hop_threshold * 0.8:
            chunk_count += 1
            chunk_doc = self._create_chunk_document(
                current_chunk, label, chunk_count, current_tokens, "multi_hop_optimized"
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def _aggregate_balanced(self, docs, label, target_tokens, min_chunk_size, max_chunk_size, count_tokens, overlap_tokens):
        """Standard balanced aggregation strategy"""
        chunks = []
        current_chunk = ""
        current_tokens = 0
        chunk_count = 0
        
        for doc in docs:
            content = doc.get('content', doc.get('page_content', '')) if isinstance(doc, dict) else getattr(doc, 'content', getattr(doc, 'page_content', ''))
            doc_tokens = count_tokens(content)
            
            # Standard aggregation logic
            should_save = False
            if current_tokens > 0 and (current_tokens + doc_tokens) > max_chunk_size:
                should_save = True
            elif current_tokens >= min_chunk_size and (current_tokens + doc_tokens) > target_tokens:
                should_save = True
            
            if should_save and current_chunk:
                chunk_count += 1
                chunk_doc = self._create_chunk_document(
                    current_chunk, label, chunk_count, current_tokens, "balanced"
                )
                chunks.append(chunk_doc)
                
                # Handle overlap
                if overlap_tokens > 0:
                    overlap_content = current_chunk[-overlap_tokens*4:]
                    current_chunk = overlap_content
                    current_tokens = count_tokens(current_chunk)
                else:
                    current_chunk = ""
                    current_tokens = 0
            
            if current_chunk:
                current_chunk += "\n\n--- Document ---\n\n"
            current_chunk += content
            current_tokens += doc_tokens
        
        # Final chunk
        if current_chunk and current_tokens >= min_chunk_size:
            chunk_count += 1
            chunk_doc = self._create_chunk_document(
                current_chunk, label, chunk_count, current_tokens, "balanced"
            )
            chunks.append(chunk_doc)
        
        return chunks
    
    def _create_chunk_document(self, content, label, chunk_id, token_count, strategy):
        """Create a standardized chunk document"""
        return {
            'content': content.strip(),
            'metadata': {
                'aggregated': True,
                'chunk_id': f"{label}_chunk_{chunk_id}",
                'token_count': token_count,
                'category': label,
                'strategy': strategy,
                'multi_hop_ready': token_count >= 1000  # Mark as multi-hop ready
            },
            'name': f"{label}_aggregated_chunk_{chunk_id}",
            'path': f"aggregated/{label}_chunk_{chunk_id}",
            'source_file': 'aggregated_documents'
        }

    def _generate_with_full_ragas_implementation(self, 
                                               processed_docs: List[Dict[str, Any]], 
                                               output_dir: Path) -> Dict[str, Any]:
        """Generate testset using Full RAGAS Knowledge Graph implementation with multi-hop support"""
        try:
            print("🚀 DEBUG: Starting Full RAGAS Knowledge Graph Pipeline with Multi-Hop Support...")
            self.logger.info("🚀 Starting Full RAGAS Knowledge Graph Pipeline with Multi-Hop Support...")
            
            # STEP 1: Aggregate documents for better clustering
            print("📋 DEBUG: Step 1: Aggregating documents for improved multi-hop relationships...")
            self.logger.info("📋 Step 1: Aggregating documents for improved multi-hop relationships...")
            print(f"📋 DEBUG: Input documents count: {len(processed_docs)}")
            self.logger.info(f"📋 Input documents count: {len(processed_docs)}")
            
            try:
                if processed_docs:
                    sample_doc = processed_docs[0]
                    print(f"📋 DEBUG: Sample input document keys: {list(sample_doc.keys())}")
                    print(f"📋 DEBUG: Sample content preview: {str(sample_doc.get('content', ''))[:100]}...")
                    self.logger.info(f"📋 Sample input document keys: {list(sample_doc.keys())}")
                    self.logger.info(f"📋 Sample content preview: {str(sample_doc.get('content', ''))[:100]}...")
                
                print("📋 DEBUG: About to call aggregation function...")
                aggregated_docs = self._aggregate_documents_for_better_clusters(
                    processed_docs
                    # target_token_count now comes from configuration
                )
                print(f"📋 DEBUG: Aggregation completed, got {len(aggregated_docs) if aggregated_docs else 0} docs")
                
            except Exception as e:
                print(f"❌ DEBUG: Aggregation failed with error: {e}")
                print(f"❌ DEBUG: Error type: {type(e).__name__}")
                import traceback
                print(f"❌ DEBUG: Traceback: {traceback.format_exc()}")
                aggregated_docs = processed_docs  # Fallback to original docs            
            print(f"📋 DEBUG: Output aggregated documents count: {len(aggregated_docs)}")
            self.logger.info(f"📋 Output aggregated documents count: {len(aggregated_docs)}")
            if aggregated_docs:
                sample_agg_doc = aggregated_docs[0]
                print(f"📋 DEBUG: Sample aggregated document keys: {list(sample_agg_doc.keys())}")
                print(f"📋 DEBUG: Sample aggregated content preview: {str(sample_agg_doc.get('content', ''))[:100]}...")
                self.logger.info(f"📋 Sample aggregated document keys: {list(sample_agg_doc.keys())}")
                self.logger.info(f"📋 Sample aggregated content preview: {str(sample_agg_doc.get('content', ''))[:100]}...")

            # Convert processed documents to LangChain format for multi-hop KG
            print("📋 DEBUG: Converting aggregated docs to LangChain format...")
            langchain_docs = self._convert_to_langchain_docs(aggregated_docs)
            print(f"📋 DEBUG: LangChain conversion completed, got {len(langchain_docs)} docs")
            
            if not langchain_docs:
                raise ValueError("No valid documents for multi-hop knowledge graph creation")
            
            # Create multi-hop compatible knowledge graph using the fixed implementation
            multi_hop_kg = self._create_multi_hop_compatible_knowledge_graph(
                langchain_docs, 
                max_nodes=self.generation_config.get('max_documents_for_generation', 20)
            )
            
            if not multi_hop_kg or len(multi_hop_kg.nodes) == 0:
                raise ValueError("Failed to create multi-hop compatible knowledge graph")
            
            # Get LLM for generation
            llm = None
            ragas_config = self.generation_config.get('ragas_config', {})
            if ragas_config.get('use_custom_llm', False):
                custom_llm_config = ragas_config.get('custom_llm', {})
                llm = self._create_custom_llm(custom_llm_config, temperature=0.7)
                
            if not llm:
                raise ValueError("No LLM available for multi-hop generation")
            
            # Get question distribution configuration
            distribution_config = self.generation_config.get('question_distribution', {
                "single_hop_entities": 0.4,
                "multi_hop_abstract": 0.3,
                "multi_hop_specific": 0.3
            })
            
            # Generate testset with configurable distribution
            testset_size = min(
                self.samples_per_doc * len(processed_docs),
                self.max_total_samples
            )
            
            testset = self._generate_testset_with_configurable_distribution(
                kg=multi_hop_kg,
                llm=llm,
                testset_size=testset_size,
                distribution_config=distribution_config
            )
            
            if not testset or not hasattr(testset, 'samples') or len(testset.samples) == 0:
                raise ValueError("Multi-hop testset generation returned no samples")
            
            # Convert to DataFrame
            testset_df = testset.to_pandas()
            
            # Normalize field names and ensure all required fields exist
            field_mapping = {
                'user_input': 'question',
                'reference_contexts': 'contexts', 
                'response': 'answer',
                'reference': 'ground_truth'
            }
            
            existing_mapping = {k: v for k, v in field_mapping.items() if k in testset_df.columns}
            if existing_mapping:
                testset_df = testset_df.rename(columns=existing_mapping)
            
            # Ensure all required fields exist - if answer is missing, use ground_truth
            if 'answer' not in testset_df.columns:
                if 'ground_truth' in testset_df.columns:
                    testset_df['answer'] = testset_df['ground_truth']
                    self.logger.info("📝 Generated 'answer' field from 'ground_truth' for testset compatibility")
                elif 'response' in testset_df.columns:
                    testset_df['answer'] = testset_df['response']
                    self.logger.info("📝 Generated 'answer' field from 'response' for testset compatibility")
                else:
                    # Generate placeholder answers if none exist
                    testset_df['answer'] = testset_df.apply(
                        lambda row: f"Based on the provided context, this question requires information from: {row.get('contexts', 'the documents')}",
                        axis=1
                    )
                    self.logger.info("📝 Generated placeholder 'answer' field for testset compatibility")
            
            # Ensure contexts is properly formatted
            if 'contexts' in testset_df.columns:
                testset_df['contexts'] = testset_df['contexts'].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x) if x is not None else ""
                )
            
            # Enhance with metadata
            enhanced_testset = self._enhance_testset_with_metadata(
                testset_df, processed_docs, method='full_ragas_multi_hop'
            )
            
            # Save testset
            self._save_testset_files(enhanced_testset, output_dir)
            
            self.logger.info(f"✅ Multi-hop RAGAS pipeline completed: {len(enhanced_testset)} samples")
            self.logger.info(f"📊 Knowledge Graph: {len(multi_hop_kg.nodes)} nodes, {len(multi_hop_kg.relationships)} relationships")
            
            # Debug: Check for clusters and analyze knowledge graph structure
            try:
                # Try multiple import paths for DirectedGraph based on RAGAS version
                DirectedGraph = None
                try:
                    from ragas.testset.graph import DirectedGraph
                except ImportError:
                    try:
                        from ragas.testset.common.graph import DirectedGraph
                    except ImportError:
                        try:
                            # Custom minimal DirectedGraph implementation as fallback
                            class DirectedGraph:
                                def __init__(self):
                                    self.edges = []
                                
                                def add_edge(self, source, target):
                                    self.edges.append((source, target))
                                
                                def connected_components(self):
                                    # Simple connected components using DFS
                                    nodes = set()
                                    for source, target in self.edges:
                                        nodes.add(source)
                                        nodes.add(target)
                                    
                                    visited = set()
                                    components = []
                                    
                                    def dfs(node, component):
                                        if node in visited:
                                            return
                                        visited.add(node)
                                        component.add(node)
                                        
                                        for source, target in self.edges:
                                            if source == node:
                                                dfs(target, component)
                                            elif target == node:
                                                dfs(source, component)
                                    
                                    for node in nodes:
                                        if node not in visited:
                                            component = set()
                                            dfs(node, component)
                                            if component:
                                                components.append(component)
                                    
                                    return components
                        except Exception:
                            DirectedGraph = None
                
                if DirectedGraph and hasattr(multi_hop_kg, 'relationships') and multi_hop_kg.relationships:
                    edges = [(rel.source.get_property("id"), rel.target.get_property("id")) for rel in multi_hop_kg.relationships]
                    self.logger.info(f"🔍 Debug: Found {len(edges)} edges for clustering")
                    self.logger.info(f"🔍 Debug: Sample edges: {edges[:5]}")
                    
                    # Create a directed graph for clustering analysis
                    graph = DirectedGraph()
                    for edge in edges:
                        graph.add_edge(edge[0], edge[1])
                    
                    clusters = graph.connected_components()
                    self.logger.info(f"🔍 Debug: Found {len(clusters)} clusters")
                    for i, cluster in enumerate(clusters):
                        self.logger.info(f"  Cluster {i}: {len(cluster)} nodes - {list(cluster)[:3]}")
                        
                    # Check relationship types
                    rel_types = {}
                    for rel in multi_hop_kg.relationships:
                        rel_type = rel.type
                        rel_types[rel_type] = rel_types.get(rel_type, 0) + 1
                    self.logger.info(f"🔍 Debug: Relationship types: {rel_types}")
                else:
                    self.logger.warning("🔍 Debug: No relationships found in knowledge graph or DirectedGraph not available")
                    self.logger.info(f"🔍 Debug: KG has {len(multi_hop_kg.nodes)} nodes")
                    if hasattr(multi_hop_kg, 'relationships'):
                        self.logger.info(f"🔍 Debug: KG has {len(multi_hop_kg.relationships)} relationships")
            except Exception as e:
                self.logger.warning(f"🔍 Debug clustering failed: {e}")
                import traceback
                self.logger.warning(f"🔍 Debug traceback: {traceback.format_exc()}")
            
            # Save the knowledge graph if enabled
            if self.generation_config.get('ragas_config', {}).get('enable_kg_saving', False):
                try:
                    self.save_knowledge_graph(multi_hop_kg, run_id=self.run_id)
                    self.logger.info("💾 Knowledge graph saved successfully")
                except Exception as e:
                    self.logger.error(f"❌ Failed to save knowledge graph: {e}")
            
            return {
                'method': 'full_ragas_multi_hop',
                'testset': enhanced_testset,
                'samples_generated': len(enhanced_testset),
                'success': True,
                'generation_config': self.generation_config,
                'knowledge_graph_stats': {
                    'nodes': len(multi_hop_kg.nodes),
                    'relationships': len(multi_hop_kg.relationships),
                    'source_documents': len(processed_docs)
                },
                'question_distribution_used': distribution_config
            }
                
        except Exception as e:
            self.logger.error(f"❌ Multi-hop RAGAS implementation failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'method': 'full_ragas_multi_hop',
                'testset': pd.DataFrame(),
                'samples_generated': 0,
                'success': False,
                'error': str(e)
            }

    def _generate_with_csv_ragas_integration(self, 
                                           processed_docs: List[Dict[str, Any]], 
                                           output_dir: Path) -> Dict[str, Any]:
        """Generate testset using CSV-to-RAGAS integration (Legacy)"""
        try:
            self.logger.info("🚀 Starting CSV-to-RAGAS integration...")
            
            # Import the CSV-to-RAGAS converter
            from data.csv_ragas_converter import CSVToRagasConverter
            
            # Create converter with FULL pipeline configuration (not just generation_config)
            # The CSV converter needs access to data_sources.csv configuration
            full_config = {}
            
            # Check if we have access to the complete config via _full_config reference
            if '_full_config' in self.config:
                full_config = self.config['_full_config']
                self.logger.info("📊 Using complete pipeline configuration for CSV converter")
            elif hasattr(self, '_full_config'):
                full_config = self._full_config
                self.logger.info("📊 Using stored full configuration for CSV converter")
            else:
                # Fallback: try to construct a working config
                full_config = {
                    'testset_generation': self.generation_config,
                    'data_sources': getattr(self, 'data_sources_config', {}),
                }
                self.logger.warning("📊 Using partial configuration for CSV converter - may not work")
            
            converter = CSVToRagasConverter(full_config)
            
            # Convert CSV to RAGAS testset
            result = converter.convert_csv_to_ragas_testset()
            
            # Extract testset DataFrame and metadata
            testset_df = result['testset_df']
            metadata = result['metadata']
            
            self.logger.info(f"✅ CSV-to-RAGAS conversion completed: {len(testset_df)} samples")
            
            # Save testset
            self._save_testset_files(testset_df, output_dir)
            
            return {
                'method': 'csv_ragas',
                'testset': testset_df,
                'samples_generated': len(testset_df),
                'success': True,
                'metadata': metadata,
                'generation_config': self.generation_config
            }
            
        except Exception as e:
            self.logger.error(f"❌ CSV-to-RAGAS integration failed: {e}")
            import traceback
            traceback.print_exc()
            
            return {
                'method': 'csv_ragas',
                'testset': pd.DataFrame(),
                'samples_generated': 0,
                'success': False,
                'error': str(e)
            }
    
    def _generate_with_standard_ragas(self, 
                                    processed_docs: List[Dict[str, Any]], 
                                    output_dir: Path) -> Dict[str, Any]:
        """Generate testset using standard RAGAS method"""
        self.logger.info("🔄 Generating testset with standard RAGAS method...")
        
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
            
            # Debug: Check document content
            self.logger.info(f"📄 Processing {len(langchain_docs)} documents for RAGAS")
            for i, doc in enumerate(langchain_docs[:3]):  # Show first 3 docs
                content_length = len(doc.page_content) if hasattr(doc, 'page_content') else 0
                self.logger.info(f"  Document {i+1}: {content_length} characters")
                if content_length > 0:
                    # Show first 200 characters
                    preview = doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content
                    self.logger.info(f"  Preview: {preview}")
                    
                    # Check for technical content that might confuse RAGAS
                    technical_indicators = ['API', 'HTTP', 'GET', 'POST', 'JSON', 'REST', 'endpoint', 'RedFish']
                    found_indicators = [ind for ind in technical_indicators if ind.lower() in doc.page_content.lower()]
                    self.logger.info(f"  Technical indicators found: {found_indicators}")
            
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
            
            # Generate testset with RAGAS            # Generate testset using RAGAS
            # Note: Current RAGAS API doesn't support custom distributions
            # It uses default query scenarios internally
            
            # Progressive retry with smaller sample sizes
            sample_sizes_to_try = [total_samples, max(5, total_samples // 2), 5, 2, 1]
            
            testset = None
            for attempt, sample_size in enumerate(sample_sizes_to_try):
                self.logger.info(f"🚀 Attempt {attempt + 1}: Starting RAGAS generation with {sample_size} samples...")
                try:
                    # ✅ SOLUTION 1: Create completely custom RAGAS generation that bypasses all problematic transforms
                    self.logger.info("🔧 Using custom RAGAS generation with CSV-compatible approach")
                    
                    # Instead of using generate_with_langchain_docs which applies default transforms,
                    # we'll use the RAGAS synthesizers directly to avoid transform pipeline issues
                    testset = self._generate_ragas_testset_custom(langchain_docs, sample_size)
                    
                    self.logger.info(f"🔍 Custom RAGAS generation completed. Testset type: {type(testset)}")
                    
                    # Check if testset has samples
                    if hasattr(testset, 'samples'):
                        self.logger.info(f"📊 RAGAS generated {len(testset.samples)} samples")
                        if len(testset.samples) > 0:
                            self.logger.info(f"✅ Success! RAGAS generated {len(testset.samples)} samples with size {sample_size}")
                            break
                        else:
                            self.logger.warning(f"⚠️ Attempt {attempt + 1} generated 0 samples with size {sample_size}")
                            if attempt < len(sample_sizes_to_try) - 1:
                                self.logger.info("🔄 Trying with smaller sample size...")
                                continue
                            else:
                                raise ValueError("RAGAS failed to generate samples even with minimal requests")
                    else:
                        self.logger.warning(f"⚠️ RAGAS testset has no 'samples' attribute: {dir(testset)}")
                        if attempt < len(sample_sizes_to_try) - 1:
                            self.logger.info("🔄 Trying with smaller sample size...")
                            continue
                        else:
                            raise ValueError("RAGAS testset structure unexpected")
                            
                except Exception as e:
                    self.logger.error(f"❌ Attempt {attempt + 1} failed: {e}")
                    if attempt < len(sample_sizes_to_try) - 1:
                        self.logger.info("🔄 Retrying with smaller sample size...")
                        continue
                    else:
                        # Re-raise the last exception
                        raise e
                        
            # Convert to DataFrame and normalize field names
            self.logger.info("🔄 Converting RAGAS testset to pandas DataFrame...")
            testset_df = testset.to_pandas()
            
            self.logger.info(f"📋 DataFrame shape: {testset_df.shape}")
            self.logger.info(f"📋 DataFrame columns: {list(testset_df.columns)}")
            
            # Check if DataFrame is empty
            if testset_df.empty:
                self.logger.error("❌ RAGAS returned an empty DataFrame!")
                self.logger.error("This indicates RAGAS sample generation failed silently")
                raise ValueError("RAGAS generated empty testset - no samples created")
            
            # Normalize field names from RAGAS to expected format
            field_mapping = {
                'user_input': 'question',
                'reference_contexts': 'contexts', 
                'response': 'answer',
                'reference': 'ground_truth'
            }
            
            self.logger.info(f"🔄 Applying field mapping: {field_mapping}")
            # Only rename columns that actually exist
            existing_mapping = {k: v for k, v in field_mapping.items() if k in testset_df.columns}
            if existing_mapping:
                testset_df = testset_df.rename(columns=existing_mapping)
                self.logger.info(f"✅ Renamed columns: {existing_mapping}")
            else:
                self.logger.warning(f"⚠️ No expected RAGAS columns found for mapping: {list(testset_df.columns)}")
            
            # Ensure contexts is properly formatted (RAGAS returns list, we might need string)
            if 'contexts' in testset_df.columns:
                self.logger.info("🔄 Converting contexts from list to string format...")
                testset_df['contexts'] = testset_df['contexts'].apply(
                    lambda x: x[0] if isinstance(x, list) and len(x) > 0 else str(x) if x is not None else ""
                )
            
            # Validate required columns before enhancement
            required_columns = ['question', 'contexts', 'answer']
            missing_columns = [col for col in required_columns if col not in testset_df.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                self.logger.error(f"Available columns: {list(testset_df.columns)}")
                raise ValueError(f"Missing required columns for enhancement: {missing_columns}")
            
            self.logger.info("✅ All required columns present, proceeding with enhancement...")
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
            self.logger.error(f"RAGAS generation failed: {e}")
            self.logger.error(f"Exception type: {type(e)}")
            
            # Add more specific error handling
            if "empty" in str(e).lower() or "no samples" in str(e).lower():
                self.logger.error("🔍 RAGAS failed to generate any samples from the documents")
                self.logger.error("Possible causes:")
                self.logger.error("  1. Documents too short (RAGAS needs >100 tokens per doc)")
                self.logger.error("  2. Document content not suitable for question generation")
                self.logger.error("  3. LLM/API configuration issues")
                self.logger.error("  4. RAGAS internal generation failure")
                
                # Try to provide fallback suggestions
                self.logger.info("💡 Suggested solutions:")
                self.logger.info("  1. Check document content and length")
                self.logger.info("  2. Try reducing testset_size")
                self.logger.info("  3. Use LOCAL generation method instead")
            
            if self.config.get('logging', {}).get('show_progress', True):
                print(f"❌ RAGAS generation failed: {e}")
            
            # Add debug information about the error
            if hasattr(e, '__traceback__'):
                import traceback
                self.logger.debug(f"RAGAS generation traceback: {traceback.format_exc()}")
            
            # Return structured error result instead of falling back
            return {
                'method': 'ragas',
                'testset': pd.DataFrame(),
                'samples_generated': 0,
                'success': False,
                'error': str(e),
                'error_type': type(e).__name__
            }
    
    def _generate_ragas_testset_custom(self, documents: List, testset_size: int):
        """
        ✅ SOLUTION 1: Custom RAGAS generation that bypasses all problematic transforms
        
        This method creates a direct RAGAS testset generation approach that:
        1. Bypasses default_transforms() which includes HeadlineSplitter and CustomNodeFilter
        2. Uses RAGAS synthesizers directly with CSV-compatible documents
        3. Avoids all the CSV incompatibility issues
        
        Args:
            documents: List of LangChain Document objects
            testset_size: Number of samples to generate
            
        Returns:
            RAGAS Testset object with generated samples
        """
        self.logger.info("🔧 Starting custom RAGAS generation with CSV-compatible approach")
        
        try:
            # Import RAGAS components we need
            from ragas.testset.synthesizers.generate import TestsetGenerator
            from ragas.testset.graph import KnowledgeGraph, Node, NodeType
            from ragas.testset.synthesizers import default_query_distribution
            from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample
            from ragas.dataset_schema import SingleTurnSample
            
            self.logger.info("🔧 Creating minimal knowledge graph manually...")
            
            # ✅ KEY FIX: Create a minimal knowledge graph manually to bypass transform pipeline
            kg = KnowledgeGraph()
            
            # Add documents as simple nodes without requiring headlines/summary properties
            for i, doc in enumerate(documents[:testset_size]):  # Limit documents processed
                try:
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    if len(content.strip()) > 50:  # Only add substantial content
                        node = Node(
                            type=NodeType.DOCUMENT,
                            properties={
                                "page_content": content,
                                "document_metadata": doc.metadata if hasattr(doc, 'metadata') else {},
                                # ✅ Add required properties to avoid transform errors
                                "summary": content[:200] + "..." if len(content) > 200 else content,
                                "headlines": [],  # Empty but present to avoid HeadlineSplitter errors
                                "keyphrases": [],  # Empty but present to avoid KeyphrasesExtractor errors
                                "node_id": f"doc_{i}",
                                "csv_compatible": True
                            }
                        )
                        kg.nodes.append(node)
                        self.logger.debug(f"  Added document node {i+1}: {len(content)} chars")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to add document {i} to KG: {e}")
                    continue
            
            if len(kg.nodes) == 0:
                raise ValueError("No valid documents could be added to knowledge graph")
                
            self.logger.info(f"✅ Created knowledge graph with {len(kg.nodes)} nodes")
            
            # ✅ Create TestsetGenerator with our pre-built knowledge graph
            generator = TestsetGenerator(
                llm=self.ragas_generator.llm,
                embedding_model=self.ragas_generator.embedding_model,
                knowledge_graph=kg  # Use our pre-built KG to bypass transforms
            )
            
            self.logger.info("🎯 Generating testset using RAGAS synthesizers...")
            
            # Use default query distribution
            query_distribution = default_query_distribution(self.ragas_generator.llm)
            
            # Generate testset directly - this bypasses the document processing pipeline
            testset = generator.generate(
                testset_size=testset_size,
                query_distribution=query_distribution
            )
            
            self.logger.info(f"✅ Custom RAGAS generation completed with {len(testset.samples) if hasattr(testset, 'samples') else 0} samples")
            
            return testset
            
        except ImportError as e:
            self.logger.error(f"❌ Missing RAGAS imports: {e}")
            raise ImportError(f"Required RAGAS components not available: {e}")
            
        except Exception as e:
            self.logger.error(f"❌ Custom RAGAS generation failed: {e}")
            self.logger.error(f"Error type: {type(e).__name__}")
            
            # More specific error handling
            if "agenerate_prompt" in str(e):
                self.logger.error("💡 This is a RAGAS LLM wrapper compatibility issue")
                self.logger.error("   The LangchainLLMWrapper is missing required async methods")
                self.logger.info("🔄 Attempting fallback with compatible LLM wrapper...")
                
                # Try creating a fallback testset using custom generation
                return self._create_fallback_ragas_testset(documents, testset_size)
                
            elif "No nodes that satisfied the given filer" in str(e):
                self.logger.error("💡 RAGAS filtering is still failing despite custom knowledge graph")
                self.logger.error("   This indicates RAGAS has internal filters that can't be bypassed")
                self.logger.info("🔄 Creating fallback testset using custom generation...")
                
                # Use fallback generation
                return self._create_fallback_ragas_testset(documents, testset_size)
            
            elif "headlines" in str(e).lower():
                self.logger.error("💡 Headlines property issue - this shouldn't happen with custom approach")
                return self._create_fallback_ragas_testset(documents, testset_size)
            
            elif "summary" in str(e).lower():
                self.logger.error("💡 Summary property issue - this shouldn't happen with custom approach")
                return self._create_fallback_ragas_testset(documents, testset_size)
            
            else:
                # For any other error, try fallback
                self.logger.warning(f"💡 Unexpected error in custom RAGAS generation: {e}")
                self.logger.info("🔄 Attempting fallback generation...")
                return self._create_fallback_ragas_testset(documents, testset_size)
    
    def _create_fallback_ragas_testset(self, documents: List, testset_size: int):
        """
        Create a fallback RAGAS-style testset when the main RAGAS generation fails
        
        This method creates a testset that matches RAGAS format but uses
        our custom LLM directly to avoid compatibility issues.
        """
        self.logger.info("🔄 Creating fallback RAGAS-style testset...")
        
        try:
            from ragas.testset.synthesizers.testset_schema import Testset, TestsetSample
            from ragas.dataset_schema import SingleTurnSample
            
            samples = []
            
            # Use our documents to create synthetic Q&A pairs
            for i, doc in enumerate(documents[:testset_size]):
                try:
                    content = doc.page_content if hasattr(doc, 'page_content') else str(doc)
                    if len(content.strip()) < 50:
                        continue
                        
                    # Create a simple question based on content
                    # Extract key concepts from content
                    words = content.split()
                    key_concepts = [w for w in words if len(w) > 6 and w.isalpha()][:3]
                    
                    if key_concepts:
                        concept = key_concepts[0]
                        question = f"What does the document explain about {concept}?"
                    else:
                        question = f"What is the main topic discussed in this document?"
                    
                    # Use content as context and create a simple answer
                    contexts = [content[:500]]  # Limit context size
                    answer = f"Based on the document, {content[:200]}..."
                    ground_truth = content[:300]
                    
                    # Create RAGAS-compatible sample
                    eval_sample = SingleTurnSample(
                        user_input=question,
                        response=answer,
                        reference_contexts=contexts,
                        reference=ground_truth
                    )
                    
                    testset_sample = TestsetSample(
                        eval_sample=eval_sample,
                        synthesizer_name="custom_fallback"
                    )
                    
                    samples.append(testset_sample)
                    
                    if len(samples) >= testset_size:
                        break
                        
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to create fallback sample {i}: {e}")
                    continue
            
            if not samples:
                raise ValueError("Failed to create any fallback samples")
            
            # Create RAGAS Testset
            testset = Testset(samples=samples)
            
            self.logger.info(f"✅ Fallback testset created with {len(samples)} samples")
            return testset
            
        except Exception as e:
            self.logger.error(f"❌ Fallback testset creation failed: {e}")
            raise ValueError(f"Both RAGAS generation and fallback failed: {e}")
    
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
    
    def _enhance_testset_with_metadata(self, testset_df: pd.DataFrame, 
                                     processed_docs: List[Dict], method: str = 'unknown') -> pd.DataFrame:
        """
        Enhance testset with additional metadata and auto-generated keywords
        
        Args:
            testset_df: DataFrame containing the testset
            processed_docs: List of processed documents for context
            method: Method used to generate the testset
        """
        self.logger.info(f"Enhancing testset with {len(testset_df)} samples using {method} method")
        
        # Debug info
        self.logger.info(f"🔍 Debug: Input testset shape: {testset_df.shape}")
        self.logger.info(f"🔍 Debug: Input testset columns: {list(testset_df.columns)}")
        
        # Validate required columns exist - for testset generation, we use ground_truth instead of answer
        required_columns = ['question', 'contexts', 'ground_truth']
        missing_columns = [col for col in required_columns if col not in testset_df.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            self.logger.error(f"Available columns: {list(testset_df.columns)}")
            raise ValueError(f"Missing required columns for enhancement: {missing_columns}")
        
        try:
            
            def enhance_row(row):
                # Safely access row data with defaults - use bracket notation for pandas Series
                try:
                    question = row['question'] if 'question' in row and pd.notna(row['question']) else ''
                    contexts = row['contexts'] if 'contexts' in row and pd.notna(row['contexts']) else ''
                    # For testset generation, use ground_truth as the reference answer
                    answer = row['ground_truth'] if 'ground_truth' in row and pd.notna(row['ground_truth']) else ''
                    ground_truth = row['ground_truth'] if 'ground_truth' in row and pd.notna(row['ground_truth']) else ''
                except KeyError as e:
                    print(f"❌ KeyError accessing row data: {e}")
                    # Return original row on error
                    return row
                
                # Ensure all fields are strings
                if not isinstance(question, str):
                    question = str(question) if question is not None else ''
                if not isinstance(contexts, str):
                    contexts = str(contexts) if contexts is not None else ''
                if not isinstance(answer, str):
                    answer = str(answer) if answer is not None else ''
                if not isinstance(ground_truth, str):
                    ground_truth = str(ground_truth) if ground_truth is not None else ''
                
                try:
                    # Extract keywords from question and answer
                    question_keywords = self._extract_keywords_safe(question)
                    answer_keywords = self._extract_keywords_safe(answer)
                    
                    # Combine and deduplicate keywords
                    all_keywords = list(set(question_keywords + answer_keywords))
                    
                    # Calculate keyword score based on presence and relevance
                    keyword_score = self._calculate_keyword_score(all_keywords, contexts, answer)
                    
                    # Add metadata to a copy of the row, preserving original dtypes
                    enhanced_row = row.copy()
                    enhanced_row['auto_keywords'] = all_keywords
                    enhanced_row['keyword_score'] = float(keyword_score)  # Ensure numeric type
                    enhanced_row['generation_method'] = str(method)  # Ensure string type
                    enhanced_row['enhanced_at'] = str(pd.Timestamp.now())  # Store as string to avoid dtype issues
                    
                    print(f"✅ Row enhanced successfully")
                    return enhanced_row
                    
                except Exception as e:
                    print(f"❌ Error in row enhancement: {e}")
                    import traceback
                    traceback.print_exc()
                    # Return original row on error
                    return row
            
            # Apply enhancement to each row
            print("🔄 Applying enhancement to rows...")
            enhanced_testset = testset_df.apply(enhance_row, axis=1)
            print(f"✅ Enhancement completed, result shape: {enhanced_testset.shape}")
            
            # CRITICAL FIX: Restore numeric dtypes that may have been converted to object during apply()
            self.logger.info("🔧 Preserving numeric column types after enhancement...")
            
            # Get numeric columns from original DataFrame
            original_numeric_cols = testset_df.select_dtypes(include=['number']).columns
            
            # Restore numeric types for columns that should be numeric
            for col in original_numeric_cols:
                if col in enhanced_testset.columns:
                    try:
                        # Convert back to numeric, handling any conversion errors
                        enhanced_testset[col] = pd.to_numeric(enhanced_testset[col], errors='coerce')
                        self.logger.debug(f"   Restored numeric type for column: {col}")
                    except Exception as e:
                        self.logger.warning(f"   Could not restore numeric type for {col}: {e}")
            
            # Also ensure timestamp column is proper datetime
            if 'enhanced_at' in enhanced_testset.columns:
                try:
                    enhanced_testset['enhanced_at'] = pd.to_datetime(enhanced_testset['enhanced_at'])
                except Exception as e:
                    self.logger.warning(f"Could not convert enhanced_at to datetime: {e}")
            
            # Log the final column types for debugging
            numeric_cols_after = enhanced_testset.select_dtypes(include=['number']).columns
            self.logger.info(f"   Numeric columns after enhancement: {list(numeric_cols_after)}")
            
            return enhanced_testset
        
        except Exception as e:
            self.logger.error(f"❌ Error enhancing testset: {e}")
            return testset_df  # Return original testset on error
    
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
                stop_words='english'
            )
            
            # Take top 5 keywords and extract just the text
            return [kw[0] for kw in keywords[:5]]
            
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
    
    def _save_testset_files(self, testset_df: pd.DataFrame, output_dir):
        """Save testset in multiple formats with dtype preservation"""
        # Ensure output_dir is a Path object
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Save dtype information for later restoration
            dtype_info = {}
            numeric_columns = testset_df.select_dtypes(include=['number']).columns
            for col in numeric_columns:
                dtype_info[col] = 'numeric'
            
            # Save datetime columns
            datetime_columns = testset_df.select_dtypes(include=['datetime']).columns  
            for col in datetime_columns:
                dtype_info[col] = 'datetime'
                
            self.logger.info(f"🔧 Preserving dtypes for {len(dtype_info)} columns: {list(dtype_info.keys())}")
            
            # Save as Excel (primary format)
            excel_file = output_dir / f"hybrid_testset_{timestamp}.xlsx"
            testset_df.to_excel(excel_file, index=False, engine='openpyxl')
            self.logger.info(f"💾 Saved Excel testset: {excel_file}")
            
            # Save as CSV (backup format)
            csv_file = output_dir / f"hybrid_testset_{timestamp}.csv"
            testset_df.to_csv(csv_file, index=False, encoding='utf-8')
            self.logger.info(f"💾 Saved CSV testset: {csv_file}")
            
            # CRITICAL: Save dtype information alongside data
            dtype_file = output_dir / f"testset_dtypes_{timestamp}.json"
            with open(dtype_file, 'w', encoding='utf-8') as f:
                json.dump(dtype_info, f, indent=2)
            self.logger.info(f"💾 Saved dtype info: {dtype_file}")
            
            # Update metadata to include dtype information
            self.metadata['column_dtypes'] = dtype_info
            self.metadata['numeric_columns'] = list(numeric_columns)
            
            # Save metadata
            metadata_file = output_dir / f"testset_metadata_{timestamp}.json"
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2, default=str)
            self.logger.info(f"💾 Saved metadata: {metadata_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Error saving testset files: {e}")
    
    def _convert_to_langchain_docs(self, processed_docs: List[Dict[str, Any]]) -> List:
        """Convert processed documents to LangChain document format"""
        try:
            from langchain.schema import Document
            
            def count_tokens(text: str) -> int:
                """Simple token counting approximation"""
                # Rough approximation: 1 token ≈ 0.75 words
                words = len(text.split())
                return int(words * 0.75)
            
            langchain_docs = []
            for doc in processed_docs:
                content = doc.get('content', '')
                if content and len(content.strip()) > 0:
                    # Check token count - RAGAS requires >100 tokens
                    token_count = count_tokens(content)
                    word_count = len(content.split())
                    self.logger.info(f"Document {doc.get('name', 'unknown')}: ~{token_count} tokens, {word_count} words, {len(content)} characters")
                    
                    if token_count > 20:  # Further lowered from 50 to 20 to include more CSV documents
                        langchain_doc = Document(
                            page_content=content,
                            metadata={
                                'source': doc.get('path', ''),
                                'name': doc.get('name', ''),
                                'token_count': token_count,
                                'word_count': word_count,
                                **doc.get('metadata', {})
                            }
                        )
                        langchain_docs.append(langchain_doc)
                        self.logger.info(f"✅ Added document {doc.get('name', 'unknown')} with ~{token_count} tokens")
                    else:
                        self.logger.warning(f"⚠️ Skipping document {doc.get('name', 'unknown')} - only ~{token_count} tokens (minimum: 21)")
                else:
                    self.logger.warning(f"⚠️ Skipping document {doc.get('name', 'unknown')} - empty content")
            
            self.logger.info(f"📄 Converted {len(langchain_docs)} documents for RAGAS (from {len(processed_docs)} processed)")
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

    def _load_full_pdf_content(self, doc_path: str) -> str:
        """Load full PDF content without chunking for RAGAS"""
        try:
            import PyPDF2
            import pdfplumber
            
            full_text = ""
            
            # Try pdfplumber first (better text extraction)
            try:
                with pdfplumber.open(doc_path) as pdf:
                    self.logger.info(f"📄 Extracting text from {len(pdf.pages)} pages using pdfplumber...")
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text:
                            # Clean up the text
                            cleaned_text = text.strip()
                            if cleaned_text:
                                full_text += cleaned_text + "\n\n"
                        
                        if (i + 1) % 10 == 0:  # Progress every 10 pages
                            self.logger.info(f"  Processed {i + 1}/{len(pdf.pages)} pages...")
                
                if full_text.strip():
                    word_count = len(full_text.split())
                    self.logger.info(f"✅ pdfplumber extracted {len(full_text)} characters, ~{word_count} words")
                    return full_text.strip()
                else:
                    self.logger.warning("⚠️ pdfplumber extracted no text content")
                    
            except Exception as e:
                self.logger.warning(f"⚠️ pdfplumber failed: {e}, trying PyPDF2...")
            
            # Fallback to PyPDF2
            try:
                with open(doc_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    self.logger.info(f"📄 Extracting text from {len(pdf_reader.pages)} pages using PyPDF2...")
                    
                    for i, page in enumerate(pdf_reader.pages):
                        try:
                            text = page.extract_text()
                            if text:
                                cleaned_text = text.strip()
                                if cleaned_text:
                                    full_text += cleaned_text + "\n\n"
                        except Exception as page_error:
                            self.logger.warning(f"⚠️ Error extracting page {i}: {page_error}")
                            continue
                        
                        if (i + 1) % 10 == 0:  # Progress every 10 pages
                            self.logger.info(f"  Processed {i + 1}/{len(pdf_reader.pages)} pages...")
                
                if full_text.strip():
                    word_count = len(full_text.split())
                    self.logger.info(f"✅ PyPDF2 extracted {len(full_text)} characters, ~{word_count} words")
                    return full_text.strip()
                else:
                    self.logger.error("❌ PyPDF2 also extracted no text content")
                    
            except Exception as e:
                self.logger.error(f"❌ PyPDF2 also failed: {e}")
            
            return ""
            
        except ImportError as e:
            self.logger.error(f"❌ PDF libraries not available: {e}")
            return ""

    def _extract_keywords_safe(self, text: str) -> List[str]:
        """
        Safely extract keywords from text with fallback handling
        """
        if not text or not isinstance(text, str):
            return []
        
        try:
            # Try to use configured keyword extraction method
            if hasattr(self, 'keyword_extractor') and self.keyword_extractor:
                return self.keyword_extractor.extract_keywords(text)
            else:
                # Simple fallback: extract meaningful words
                import re
                words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
                return list(set(words))[:10]  # Limit to 10 keywords
        except Exception as e:
            self.logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def _calculate_keyword_score(self, keywords: List[str], contexts: str, answer: str) -> float:
        """
        Calculate a simple keyword relevance score
        """
        if not keywords:
            return 0.0
        
        try:
            combined_text = f"{contexts} {answer}".lower()
            present_keywords = sum(1 for kw in keywords if kw.lower() in combined_text)
            return round(present_keywords / len(keywords), 3)
        except Exception as e:
            self.logger.warning(f"Keyword score calculation failed: {e}")
            return 0.5  # Default score

    def generate(self, processed_documents: List[Dict[str, Any]] = None) -> pd.DataFrame:
        """
        Generate testset using the configured method.
        This method provides a simple interface that's compatible with the existing pipeline.
        
        Args:
            processed_documents: Optional list of processed document dictionaries.
                               If None, will process documents using DocumentProcessor.
            
        Returns:
            DataFrame containing the generated testset
        """
        # If no processed documents provided, process them internally
        if processed_documents is None:
            self.logger.info("📄 No processed documents provided, processing documents internally...")
            processed_documents = self._process_documents([])  # Empty list triggers DocumentProcessor
        
        self.logger.info(f"🚀 Starting testset generation with method '{self.method}' and {len(processed_documents)} documents")
        
        try:
            # Store the processed documents
            self.processed_documents = processed_documents
            
            # Initialize generators if not already done
            self.initialize_generators()
            
            # Route to appropriate generator based on method
            if self.method == 'ragas':
                self.logger.info("🔧 Using RAGAS generation method...")
                if self.ragas_generator:
                    return self._generate_with_ragas_simple(processed_documents)
                else:
                    self.logger.warning("⚠️ RAGAS generator not available, falling back to configurable")
                    # Fall through to configurable method
            
            elif self.method in ['configurable', 'hybrid']:
                self.logger.info("🔧 Using configurable generation method...")
                if self.configurable_generator:
                    # Inject documents into the configurable generator
                    if self.processed_documents:
                        processed_content = [doc['content'] for doc in self.processed_documents if doc.get('content')]
                        if processed_content:
                            self.configurable_generator.custom_documents = processed_content
                            if hasattr(self.configurable_generator, 'generator'):
                                self.configurable_generator.generator.documents = processed_content
                            self.logger.info(f"✅ Injected {len(processed_content)} processed documents")
                        else:
                            self.logger.warning("⚠️ No valid document content found")
                    
                    # Use the generate_dataset method from ConfigurableDatasetGenerator
                    result_df = self.configurable_generator.generate_dataset()
                    
                    if result_df is not None and len(result_df) > 0:
                        self.logger.info(f"✅ Generated {len(result_df)} samples using configurable method")
                        return result_df
                    else:
                        self.logger.warning("⚠️ Configurable generator returned empty result")
                else:
                    self.logger.warning("⚠️ Configurable generator not available")
            
            else:
                self.logger.warning(f"⚠️ Unknown method '{self.method}', using fallback")
                    
            # Fallback: create a minimal testset
            self.logger.warning("⚠️ Using fallback testset generation")
            return self._generate_fallback_testset()
            
        except Exception as e:
            self.logger.error(f"❌ Generation failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._generate_fallback_testset()

    def _generate_with_ragas_simple(self, processed_documents: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Simple wrapper for RAGAS generation that works with the pipeline
        """
        try:
            self.logger.info(f"🔥 Starting RAGAS generation with {len(processed_documents)} documents")
            
            # Convert processed documents to DataFrame for compatibility
            processed_content = [doc['content'] for doc in processed_documents if doc.get('content')]
            
            if not processed_content:
                self.logger.warning("⚠️ No content found in processed documents")
                return self._generate_fallback_testset()
            
            self.logger.info(f"📄 Processing {len(processed_content)} content items with RAGAS")
            
            # Create a temporary output directory
            from pathlib import Path
            import tempfile
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_output = Path(temp_dir)
                
                # Call the existing _generate_with_ragas method
                result = self._generate_with_ragas(processed_content, temp_output)
                
                if isinstance(result, pd.DataFrame) and len(result) > 0:
                    self.logger.info(f"✅ RAGAS generated {len(result)} samples")
                    return result
                else:
                    self.logger.warning("⚠️ RAGAS generation returned empty result")
                    return self._generate_fallback_testset()
                    
        except Exception as e:
            self.logger.error(f"❌ RAGAS generation failed: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return self._generate_fallback_testset()

    def _should_use_csv_input(self) -> bool:
        """Check if CSV input should be used based on configuration"""
        try:
            # Check if data_sources.input_type is set to "csv"
            full_config = self.config
            if isinstance(self.config, dict) and 'data_sources' not in self.config:
                # Load full config if we only have testset_generation section
                from pipeline.config_manager import ConfigManager
                config_manager = ConfigManager('config/pipeline_config.yaml')
                full_config = config_manager.load_config()
            
            input_type = full_config.get('data_sources', {}).get('input_type', 'documents')
            return input_type == 'csv'
            
        except Exception as e:
            self.logger.warning(f"Could not determine input type: {e}")
            return False
    
    def _generate_from_csv(self, output_dir: Path) -> Dict[str, Any]:
        """Generate testset from CSV input files using existing DocumentLoader"""
        self.logger.info("📊 Generating testset from CSV input")
        
        try:
            # Load full config
            from pipeline.config_manager import ConfigManager
            config_manager = ConfigManager('config/pipeline_config.yaml')
            full_config = config_manager.load_config()
            
            # Use existing DocumentLoader to process CSV files
            loader = DocumentLoader(full_config)
            documents, metadata = loader.load_all_documents()
            
            if not documents:
                raise ValueError("No documents loaded from CSV files")
            
            self.logger.info(f"📄 Loaded {len(documents)} documents from CSV")
            
            # Convert documents to processed format for configurable generator
            processed_docs = []
            for i, (doc, meta) in enumerate(zip(documents, metadata)):
                processed_docs.append({
                    'content': doc,
                    'metadata': meta,
                    'source_file': meta.get('source_file', 'csv'),
                    'name': f"CSV_Document_{meta.get('csv_id', i)}",
                    'path': meta.get('source_file', 'csv')
                })
            
            # Use the configured method to generate testset from CSV documents
            self.logger.info(f"🔧 Using {self.method} method for CSV testset generation")
            
            if self.method == 'ragas':
                result = self._generate_with_ragas(processed_docs, output_dir)
            elif self.method == 'configurable':
                result = self._generate_with_configurable(processed_docs, output_dir)
            elif self.method == 'hybrid':
                result = self._generate_hybrid(processed_docs, output_dir)
            else:
                self.logger.warning(f"⚠️ Unknown method '{self.method}', falling back to configurable")
                result = self._generate_with_configurable(processed_docs, output_dir)
            
            # Update result to indicate CSV source
            if result['success']:
                result['method'] = 'csv_configurable'
                result['source_type'] = 'csv'
                result['csv_documents_processed'] = len(documents)
                
                # Add CSV-specific metadata to testset
                if 'testset' in result and len(result['testset']) > 0:
                    testset_df = result['testset'].copy()
                    testset_df['source_type'] = 'csv'
                    testset_df['generation_method'] = 'csv_configurable'
                    result['testset'] = testset_df
            
            self.logger.info(f"✅ CSV method generated {result.get('samples_generated', 0)} samples")
            return result
            
        except Exception as e:
            self.logger.error(f"❌ CSV generation failed: {e}")
            import traceback
            traceback.print_exc()
            
            # Return fallback result
            fallback_testset = self._generate_fallback_testset()
            return {
                'method': 'csv',
                'testset': fallback_testset,
                'samples_generated': len(fallback_testset),
                'success': False,
                'error': str(e)
            }
    
    def _create_multi_hop_compatible_knowledge_graph(self, documents, max_nodes=20):
        """
        Create a knowledge graph that guarantees multi-hop scenario generation
        
        This implementation fixes the root causes identified in the analysis:
        1. Creates nodes with ALL required properties for RAGAS multi-hop
        2. Builds relationships with specific types needed by synthesizers
        3. Ensures minimum clustering requirements are met
        4. Supports saving/loading for performance optimization
        """
        try:
            from ragas.testset.graph import KnowledgeGraph, Node, NodeType, Relationship
            import numpy as np
            
            # Try to load existing knowledge graph first
            source_info = f"csv_docs_{len(documents)}"
            existing_kg = self._load_knowledge_graph(documents, source_info)
            if existing_kg is not None:
                self.logger.info("🎉 Using existing knowledge graph from cache!")
                return existing_kg
            
            # Clean up old KG files
            self._cleanup_old_knowledge_graphs()
            
            self.logger.info(f"🏗️ Creating new multi-hop compatible KG from {min(len(documents), max_nodes)} documents...")
            
            kg = KnowledgeGraph()
            
            # Step 1: Create embeddings model for summary similarity
            try:
                from langchain_community.embeddings import HuggingFaceEmbeddings
                embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                self.logger.info("✅ Created embeddings model for similarity calculations")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not create embeddings model: {e}")
                embeddings_model = None
            
            # Step 2: Create enhanced nodes with ALL required properties
            for i, doc in enumerate(documents[:max_nodes]):
                
                # Extract content and create summary
                content = doc.page_content.strip()
                summary = content[:200] + "..." if len(content) > 200 else content
                
                # Extract entities (for MultiHopSpecificQuerySynthesizer)
                import re
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
                        self.logger.warning(f"⚠️ Could not create embedding for node {i}: {e}")
                
                # ✅ Create node with ALL properties required by RAGAS multi-hop synthesizers
                node_properties = {
                    # Core content
                    "page_content": content,
                    "summary": summary,
                    
                    # ✅ CRITICAL: Properties required by multi-hop synthesizers
                    "entities": entities,           # Required by MultiHopSpecificQuerySynthesizer
                    "keyphrases": keyphrases,       # Alternative property for synthesis
                    "headlines": headlines,         # Required by some transforms
                    
                    # ✅ CRITICAL: Embedding required by MultiHopAbstractQuerySynthesizer
                    "summary_embedding": summary_embedding,
                    
                    # Node identification
                    "id": f"csv_doc_{i}",
                    "title": doc.metadata.get('title', f'Document {i+1}'),
                    "source": doc.metadata.get('source', 'csv_input'),
                    
                    # Enhanced metadata
                    "document_metadata": doc.metadata,
                    "content_length": len(content),
                    "language": doc.metadata.get('language', 'unknown'),
                    "csv_row_index": i,
                    
                    # Quality indicators
                    "has_entities": len(entities) > 0,
                    "has_summary_embedding": summary_embedding is not None,
                    "multi_hop_ready": True
                }
                
                node = Node(
                    type=NodeType.DOCUMENT,  # Use DOCUMENT type for CSV data
                    properties=node_properties
                )
                kg.nodes.append(node)
            
            self.logger.info(f"✅ Created {len(kg.nodes)} enhanced nodes with multi-hop properties")
            
            # Step 3: Create relationships required by multi-hop synthesizers
            relationships_created = 0
            
            self.logger.info("🔗 Creating relationships for multi-hop clustering...")
            
            for i in range(len(kg.nodes)):
                for j in range(i + 1, len(kg.nodes)):
                    node_a = kg.nodes[i]
                    node_b = kg.nodes[j]
                    
                    # ✅ Relationship Type 1: summary_similarity (for MultiHopAbstractQuerySynthesizer)
                    if (node_a.get_property("summary_embedding") is not None and 
                        node_b.get_property("summary_embedding") is not None):
                        
                        try:
                            # Calculate cosine similarity
                            emb_a = np.array(node_a.get_property("summary_embedding"))
                            emb_b = np.array(node_b.get_property("summary_embedding"))
                            
                            similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
                            
                            # Create relationship if similarity > threshold
                            if similarity > 0.3:  # Lower threshold to ensure relationships
                                relationship = Relationship(
                                    source=node_a,
                                    target=node_b,
                                    type="summary_similarity",  # Required by MultiHopAbstractQuerySynthesizer
                                    properties={
                                        "summary_similarity": float(similarity),
                                        "similarity_score": float(similarity)
                                    }
                                )
                                kg.relationships.append(relationship)
                                relationships_created += 1
                                
                        except Exception as e:
                            self.logger.warning(f"⚠️ Error calculating similarity for nodes {i}-{j}: {e}")
                    
                    # ✅ Relationship Type 2: entities_overlap (for MultiHopSpecificQuerySynthesizer)
                    entities_a = set(node_a.get_property("entities") or [])
                    entities_b = set(node_b.get_property("entities") or [])
                    
                    if entities_a and entities_b:
                        overlap = entities_a.intersection(entities_b)
                        if len(overlap) > 0:  # Any overlap creates relationship
                            overlap_ratio = len(overlap) / len(entities_a.union(entities_b))
                            
                            relationship = Relationship(
                                source=node_a,
                                target=node_b,
                                type="entities_overlap",  # Required by MultiHopSpecificQuerySynthesizer
                                properties={
                                    "overlapped_items": {entity: entity for entity in overlap},  # Dictionary format expected by RAGAS
                                    "overlap_count": len(overlap),
                                    "overlap_ratio": float(overlap_ratio)
                                }
                            )
                            kg.relationships.append(relationship)
                            relationships_created += 1
                    
                    # ✅ Relationship Type 3: Content similarity (additional relationships)
                    content_a_words = set(node_a.get_property("page_content").lower().split())
                    content_b_words = set(node_b.get_property("page_content").lower().split())
                    
                    if content_a_words and content_b_words:
                        word_overlap = content_a_words.intersection(content_b_words)
                        if len(word_overlap) > 5:  # Minimum word overlap
                            content_similarity = len(word_overlap) / len(content_a_words.union(content_b_words))
                            
                            if content_similarity > 0.2:  # Lower threshold
                                relationship = Relationship(
                                    source=node_a,
                                    target=node_b,
                                    type="content_similarity",
                                    properties={
                                        "content_similarity": float(content_similarity),
                                        "shared_words": len(word_overlap)
                                    }
                                )
                                kg.relationships.append(relationship)
                                relationships_created += 1
            
            self.logger.info(f"✅ Created {relationships_created} relationships")
            self.logger.info(f"📊 Final KG: {len(kg.nodes)} nodes, {len(kg.relationships)} relationships")
            
            # Step 4: Validate multi-hop requirements
            self.logger.info("🔍 Validating multi-hop requirements...")
            
            # Check for summary_similarity relationships (MultiHopAbstractQuerySynthesizer)
            summary_rels = [r for r in kg.relationships if r.type == "summary_similarity"]
            self.logger.info(f"   Summary similarity relationships: {len(summary_rels)}")
            
            # Check for entities_overlap relationships (MultiHopSpecificQuerySynthesizer)
            entity_rels = [r for r in kg.relationships if r.type == "entities_overlap"]
            self.logger.info(f"   Entity overlap relationships: {len(entity_rels)}")
            
            # Check node properties
            nodes_with_entities = sum(1 for n in kg.nodes if n.get_property("entities"))
            nodes_with_embeddings = sum(1 for n in kg.nodes if n.get_property("summary_embedding") is not None)
            
            self.logger.info(f"   Nodes with entities: {nodes_with_entities}/{len(kg.nodes)}")
            self.logger.info(f"   Nodes with embeddings: {nodes_with_embeddings}/{len(kg.nodes)}")
            
            if len(summary_rels) > 0 and len(entity_rels) > 0:
                self.logger.info("🎉 SUCCESS: Knowledge graph is multi-hop compatible!")
            else:
                self.logger.warning("⚠️ WARNING: Knowledge graph may not support full multi-hop generation")
            
            # Save the knowledge graph for future reuse
            saved_path = self._save_knowledge_graph(kg, documents, source_info)
            if saved_path:
                self.logger.info(f"💾 Knowledge graph saved for future reuse")
            
            return kg
            
        except Exception as e:
            self.logger.error(f"❌ Error creating multi-hop compatible KG: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise

    def _generate_testset_with_configurable_distribution(self, kg, llm, testset_size=10, distribution_config=None):
        """
        Generate testset with configurable question type distribution
        
        Args:
            kg: Knowledge graph
            llm: Language model
            testset_size: Total number of questions to generate
            distribution_config: Dict specifying question type distribution
                               Example: {
                                   "single_hop_entities": 0.4,      # 40% single-hop based on entities
                                   "single_hop_keyphrases": 0.2,    # 20% single-hop based on keyphrases
                                   "multi_hop_abstract": 0.25,      # 25% multi-hop abstract
                                   "multi_hop_specific": 0.15       # 15% multi-hop specific
                               }
        """
        
        if distribution_config is None:
            # Default balanced distribution
            distribution_config = {
                "single_hop_entities": 0.5,
                "multi_hop_abstract": 0.25,
                "multi_hop_specific": 0.25
            }
        
        self.logger.info(f"🎯 Generating testset with configurable distribution...")
        self.logger.info(f"   Testset size: {testset_size}")
        self.logger.info(f"   Distribution config: {distribution_config}")
        
        # Validate distribution
        total_weight = sum(distribution_config.values())
        if abs(total_weight - 1.0) > 0.01:
            self.logger.warning(f"⚠️ Warning: Distribution weights sum to {total_weight:.3f}, should be 1.0")
            # Normalize weights
            distribution_config = {k: v/total_weight for k, v in distribution_config.items()}
            self.logger.info(f"   Normalized distribution: {distribution_config}")
        
        # Create synthesizers
        synthesizers = {}
        
        try:
            from ragas.testset.synthesizers import (
                SingleHopSpecificQuerySynthesizer,
                MultiHopAbstractQuerySynthesizer, 
                MultiHopSpecificQuerySynthesizer
            )
            
            # Single-hop synthesizers
            if "single_hop_entities" in distribution_config:
                synthesizers["single_hop_entities"] = SingleHopSpecificQuerySynthesizer(
                    llm=llm, 
                    property_name="entities"
                )
            
            if "single_hop_keyphrases" in distribution_config:
                synthesizers["single_hop_keyphrases"] = SingleHopSpecificQuerySynthesizer(
                    llm=llm, 
                    property_name="keyphrases"
                )
                
            if "single_hop_headlines" in distribution_config:
                synthesizers["single_hop_headlines"] = SingleHopSpecificQuerySynthesizer(
                    llm=llm, 
                    property_name="headlines"
                )
            
            # Multi-hop synthesizers
            if "multi_hop_abstract" in distribution_config:
                synthesizers["multi_hop_abstract"] = MultiHopAbstractQuerySynthesizer(llm=llm)
                
            if "multi_hop_specific" in distribution_config:
                synthesizers["multi_hop_specific"] = MultiHopSpecificQuerySynthesizer(llm=llm)
            
            self.logger.info(f"📊 Created {len(synthesizers)} synthesizers")
            
        except ImportError as e:
            self.logger.error(f"❌ Failed to import RAGAS synthesizers: {e}")
            raise ImportError(f"RAGAS synthesizers not available: {e}")
        
        # Check synthesizer compatibility
        self.logger.info("🔍 Checking synthesizer compatibility...")
        query_distribution = []
        
        for synth_name, synthesizer in synthesizers.items():
            try:
                if hasattr(synthesizer, 'get_node_clusters'):
                    clusters = synthesizer.get_node_clusters(kg)
                    cluster_count = len(clusters)
                    
                    if cluster_count > 0:
                        weight = distribution_config.get(synth_name, 0)
                        if weight > 0:
                            query_distribution.append((synthesizer, weight))
                            self.logger.info(f"   ✅ {synth_name}: {cluster_count} clusters available, weight: {weight:.1%}")
                        else:
                            self.logger.info(f"   ⚠️ {synth_name}: {cluster_count} clusters but weight is 0")
                    else:
                        self.logger.warning(f"   ❌ {synth_name}: No clusters found, skipping")
                else:
                    self.logger.warning(f"   ⚠️ {synth_name}: No get_node_clusters method")
                    
            except Exception as e:
                self.logger.error(f"   ❌ {synth_name}: Error checking compatibility: {e}")
        
        if not query_distribution:
            self.logger.error("❌ No compatible synthesizers found!")
            raise ValueError("No compatible synthesizers available for testset generation")
        
        # Normalize weights for available synthesizers
        total_available_weight = sum(weight for _, weight in query_distribution)
        query_distribution = [(synth, weight/total_available_weight) for synth, weight in query_distribution]
        
        self.logger.info(f"📊 Final query distribution:")
        for synthesizer, weight in query_distribution:
            synth_name = type(synthesizer).__name__
            self.logger.info(f"   {synth_name}: {weight:.1%}")
        
        # Generate testset using RAGAS
        try:
            from ragas.testset import TestsetGenerator
            from ragas.testset.persona import Persona
            
            # Create personas for diversity
            personas = [
                Persona(name="Technical Engineer", role_description="Engineer working with manufacturing processes"),
                Persona(name="Quality Inspector", role_description="Quality control specialist"),
                Persona(name="Production Manager", role_description="Manager overseeing production operations")
            ]
            
            # Create embeddings if needed
            try:
                from sentence_transformers import SentenceTransformer
                from ragas.embeddings import LangchainEmbeddingsWrapper
                from langchain_community.embeddings import HuggingFaceEmbeddings
                
                langchain_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                ragas_embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
                self.logger.info("✅ Created embeddings for generation")
            except Exception as e:
                self.logger.warning(f"⚠️ Could not create embeddings: {e}")
                ragas_embeddings = None
            
            # Create generator
            generator = TestsetGenerator(
                llm=llm,
                embedding_model=ragas_embeddings,
                knowledge_graph=kg,
                persona_list=personas
            )
            
            self.logger.info("🚀 Generating testset...")
            
            # Generate with configured distribution
            testset = generator.generate(
                testset_size=testset_size,
                query_distribution=query_distribution
            )
            
            self.logger.info(f"✅ Generated testset with {len(testset.samples) if hasattr(testset, 'samples') else 'unknown'} questions")
            
            return testset
            
        except Exception as e:
            self.logger.error(f"❌ Error generating testset: {e}")
            import traceback
            traceback.print_exc()
            raise

    def _generate_fallback_testset(self) -> pd.DataFrame:
        """Generate a minimal fallback testset when main generation fails"""
        fallback_data = {
            'question': ['What is the main topic of this document?'],
            'answer': ['This document contains technical information.'],
            'contexts': ['Document content not available'],
            'context_precision': [0.5],
            'context_recall': [0.5],
            'faithfulness': [0.5],
            'answer_relevancy': [0.5]
        }
        return pd.DataFrame(fallback_data)

    # ============================================================================
    # KNOWLEDGE GRAPH PERSISTENCE METHODS
    # ============================================================================

    def _get_content_hash(self, documents: List[Any]) -> str:
        """Generate a hash from document contents for caching"""
        try:
            content_strings = []
            for doc in documents:
                if hasattr(doc, 'page_content'):
                    content_strings.append(doc.page_content)
                elif isinstance(doc, str):
                    content_strings.append(doc)
                else:
                    content_strings.append(str(doc))
            
            combined_content = "".join(sorted(content_strings))
            return hashlib.md5(combined_content.encode()).hexdigest()[:12]
        except Exception as e:
            self.logger.warning(f"⚠️ Could not generate content hash: {e}")
            return datetime.now().strftime("%Y%m%d_%H%M%S")

    def _get_config_hash(self) -> str:
        """Generate a hash from the current configuration"""
        try:
            config_str = json.dumps(self.config, sort_keys=True, default=str)
            return hashlib.md5(config_str.encode()).hexdigest()[:8]
        except Exception as e:
            self.logger.warning(f"⚠️ Could not generate config hash: {e}")
            return "unknown"

    def _get_kg_filename(self, documents: List[Any], source_info: str = "") -> str:
        """Generate knowledge graph filename based on configuration"""
        try:
            kg_config = self.config.get('testset_generation', {}).get('ragas_config', {}).get('knowledge_graph_config', {})
            naming_config = kg_config.get('kg_naming', {})
            
            # Base components
            prefix = naming_config.get('prefix', 'knowledge_graph')
            content_hash = self._get_content_hash(documents)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Optional components
            parts = [prefix]
            
            if naming_config.get('include_source_info', True) and source_info:
                # Clean source info for filename
                clean_source = "".join(c for c in source_info if c.isalnum() or c in "._-")[:20]
                parts.append(clean_source)
            
            parts.append(content_hash)
            
            if naming_config.get('use_timestamp', True):
                parts.append(timestamp)
            
            if naming_config.get('include_config_hash', True):
                config_hash = self._get_config_hash()
                parts.append(f"cfg_{config_hash}")
            
            # Combine and limit length
            filename = "_".join(parts)
            max_length = naming_config.get('max_filename_length', 200)
            if len(filename) > max_length:
                filename = filename[:max_length-4]  # Leave room for extension
            
            return filename + ".json"
        
        except Exception as e:
            self.logger.warning(f"⚠️ Error generating KG filename: {e}")
            return f"knowledge_graph_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    def _get_kg_storage_path(self, filename: str) -> Path:
        """Get the full storage path for a knowledge graph file"""
        try:
            kg_config = self.config.get('testset_generation', {}).get('ragas_config', {}).get('knowledge_graph_config', {})
            base_dir = kg_config.get('kg_storage', {}).get('base_directory', './outputs/knowledge_graphs')
            
            storage_path = Path(base_dir)
            storage_path.mkdir(parents=True, exist_ok=True)
            
            return storage_path / filename
        
        except Exception as e:
            self.logger.warning(f"⚠️ Error getting KG storage path: {e}")
            fallback_dir = Path('./outputs/knowledge_graphs')
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return fallback_dir / filename

    def _save_knowledge_graph(self, kg, documents: List[Any], source_info: str = "") -> Optional[str]:
        """Save knowledge graph to disk with metadata"""
        try:
            # Debug: print the full config to see what's available
            self.logger.info(f"🔍 Debug: Full config keys: {list(self.config.keys())}")
            
            kg_config = self.config.get('testset_generation', {}).get('ragas_config', {}).get('knowledge_graph_config', {})
            self.logger.info(f"🔍 Debug: KG config found: {kg_config}")
            
            enable_saving = kg_config.get('enable_kg_saving', False)
            self.logger.info(f"🔍 Debug: enable_kg_saving = {enable_saving}")
            
            if not enable_saving:
                self.logger.debug("Knowledge graph saving is disabled")
                return None
            
            # Generate filename and path
            filename = self._get_kg_filename(documents, source_info)
            file_path = self._get_kg_storage_path(filename)
            
            self.logger.info(f"💾 Saving knowledge graph to: {file_path}")
            
            # Create metadata
            metadata = {
                'created_at': datetime.now().isoformat(),
                'source_info': source_info,
                'document_count': len(documents),
                'content_hash': self._get_content_hash(documents),
                'config_hash': self._get_config_hash(),
                'nodes_count': len(kg.nodes) if hasattr(kg, 'nodes') else 0,
                'relationships_count': len(kg.relationships) if hasattr(kg, 'relationships') else 0,
                'ragas_version': self._get_ragas_version(),
                'pipeline_version': '1.0.0'
            }
            
            # Save using RAGAS built-in save method
            if hasattr(kg, 'save'):
                kg.save(str(file_path))
                
                # Save metadata separately
                metadata_path = file_path.with_suffix('.metadata.json')
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                self.logger.info(f"✅ Knowledge graph saved successfully")
                self.logger.info(f"   📄 Graph file: {file_path}")
                self.logger.info(f"   📊 Metadata: {metadata_path}")
                self.logger.info(f"   🔢 Nodes: {metadata['nodes_count']}, Relationships: {metadata['relationships_count']}")
                
                return str(file_path)
            else:
                self.logger.warning("⚠️ Knowledge graph does not support saving")
                return None
                
        except Exception as e:
            self.logger.error(f"❌ Failed to save knowledge graph: {e}")
            return None

    def _load_knowledge_graph(self, documents: List[Any], source_info: str = "") -> Optional[Any]:
        """Load existing knowledge graph if available and valid"""
        try:
            kg_config = self.config.get('testset_generation', {}).get('ragas_config', {}).get('knowledge_graph_config', {})
            
            if not kg_config.get('enable_kg_loading', False):
                self.logger.debug("Knowledge graph loading is disabled")
                return None
            
            if kg_config.get('kg_reuse', {}).get('force_regenerate', False):
                self.logger.info("🔄 Force regeneration enabled, skipping KG loading")
                return None
            
            # Generate expected filename
            filename = self._get_kg_filename(documents, source_info)
            file_path = self._get_kg_storage_path(filename)
            metadata_path = file_path.with_suffix('.metadata.json')
            
            if not file_path.exists():
                self.logger.debug(f"Knowledge graph file not found: {file_path}")
                return None
            
            # Check cache TTL
            cache_config = kg_config.get('kg_cache', {})
            if cache_config.get('enable_cache', True):
                ttl_hours = cache_config.get('cache_ttl_hours', 24)
                if file_path.stat().st_mtime < (datetime.now() - timedelta(hours=ttl_hours)).timestamp():
                    self.logger.info(f"🕒 Knowledge graph cache expired (TTL: {ttl_hours}h)")
                    return None
            
            # Load and validate metadata
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                # Validate content hash if enabled
                reuse_config = kg_config.get('kg_reuse', {})
                if reuse_config.get('content_hash_matching', True):
                    current_hash = self._get_content_hash(documents)
                    if metadata.get('content_hash') != current_hash:
                        self.logger.info("🔄 Content hash mismatch, KG needs regeneration")
                        return None
                
                # Validate config hash if enabled
                if reuse_config.get('metadata_matching', True):
                    current_config_hash = self._get_config_hash()
                    if metadata.get('config_hash') != current_config_hash:
                        self.logger.info("🔄 Configuration changed, KG needs regeneration")
                        return None
            
            # Load knowledge graph
            self.logger.info(f"📖 Loading existing knowledge graph: {file_path}")
            
            try:
                from ragas.testset.graph import KnowledgeGraph
                kg = KnowledgeGraph.load(str(file_path))
                
                self.logger.info(f"✅ Knowledge graph loaded successfully")
                if metadata_path.exists():
                    self.logger.info(f"   🔢 Nodes: {metadata.get('nodes_count', 'unknown')}, Relationships: {metadata.get('relationships_count', 'unknown')}")
                    self.logger.info(f"   📅 Created: {metadata.get('created_at', 'unknown')}")
                
                return kg
                
            except Exception as load_error:
                self.logger.warning(f"⚠️ Failed to load knowledge graph: {load_error}")
                return None
                
        except Exception as e:
            self.logger.warning(f"⚠️ Error loading knowledge graph: {e}")
            return None

    def _get_ragas_version(self) -> str:
        """Get RAGAS version for metadata"""
        try:
            import ragas
            return ragas.__version__ if hasattr(ragas, '__version__') else 'unknown'
        except:
            return 'unknown'

    def _cleanup_old_knowledge_graphs(self):
        """Clean up old knowledge graph files based on cache settings"""
        try:
            kg_config = self.config.get('testset_generation', {}).get('ragas_config', {}).get('knowledge_graph_config', {})
            cache_config = kg_config.get('kg_cache', {})
            
            if not cache_config.get('enable_cache', True):
                return
            
            base_dir = Path(kg_config.get('kg_storage', {}).get('base_directory', './outputs/knowledge_graphs'))
            if not base_dir.exists():
                return
            
            max_size_mb = cache_config.get('max_cache_size_mb', 100)
            ttl_hours = cache_config.get('cache_ttl_hours', 24)
            cutoff_time = datetime.now() - timedelta(hours=ttl_hours)
            
            total_size_mb = 0
            kg_files = []
            
            # Collect all KG files with their info
            for file_path in base_dir.glob("knowledge_graph_*.json"):
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    # Delete expired files
                    try:
                        file_path.unlink()
                        metadata_path = file_path.with_suffix('.metadata.json')
                        if metadata_path.exists():
                            metadata_path.unlink()
                        self.logger.debug(f"🗑️ Deleted expired KG: {file_path.name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not delete {file_path}: {e}")
                else:
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    total_size_mb += size_mb
                    kg_files.append((file_path, size_mb, file_path.stat().st_mtime))
            
            # If total size exceeds limit, delete oldest files
            if total_size_mb > max_size_mb:
                kg_files.sort(key=lambda x: x[2])  # Sort by modification time
                
                for file_path, size_mb, _ in kg_files:
                    if total_size_mb <= max_size_mb:
                        break
                    try:
                        file_path.unlink()
                        metadata_path = file_path.with_suffix('.metadata.json')
                        if metadata_path.exists():
                            metadata_path.unlink()
                        total_size_mb -= size_mb
                        self.logger.debug(f"🗑️ Deleted old KG for space: {file_path.name}")
                    except Exception as e:
                        self.logger.warning(f"⚠️ Could not delete {file_path}: {e}")
            
            self.logger.debug(f"📊 KG cache cleanup completed. Size: {total_size_mb:.1f}MB")
            
        except Exception as e:
            self.logger.warning(f"⚠️ Error during KG cleanup: {e}")

    def save_testsets(self, testsets: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Save testsets to CSV format with proper error handling
        
        Args:
            testsets: List of testset dictionaries
            output_path: Base output directory
        """
        if not testsets:
            self.logger.warning("❌ No testsets to save")
            return
        
        try:
            # Import the comprehensive file saver
            sys.path.append(str(Path(__file__).parent.parent / 'utils'))
            from pipeline_file_saver import PipelineFileSaver
            
            # Use the standardized file saver
            file_saver = PipelineFileSaver(output_path)
            
            # Save each testset
            saved_files = []
            for i, testset in enumerate(testsets):
                # Extract test samples from testset
                test_samples = testset.get('test_samples', [])
                if not test_samples:
                    # Try other possible keys
                    test_samples = testset.get('samples', [])
                    if not test_samples:
                        # If the testset itself is a list of samples
                        if isinstance(testset, list):
                            test_samples = testset
                        else:
                            self.logger.warning(f"No test samples found in testset {i+1}")
                            continue
                
                # Save as CSV
                filename_prefix = f"hybrid_testset_{i+1}"
                csv_path = file_saver.save_testset_csv(test_samples, filename_prefix)
                if csv_path:
                    saved_files.append(csv_path)
                    
                # Save knowledge graph if available
                kg_data = testset.get('knowledge_graph', None)
                if kg_data:
                    kg_filename_prefix = f"hybrid_kg_{i+1}"
                    kg_path = file_saver.save_knowledge_graph_json(kg_data, kg_filename_prefix)
                    if kg_path:
                        saved_files.append(kg_path)
            
            # Save personas and scenarios as well
            try:
                # Create default personas for hybrid generator
                default_personas = [
                    {
                        "id": "configurable_persona_1",
                        "name": "Technical Specialist",
                        "description": "Specialist asking technical questions based on configurable parameters",
                        "question_style": "technical",
                        "complexity_preference": "high",
                        "generation_method": "configurable"
                    },
                    {
                        "id": "ragas_persona_1", 
                        "name": "Quality Assessor",
                        "description": "Assessor asking quality-focused questions using RAGAS methods",
                        "question_style": "analytical",
                        "complexity_preference": "medium", 
                        "generation_method": "ragas"
                    }
                ]
                
                # Create scenarios based on generation methods
                default_scenarios = [
                    {
                        "id": "configurable_scenario",
                        "name": "Configurable Dataset Scenario",
                        "description": "Questions generated using configurable dataset generator",
                        "complexity": "medium",
                        "generation_method": "configurable"
                    },
                    {
                        "id": "ragas_scenario",
                        "name": "RAGAS Generation Scenario", 
                        "description": "Questions generated using RAGAS TestsetGenerator",
                        "complexity": "varied",
                        "generation_method": "ragas"
                    }
                ]
                
                personas_path = file_saver.save_personas_json(default_personas)
                scenarios_path = file_saver.save_scenarios_json(default_scenarios)
                
                if personas_path:
                    saved_files.append(personas_path)
                if scenarios_path:
                    saved_files.append(scenarios_path)
                    
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to save personas/scenarios: {e}")
            
            self.logger.info(f"✅ Saved {len(saved_files)} hybrid testset files using PipelineFileSaver")
            
        except ImportError as e:
            self.logger.warning(f"⚠️ Could not import PipelineFileSaver: {e}, using fallback method")
            self._save_testsets_fallback(testsets, output_path)
        except Exception as e:
            self.logger.error(f"❌ Failed to save testsets using PipelineFileSaver: {e}")
            self._save_testsets_fallback(testsets, output_path)
    
    def _save_testsets_fallback(self, testsets: List[Dict[str, Any]], output_path: Path) -> None:
        """Fallback method for saving testsets when PipelineFileSaver is not available"""
        try:
            output_path = Path(output_path)
            testsets_dir = output_path / "testsets"
            testsets_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Save each testset as CSV
            for i, testset in enumerate(testsets):
                test_samples = testset.get('test_samples', [])
                if not test_samples:
                    test_samples = testset.get('samples', [])
                    if not test_samples and isinstance(testset, list):
                        test_samples = testset
                
                if test_samples:
                    df = pd.DataFrame(test_samples)
                    csv_filename = f"hybrid_testset_{i+1}_{timestamp}.csv"
                    csv_path = testsets_dir / csv_filename
                    df.to_csv(csv_path, index=False, encoding='utf-8')
                    self.logger.info(f"✅ Fallback: Saved testset to {csv_path}")
            
        except Exception as e:
            self.logger.error(f"❌ Fallback save method also failed: {e}")
            raise
