#!/usr/bin/env python3
"""
Production-Ready RAGAS Custom LLM Fix
====================================

This module provides the production fix for using custom LLM with RAGAS
that can be integrated directly into your run_pipeline.py.
"""

import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)

class CustomLLMRAGASPatch:
    """
    Production patch for RAGAS to use custom LLM instead of OpenAI.
    This patch ensures your gpt-4o model is used for testset generation.
    """
    
    @staticmethod
    def patch_pure_ragas_generator(generator_instance, config: Dict[str, Any]):
        """
        Patch PureRAGASTestsetGenerator to handle custom LLM properly.
        
        Args:
            generator_instance: Instance of PureRAGASTestsetGenerator
            config: Pipeline configuration
        """
        logger.info("🔧 Applying custom LLM patch to RAGAS generator...")
        
        # Patch the LLM setup method
        original_setup_llm = generator_instance._setup_llm
        
        def patched_setup_llm():
            """Enhanced LLM setup with better error handling"""
            try:
                from langchain_openai import ChatOpenAI
                from ragas.llms import LangchainLLMWrapper
                
                custom_llm_config = config.get('testset_generation', {}).get('ragas_config', {}).get('custom_llm', {})
                
                # Enhanced URL handling
                base_url = custom_llm_config.get('endpoint', '')
                logger.info(f"🔗 Original endpoint: {base_url}")
                
                # Clean up URL - remove paths that ChatOpenAI adds automatically
                if '/chat/completions' in base_url:
                    base_url = base_url.replace('/chat/completions', '')
                if base_url.endswith('/v1/chat/completions'):
                    base_url = base_url.replace('/v1/chat/completions', '')
                if not base_url.endswith('/v1'):
                    base_url = base_url.rstrip('/') + '/v1'
                
                logger.info(f"🔗 Cleaned endpoint: {base_url}")
                
                # Create ChatOpenAI with enhanced configuration
                llm = ChatOpenAI(
                    base_url=base_url,
                    api_key=custom_llm_config.get('api_key', ''),
                    model=custom_llm_config.get('model', 'gpt-4o'),
                    temperature=custom_llm_config.get('temperature', 0.3),
                    max_tokens=custom_llm_config.get('max_tokens', 4096),
                    timeout=custom_llm_config.get('timeout', 180),
                    default_headers=custom_llm_config.get('headers', {}),
                    max_retries=2  # Add retries for stability
                )
                
                # Test the LLM connection
                try:
                    test_response = llm.invoke("Test connection - respond with 'OK'")
                    logger.info(f"✅ Custom LLM test successful: {test_response.content[:50]}...")
                except Exception as e:
                    logger.warning(f"⚠️ LLM test warning (continuing): {e}")
                
                # Wrap in RAGAS wrapper
                ragas_llm = LangchainLLMWrapper(llm)
                
                # Store reference to underlying LLM for direct access
                ragas_llm._underlying_llm = llm
                
                logger.info("✅ Custom LLM setup completed with patch")
                return ragas_llm
                
            except Exception as e:
                logger.error(f"❌ Custom LLM setup failed: {e}")
                raise
        
        # Apply the patch
        generator_instance._setup_llm = patched_setup_llm
        
        # Patch the generation method to handle result properly
        original_generate = generator_instance.generate_comprehensive_testset
        
        def patched_generate_comprehensive_testset(csv_files: List[str], output_dir: Path) -> Dict[str, Any]:
            """Enhanced testset generation with better error handling and result processing"""
            try:
                logger.info("🚀 Starting patched RAGAS testset generation...")
                
                # Call the original method with enhancements
                result = original_generate(csv_files, output_dir)
                
                # If original succeeds, return as-is
                if result.get('success', False):
                    logger.info("✅ Original RAGAS generation succeeded")
                    return result
                
                # If original fails, try the enhanced fallback
                logger.info("🔄 Original generation failed, trying enhanced fallback...")
                return CustomLLMRAGASPatch._enhanced_fallback_generation(
                    generator_instance, csv_files, output_dir, config
                )
                
            except Exception as e:
                logger.error(f"❌ Patched generation failed: {e}")
                # Try enhanced fallback as last resort
                return CustomLLMRAGASPatch._enhanced_fallback_generation(
                    generator_instance, csv_files, output_dir, config
                )
        
        # Apply the generation patch
        generator_instance.generate_comprehensive_testset = patched_generate_comprehensive_testset
        
        logger.info("✅ Custom LLM patches applied successfully")
        return generator_instance
    
    @staticmethod
    def _enhanced_fallback_generation(generator_instance, csv_files: List[str], 
                                    output_dir: Path, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enhanced fallback generation using direct custom LLM calls.
        This ensures testset generation works even if RAGAS filtering fails.
        """
        try:
            import pandas as pd
            import json
            from datetime import datetime
            
            logger.info("🎯 Starting enhanced fallback generation...")
            
            # Load documents from CSV
            documents = []
            for csv_file in csv_files:
                df = pd.read_csv(csv_file)
                max_docs = min(5, len(df))  # Limit for stability
                
                for idx, row in df.head(max_docs).iterrows():
                    try:
                        content_data = row.get('content', '')
                        if isinstance(content_data, str) and content_data.startswith('{'):
                            content_json = json.loads(content_data)
                            text_content = content_json.get('text', content_data)
                        else:
                            text_content = str(content_data)
                        
                        if len(text_content.strip()) > 50:
                            documents.append({
                                'content': text_content,
                                'id': str(row.get('id', idx)),
                                'source': csv_file
                            })
                    except Exception as e:
                        logger.warning(f"⚠️ Failed to process row {idx}: {e}")
                        continue
            
            if not documents:
                raise ValueError("No valid documents loaded")
            
            logger.info(f"📄 Loaded {len(documents)} documents for fallback generation")
            
            # Get the underlying LLM for direct generation
            llm = generator_instance.llm
            if hasattr(llm, '_underlying_llm'):
                direct_llm = llm._underlying_llm
            else:
                # Create direct LLM if not available
                from langchain_openai import ChatOpenAI
                custom_llm_config = config.get('testset_generation', {}).get('ragas_config', {}).get('custom_llm', {})
                base_url = custom_llm_config.get('endpoint', '').replace('/chat/completions', '').rstrip('/') + '/v1'
                
                direct_llm = ChatOpenAI(
                    base_url=base_url,
                    api_key=custom_llm_config.get('api_key', ''),
                    model=custom_llm_config.get('model', 'gpt-4o'),
                    temperature=0.3,
                    max_tokens=2000,
                    timeout=60
                )
            
            # Generate Q&A pairs using direct LLM
            testset_data = []
            max_samples = min(3, len(documents))
            
            for i, doc in enumerate(documents[:max_samples]):
                try:
                    # Generate question
                    question_prompt = f"""Based on the following technical documentation, generate a clear, specific question that can be answered using the information provided. Focus on key procedures, error codes, or important technical details.

Documentation: {doc['content'][:1200]}

Generate only the question (no explanations):"""
                    
                    question_response = direct_llm.invoke(question_prompt)
                    question = question_response.content.strip()
                    
                    # Generate answer
                    answer_prompt = f"""Based on the following technical documentation and question, provide a comprehensive and accurate answer.

Documentation: {doc['content'][:1200]}
Question: {question}

Provide a detailed, technical answer:"""
                    
                    answer_response = direct_llm.invoke(answer_prompt)
                    answer = answer_response.content.strip()
                    
                    # Create testset entry in RAGAS format
                    testset_data.append({
                        'user_input': question,
                        'reference_contexts': [doc['content']],
                        'reference': answer,
                        'auto_keywords': '',  # Will be filled by KeyBERT if available
                        'generation_method': 'enhanced_fallback_custom_llm',
                        'source_document_id': doc['id']
                    })
                    
                    logger.info(f"✅ Generated Q&A pair {i+1}/{max_samples}")
                    
                except Exception as e:
                    logger.warning(f"⚠️ Failed to generate Q&A for document {i}: {e}")
                    continue
            
            if not testset_data:
                raise ValueError("No Q&A pairs generated")
            
            # Create DataFrame
            df = pd.DataFrame(testset_data)
            
            # Add keywords using KeyBERT if available
            if hasattr(generator_instance, 'keybert_extractor') and generator_instance.keybert_extractor:
                logger.info("🔍 Adding keywords with KeyBERT...")
                for idx, row in df.iterrows():
                    try:
                        keywords_result = generator_instance.keybert_extractor.extract_for_testset_generation(
                            row['user_input']
                        )
                        if keywords_result and 'high_relevance' in keywords_result:
                            keywords_list = [kw['keyword'] for kw in keywords_result['high_relevance']]
                            df.at[idx, 'auto_keywords'] = ', '.join(keywords_list)
                    except Exception as e:
                        logger.warning(f"Failed to extract keywords for row {idx}: {e}")
            
            # Save testset
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            testset_file = output_dir / f"enhanced_fallback_testset_{timestamp}.csv"
            
            df.to_csv(testset_file, index=False)
            logger.info(f"💾 Fallback testset saved: {testset_file}")
            
            # Return result in expected format
            return {
                'success': True,
                'testset_path': str(testset_file),
                'knowledge_graph_path': None,
                'metadata': {
                    'generation_method': 'enhanced_fallback_custom_llm',
                    'samples_generated': len(df),
                    'knowledge_graph_nodes': 0,
                    'knowledge_graph_relationships': 0,
                    'documents_processed': len(documents),
                    'custom_llm_used': True,
                    'llm_model': config.get('testset_generation', {}).get('ragas_config', {}).get('custom_llm', {}).get('model', 'gpt-4o'),
                    'timestamp': timestamp
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced fallback generation failed: {e}")
            return {
                'success': False,
                'error': f'Enhanced fallback failed: {str(e)}',
                'metadata': {
                    'generation_method': 'enhanced_fallback_failed',
                    'samples_generated': 0,
                    'knowledge_graph_nodes': 0,
                    'knowledge_graph_relationships': 0,
                    'documents_processed': 0
                }
            }

def apply_custom_llm_ragas_patch(config: Dict[str, Any]):
    """
    Apply the custom LLM RAGAS patch to the pipeline configuration.
    
    This function modifies the pipeline configuration to ensure RAGAS
    uses your custom gpt-4o LLM instead of OpenAI.
    
    Args:
        config: Pipeline configuration dictionary
        
    Returns:
        Modified configuration with custom LLM patches applied
    """
    logger.info("🔧 Applying custom LLM RAGAS patches to configuration...")
    
    # Ensure custom LLM is properly configured
    testset_config = config.setdefault('testset_generation', {})
    ragas_config = testset_config.setdefault('ragas_config', {})
    
    # Force custom LLM usage
    ragas_config['use_custom_llm'] = True
    ragas_config['use_openai'] = False
    ragas_config['force_custom_llm'] = True
    
    # Ensure fallback is enabled
    error_handling = ragas_config.setdefault('error_handling', {})
    error_handling['fallback_to_enhanced_generation'] = True
    error_handling['continue_on_generation_error'] = True
    
    logger.info("✅ Custom LLM RAGAS patches applied to configuration")
    
    return config
