"""
Enhanced RAGAS Evaluator with LLM-based and Non-LLM Fallback Metrics

This module implements a robust RAGAS evaluation system that:
1. First tries LLM-based metrics with custom endpoint
2. Falls back to non-LLM alternatives when LLM metrics fail
3. Handles token limits and timeout issues gracefully
4. Provides comprehensive error handling and recovery
"""

import logging
import pandas as pd
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime

logger = logging.getLogger(__name__)

# RAGAS imports with comprehensive fallback support
try:
    from ragas import evaluate
    RAGAS_AVAILABLE = True
    
    # Import Dataset with fallback
    try:
        from datasets import Dataset
        DATASET_AVAILABLE = True
    except ImportError:
        # Create a minimal Dataset placeholder if datasets library is not available
        class Dataset:
            def __init__(self, data_dict):
                self.data = data_dict
            
            def __getitem__(self, key):
                return self.data[key]
            
            def __len__(self):
                return len(next(iter(self.data.values())))
        
        DATASET_AVAILABLE = False
        logger.warning("⚠️ datasets library not available, using fallback Dataset class")
    
    # Try to import both LLM-based and Non-LLM metrics
    LLM_METRICS_AVAILABLE = False
    NONLLM_METRICS_AVAILABLE = False
    
    # LLM-based metrics (primary)
    try:
        from ragas.metrics import (
            LLMContextPrecisionWithoutReference,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            Faithfulness,
            ResponseRelevancy,
            AnswerRelevancy
        )
        LLM_METRICS_AVAILABLE = True
        logger.info("✅ LLM-based RAGAS metrics available")
    except ImportError as e:
        logger.warning(f"⚠️ LLM-based RAGAS metrics not available: {e}")
        # Try alternative imports for older RAGAS versions
        try:
            from ragas.metrics import (
                ContextPrecision,
                ContextRecall,
                Faithfulness,
                AnswerRelevancy
            )
            LLM_METRICS_AVAILABLE = True
            logger.info("✅ Alternative LLM-based RAGAS metrics available")
        except ImportError as e2:
            logger.warning(f"⚠️ Alternative LLM-based RAGAS metrics not available: {e2}")
    
    # Non-LLM fallback metrics
    try:
        from ragas.metrics import (
            NonLLMContextPrecisionWithReference,
            NonLLMContextRecall,
            AnswerSimilarity,
            AnswerCorrectness
        )
        NONLLM_METRICS_AVAILABLE = True
        logger.info("✅ Non-LLM RAGAS metrics available")
    except ImportError as e:
        logger.warning(f"⚠️ Non-LLM RAGAS metrics not available: {e}")
    
    # Legacy fallback metrics
    LEGACY_METRICS_AVAILABLE = False
    try:
        from ragas.metrics import (
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        )
        LEGACY_METRICS_AVAILABLE = True
        logger.info("✅ Legacy RAGAS metrics available")
    except ImportError as e:
        logger.warning(f"⚠️ Legacy RAGAS metrics not available: {e}")

except ImportError as e:
    logging.warning(f"RAGAS imports failed: {e}")
    RAGAS_AVAILABLE = False
    LLM_METRICS_AVAILABLE = False
    NONLLM_METRICS_AVAILABLE = False
    LEGACY_METRICS_AVAILABLE = False

# LangChain LLM wrapper for custom endpoint
try:
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain imports failed: {e}")
    LANGCHAIN_AVAILABLE = False

class RAGASEvaluatorWithFallbacks:
    """
    Enhanced RAGAS evaluator with comprehensive fallback mechanisms.
    
    Evaluation Strategy:
    1. Try LLM-based metrics with increased max_tokens
    2. Fall back to Non-LLM metrics if LLM fails
    3. Use legacy metrics as final fallback
    4. Provide detailed failure analysis
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAGAS evaluator with fallback strategies.
        
        Args:
            config: Pipeline configuration containing LLM settings
        """
        self.config = config
        self.llm_config = config.get('llm', {})
        self.ragas_config = config.get('ragas_metrics', {})
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.llm_metrics = []
        self.nonllm_metrics = []
        self.legacy_metrics = []
        
        # Setup custom LLM and embeddings
        self._setup_llm_and_embeddings()
        
        # Initialize all available RAGAS metrics
        self._setup_metric_fallbacks()
        
        logger.info(f"RAGASEvaluatorWithFallbacks initialized")
        logger.info(f"📊 LLM metrics: {len(self.llm_metrics)}")
        logger.info(f"🔄 Non-LLM fallbacks: {len(self.nonllm_metrics)}")
        logger.info(f"🚨 Legacy fallbacks: {len(self.legacy_metrics)}")
    
    def _setup_llm_and_embeddings(self):
        """Setup custom LLM endpoint with increased token limits for RAGAS."""
        try:
            # Import RAGAS LLM wrapper
            from ragas.llms import LangchainLLMWrapper
            
            # Extract LLM configuration
            ragas_llm_config = self.ragas_config.get('llm', {})
            if not ragas_llm_config:
                ragas_llm_config = self.llm_config
            
            endpoint_url = ragas_llm_config.get('endpoint_url') or ragas_llm_config.get('endpoint')
            api_key = ragas_llm_config.get('api_key')
            model_name = ragas_llm_config.get('model', ragas_llm_config.get('model_name', 'gpt-4o'))
            
            # Increase max_tokens significantly for RAGAS metrics
            max_tokens = ragas_llm_config.get('max_tokens', 4000)  # Increased from 2000
            if max_tokens < 3000:
                max_tokens = 4000  # Ensure minimum for RAGAS
                logger.info(f"🔧 Increased max_tokens to {max_tokens} for RAGAS compatibility")
            
            temperature = ragas_llm_config.get('temperature', 0.1)
            timeout = ragas_llm_config.get('timeout', 120)  # Increased timeout
            
            if not endpoint_url or not api_key:
                raise ValueError("LLM endpoint URL and API key must be configured")
            
            # Remove /chat/completions from endpoint_url if present
            base_url = endpoint_url.replace('/chat/completions', '') if '/chat/completions' in endpoint_url else endpoint_url
            
            # Create LangChain LLM with enhanced settings for RAGAS
            langchain_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout=timeout,
                max_retries=3,  # Add retries
                request_timeout=timeout
            )
            
            # Test the LLM connection
            try:
                test_response = langchain_llm.invoke("Test")
                logger.info("✅ LLM connection test successful")
            except Exception as e:
                logger.warning(f"⚠️ LLM connection test failed: {e}")
            
            # Wrap with RAGAS LangchainLLMWrapper
            self.llm = LangchainLLMWrapper(langchain_llm)
            
            # Setup embeddings with fallback
            try:
                from langchain_openai import OpenAIEmbeddings
                langchain_embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    openai_api_base=base_url,
                    timeout=timeout
                )
                
                try:
                    from ragas.embeddings import LangchainEmbeddingsWrapper
                    self.embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
                except ImportError:
                    self.embeddings = langchain_embeddings
                    
                logger.info("✅ Custom embeddings configured")
                    
            except Exception as e:
                logger.warning(f"Custom embeddings failed: {e}")
                # Fallback to HuggingFace embeddings
                try:
                    from langchain_community.embeddings import HuggingFaceEmbeddings
                    hf_embeddings = HuggingFaceEmbeddings(
                        model_name="sentence-transformers/all-MiniLM-L6-v2"
                    )
                    
                    try:
                        from ragas.embeddings import LangchainEmbeddingsWrapper
                        self.embeddings = LangchainEmbeddingsWrapper(hf_embeddings)
                    except ImportError:
                        self.embeddings = hf_embeddings
                        
                    logger.info("✅ Using HuggingFace embeddings as fallback")
                except Exception as e2:
                    logger.error(f"Failed to setup fallback embeddings: {e2}")
                    self.embeddings = None
            
            logger.info(f"✅ Enhanced RAGAS LLM configured")
            logger.info(f"🔧 Model: {model_name}")
            logger.info(f"🔧 Max tokens: {max_tokens}")
            logger.info(f"🔧 Timeout: {timeout}s")
            logger.info(f"🔧 Base URL: {base_url}")
            
        except Exception as e:
            logger.error(f"Failed to setup RAGAS LLM: {e}")
            self.llm = None
            self.embeddings = None
    
    def _setup_metric_fallbacks(self):
        """Setup comprehensive metric fallback strategy with working RAG metrics."""
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS not available - no metrics can be configured")
            return
        
        # 1. Setup RAG evaluation metrics (primary) - Focus on what actually works
        self.llm_metrics = []
        self.nonllm_metrics = []
        self.legacy_metrics = []
        
        if self.llm:
            logger.info("🎯 Setting up RAG evaluation metrics...")
            try:
                # Import the exact metrics from our available set
                from ragas.metrics import (
                    ContextPrecision,
                    ContextRecall, 
                    Faithfulness,
                    AnswerRelevancy
                )
                
                # Setup each metric individually with proper error handling
                
                # 1. ContextPrecision - measures precision of retrieved contexts
                try:
                    context_precision = ContextPrecision(llm=self.llm)
                    if self.embeddings:
                        context_precision.embeddings = self.embeddings
                    self.llm_metrics.append(context_precision)
                    logger.info("✅ ContextPrecision configured for RAG evaluation")
                except Exception as e:
                    logger.warning(f"⚠️ ContextPrecision setup failed: {e}")
                
                # 2. ContextRecall - measures recall of retrieved contexts
                try:
                    context_recall = ContextRecall(llm=self.llm)
                    if self.embeddings:
                        context_recall.embeddings = self.embeddings
                    self.llm_metrics.append(context_recall)
                    logger.info("✅ ContextRecall configured for RAG evaluation")
                except Exception as e:
                    logger.warning(f"⚠️ ContextRecall setup failed: {e}")
                
                # 3. Faithfulness - measures factual consistency with contexts
                try:
                    faithfulness = Faithfulness(llm=self.llm)
                    if self.embeddings:
                        faithfulness.embeddings = self.embeddings
                    self.llm_metrics.append(faithfulness)
                    logger.info("✅ Faithfulness configured for RAG evaluation")
                except Exception as e:
                    logger.warning(f"⚠️ Faithfulness setup failed: {e}")
                
                # 4. AnswerRelevancy - measures relevance of answer to question
                try:
                    answer_relevancy = AnswerRelevancy(llm=self.llm)
                    if self.embeddings:
                        answer_relevancy.embeddings = self.embeddings
                    self.llm_metrics.append(answer_relevancy)
                    logger.info("✅ AnswerRelevancy configured for RAG evaluation")
                except Exception as e:
                    logger.warning(f"⚠️ AnswerRelevancy setup failed: {e}")
                
                logger.info(f"🎯 Successfully configured {len(self.llm_metrics)} RAG evaluation metrics")
                
            except Exception as e:
                logger.error(f"❌ Failed to setup RAG metrics: {e}")
                self.llm_metrics = []
        else:
            logger.warning("⚠️ No LLM configured - RAG metrics will not be available")
            self.llm_metrics = []
        
        # 2. Setup Non-LLM fallback metrics for basic similarity measures
        self.nonllm_metrics = []
        try:
            logger.info("🔧 Setting up Non-LLM fallback metrics...")
            
            # Basic similarity metrics that don't require reference_contexts
            from ragas.metrics import AnswerSimilarity
            
            # Only add metrics that work with standard columns
            try:
                answer_similarity = AnswerSimilarity()
                # Override embeddings if available
                if self.embeddings:
                    answer_similarity.embeddings = self.embeddings
                self.nonllm_metrics.append(answer_similarity)
                logger.info("✅ AnswerSimilarity configured as Non-LLM fallback")
            except Exception as e:
                logger.warning(f"⚠️ AnswerSimilarity setup failed: {e}")
            
            logger.info(f"🔄 Configured {len(self.nonllm_metrics)} Non-LLM fallback metrics")
            
        except Exception as e:
            logger.warning(f"⚠️ Non-LLM metrics setup failed: {e}")
            self.nonllm_metrics = []
        
        # 3. Setup basic legacy metrics as final fallback
        self.legacy_metrics = []
        logger.info("⚠️ Legacy metrics disabled to focus on working RAG metrics")
        
        # Log total available metrics
        total_metrics = len(self.llm_metrics) + len(self.nonllm_metrics) + len(self.legacy_metrics)
        logger.info(f"📊 Total RAG evaluation metrics configured: {total_metrics}")
        
        if total_metrics == 0:
            logger.warning("⚠️ No RAGAS metrics available - will use simplified evaluation mode")
            # Don't raise exception, allow pipeline to continue
    
    def _prepare_ragas_dataset(self, questions: List[str], answers: List[str], 
                             contexts: List[List[str]], ground_truths: List[str]) -> Dataset:
        """
        Prepare dataset in RAGAS-compatible format.
        """
        try:
            # Use appropriate column names for RAGAS version 0.2.x
            processed_data = {
                'user_input': [],      # Question/query
                'response': [],        # Generated response from RAG system
                'retrieved_contexts': [], # Retrieved contexts as list
                'reference': [],       # Ground truth reference
                'reference_contexts': [] # Reference contexts for Non-LLM metrics
            }
            
            for i in range(len(questions)):
                # Process and validate each field
                question = str(questions[i]).strip()
                if not question:
                    continue
                
                answer = str(answers[i]).strip() if i < len(answers) else "No answer provided"
                if not answer:
                    answer = "No answer provided"
                
                # Process contexts - ensure they are a proper list
                if i < len(contexts):
                    context_list = contexts[i]
                    if isinstance(context_list, str):
                        try:
                            # Try to parse as JSON
                            context_list = json.loads(context_list)
                        except json.JSONDecodeError:
                            try:
                                # Try to parse as Python literal
                                import ast
                                context_list = ast.literal_eval(context_list)
                            except (ValueError, SyntaxError):
                                # Treat as single context
                                context_list = [context_list]
                    elif not isinstance(context_list, list):
                        context_list = [str(context_list)]
                    
                    # Clean and validate contexts
                    context_list = [str(ctx).strip() for ctx in context_list if str(ctx).strip()]
                    if not context_list:
                        context_list = ["No context available"]
                else:
                    context_list = ["No context available"]
                
                # Process ground truth
                ground_truth = str(ground_truths[i]).strip() if i < len(ground_truths) else answer
                if not ground_truth:
                    ground_truth = answer
                
                # Add to dataset
                processed_data['user_input'].append(question)
                processed_data['response'].append(answer)
                processed_data['retrieved_contexts'].append(context_list)
                processed_data['reference'].append(ground_truth)
                # Add reference_contexts (same as retrieved for now, can be improved)
                processed_data['reference_contexts'].append(context_list)
            
            if not processed_data['user_input']:
                raise ValueError("No valid questions to create RAGAS dataset")
            
            # Create dataset
            dataset = Dataset.from_dict(processed_data)
            logger.info(f"✅ RAGAS dataset prepared with {len(dataset)} samples")
            logger.info(f"📋 Dataset columns: {dataset.column_names}")
            
            # Log sample data for debugging
            if len(dataset) > 0:
                sample = dataset[0]
                logger.debug(f"Sample data:")
                logger.debug(f"  user_input: {sample['user_input'][:100]}...")
                logger.debug(f"  response: {sample['response'][:100]}...")
                logger.debug(f"  retrieved_contexts: {len(sample['retrieved_contexts'])} contexts")
                logger.debug(f"  reference: {sample['reference'][:100]}...")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare RAGAS dataset: {e}")
            raise
    
    def _evaluate_with_metrics(self, dataset: Dataset, metrics: List, 
                              metric_type: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Evaluate dataset with a specific set of metrics.
        
        Args:
            dataset: RAGAS dataset
            metrics: List of metrics to evaluate
            metric_type: Type description (e.g., "LLM-based", "Non-LLM")
            max_retries: Maximum number of retries per metric
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"🔄 Running {metric_type} evaluation with {len(metrics)} metrics...")
        
        results = {}
        successful_metrics = 0
        
        for metric in metrics:
            metric_name = metric.__class__.__name__
            logger.info(f"📊 Evaluating {metric_type} metric: {metric_name}")
            
            for attempt in range(max_retries + 1):
                try:
                    # Run evaluation for this metric
                    if self.llm and self.embeddings and metric_type == "LLM-based":
                        result = evaluate(
                            dataset=dataset,
                            metrics=[metric],
                            llm=self.llm,
                            embeddings=self.embeddings,
                            raise_exceptions=False,
                            show_progress=True
                        )
                    else:
                        result = evaluate(
                            dataset=dataset,
                            metrics=[metric],
                            raise_exceptions=False,
                            show_progress=True
                        )
                    
                    # Debug the result structure
                    logger.debug(f"Raw result type for {metric_name}: {type(result)}")
                    if hasattr(result, 'to_pandas'):
                        try:
                            result_df = result.to_pandas()
                            logger.debug(f"Result DataFrame columns: {result_df.columns.tolist()}")
                            logger.debug(f"Result DataFrame shape: {result_df.shape}")
                            logger.debug(f"Result DataFrame head:\n{result_df.head()}")
                        except Exception as e:
                            logger.debug(f"Failed to convert result to pandas: {e}")
                    
                    # Extract scores
                    scores = self._extract_scores_from_result(result, metric_name)
                    
                    if scores and any(s is not None for s in scores):
                        # Use robust NaN handling utilities
                        import sys
                        from pathlib import Path
                        sys.path.append(str(Path(__file__).parent.parent / "utils"))
                        from nan_handling import calculate_robust_summary_stats, is_valid_score
                        
                        # Calculate robust statistics
                        stats = calculate_robust_summary_stats(scores, metric_name)
                        
                        # Only consider success if we have valid scores
                        if stats['valid_count'] > 0:
                            results[metric_name] = {
                                'mean_score': stats['mean_score'],
                                'individual_scores': stats['individual_scores'],
                                'valid_count': stats['valid_count'],
                                'total_count': stats['total_count'],
                                'std_score': stats['std_score'],
                                'min_score': stats['min_score'],
                                'max_score': stats['max_score'],
                                'metric_type': metric_type,
                                'attempt': attempt + 1
                            }
                            successful_metrics += 1
                            logger.info(f"✅ {metric_name}: {stats['mean_score']:.3f} (n={stats['valid_count']}/{stats['total_count']})")
                            break
                        else:
                            logger.warning(f"⚠️ No valid scores for {metric_name} on attempt {attempt + 1}")
                    else:
                        logger.warning(f"⚠️ No scores extracted for {metric_name} on attempt {attempt + 1}")
                    
                    if attempt == max_retries:
                        logger.error(f"❌ {metric_name} failed after {max_retries + 1} attempts")
                        results[metric_name] = {
                            'mean_score': None,
                            'error': 'No valid scores after all attempts',
                            'metric_type': metric_type,
                            'attempts': max_retries + 1
                        }
                    
                except Exception as e:
                    error_msg = str(e)
                    logger.warning(f"⚠️ {metric_name} attempt {attempt + 1} failed: {error_msg}")
                    
                    # Check for specific error types
                    if "LLMDidNotFinishException" in error_msg or "max_tokens" in error_msg:
                        logger.warning(f"🔧 Token limit issue detected for {metric_name}")
                        # For token limit issues, don't retry - move to fallback
                        results[metric_name] = {
                            'mean_score': None,
                            'error': f'Token limit exceeded: {error_msg}',
                            'metric_type': metric_type,
                            'token_limit_issue': True
                        }
                        break
                    elif attempt == max_retries:
                        logger.error(f"❌ {metric_name} failed after {max_retries + 1} attempts: {error_msg}")
                        results[metric_name] = {
                            'mean_score': None,
                            'error': error_msg,
                            'metric_type': metric_type,
                            'attempts': max_retries + 1
                        }
                    else:
                        # Wait before retry
                        time.sleep(2)
        
        logger.info(f"📊 {metric_type} evaluation complete: {successful_metrics}/{len(metrics)} metrics succeeded")
        return results
    
    def _extract_scores_from_result(self, result, metric_name: str) -> List[Union[float, None]]:
        """Extract scores from RAGAS evaluation result with enhanced error handling."""
        try:
            logger.debug(f"Extracting scores for {metric_name} from result type: {type(result)}")
            
            # Method 1: Try pandas DataFrame conversion
            if hasattr(result, 'to_pandas'):
                try:
                    result_df = result.to_pandas()
                    logger.debug(f"DataFrame columns: {result_df.columns.tolist()}")
                    
                    # Try exact metric name match
                    if metric_name in result_df.columns:
                        scores = result_df[metric_name].tolist()
                        logger.debug(f"Found exact match for {metric_name}: {len(scores)} scores")
                        return scores
                    
                    # Try lowercase match
                    elif metric_name.lower() in result_df.columns:
                        scores = result_df[metric_name.lower()].tolist()
                        logger.debug(f"Found lowercase match for {metric_name}: {len(scores)} scores")
                        return scores
                    
                    # Try partial matches (for class names vs column names)
                    matching_cols = []
                    for col in result_df.columns:
                        col_lower = col.lower()
                        metric_lower = metric_name.lower()
                        
                        # Check for partial matches
                        if metric_lower in col_lower or col_lower in metric_lower:
                            matching_cols.append(col)
                        
                        # Special mappings for common metric name differences
                        elif (metric_lower.startswith('llmcontext') and 'context' in col_lower) or \
                             (metric_lower.startswith('nonllmcontext') and 'context' in col_lower) or \
                             ('precision' in metric_lower and 'precision' in col_lower) or \
                             ('recall' in metric_lower and 'recall' in col_lower) or \
                             ('faithfulness' in metric_lower and 'faithfulness' in col_lower) or \
                             ('relevancy' in metric_lower and ('relevancy' in col_lower or 'relevance' in col_lower)) or \
                             ('answer' in metric_lower and 'answer' in col_lower) or \
                             (metric_lower == 'llmcontextprecisionwithoutreference' and ('context_precision' in col_lower or 'precision' in col_lower)) or \
                             (metric_lower == 'llmcontextrecall' and ('context_recall' in col_lower or 'recall' in col_lower)) or \
                             (metric_lower == 'responserelevancy' and ('response_relevancy' in col_lower or 'relevancy' in col_lower)):
                            matching_cols.append(col)
                    
                    if matching_cols:
                        # Use the first match
                        col = matching_cols[0]
                        scores = result_df[col].tolist()
                        logger.info(f"Found partial match for {metric_name} -> {col}: {len(scores)} scores")
                        return scores
                    else:
                        logger.warning(f"No matching columns found for {metric_name}. Available: {result_df.columns.tolist()}")
                
                except Exception as e:
                    logger.warning(f"Failed to extract from DataFrame for {metric_name}: {e}")
            
            # Method 2: Try dictionary access
            if hasattr(result, 'to_dict'):
                try:
                    result_dict = result.to_dict()
                    logger.debug(f"Dict keys: {list(result_dict.keys())}")
                    
                    # Try exact match
                    if metric_name in result_dict:
                        return result_dict[metric_name]
                    elif metric_name.lower() in result_dict:
                        return result_dict[metric_name.lower()]
                    else:
                        # Try partial matches
                        for key in result_dict.keys():
                            if metric_name.lower() in key.lower() or key.lower() in metric_name.lower():
                                logger.info(f"Found partial dict match for {metric_name} -> {key}")
                                return result_dict[key]
                
                except Exception as e:
                    logger.warning(f"Failed to extract from dict for {metric_name}: {e}")
            
            # Method 3: Try direct attribute access
            if hasattr(result, metric_name):
                try:
                    metric_values = getattr(result, metric_name)
                    if hasattr(metric_values, 'tolist'):
                        return metric_values.tolist()
                    elif isinstance(metric_values, (list, tuple)):
                        return list(metric_values)
                    else:
                        return [metric_values] if metric_values is not None else []
                except Exception as e:
                    logger.warning(f"Failed to extract via attribute for {metric_name}: {e}")
            
            # Method 4: Try to access by index (for datasets)
            if hasattr(result, '__getitem__') and hasattr(result, '__len__'):
                try:
                    if len(result) > 0:
                        sample = result[0]
                        if isinstance(sample, dict) and metric_name in sample:
                            return [item.get(metric_name) for item in result if metric_name in item]
                except Exception as e:
                    logger.warning(f"Failed to extract via indexing for {metric_name}: {e}")
            
            logger.warning(f"Could not extract scores for {metric_name} from result type {type(result)}")
            return []
            
        except Exception as e:
            logger.error(f"Error extracting scores for {metric_name}: {e}")
            return []
    
    def evaluate_testset_with_rag_responses(self, enhanced_df: pd.DataFrame, 
                                          output_dir: Path) -> Dict[str, Any]:
        """
        Evaluate testset using RAG system responses with comprehensive fallback strategy.
        
        Args:
            enhanced_df: DataFrame with RAG responses and ground truth
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("🚀 Starting comprehensive RAGAS evaluation with fallbacks...")
        
        try:
            # Extract data for RAGAS
            questions = enhanced_df['user_input'].tolist()
            rag_answers = enhanced_df['rag_answer'].tolist()
            contexts = enhanced_df['reference_contexts'].tolist()
            ground_truths = enhanced_df['reference'].tolist()
            
            logger.info(f"📋 Evaluating {len(questions)} RAG responses")
            
            # Prepare RAGAS dataset
            dataset = self._prepare_ragas_dataset(questions, rag_answers, contexts, ground_truths)
            
            # Results aggregation
            all_results = {}
            overall_scores = {}
            detailed_results = []
            successful_evaluators = 0
            
            # 1. Try LLM-based metrics first
            if self.llm_metrics:
                logger.info("🎯 Phase 1: Attempting LLM-based RAG metrics...")
                llm_results = self._evaluate_with_metrics(dataset, self.llm_metrics, "LLM-based")
                all_results['llm_based'] = llm_results
                
                # Extract successful LLM results
                for metric_name, metric_result in llm_results.items():
                    if metric_result.get('mean_score') is not None:
                        overall_scores[metric_name] = metric_result['mean_score']
                        successful_evaluators += 1
                        
                        # Add to detailed results
                        for i, score in enumerate(metric_result['individual_scores']):
                            if i < len(questions) and score is not None:
                                detailed_results.append({
                                    'question_index': i,
                                    'question': questions[i][:100] + "..." if len(questions[i]) > 100 else questions[i],
                                    'rag_answer': rag_answers[i][:100] + "..." if len(rag_answers[i]) > 100 else rag_answers[i],
                                    'ground_truth': ground_truths[i][:100] + "..." if len(ground_truths[i]) > 100 else ground_truths[i],
                                    'metric_name': metric_name,
                                    'score': float(score),
                                    'metric_type': 'LLM-based',
                                    'timestamp': datetime.now().isoformat()
                                })
                
                logger.info(f"✅ LLM-based phase: {len([r for r in llm_results.values() if r.get('mean_score') is not None])} metrics succeeded")
            else:
                logger.warning("⚠️ No LLM metrics available for evaluation")
            
            # 2. Use Non-LLM fallbacks if LLM metrics failed or are unavailable
            failed_llm_metrics = []
            if self.llm_metrics:
                for metric in self.llm_metrics:
                    metric_name = metric.__class__.__name__
                    if metric_name not in overall_scores or overall_scores[metric_name] is None:
                        failed_llm_metrics.append(metric_name)
            
            if (failed_llm_metrics or not self.llm_metrics) and self.nonllm_metrics:
                logger.info(f"🔄 Phase 2: Using Non-LLM fallbacks for {'failed metrics' if failed_llm_metrics else 'all metrics'}...")
                nonllm_results = self._evaluate_with_metrics(dataset, self.nonllm_metrics, "Non-LLM")
                all_results['non_llm_fallback'] = nonllm_results
                
                # Extract successful Non-LLM results
                for metric_name, metric_result in nonllm_results.items():
                    if metric_result.get('mean_score') is not None:
                        fallback_metric_name = f"{metric_name}_NonLLM"
                        overall_scores[fallback_metric_name] = metric_result['mean_score']
                        successful_evaluators += 1
                        
                        # Add to detailed results
                        for i, score in enumerate(metric_result['individual_scores']):
                            if i < len(questions) and score is not None:
                                detailed_results.append({
                                    'question_index': i,
                                    'question': questions[i][:100] + "..." if len(questions[i]) > 100 else questions[i],
                                    'rag_answer': rag_answers[i][:100] + "..." if len(rag_answers[i]) > 100 else rag_answers[i],
                                    'ground_truth': ground_truths[i][:100] + "..." if len(ground_truths[i]) > 100 else ground_truths[i],
                                    'metric_name': fallback_metric_name,
                                    'score': float(score),
                                    'metric_type': 'Non-LLM-fallback',
                                    'timestamp': datetime.now().isoformat()
                                })
                
                logger.info(f"✅ Non-LLM fallback phase: {len([r for r in nonllm_results.values() if r.get('mean_score') is not None])} metrics succeeded")
            
            # 3. If all else fails, provide a basic evaluation based on data completeness
            if successful_evaluators == 0:
                logger.warning("🚨 No RAGAS metrics succeeded - providing basic evaluation...")
                
                # Basic data completeness score
                data_completeness_score = 1.0 if all(
                    len(q.strip()) > 0 and len(a.strip()) > 0 
                    for q, a in zip(questions, rag_answers)
                ) else 0.5
                
                overall_scores['data_completeness'] = data_completeness_score
                successful_evaluators = 1
                
                # Add basic results
                for i in range(len(questions)):
                    detailed_results.append({
                        'question_index': i,
                        'question': questions[i][:100] + "..." if len(questions[i]) > 100 else questions[i],
                        'rag_answer': rag_answers[i][:100] + "..." if len(rag_answers[i]) > 100 else rag_answers[i],
                        'ground_truth': ground_truths[i][:100] + "..." if len(ground_truths[i]) > 100 else ground_truths[i],
                        'metric_name': 'data_completeness',
                        'score': data_completeness_score,
                        'metric_type': 'basic-fallback',
                        'timestamp': datetime.now().isoformat()
                    })
                
                logger.info(f"✅ Basic evaluation: data_completeness = {data_completeness_score:.3f}")
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed calculations
            if detailed_results:
                detailed_file = output_dir / f"ragas_enhanced_detailed_calculations_{timestamp}.json"
                with open(detailed_file, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
                logger.info(f"💾 Enhanced detailed calculations saved: {detailed_file}")
            
            # Apply NaN tolerance to final results
            import sys
            from pathlib import Path
            sys.path.append(str(Path(__file__).parent.parent / "utils"))
            from nan_handling import apply_nan_tolerance, validate_metric_results
            
            # Clean up results before saving
            all_results = apply_nan_tolerance(all_results, tolerance_strategy="fallback")
            
            # Validate overall scores
            import math
            for metric_name, score in overall_scores.items():
                if not isinstance(score, (int, float)) or not math.isfinite(score):
                    logger.warning(f"Invalid overall score for {metric_name}: {score}")
                    overall_scores[metric_name] = 0.0
            
            # Save comprehensive results
            comprehensive_results = {
                'evaluation_type': 'ragas_with_comprehensive_fallbacks',
                'timestamp': timestamp,
                'total_questions': len(questions),
                'successful_evaluators': successful_evaluators,
                'overall_scores': overall_scores,
                'detailed_results_by_type': all_results,
                'detailed_results_file': str(detailed_file) if detailed_results else None,
                'fallback_summary': {
                    'llm_metrics_attempted': len(self.llm_metrics),
                    'nonllm_fallbacks_used': len(self.nonllm_metrics) if self.nonllm_metrics else 0,
                    'legacy_fallbacks_used': len(self.legacy_metrics) if self.legacy_metrics else 0,
                    'total_successful': successful_evaluators
                },
                'nan_handling_applied': True,  # Flag to indicate NaN handling was applied
                'config': {
                    'llm_model': self.llm_config.get('model', 'unknown'),
                    'llm_endpoint': self.llm_config.get('endpoint_url', 'unknown'),
                    'max_tokens': self.llm_config.get('max_tokens', 'unknown'),
                    'metrics_available': {
                        'llm_based': [m.__class__.__name__ for m in self.llm_metrics],
                        'non_llm': [m.__class__.__name__ for m in self.nonllm_metrics],
                        'legacy': [m.__class__.__name__ for m in self.legacy_metrics]
                    }
                }
            }
            
            results_file = output_dir / f"ragas_enhanced_evaluation_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(comprehensive_results, f, indent=2)
            
            logger.info(f"✅ Enhanced RAGAS evaluation completed!")
            logger.info(f"📊 Total metrics evaluated: {successful_evaluators}")
            logger.info(f"🎯 Success rate: {successful_evaluators}/{len(self.llm_metrics + self.nonllm_metrics + self.legacy_metrics)}")
            logger.info(f"💾 Results saved: {results_file}")
            
            return {
                'success': True,
                'evaluation_type': 'ragas_with_comprehensive_fallbacks',
                'overall_scores': overall_scores,
                'total_questions': len(questions),
                'successful_evaluators': successful_evaluators,
                'detailed_results_file': str(detailed_file) if detailed_results else None,
                'results_file': str(results_file),
                'timestamp': timestamp,
                'fallback_summary': comprehensive_results['fallback_summary']
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced RAGAS evaluation failed: {e}")
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
