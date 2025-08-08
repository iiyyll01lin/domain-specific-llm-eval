"""
Fixed RAGAS Evaluator with proper data format handling and custom LLM endpoint support

This module addresses the model_dump errors and RAGAS compatibility issues.
Now includes comprehensive NaN handling for all 4 RAGAS metrics.
"""
import logging
import pandas as pd
import json
import math
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# RAGAS imports with proper error handling
try:
    from ragas import evaluate
    from ragas.metrics import (
        LLMContextPrecisionWithoutReference,  # Updated for 0.2.x
        LLMContextRecallWithReference,        # Updated for 0.2.x
        Faithfulness,                         # Updated for 0.2.x
        ResponseRelevancy                     # Updated for 0.2.x
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
    RAGAS_VERSION = "0.2.x"
except ImportError:
    # Fallback to older metrics
    try:
        from ragas import evaluate
        from ragas.metrics import (
            context_precision,
            context_recall, 
            faithfulness,
            answer_relevancy
        )
        from datasets import Dataset
        RAGAS_AVAILABLE = True
        RAGAS_VERSION = "legacy"
    except ImportError as e:
        logging.warning(f"RAGAS imports failed: {e}")
        RAGAS_AVAILABLE = False
        RAGAS_VERSION = "none"

# LangChain LLM wrapper for custom endpoint
try:
    from langchain_openai import ChatOpenAI
    from langchain_openai import OpenAIEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    logging.warning(f"LangChain imports failed: {e}")
    LANGCHAIN_AVAILABLE = False

logger = logging.getLogger(__name__)

class RAGASEvaluatorFixed:
    """
    Fixed RAGAS evaluator that properly handles:
    1. Custom LLM endpoint configuration
    2. Pydantic model_dump compatibility issues
    3. Correct dataset format for RAGAS
    4. Detailed calculation tracking
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAGAS evaluator with custom LLM endpoint.
        
        Args:
            config: Pipeline configuration containing LLM settings
        """
        self.config = config
        self.llm_config = config.get('llm', {})
        self.ragas_config = config.get('ragas_metrics', {})
        
        # Initialize components
        self.llm = None
        self.embeddings = None
        self.available_metrics = []
        
        # Setup custom LLM and embeddings
        self._setup_llm_and_embeddings()
        
        # Initialize RAGAS metrics
        self._setup_ragas_metrics()
        
        logger.info(f"RAGASEvaluatorFixed initialized with {len(self.available_metrics)} metrics")
    
    def _setup_llm_and_embeddings(self):
        """Setup custom LLM endpoint and embeddings for RAGAS using LangchainLLMWrapper."""
        try:
            # Import RAGAS LLM wrapper for this version
            from ragas.llms import LangchainLLMWrapper
            
            # Extract LLM configuration - check multiple possible config paths
            ragas_llm_config = self.ragas_config.get('llm', {})
            if not ragas_llm_config:
                ragas_llm_config = self.llm_config
            
            endpoint_url = ragas_llm_config.get('endpoint_url') or ragas_llm_config.get('endpoint')
            api_key = ragas_llm_config.get('api_key')
            model_name = ragas_llm_config.get('model', ragas_llm_config.get('model_name', 'gpt-4o'))
            max_tokens = ragas_llm_config.get('max_tokens', 2000)  # Increased default
            temperature = ragas_llm_config.get('temperature', 0.1)
            timeout = ragas_llm_config.get('timeout', 60)
            
            if not endpoint_url or not api_key:
                raise ValueError("LLM endpoint URL and API key must be configured")
            
            # Remove /chat/completions from endpoint_url if present for base URL
            base_url = endpoint_url.replace('/chat/completions', '') if '/chat/completions' in endpoint_url else endpoint_url
            
            # Create LangChain LLM first with increased max_tokens
            from langchain_openai import ChatOpenAI
            langchain_llm = ChatOpenAI(
                model=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=temperature,
                max_tokens=max_tokens,  # Use configured max_tokens
                timeout=timeout
            )
            
            # Wrap it with RAGAS LangchainLLMWrapper
            self.llm = LangchainLLMWrapper(langchain_llm)
            
            # Setup embeddings
            try:
                from langchain_openai import OpenAIEmbeddings
                langchain_embeddings = OpenAIEmbeddings(
                    openai_api_key=api_key,
                    openai_api_base=base_url
                )
                
                # Check if RAGAS has embeddings wrapper
                try:
                    from ragas.embeddings import LangchainEmbeddingsWrapper
                    self.embeddings = LangchainEmbeddingsWrapper(langchain_embeddings)
                except ImportError:
                    # Use LangChain embeddings directly
                    self.embeddings = langchain_embeddings
                    
            except Exception as e:
                logger.warning(f"Failed to setup custom embeddings: {e}")
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
            
            logger.info(f"✅ RAGAS LangchainLLMWrapper configured: {model_name} at {base_url}")
            logger.info(f"🔧 Max tokens: {max_tokens}, Temperature: {temperature}")
            logger.debug(f"🔧 Original endpoint: {endpoint_url}")
            logger.debug(f"🔧 Base URL: {base_url}")
            
        except Exception as e:
            logger.error(f"Failed to setup RAGAS LLM wrapper: {e}")
            # Fallback to direct LangChain
            self._setup_langchain_fallback()
    
    def _setup_langchain_fallback(self):
        """Fallback to LangChain LLM if RAGAS wrapper fails"""
        if not LANGCHAIN_AVAILABLE:
            logger.error("LangChain not available - cannot setup fallback LLM")
            return
        
        try:
            endpoint_url = self.llm_config.get('endpoint_url')
            api_key = self.llm_config.get('api_key')
            model_name = self.llm_config.get('model', 'gpt-4o')
            
            base_url = endpoint_url.replace('/chat/completions', '') if '/chat/completions' in endpoint_url else endpoint_url
            
            self.llm = ChatOpenAI(
                model_name=model_name,
                openai_api_key=api_key,
                openai_api_base=base_url,
                temperature=0.1,
                max_tokens=1000,
                timeout=30
            )
            
            # Setup embeddings
            self.embeddings = OpenAIEmbeddings(
                openai_api_key=api_key,
                openai_api_base=base_url if "embeddings" in base_url else None
            )
            
            logger.info(f"✅ LangChain fallback LLM configured: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to setup LangChain fallback: {e}")
            self.llm = None
            self.embeddings = None
    
    def _setup_ragas_metrics(self):
        """Setup RAGAS metrics with custom LLM."""
        if not RAGAS_AVAILABLE:
            logger.error("RAGAS not available")
            return
        
        try:
            logger.info(f"🔧 Setting up RAGAS metrics (version: {RAGAS_VERSION})")
            
            # Initialize metrics based on available version
            if RAGAS_VERSION == "0.2.x":
                # For RAGAS 0.2.x, initialize metrics with LLM if available
                if self.llm:
                    self.available_metrics = [
                        LLMContextPrecisionWithoutReference(llm=self.llm),
                        LLMContextRecallWithReference(llm=self.llm),
                        Faithfulness(llm=self.llm),
                        ResponseRelevancy(llm=self.llm)
                    ]
                else:
                    # Use default initialization
                    self.available_metrics = [
                        LLMContextPrecisionWithoutReference(),
                        LLMContextRecallWithReference(),
                        Faithfulness(),
                        ResponseRelevancy()
                    ]
                logger.info(f"✅ RAGAS 0.2.x metrics configured")
            else:
                # For legacy RAGAS, use original metrics
                self.available_metrics = [
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy
                ]
                logger.info(f"✅ RAGAS legacy metrics configured")
            
            logger.info(f"📊 Available metrics: {[m.__class__.__name__ for m in self.available_metrics]}")
            
        except Exception as e:
            logger.error(f"Failed to setup RAGAS metrics: {e}")
            # Fallback: try legacy metrics even if 0.2.x was detected
            try:
                self.available_metrics = [
                    context_precision,
                    context_recall,
                    faithfulness,
                    answer_relevancy
                ]
                logger.info(f"✅ Fallback to legacy RAGAS metrics")
            except Exception as e2:
                logger.error(f"Failed to setup fallback metrics: {e2}")
                self.available_metrics = []
    
    def _prepare_ragas_dataset(self, questions: List[str], answers: List[str], 
                             contexts: List[List[str]], ground_truths: List[str]) -> Dataset:
        """
        Prepare dataset in RAGAS-compatible format using the new SingleTurnSample schema.
        
        For RAGAS 0.2.x, we need:
        - 'user_input' instead of 'question'
        - 'response' for the generated answer
        - 'retrieved_contexts' for the contexts
        - 'reference' for the ground truth
        """
        try:
            # Ensure all inputs are properly formatted with RAGAS column names
            processed_data = {
                'user_input': [],      # RAGAS 0.2.x expects 'user_input'
                'response': [],        # Generated response
                'retrieved_contexts': [], # Retrieved contexts
                'reference': []        # Ground truth reference
            }
            
            for i in range(len(questions)):
                # Clean and validate question -> user_input
                question = str(questions[i]).strip()
                if not question:
                    logger.warning(f"Empty question at index {i}, skipping")
                    continue
                
                # Clean and validate answer -> response
                answer = str(answers[i]).strip() if i < len(answers) else ""
                if not answer:
                    logger.warning(f"Empty answer at index {i}, using placeholder")
                    answer = "No answer provided"
                
                # Process contexts -> retrieved_contexts (ensure it's a list of strings)
                if i < len(contexts):
                    context_list = contexts[i]
                    if isinstance(context_list, str):
                        # Handle single context string
                        try:
                            # First try to parse as JSON list
                            context_list = json.loads(context_list)
                        except json.JSONDecodeError:
                            try:
                                # Try to evaluate as Python literal (safer than eval)
                                import ast
                                context_list = ast.literal_eval(context_list)
                            except (ValueError, SyntaxError):
                                # Treat as single context
                                context_list = [context_list]
                    elif not isinstance(context_list, list):
                        context_list = [str(context_list)]
                    
                    # Clean each context
                    context_list = [str(ctx).strip() for ctx in context_list if str(ctx).strip()]
                    if not context_list:
                        context_list = ["No context available"]
                else:
                    context_list = ["No context available"]
                
                # Clean and validate ground truth -> reference
                ground_truth = str(ground_truths[i]).strip() if i < len(ground_truths) else answer
                if not ground_truth:
                    ground_truth = answer
                
                # Add to processed data with RAGAS 0.2.x compatible column names
                processed_data['user_input'].append(question)
                processed_data['response'].append(answer)
                processed_data['retrieved_contexts'].append(context_list)
                processed_data['reference'].append(ground_truth)
            
            # Ensure we have data to create dataset
            if not processed_data['user_input']:
                logger.error("No valid questions found in dataset")
                raise ValueError("No valid questions to create RAGAS dataset")
            
            # Create RAGAS dataset
            dataset = Dataset.from_dict(processed_data)
            
            logger.info(f"✅ RAGAS dataset prepared with {len(dataset)} samples")
            logger.info(f"📊 Schema: {list(processed_data.keys())}")
            logger.debug(f"Sample data: {processed_data['user_input'][0][:100]}...")
            
            return dataset
            
        except Exception as e:
            logger.error(f"Failed to prepare RAGAS dataset: {e}")
            raise
    
    def evaluate_testset(self, testset_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Evaluate testset using RAGAS metrics with detailed calculation tracking.
        
        Args:
            testset_file: Path to enhanced testset CSV
            output_dir: Directory to save evaluation results
            
        Returns:
            Dictionary with evaluation results and detailed calculations
        """
        logger.info(f"🎯 Starting RAGAS evaluation of testset: {testset_file}")
        
        if not RAGAS_AVAILABLE:
            return {
                'success': False,
                'error': 'RAGAS not available - please install ragas package'
            }
        
        if not self.available_metrics:
            return {
                'success': False,
                'error': 'No RAGAS metrics available - check LLM configuration'
            }
        
        try:
            # Load testset
            df = pd.read_csv(testset_file)
            logger.info(f"📊 Loaded testset with {len(df)} rows")
            
            # Validate required columns
            required_columns = ['user_input', 'reference_contexts', 'reference']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Extract data for RAGAS
            questions = df['user_input'].tolist()
            answers = df['reference'].tolist()  # Use reference as answer for now
            contexts = df['reference_contexts'].tolist()
            ground_truths = df['reference'].tolist()  # Use reference as ground truth
            
            logger.info(f"📋 Extracted {len(questions)} questions for evaluation")
            
            # Prepare RAGAS dataset
            dataset = self._prepare_ragas_dataset(questions, answers, contexts, ground_truths)
            
            # Run RAGAS evaluation with detailed tracking
            logger.info("🚀 Running RAGAS evaluation...")
            
            detailed_results = []
            overall_scores = {}
            
            # Evaluate each metric individually to track details
            for metric in self.available_metrics:
                metric_name = metric.__class__.__name__
                logger.info(f"📊 Evaluating metric: {metric_name}")
                
                try:
                    # Run evaluation for this metric with retry logic
                    max_retries = 3
                    result = None
                    
                    for attempt in range(max_retries):
                        try:
                            # For RAGAS 0.2.x, pass LLM and embeddings in evaluate() call
                            if self.llm and self.embeddings:
                                result = evaluate(
                                    dataset=dataset,
                                    metrics=[metric],
                                    llm=self.llm,
                                    embeddings=self.embeddings,
                                    raise_exceptions=False
                                )
                            else:
                                result = evaluate(
                                    dataset=dataset,
                                    metrics=[metric],
                                    raise_exceptions=False
                                )
                            break  # Success, exit retry loop
                            
                        except Exception as eval_error:
                            logger.warning(f"Attempt {attempt + 1}/{max_retries} failed for {metric_name}: {eval_error}")
                            if attempt == max_retries - 1:
                                logger.error(f"All {max_retries} attempts failed for {metric_name}")
                                result = None
                                break
                            import time
                            time.sleep(2)  # Wait 2 seconds before retry
                    
                    if result is None:
                        logger.error(f"❌ Failed to evaluate {metric_name} after {max_retries} attempts")
                        overall_scores[metric_name] = None
                        continue
                    
                    # Extract scores - enhanced error handling for RAGAS 0.2.x
                    scores = []
                    try:
                        if hasattr(result, 'to_pandas'):
                            result_df = result.to_pandas()
                            # Try multiple column name variations
                            if metric_name in result_df.columns:
                                scores = result_df[metric_name].tolist()
                            elif metric_name.lower() in result_df.columns:
                                scores = result_df[metric_name.lower()].tolist()
                            else:
                                # Get first available column that contains the metric name
                                matching_cols = [col for col in result_df.columns if metric_name.lower() in col.lower()]
                                if matching_cols:
                                    scores = result_df[matching_cols[0]].tolist()
                                    logger.warning(f"Using column {matching_cols[0]} for metric {metric_name}")
                                else:
                                    logger.warning(f"No matching columns found for {metric_name}. Available: {result_df.columns.tolist()}")
                        elif hasattr(result, 'to_dict'):
                            result_dict = result.to_dict()
                            scores = result_dict.get(metric_name, result_dict.get(metric_name.lower(), []))
                        elif hasattr(result, metric_name):
                            metric_values = getattr(result, metric_name)
                            if hasattr(metric_values, 'tolist'):
                                scores = metric_values.tolist()
                            elif isinstance(metric_values, (list, tuple)):
                                scores = list(metric_values)
                            else:
                                scores = [metric_values] if metric_values is not None else []
                        else:
                            logger.warning(f"Cannot extract scores for {metric_name} from result type {type(result)}")
                    except Exception as score_error:
                        logger.error(f"Error extracting scores for {metric_name}: {score_error}")
                        scores = []
                    
                    if scores:
                        # Enhanced NaN handling for all RAGAS metrics
                        clean_scores = self._clean_metric_scores(scores, metric_name)
                        
                        if clean_scores:
                            # Calculate statistics with NaN handling
                            stats = self._calculate_enhanced_statistics(clean_scores, metric_name)
                            
                            if stats['valid_count'] > 0:
                                overall_scores[metric_name] = stats['mean_score']
                                
                                # Store detailed results
                                for i, score in enumerate(clean_scores):
                                    if i < len(questions) and score is not None:
                                        detailed_results.append({
                                            'question_index': i,
                                            'question': questions[i][:100] + "..." if len(questions[i]) > 100 else questions[i],
                                            'metric_name': metric_name,
                                            'score': float(score),
                                        'timestamp': datetime.now().isoformat()
                                    })
                            
                            logger.info(f"✅ {metric_name}: {stats['mean_score']:.3f} (n={stats['valid_count']}/{stats['total_count']})")
                        else:
                            logger.warning(f"⚠️ No valid scores for {metric_name}")
                            overall_scores[metric_name] = 0.0  # Use fallback instead of None
                    else:
                        logger.warning(f"⚠️ No scores for {metric_name}")
                        overall_scores[metric_name] = None
                
                except Exception as e:
                    logger.error(f"❌ Failed to evaluate {metric_name}: {e}")
                    import traceback
                    logger.debug(f"Traceback for {metric_name}: {traceback.format_exc()}")
                    overall_scores[metric_name] = None
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save individual calculations
            if detailed_results:
                detailed_file = output_dir / f"ragas_detailed_calculations_{timestamp}.json"
                with open(detailed_file, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
                logger.info(f"💾 Detailed calculations saved to: {detailed_file}")
            
            # Save overall results
            overall_results = {
                'testset_file': str(testset_file),
                'timestamp': timestamp,
                'total_questions': len(questions),
                'metrics_evaluated': len(self.available_metrics),
                'overall_scores': overall_scores,
                'detailed_results_file': str(detailed_file) if detailed_results else None,
                'config': {
                    'llm_model': self.llm_config.get('model', 'unknown'),
                    'llm_endpoint': self.llm_config.get('endpoint_url', 'unknown'),
                    'metrics_used': [m.__class__.__name__ for m in self.available_metrics]
                }
            }
            
            results_file = output_dir / f"ragas_evaluation_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(overall_results, f, indent=2)
            
            logger.info(f"✅ RAGAS evaluation completed successfully!")
            logger.info(f"📊 Overall scores: {overall_scores}")
            logger.info(f"💾 Results saved to: {results_file}")
            
            return {
                'success': True,
                'overall_scores': overall_scores,
                'total_questions': len(questions),
                'detailed_results_file': str(detailed_file) if detailed_results else None,
                'results_file': str(results_file),
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def evaluate_testset_with_rag_responses(self, enhanced_df: pd.DataFrame, 
                                          output_dir: Path) -> Dict[str, Any]:
        """
        Evaluate testset using RAG system responses vs ground truth.
        
        Args:
            enhanced_df: DataFrame with RAG responses and ground truth
            output_dir: Directory to save results
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info("🚀 Evaluating RAG responses with RAGAS metrics...")
        
        try:
            # Extract data for RAGAS - use RAG answers vs ground truth
            questions = enhanced_df['user_input'].tolist()
            rag_answers = enhanced_df['rag_answer'].tolist()  # RAG system answers
            contexts = enhanced_df['reference_contexts'].tolist()  # Original contexts
            ground_truths = enhanced_df['reference'].tolist()  # Ground truth answers
            
            logger.info(f"📋 Evaluating {len(questions)} RAG responses")
            
            # Prepare RAGAS dataset
            dataset = self._prepare_ragas_dataset(questions, rag_answers, contexts, ground_truths)
            
            # Run RAGAS evaluation with detailed tracking
            logger.info("🚀 Running RAGAS evaluation on RAG responses...")
            
            detailed_results = []
            overall_scores = {}
            
            # Evaluate each metric individually to track details
            for metric in self.available_metrics:
                metric_name = metric.__class__.__name__
                logger.info(f"📊 Evaluating RAG responses with metric: {metric_name}")
                
                try:
                    # Run evaluation for this metric
                    # For RAGAS 0.2.x, pass LLM and embeddings in evaluate() call
                    if self.llm and self.embeddings:
                        result = evaluate(
                            dataset=dataset,
                            metrics=[metric],
                            llm=self.llm,
                            embeddings=self.embeddings,
                            raise_exceptions=False
                        )
                    else:
                        result = evaluate(
                            dataset=dataset,
                            metrics=[metric],
                            raise_exceptions=False
                        )
                    
                    # Extract scores - enhanced error handling for RAGAS 0.2.x
                    scores = []
                    try:
                        if hasattr(result, 'to_pandas'):
                            result_df = result.to_pandas()
                            # Try multiple column name variations
                            if metric_name in result_df.columns:
                                scores = result_df[metric_name].tolist()
                            elif metric_name.lower() in result_df.columns:
                                scores = result_df[metric_name.lower()].tolist()
                            else:
                                # Get first available column that contains the metric name
                                matching_cols = [col for col in result_df.columns if metric_name.lower() in col.lower()]
                                if matching_cols:
                                    scores = result_df[matching_cols[0]].tolist()
                                    logger.warning(f"Using column {matching_cols[0]} for metric {metric_name}")
                        elif hasattr(result, 'to_dict'):
                            result_dict = result.to_dict()
                            scores = result_dict.get(metric_name, result_dict.get(metric_name.lower(), []))
                        elif hasattr(result, metric_name):
                            metric_values = getattr(result, metric_name)
                            if hasattr(metric_values, 'tolist'):
                                scores = metric_values.tolist()
                            elif isinstance(metric_values, (list, tuple)):
                                scores = list(metric_values)
                            else:
                                scores = [metric_values] if metric_values is not None else []
                        else:
                            logger.warning(f"Cannot extract scores for {metric_name} from result type {type(result)}")
                    except Exception as score_error:
                        logger.error(f"Error extracting scores for {metric_name}: {score_error}")
                        scores = []
                    
                    if scores:
                        # Use robust NaN handling
                        import sys
                        from pathlib import Path
                        sys.path.append(str(Path(__file__).parent.parent / "utils"))
                        from nan_handling import calculate_robust_summary_stats
                        
                        # Calculate robust statistics
                        stats = calculate_robust_summary_stats(scores, metric_name)
                        
                        if stats['valid_count'] > 0:
                            overall_scores[metric_name] = stats['mean_score']
                            
                            # Store detailed results
                            for i, score in enumerate(scores):
                                if i < len(questions):
                                    detailed_results.append({
                                        'question_index': i,
                                        'question': questions[i][:100] + "..." if len(questions[i]) > 100 else questions[i],
                                        'rag_answer': rag_answers[i][:100] + "..." if len(rag_answers[i]) > 100 else rag_answers[i],
                                        'ground_truth': ground_truths[i][:100] + "..." if len(ground_truths[i]) > 100 else ground_truths[i],
                                        'metric_name': metric_name,
                                        'score': float(score) if score is not None and isinstance(score, (int, float)) else None,
                                        'timestamp': datetime.now().isoformat()
                                    })
                            
                            logger.info(f"✅ {metric_name}: {stats['mean_score']:.3f} (n={stats['valid_count']}/{stats['total_count']} RAG vs Ground Truth)")
                        else:
                            logger.warning(f"⚠️ No valid scores for {metric_name}")
                            overall_scores[metric_name] = 0.0  # Use fallback instead of None
                    else:
                        logger.warning(f"⚠️ No scores for {metric_name}")
                        overall_scores[metric_name] = 0.0  # Use fallback instead of None
                
                except Exception as e:
                    logger.error(f"❌ Failed to evaluate {metric_name}: {e}")
                    overall_scores[metric_name] = None
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save individual calculations
            if detailed_results:
                detailed_file = output_dir / f"ragas_rag_detailed_calculations_{timestamp}.json"
                with open(detailed_file, 'w') as f:
                    json.dump(detailed_results, f, indent=2)
                logger.info(f"💾 Detailed RAG evaluation saved to: {detailed_file}")
            
            # Save overall results
            overall_results = {
                'evaluation_type': 'ragas_with_rag_responses',
                'timestamp': timestamp,
                'total_questions': len(questions),
                'metrics_evaluated': len(self.available_metrics),
                'overall_scores': overall_scores,
                'detailed_results_file': str(detailed_file) if detailed_results else None,
                'config': {
                    'llm_model': self.llm_config.get('model', 'unknown'),
                    'llm_endpoint': self.llm_config.get('endpoint_url', 'unknown'),
                    'metrics_used': [m.__class__.__name__ for m in self.available_metrics]
                }
            }
            
            results_file = output_dir / f"ragas_rag_evaluation_results_{timestamp}.json"
            with open(results_file, 'w') as f:
                json.dump(overall_results, f, indent=2)
            
            logger.info(f"✅ RAGAS RAG evaluation completed successfully!")
            logger.info(f"📊 RAG vs Ground Truth scores: {overall_scores}")
            logger.info(f"💾 Results saved to: {results_file}")
            
            return {
                'success': True,
                'evaluation_type': 'ragas_with_rag_responses',
                'overall_scores': overall_scores,
                'total_questions': len(questions),
                'detailed_results_file': str(detailed_file) if detailed_results else None,
                'results_file': str(results_file),
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ RAGAS RAG evaluation failed: {e}")
            import traceback
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _clean_metric_scores(self, scores: List[Any], metric_name: str) -> List[float]:
        """
        Clean metric scores with comprehensive NaN handling for all 4 RAGAS metrics:
        - Faithfulness
        - AnswerRelevancy (ResponseRelevancy)  
        - ContextPrecision
        - ContextRecall
        """
        clean_scores = []
        nan_count = 0
        
        for score in scores:
            try:
                if score is None:
                    nan_count += 1
                    clean_scores.append(0.0)
                    continue
                    
                float_score = float(score)
                
                if math.isnan(float_score) or math.isinf(float_score):
                    logger.warning(f"NaN/Inf value found in {metric_name}: {score}")
                    nan_count += 1
                    clean_scores.append(0.0)
                    continue
                    
                # Ensure score is within valid range [0, 1]
                if not (0.0 <= float_score <= 1.0):
                    logger.warning(f"Score {float_score} for {metric_name} outside valid range [0,1], clipping")
                    float_score = np.clip(float_score, 0.0, 1.0)
                    
                clean_scores.append(float_score)
                
            except (ValueError, TypeError) as e:
                logger.error(f"Could not convert score {score} to float for {metric_name}: {e}")
                nan_count += 1
                clean_scores.append(0.0)
        
        if nan_count > 0:
            logger.warning(f"Replaced {nan_count}/{len(scores)} NaN values with 0.0 for {metric_name}")
            
        return clean_scores
    
    def _calculate_enhanced_statistics(self, scores: List[float], metric_name: str) -> Dict[str, Any]:
        """Calculate enhanced statistics with NaN handling"""
        if not scores:
            return {
                'mean_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'valid_count': 0,
                'total_count': 0
            }
        
        # Remove any remaining NaN values
        valid_scores = [s for s in scores if not (math.isnan(s) if isinstance(s, float) else False)]
        
        if not valid_scores:
            logger.warning(f"No valid scores found for {metric_name}")
            return {
                'mean_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'valid_count': 0,
                'total_count': len(scores)
            }
        
        try:
            mean_score = float(np.mean(valid_scores))
            std_score = float(np.std(valid_scores)) if len(valid_scores) > 1 else 0.0
            min_score = float(min(valid_scores))
            max_score = float(max(valid_scores))
            
            return {
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'valid_count': len(valid_scores),
                'total_count': len(scores)
            }
            
        except Exception as e:
            logger.error(f"Error calculating statistics for {metric_name}: {e}")
            return {
                'mean_score': 0.0,
                'std_score': 0.0,
                'min_score': 0.0,
                'max_score': 0.0,
                'valid_count': 0,
                'total_count': len(scores)
            }
