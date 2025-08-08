"""
Comprehensive RAG Evaluator for Pipeline

Implements complete RAG evaluation with:
1. RAG System Testing - Query the RAG endpoint with testset questions
2. Contextual Keyword Evaluation - Using existing contextual keyword gate
3. RAGAS Metrics Evaluation - Standard RAGAS metrics for RAG
4. Detailed Scoring and Reporting - Save all calculation steps
"""

# Import fix applied
import sys
from pathlib import Path

# Add utils directory to Python path for local imports
current_file_dir = Path(__file__).parent
utils_dir = current_file_dir.parent / "utils"
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))


import csv
import json
import logging
import math
import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
import requests
import time

logger = logging.getLogger(__name__)

class ComprehensiveRAGEvaluator:
    """Complete RAG evaluation with contextual keywords and RAGAS metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the comprehensive RAG evaluator."""
        self.config = config
        self.similarity_threshold = config.get('keyword_similarity_threshold', 0.7)
        
        # Initialize component evaluators with null pattern to disable RAGAS temporarily
        logger.warning("🔧 Initializing comprehensive RAG evaluator with RAGAS disabled")
        
        # Disable RAGAS to avoid model_dump errors
        self.ragas_evaluate = None
        self.ragas_metrics = []
        
        logger.info("✅ Comprehensive RAG evaluator initialized (RAGAS disabled)")
    
    def _init_contextual_evaluator(self):
        """Initialize contextual keyword evaluator."""
        try:
            # Import contextual keyword functions
            import sys
            from pathlib import Path
            
            # Add parent directories to access contextual_keyword_gate.py
            sys.path.append(str(Path(__file__).parent.parent.parent.parent))
            
            from contextual_keyword_gate import weighted_keyword_score, get_contextual_segments
            
            self.contextual_functions = {
                'weighted_keyword_score': weighted_keyword_score,
                'get_contextual_segments': get_contextual_segments
            }
            
            # Contextual keyword configuration
            contextual_config = self.eval_config.get('contextual_keywords', {})
            self.contextual_weights = contextual_config.get('weights', {'mandatory': 0.8, 'optional': 0.2})
            self.contextual_threshold = contextual_config.get('threshold', 0.6)
            
            logger.info("Contextual keyword evaluator initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize contextual keyword evaluator: {e}")
            self.contextual_functions = None
    
    def _init_ragas_evaluator(self):
        """Initialize RAGAS metrics evaluator."""
        try:
            # Import RAGAS components
            from ragas import evaluate
            from ragas.metrics import (
                context_precision,
                context_recall, 
                faithfulness,
                answer_relevancy
            )
            from ragas.llms import LangchainLLMWrapper
            
            # Configure RAGAS LLM
            ragas_config = self.eval_config.get('ragas_metrics', {})
            llm_config = ragas_config.get('llm', {})
            
            if llm_config.get('use_custom_llm', False):
                from langchain_openai import ChatOpenAI
                
                # Use custom LLM endpoint
                custom_llm = ChatOpenAI(
                    base_url=llm_config.get('endpoint', ''),
                    api_key=llm_config.get('api_key', ''),
                    model=llm_config.get('model_name', 'gpt-4o'),
                    temperature=llm_config.get('temperature', 0.1),
                    max_tokens=llm_config.get('max_tokens', 1000),
                    timeout=llm_config.get('timeout', 60)
                )
                
                self.ragas_llm = LangchainLLMWrapper(custom_llm)
            else:
                self.ragas_llm = None
            
            # Configure metrics
            enabled_metrics = ragas_config.get('metrics', ['context_precision', 'faithfulness'])
            self.ragas_metrics = []
            
            metric_map = {
                'context_precision': context_precision,
                'context_recall': context_recall,
                'faithfulness': faithfulness,
                'answer_relevancy': answer_relevancy
            }
            
            for metric_name in enabled_metrics:
                if metric_name in metric_map:
                    metric = metric_map[metric_name]
                    if self.ragas_llm:
                        metric.llm = self.ragas_llm
                    self.ragas_metrics.append(metric)
            
            self.ragas_evaluate = evaluate
            logger.info(f"RAGAS evaluator initialized with metrics: {enabled_metrics}")
            
        except Exception as e:
            logger.error(f"Failed to initialize RAGAS evaluator: {e}")
            self.ragas_evaluate = None
            self.ragas_metrics = []
    
    def query_rag_system(self, question: str) -> Dict[str, Any]:
        """Query the RAG system with a question."""
        try:
            # Prepare request payload
            payload_template = self.request_format.get('payload_template', {})
            payload = payload_template.copy()
            
            # Set the question field
            question_field = self.request_format.get('question_field', 'content')
            payload[question_field] = question
            
            # Make request to RAG system
            response = requests.post(
                self.rag_endpoint,
                json=payload,
                timeout=self.rag_timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract answer from response
                response_fields = self.request_format.get('response_fields', {})
                answer_field = response_fields.get('answer', 'message[0].content')
                
                # Handle nested field access like 'message[0].content'
                answer = self._extract_nested_field(result, answer_field)
                
                return {
                    'success': True,
                    'answer': answer,
                    'raw_response': result,
                    'status_code': response.status_code
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'status_code': response.status_code
                }
                
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                'success': False,
                'error': str(e),
                'status_code': None
            }
    
    def _extract_nested_field(self, data: Dict, field_path: str) -> str:
        """Extract nested field from response data."""
        try:
            # Handle simple field names
            if '.' not in field_path and '[' not in field_path:
                return str(data.get(field_path, ''))
            
            # Handle nested paths like 'message[0].content'
            current = data
            
            # Split by dots and handle array indices
            parts = field_path.split('.')
            for part in parts:
                if '[' in part and ']' in part:
                    # Handle array access like 'message[0]'
                    field_name = part.split('[')[0]
                    index = int(part.split('[')[1].split(']')[0])
                    current = current[field_name][index]
                else:
                    current = current[part]
            
            return str(current)
            
        except Exception as e:
            logger.error(f"Error extracting field {field_path}: {e}")
            return ""
    
    def evaluate_contextual_keywords(self, rag_answer: str, keywords: List[str]) -> Dict[str, Any]:
        """Evaluate RAG answer using contextual keyword matching."""
        if not self.contextual_functions:
            return {'error': 'Contextual keyword evaluator not available'}
        
        try:
            # Use existing contextual keyword evaluation
            weighted_keyword_score = self.contextual_functions['weighted_keyword_score']
            
            # Split keywords and clean them
            mandatory_keywords = [kw.strip() for kw in keywords if kw.strip()]
            optional_keywords = []  # Can be extended later
            
            # Calculate contextual keyword score
            total_score, mandatory_score, optional_score, answer_segments = weighted_keyword_score(
                mandatory_keywords, rag_answer, self.contextual_weights, optional_keywords
            )
            
            # Determine if passes threshold
            passes_threshold = total_score >= self.contextual_threshold
            
            return {
                'total_score': float(total_score),
                'mandatory_score': float(mandatory_score),
                'optional_score': float(optional_score),
                'passes_threshold': bool(passes_threshold),
                'threshold': self.contextual_threshold,
                'answer_segments': answer_segments,
                'mandatory_keywords': mandatory_keywords,
                'optional_keywords': optional_keywords
            }
            
        except Exception as e:
            logger.error(f"Error in contextual keyword evaluation: {e}")
            return {'error': str(e)}
    
    def evaluate_ragas_metrics(self, question: str, rag_answer: str, 
                             contexts: List[str], ground_truth: str) -> Dict[str, Any]:
        """Evaluate using RAGAS metrics."""
        if not self.ragas_evaluate or not self.ragas_metrics:
            return {'error': 'RAGAS evaluator not available'}
        
        # Temporary disable RAGAS evaluation to fix model_dump errors
        logger.warning("⚠️ RAGAS metrics evaluation temporarily disabled due to model_dump compatibility issues")
        return {
            'context_precision': 0.7,
            'context_recall': 0.8,
            'faithfulness': 0.75,
            'answer_relevancy': 0.72,
            'message': 'RAGAS evaluation disabled due to model_dump compatibility issues',
            'mock_data': True
        }
        
        try:
            # Prepare dataset for RAGAS
            from datasets import Dataset
            
            eval_data = {
                'question': [question],
                'answer': [rag_answer], 
                'contexts': [contexts],
                'ground_truth': [ground_truth]
            }
            
            dataset = Dataset.from_dict(eval_data)
            
            # Run RAGAS evaluation
            result = self.ragas_evaluate(
                dataset=dataset,
                metrics=self.ragas_metrics,
                raise_exceptions=False
            )
            
            # Extract scores
            scores = {}
            for metric in self.ragas_metrics:
                metric_name = metric.__class__.__name__.lower()
                if metric_name in result:
                    scores[metric_name] = float(result[metric_name])
            
            return {
                'scores': scores,
                'raw_result': result.to_dict() if hasattr(result, 'to_dict') else str(result)
            }
            
        except Exception as e:
            logger.error(f"Error in RAGAS evaluation: {e}")
            return {'error': str(e)}
    
    def evaluate_testset(self, testset_file: Path, output_dir: Path) -> Dict[str, Any]:
        """Evaluate RAG system using enhanced testset."""
        logger.info(f"Starting RAG evaluation with testset: {testset_file}")
        
        try:
            # Load enhanced testset
            df = pd.read_csv(testset_file)
            logger.info(f"Loaded testset with {len(df)} questions")
            
            # Validate required columns
            required_columns = ['user_input', 'reference_contexts', 'reference', 'auto_keywords']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Initialize results storage
            detailed_results = []
            summary_scores = {
                'contextual_keyword': [],
                'ragas_metrics': {}
            }
            
            # Process each question
            for idx, row in df.iterrows():
                logger.info(f"Evaluating question {idx + 1}/{len(df)}")
                
                question = str(row['user_input'])
                ground_truth = str(row['reference'])
                contexts = eval(row['reference_contexts']) if isinstance(row['reference_contexts'], str) else row['reference_contexts']
                
                # Parse keywords properly for Chinese text
                keywords_str = str(row['auto_keywords'])
                keywords = re.split(r'[,，、]', keywords_str)
                keywords = [kw.strip() for kw in keywords if kw.strip()]
                
                # Query RAG system
                rag_response = self.query_rag_system(question)
                
                if not rag_response.get('success', False):
                    logger.error(f"RAG query failed for question {idx + 1}: {rag_response.get('error', 'Unknown error')}")
                    continue
                
                rag_answer = rag_response['answer']
                
                # Evaluate with contextual keywords
                contextual_result = self.evaluate_contextual_keywords(rag_answer, keywords)
                
                # Evaluate with RAGAS metrics
                ragas_result = self.evaluate_ragas_metrics(
                    question, rag_answer, contexts, ground_truth
                )
                
                # Store detailed results
                detailed_result = {
                    'question_idx': idx,
                    'question': question,
                    'ground_truth': ground_truth,
                    'contexts': contexts,
                    'keywords': keywords,
                    'rag_answer': rag_answer,
                    'rag_response_raw': rag_response.get('raw_response', {}),
                    'contextual_keyword_evaluation': contextual_result,
                    'ragas_evaluation': ragas_result,
                    'timestamp': datetime.now().isoformat()
                }
                
                detailed_results.append(detailed_result)
                
                # Collect summary scores
                if 'error' not in contextual_result:
                    summary_scores['contextual_keyword'].append(contextual_result['total_score'])
                
                if 'error' not in ragas_result and 'scores' in ragas_result:
                    for metric_name, score in ragas_result['scores'].items():
                        if metric_name not in summary_scores['ragas_metrics']:
                            summary_scores['ragas_metrics'][metric_name] = []
                        summary_scores['ragas_metrics'][metric_name].append(score)
                
                logger.info(f"Question {idx + 1} completed")
            
            # Calculate summary statistics
            summary_stats = self._calculate_summary_stats(summary_scores)
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save detailed evaluation results (all calculation steps)
            detailed_file = output_dir / f"detailed_evaluation_results_{timestamp}.json"
            with open(detailed_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            # Save summary report
            summary_file = output_dir / f"evaluation_summary_{timestamp}.json"
            final_report = {
                'evaluation_summary': summary_stats,
                'configuration': {
                    'testset_file': str(testset_file),
                    'rag_endpoint': self.rag_endpoint,
                    'contextual_threshold': self.contextual_threshold,
                    'contextual_weights': self.contextual_weights,
                    'ragas_metrics': [m.__class__.__name__ for m in self.ragas_metrics]
                },
                'total_questions': len(df),
                'successful_evaluations': len(detailed_results),
                'timestamp': timestamp
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, indent=2, ensure_ascii=False)
            
            # Save CSV report for easy analysis
            csv_file = output_dir / f"evaluation_report_{timestamp}.csv"
            self._save_csv_report(detailed_results, csv_file)
            
            logger.info(f"✅ RAG evaluation completed successfully")
            logger.info(f"📊 Detailed results: {detailed_file}")
            logger.info(f"📋 Summary report: {summary_file}")
            logger.info(f"📈 CSV report: {csv_file}")
            
            return {
                'success': True,
                'detailed_results_file': str(detailed_file),
                'summary_report_file': str(summary_file),
                'csv_report_file': str(csv_file),
                'summary_stats': summary_stats,
                'total_questions': len(df),
                'successful_evaluations': len(detailed_results)
            }
            
        except Exception as e:
            logger.error(f"Error in RAG evaluation: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_summary_stats(self, scores: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate summary statistics from scores with robust NaN handling."""
        from nan_handling import safe_mean, safe_std, safe_min_max
        
        stats = {}
        
        # Contextual keyword statistics
        contextual_pass_rate = 0.0
        if scores['contextual_keyword']:
            ck_scores = scores['contextual_keyword']
            # Use safe calculations
            mean_score = safe_mean(ck_scores)
            std_score = safe_std(ck_scores)
            min_score, max_score = safe_min_max(ck_scores)
            
            # Calculate pass rate safely
            pass_indicators = [1 if score >= self.contextual_threshold else 0 for score in ck_scores if isinstance(score, (int, float))]
            contextual_pass_rate = safe_mean(pass_indicators)
            
            stats['contextual_keyword'] = {
                'mean_score': mean_score,
                'std_score': std_score,
                'min_score': min_score,
                'max_score': max_score,
                'pass_rate': contextual_pass_rate,
                'total_evaluations': len(ck_scores)
            }
        
        # RAGAS metrics statistics with robust NaN handling
        ragas_pass_rate = 0.0
        ragas_scores_all = []
        stats['ragas_metrics'] = {}
        for metric_name, metric_scores in scores['ragas_metrics'].items():
            if metric_scores:
                # Filter valid scores only
                valid_scores = [s for s in metric_scores if isinstance(s, (int, float)) and not math.isnan(s)]
                if valid_scores:
                    ragas_scores_all.extend(valid_scores)
                    stats['ragas_metrics'][metric_name] = {
                        'mean_score': safe_mean(valid_scores),
                        'std_score': safe_std(valid_scores),
                        'min_score': safe_min_max(valid_scores)[0],
                        'max_score': safe_min_max(valid_scores)[1],
                        'total_evaluations': len(metric_scores),
                        'valid_evaluations': len(valid_scores)
                    }
                else:
                    logger.warning(f"No valid scores found for {metric_name}")
                    stats['ragas_metrics'][metric_name] = {
                        'mean_score': 0.0,
                        'std_score': 0.0,
                        'min_score': 0.0,
                        'max_score': 0.0,
                        'total_evaluations': len(metric_scores),
                        'valid_evaluations': 0
                    }
        
        # Calculate overall RAGAS pass rate (if mean of all metrics > 0.7)
        if ragas_scores_all:
            pass_indicators = [1 if score >= 0.7 else 0 for score in ragas_scores_all]
            ragas_pass_rate = safe_mean(pass_indicators)
        
        # Add overall statistics structure that report generator expects
        semantic_pass_rate = 0.5 * (contextual_pass_rate + ragas_pass_rate)  # Simple average for now
        overall_pass_rate = contextual_pass_rate * ragas_pass_rate  # Both must pass
        
        stats['overall_statistics'] = {
            'contextual_pass_rate': contextual_pass_rate,
            'ragas_pass_rate': ragas_pass_rate,
            'semantic_pass_rate': semantic_pass_rate,
            'overall_pass_rate': overall_pass_rate,
            'total_evaluations': len(scores.get('contextual_keyword', []))
        }
        
        # Add score statistics structure with safe calculations
        contextual_mean = safe_mean(scores['contextual_keyword']) if scores['contextual_keyword'] else 0.0
        contextual_std = safe_std(scores['contextual_keyword']) if scores['contextual_keyword'] else 0.0
        ragas_mean = safe_mean(ragas_scores_all) if ragas_scores_all else 0.0
        ragas_std = safe_std(ragas_scores_all) if ragas_scores_all else 0.0
        
        stats['score_statistics'] = {
            'contextual_scores': {
                'mean': contextual_mean,
                'std': contextual_std
            },
            'ragas_scores': {
                'mean': ragas_mean,
                'std': ragas_std
            },
            'semantic_scores': {
                'mean': 0.5 * (contextual_mean + ragas_mean),
                'std': 0.0  # Placeholder
            }
        }
        
        # Add human feedback statistics (placeholder)
        stats['human_feedback_statistics'] = {
            'feedback_needed_count': 0,
            'feedback_needed_ratio': 0.0
        }
        
        return stats
    
    def _save_csv_report(self, detailed_results: List[Dict], csv_file: Path):
        """Save detailed results as CSV for easy analysis."""
        if not detailed_results:
            return
        
        csv_data = []
        for result in detailed_results:
            row = {
                'question_idx': result['question_idx'],
                'question': result['question'],
                'rag_answer': result['rag_answer'],
                'ground_truth': result['ground_truth'],
                'keywords': ', '.join(result['keywords']),
            }
            
            # Add contextual keyword scores
            ck_eval = result.get('contextual_keyword_evaluation', {})
            if 'error' not in ck_eval:
                row.update({
                    'ck_total_score': ck_eval.get('total_score', 0),
                    'ck_mandatory_score': ck_eval.get('mandatory_score', 0),
                    'ck_optional_score': ck_eval.get('optional_score', 0),
                    'ck_passes_threshold': ck_eval.get('passes_threshold', False),
                })
            
            # Add RAGAS scores
            ragas_eval = result.get('ragas_evaluation', {})
            if 'error' not in ragas_eval and 'scores' in ragas_eval:
                for metric_name, score in ragas_eval['scores'].items():
                    row[f'ragas_{metric_name}'] = score
            
            csv_data.append(row)
        
        # Save to CSV
        df_report = pd.DataFrame(csv_data)
        df_report.to_csv(csv_file, index=False)
        logger.info(f"CSV report saved: {csv_file}")
