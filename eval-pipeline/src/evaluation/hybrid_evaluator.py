#!/usr/bin/env python3
"""
Hybrid Evaluator - Combines Contextual Keyword Gate + RAGAS + Human Feedback

This module integrates:
1. Your existing contextual keyword gate system
2. RAGAS metrics for comprehensive evaluation
3. Dynamic human feedback integration
4. Multi-source testset evaluation
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging
from datetime import datetime

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import your existing evaluation systems
from contextual_keyword_gate import (
    weighted_keyword_score, 
    get_contextual_segments,
    ContextualKeywordGate
)
from dynamic_ragas_gate_with_human_feedback import (
    needs_human_feedback_dynamic,
    adaptive_exponential_smoothing,
    calculate_feedback_consistency,
    calculate_adaptive_window_size
)

# Import RAGAS components
sys.path.append(str(Path(__file__).parent.parent.parent.parent / "ragas"))
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
    answer_similarity,
    answer_correctness
)

class HybridEvaluator:
    """
    Comprehensive hybrid evaluator that combines multiple evaluation approaches:
    - Contextual keyword matching with semantic similarity
    - RAGAS metrics for retrieval-augmented generation
    - Dynamic human feedback integration
    - Multi-dimensional scoring and reporting
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluation components
        self.contextual_gate = ContextualKeywordGate()
        
        # Evaluation thresholds and weights
        self.evaluation_config = config.get('evaluation', {})
        self.contextual_threshold = self.evaluation_config.get('contextual_threshold', 0.6)
        self.ragas_threshold = self.evaluation_config.get('ragas_threshold', 0.7)
        
        # Keyword weights
        self.keyword_weights = self.evaluation_config.get('keyword_weights', {
            "mandatory": 0.8,
            "optional": 0.2
        })
        
        # Human feedback tracking
        self.feedback_history = []
        self.threshold_history = [self.ragas_threshold]
        self.all_ragas_scores = []
        
        # Results tracking
        self.evaluation_results = []
        self.metadata = {
            'start_time': datetime.now(),
            'total_evaluations': 0,
            'threshold_adjustments': 0
        }
        
        self.logger.info("HybridEvaluator initialized with contextual and RAGAS metrics")
    
    def evaluate_rag_response(self, 
                            question: str,
                            rag_answer: str,
                            expected_answer: str,
                            auto_keywords: List[str],
                            contexts: List[str],
                            metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive evaluation of a single RAG response using hybrid approach
        
        Args:
            question: The input question
            rag_answer: Response from RAG system
            expected_answer: Ground truth answer
            auto_keywords: Auto-extracted keywords from document
            contexts: Retrieved contexts from RAG system
            metadata: Additional metadata (source file, question type, etc.)
        
        Returns:
            Comprehensive evaluation results
        """
        evaluation_id = len(self.evaluation_results)
        self.logger.debug(f"Evaluating response {evaluation_id}")
        
        # 1. Contextual keyword evaluation using your existing system
        contextual_results = self._evaluate_contextual_keywords(
            rag_answer, auto_keywords
        )
        
        # 2. RAGAS evaluation
        ragas_results = self._evaluate_with_ragas(
            question, rag_answer, expected_answer, contexts
        )
        
        # 3. Semantic similarity evaluation
        semantic_results = self._evaluate_semantic_similarity(
            rag_answer, expected_answer
        )
        
        # 4. Dynamic human feedback assessment
        feedback_results = self._assess_human_feedback_need(
            ragas_results, evaluation_id
        )
        
        # 5. Compile comprehensive results
        comprehensive_results = {
            'evaluation_id': evaluation_id,
            'timestamp': datetime.now().isoformat(),
            'question': question,
            'rag_answer': rag_answer,
            'expected_answer': expected_answer,
            'auto_keywords': auto_keywords,
            'contexts': contexts,
            'metadata': metadata or {},
            
            # Contextual keyword results
            **contextual_results,
            
            # RAGAS results
            **ragas_results,
            
            # Semantic similarity results
            **semantic_results,
            
            # Human feedback results
            **feedback_results,
            
            # Overall assessment
            'overall_score': self._calculate_overall_score(
                contextual_results, ragas_results, semantic_results
            ),
            'overall_pass': self._determine_overall_pass(
                contextual_results, ragas_results, semantic_results
            )
        }
        
        # Store results
        self.evaluation_results.append(comprehensive_results)
        self.metadata['total_evaluations'] += 1
        
        return comprehensive_results
    
    def _evaluate_contextual_keywords(self, rag_answer: str, 
                                    auto_keywords: List[str]) -> Dict[str, Any]:
        """Evaluate using your existing contextual keyword system"""
        try:
            # Use your existing weighted keyword scoring
            total_score, mandatory_score, optional_score, answer_segments = (
                weighted_keyword_score(auto_keywords, rag_answer, self.keyword_weights)
            )
            
            # Get contextual segments
            contextual_segments = get_contextual_segments(rag_answer)
            
            return {
                'contextual_total_score': total_score,
                'contextual_mandatory_score': mandatory_score,
                'contextual_optional_score': optional_score,
                'contextual_segments': contextual_segments,
                'answer_segments': answer_segments,
                'contextual_keyword_pass': total_score >= self.contextual_threshold,
                'keyword_coverage_ratio': len(answer_segments) / len(auto_keywords) if auto_keywords else 0
            }
        except Exception as e:
            self.logger.error(f"Contextual keyword evaluation error: {e}")
            return {
                'contextual_total_score': 0.0,
                'contextual_mandatory_score': 0.0,
                'contextual_optional_score': 0.0,
                'contextual_segments': [],
                'answer_segments': [],
                'contextual_keyword_pass': False,
                'keyword_coverage_ratio': 0.0,
                'contextual_error': str(e)
            }
    
    def _evaluate_with_ragas(self, question: str, rag_answer: str, 
                           expected_answer: str, contexts: List[str]) -> Dict[str, Any]:
        """Evaluate using RAGAS metrics"""
        try:
            # Prepare data for RAGAS evaluation
            evaluation_data = {
                'question': [question],
                'answer': [rag_answer],
                'contexts': [contexts],
                'ground_truth': [expected_answer]
            }
            
            # Convert to DataFrame
            eval_df = pd.DataFrame(evaluation_data)
            
            # Define RAGAS metrics to evaluate
            metrics = [
                faithfulness,
                answer_relevancy,
                context_precision,
                context_recall,
                answer_similarity,
                answer_correctness
            ]
            
            # Run RAGAS evaluation
            ragas_results = evaluate(eval_df, metrics=metrics)
            
            # Extract individual scores
            ragas_scores = {
                'ragas_faithfulness': ragas_results['faithfulness'].iloc[0] if 'faithfulness' in ragas_results else 0.0,
                'ragas_answer_relevancy': ragas_results['answer_relevancy'].iloc[0] if 'answer_relevancy' in ragas_results else 0.0,
                'ragas_context_precision': ragas_results['context_precision'].iloc[0] if 'context_precision' in ragas_results else 0.0,
                'ragas_context_recall': ragas_results['context_recall'].iloc[0] if 'context_recall' in ragas_results else 0.0,
                'ragas_answer_similarity': ragas_results['answer_similarity'].iloc[0] if 'answer_similarity' in ragas_results else 0.0,
                'ragas_answer_correctness': ragas_results['answer_correctness'].iloc[0] if 'answer_correctness' in ragas_results else 0.0
            }
            
            # Calculate composite RAGAS score
            composite_score = np.mean([score for score in ragas_scores.values() if score > 0])
            self.all_ragas_scores.append(composite_score)
            
            ragas_scores.update({
                'ragas_composite_score': composite_score,
                'ragas_pass': composite_score >= self.ragas_threshold,
                'ragas_threshold_current': self.ragas_threshold
            })
            
            return ragas_scores
            
        except Exception as e:
            self.logger.error(f"RAGAS evaluation error: {e}")
            return {
                'ragas_faithfulness': 0.0,
                'ragas_answer_relevancy': 0.0,
                'ragas_context_precision': 0.0,
                'ragas_context_recall': 0.0,
                'ragas_answer_similarity': 0.0,
                'ragas_answer_correctness': 0.0,
                'ragas_composite_score': 0.0,
                'ragas_pass': False,
                'ragas_threshold_current': self.ragas_threshold,
                'ragas_error': str(e)
            }
    
    def _evaluate_semantic_similarity(self, rag_answer: str, 
                                    expected_answer: str) -> Dict[str, Any]:
        """Evaluate semantic similarity using sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer, util
            
            # Use the same model as your contextual system
            model = SentenceTransformer("all-MiniLM-L6-v2")
            
            # Get embeddings
            rag_embedding = model.encode([rag_answer])
            expected_embedding = model.encode([expected_answer])
            
            # Calculate similarity
            similarity_score = util.pytorch_cos_sim(rag_embedding, expected_embedding).item()
            
            # Additional semantic metrics
            rag_segments = get_contextual_segments(rag_answer)
            expected_segments = get_contextual_segments(expected_answer)
            
            segment_similarities = []
            if rag_segments and expected_segments:
                rag_seg_embeddings = model.encode(rag_segments)
                exp_seg_embeddings = model.encode(expected_segments)
                
                for exp_emb in exp_seg_embeddings:
                    seg_sims = [
                        util.pytorch_cos_sim([exp_emb], [rag_emb]).item()
                        for rag_emb in rag_seg_embeddings
                    ]
                    segment_similarities.append(max(seg_sims) if seg_sims else 0.0)
            
            return {
                'semantic_similarity': similarity_score,
                'segment_similarity_avg': np.mean(segment_similarities) if segment_similarities else 0.0,
                'segment_similarity_max': max(segment_similarities) if segment_similarities else 0.0,
                'semantic_pass': similarity_score >= 0.6,
                'segment_count_ratio': len(rag_segments) / len(expected_segments) if expected_segments else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Semantic similarity evaluation error: {e}")
            return {
                'semantic_similarity': 0.0,
                'segment_similarity_avg': 0.0,
                'segment_similarity_max': 0.0,
                'semantic_pass': False,
                'segment_count_ratio': 0.0,
                'semantic_error': str(e)
            }
    
    def _assess_human_feedback_need(self, ragas_results: Dict[str, Any], 
                                  evaluation_id: int) -> Dict[str, Any]:
        """Assess if human feedback is needed using your dynamic system"""
        try:
            ragas_score = ragas_results.get('ragas_composite_score', 0.0)
            
            # Use your existing dynamic human feedback system
            needs_feedback = needs_human_feedback_dynamic(
                ragas_score, 
                self.all_ragas_scores, 
                evaluation_id,
                uncertainty_min=0.3,
                uncertainty_max=0.9,
                uncertainty_buffer=0.1
            )
            
            # Calculate feedback consistency if we have enough data
            feedback_consistency = 0.0
            if len(self.feedback_history) >= 5:
                feedback_consistency = calculate_feedback_consistency(
                    self.feedback_history[-5:]
                )
            
            return {
                'needs_human_feedback': needs_feedback,
                'feedback_consistency': feedback_consistency,
                'uncertainty_score': abs(ragas_score - 0.5) * 2,  # Distance from maximum uncertainty
                'feedback_confidence': 1.0 - (abs(ragas_score - 0.5) * 2) if ragas_score else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Human feedback assessment error: {e}")
            return {
                'needs_human_feedback': False,
                'feedback_consistency': 0.0,
                'uncertainty_score': 0.0,
                'feedback_confidence': 0.0,
                'feedback_error': str(e)
            }
    
    def _calculate_overall_score(self, contextual_results: Dict, 
                               ragas_results: Dict, semantic_results: Dict) -> float:
        """Calculate weighted overall score"""
        weights = self.evaluation_config.get('score_weights', {
            'contextual': 0.3,
            'ragas': 0.4,
            'semantic': 0.3
        })
        
        contextual_score = contextual_results.get('contextual_total_score', 0.0)
        ragas_score = ragas_results.get('ragas_composite_score', 0.0)
        semantic_score = semantic_results.get('semantic_similarity', 0.0)
        
        overall_score = (
            contextual_score * weights['contextual'] +
            ragas_score * weights['ragas'] +
            semantic_score * weights['semantic']
        )
        
        return round(overall_score, 4)
    
    def _determine_overall_pass(self, contextual_results: Dict, 
                              ragas_results: Dict, semantic_results: Dict) -> bool:
        """Determine if evaluation passes overall criteria"""
        contextual_pass = contextual_results.get('contextual_keyword_pass', False)
        ragas_pass = ragas_results.get('ragas_pass', False)
        semantic_pass = semantic_results.get('semantic_pass', False)
        
        # Configurable pass criteria
        pass_criteria = self.evaluation_config.get('pass_criteria', 'all')
        
        if pass_criteria == 'all':
            return contextual_pass and ragas_pass and semantic_pass
        elif pass_criteria == 'majority':
            return sum([contextual_pass, ragas_pass, semantic_pass]) >= 2
        elif pass_criteria == 'any':
            return contextual_pass or ragas_pass or semantic_pass
        else:
            return contextual_pass and ragas_pass  # Default
    
    def batch_evaluate(self, evaluation_data: pd.DataFrame) -> pd.DataFrame:
        """Evaluate multiple RAG responses in batch"""
        self.logger.info(f"Starting batch evaluation of {len(evaluation_data)} samples")
        
        results = []
        for idx, row in evaluation_data.iterrows():
            # Extract keywords from the row
            keywords = []
            if 'kw' in row and pd.notna(row['kw']):
                kw_str = str(row['kw']).strip("[]")
                keywords = [k.strip().strip("'\"") for k in kw_str.split(",")]
            elif 'auto_keywords' in row:
                keywords = row['auto_keywords']
            
            # Extract contexts
            contexts = []
            if 'contexts' in row and pd.notna(row['contexts']):
                contexts = row['contexts'] if isinstance(row['contexts'], list) else [str(row['contexts'])]
            
            # Extract metadata
            metadata = {
                'source_file': row.get('source_file', 'unknown'),
                'question_type': row.get('question_type', 'unknown'),
                'batch_index': idx
            }
            
            # Evaluate single response
            eval_result = self.evaluate_rag_response(
                question=row['question'],
                rag_answer=row.get('rag_answer', ''),
                expected_answer=row['answer'],
                auto_keywords=keywords,
                contexts=contexts,
                metadata=metadata
            )
            
            results.append(eval_result)
            
            # Progress logging
            if (idx + 1) % 10 == 0:
                self.logger.info(f"Completed {idx + 1}/{len(evaluation_data)} evaluations")
        
        results_df = pd.DataFrame(results)
        self.logger.info(f"Batch evaluation completed. {len(results_df)} results generated.")
        
        return results_df
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation summary"""
        if not self.evaluation_results:
            return {'error': 'No evaluation results available'}
        
        results_df = pd.DataFrame(self.evaluation_results)
        
        # Overall statistics
        total_evaluations = len(results_df)
        overall_pass_rate = results_df['overall_pass'].mean()
        
        # Component pass rates
        contextual_pass_rate = results_df['contextual_keyword_pass'].mean()
        ragas_pass_rate = results_df['ragas_pass'].mean()
        semantic_pass_rate = results_df['semantic_pass'].mean()
        
        # Score distributions
        score_stats = {
            'contextual_scores': {
                'mean': results_df['contextual_total_score'].mean(),
                'std': results_df['contextual_total_score'].std(),
                'min': results_df['contextual_total_score'].min(),
                'max': results_df['contextual_total_score'].max()
            },
            'ragas_scores': {
                'mean': results_df['ragas_composite_score'].mean(),
                'std': results_df['ragas_composite_score'].std(),
                'min': results_df['ragas_composite_score'].min(),
                'max': results_df['ragas_composite_score'].max()
            },
            'semantic_scores': {
                'mean': results_df['semantic_similarity'].mean(),
                'std': results_df['semantic_similarity'].std(),
                'min': results_df['semantic_similarity'].min(),
                'max': results_df['semantic_similarity'].max()
            }
        }
        
        # Human feedback statistics
        feedback_needed = results_df['needs_human_feedback'].sum()
        feedback_ratio = feedback_needed / total_evaluations
        
        return {
            'evaluation_metadata': self.metadata,
            'overall_statistics': {
                'total_evaluations': total_evaluations,
                'overall_pass_rate': overall_pass_rate,
                'contextual_pass_rate': contextual_pass_rate,
                'ragas_pass_rate': ragas_pass_rate,
                'semantic_pass_rate': semantic_pass_rate
            },
            'score_statistics': score_stats,
            'human_feedback_statistics': {
                'feedback_needed_count': feedback_needed,
                'feedback_needed_ratio': feedback_ratio,
                'current_ragas_threshold': self.ragas_threshold,
                'threshold_adjustments': self.metadata['threshold_adjustments']
            },
            'detailed_results': results_df
        }
    
    def update_human_feedback(self, evaluation_id: int, feedback_score: float, 
                            feedback_comments: str = ""):
        """Update system with human feedback"""
        if evaluation_id < len(self.evaluation_results):
            # Update the specific evaluation result
            self.evaluation_results[evaluation_id]['human_feedback'] = {
                'score': feedback_score,
                'comments': feedback_comments,
                'timestamp': datetime.now().isoformat()
            }
            
            # Add to feedback history
            self.feedback_history.append(feedback_score)
            
            # Update threshold using your adaptive system
            if len(self.all_ragas_scores) > evaluation_id:
                old_threshold = self.ragas_threshold
                self.ragas_threshold = adaptive_exponential_smoothing(
                    self.all_ragas_scores[evaluation_id],
                    old_threshold,
                    alpha=0.1
                )
                
                if abs(self.ragas_threshold - old_threshold) > 0.01:
                    self.metadata['threshold_adjustments'] += 1
                    self.threshold_history.append(self.ragas_threshold)
                    self.logger.info(f"Threshold adjusted: {old_threshold:.3f} -> {self.ragas_threshold:.3f}")
            
            self.logger.info(f"Human feedback updated for evaluation {evaluation_id}")
        else:
            self.logger.error(f"Invalid evaluation_id: {evaluation_id}")
