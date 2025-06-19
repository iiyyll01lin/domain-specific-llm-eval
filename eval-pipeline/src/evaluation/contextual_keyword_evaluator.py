"""
Contextual Keyword Evaluator for RAG Evaluation Pipeline

Leverages existing contextual_keyword_gate.py functionality to evaluate
RAG responses using contextual keyword matching.
"""

import logging
import sys
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

# Add parent directories to path to import existing code
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from contextual_keyword_gate import weighted_keyword_score, get_contextual_segments
    CONTEXTUAL_GATE_AVAILABLE = True
except ImportError:
    logging.warning("Could not import contextual_keyword_gate functions")
    CONTEXTUAL_GATE_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer, util
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logging.warning("sentence_transformers not available")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    logging.warning("spaCy not available")
    SPACY_AVAILABLE = False

logger = logging.getLogger(__name__)

class ContextualKeywordEvaluator:
    """
    Evaluates RAG responses using contextual keyword matching.
    
    This class leverages your existing contextual keyword evaluation system
    to assess whether RAG responses contain the required keywords in context.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize contextual keyword evaluator.
        
        Args:
            config: Contextual keyword evaluation configuration
        """
        self.config = config
        
        # Load configuration
        self.weights = config.get('weights', {'mandatory': 0.8, 'optional': 0.2})
        self.threshold = config.get('threshold', 0.6)
        self.similarity_model_name = config.get('similarity_model', 'all-MiniLM-L6-v2')
        self.spacy_model_name = config.get('spacy_model', 'en_core_web_sm')
        
        # Initialize components
        self._initialize_models()
        
        logger.info("Contextual keyword evaluator initialized")
    
    def _initialize_models(self) -> None:
        """Initialize required models."""
        # Initialize sentence transformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.similarity_model = SentenceTransformer(self.similarity_model_name)
                logger.info(f"✅ Loaded sentence transformer: {self.similarity_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.similarity_model = None
        else:
            self.similarity_model = None
        
        # Initialize spaCy model
        if SPACY_AVAILABLE:
            try:
                self.nlp = spacy.load(self.spacy_model_name)
                logger.info(f"✅ Loaded spaCy model: {self.spacy_model_name}")
            except Exception as e:
                logger.warning(f"Failed to load spaCy model: {e}")
                self.nlp = None
        else:
            self.nlp = None
    
    def evaluate_response(self, rag_response: str, expected_keywords: List[str], 
                         optional_keywords: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate a single RAG response using contextual keyword matching.
        
        Args:
            rag_response: RAG system response text
            expected_keywords: List of mandatory keywords
            optional_keywords: List of optional keywords
            
        Returns:
            Dictionary with evaluation results
        """
        if optional_keywords is None:
            optional_keywords = []
        
        # Use existing contextual keyword evaluation if available
        if CONTEXTUAL_GATE_AVAILABLE and self.similarity_model:
            try:
                return self._evaluate_with_existing_method(
                    rag_response, expected_keywords, optional_keywords
                )
            except Exception as e:
                logger.warning(f"Existing method failed, using fallback: {e}")
        
        # Fallback evaluation method
        return self._evaluate_with_fallback(
            rag_response, expected_keywords, optional_keywords
        )
    
    def _evaluate_with_existing_method(self, response: str, mandatory_keywords: List[str], 
                                     optional_keywords: List[str]) -> Dict[str, Any]:
        """
        Evaluate using your existing contextual keyword gate method.
        
        Args:
            response: RAG response text
            mandatory_keywords: Mandatory keywords to check
            optional_keywords: Optional keywords to check
            
        Returns:
            Evaluation results dictionary
        """
        # Use your existing weighted_keyword_score function
        total_score, mandatory_score, optional_score, answer_segments = weighted_keyword_score(
            mandatory_keywords, response, self.weights, optional_keywords
        )
        
        # Determine if response passes threshold
        passes_threshold = total_score >= self.threshold
        
        return {
            'total_score': float(total_score),
            'mandatory_score': float(mandatory_score),
            'optional_score': float(optional_score),
            'passes_threshold': passes_threshold,
            'threshold': self.threshold,
            'answer_segments': answer_segments,
            'mandatory_keywords': mandatory_keywords,
            'optional_keywords': optional_keywords,
            'evaluation_method': 'existing_contextual_gate'
        }
    
    def _evaluate_with_fallback(self, response: str, mandatory_keywords: List[str], 
                              optional_keywords: List[str]) -> Dict[str, Any]:
        """
        Fallback evaluation method when existing system is not available.
        
        Args:
            response: RAG response text
            mandatory_keywords: Mandatory keywords to check
            optional_keywords: Optional keywords to check
            
        Returns:
            Evaluation results dictionary
        """
        # Simple keyword presence check
        response_lower = response.lower()
        
        # Check mandatory keywords
        mandatory_found = 0
        for keyword in mandatory_keywords:
            if keyword.lower() in response_lower:
                mandatory_found += 1
        
        mandatory_score = (mandatory_found / len(mandatory_keywords)) if mandatory_keywords else 0.0
        
        # Check optional keywords
        optional_found = 0
        if optional_keywords:
            for keyword in optional_keywords:
                if keyword.lower() in response_lower:
                    optional_found += 1
            optional_score = (optional_found / len(optional_keywords))
        else:
            optional_score = 0.0
        
        # Calculate weighted total score
        total_score = (
            mandatory_score * self.weights['mandatory'] + 
            optional_score * self.weights['optional']
        )
        
        passes_threshold = total_score >= self.threshold
        
        return {
            'total_score': float(total_score),
            'mandatory_score': float(mandatory_score),
            'optional_score': float(optional_score),
            'passes_threshold': passes_threshold,
            'threshold': self.threshold,
            'answer_segments': self._extract_segments_fallback(response),
            'mandatory_keywords': mandatory_keywords,
            'optional_keywords': optional_keywords,
            'evaluation_method': 'fallback_keyword_presence'
        }
    
    def _extract_segments_fallback(self, text: str) -> List[str]:
        """
        Fallback method to extract contextual segments.
        
        Args:
            text: Input text
            
        Returns:
            List of text segments
        """
        if self.nlp:
            # Use spaCy if available
            doc = self.nlp(text.lower())
            segments = []
            
            # Get noun phrases
            segments.extend([chunk.text for chunk in doc.noun_chunks])
            
            # Get named entities
            segments.extend([ent.text for ent in doc.ents])
            
            return segments
        else:
            # Very simple fallback - split by sentences
            import re
            sentences = re.split(r'[.!?]+', text)
            return [s.strip() for s in sentences if len(s.strip()) > 10]
    
    def evaluate_batch(self, evaluations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate a batch of RAG responses.
        
        Args:
            evaluations: List of evaluation dictionaries with keys:
                        - 'response': RAG response text
                        - 'mandatory_keywords': List of mandatory keywords
                        - 'optional_keywords': List of optional keywords (optional)
        
        Returns:
            Batch evaluation results
        """
        individual_results = []
        
        for eval_item in evaluations:
            response = eval_item['response']
            mandatory_keywords = eval_item['mandatory_keywords']
            optional_keywords = eval_item.get('optional_keywords', [])
            
            result = self.evaluate_response(response, mandatory_keywords, optional_keywords)
            individual_results.append(result)
        
        # Calculate aggregate statistics
        total_scores = [r['total_score'] for r in individual_results]
        mandatory_scores = [r['mandatory_score'] for r in individual_results]
        optional_scores = [r['optional_score'] for r in individual_results]
        pass_rates = [r['passes_threshold'] for r in individual_results]
        
        return {
            'individual_results': individual_results,
            'aggregate_stats': {
                'mean_total_score': float(np.mean(total_scores)),
                'mean_mandatory_score': float(np.mean(mandatory_scores)),
                'mean_optional_score': float(np.mean(optional_scores)),
                'pass_rate': float(np.mean(pass_rates)),
                'total_evaluations': len(individual_results),
                'passed_evaluations': sum(pass_rates),
                'failed_evaluations': len(individual_results) - sum(pass_rates)
            },
            'evaluation_config': {
                'threshold': self.threshold,
                'weights': self.weights,
                'similarity_model': self.similarity_model_name,
                'spacy_model': self.spacy_model_name
            }
        }
    
    def get_evaluation_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of evaluation results.
        
        Args:
            results: Evaluation results from evaluate_batch
            
        Returns:
            Formatted summary string
        """
        stats = results['aggregate_stats']
        
        summary = f"""
Contextual Keyword Evaluation Summary
=====================================
Total Evaluations: {stats['total_evaluations']}
Passed Threshold: {stats['passed_evaluations']} ({stats['pass_rate']:.1%})
Failed Threshold: {stats['failed_evaluations']}

Score Statistics:
- Mean Total Score: {stats['mean_total_score']:.3f}
- Mean Mandatory Score: {stats['mean_mandatory_score']:.3f}
- Mean Optional Score: {stats['mean_optional_score']:.3f}

Configuration:
- Threshold: {results['evaluation_config']['threshold']}
- Mandatory Weight: {results['evaluation_config']['weights']['mandatory']}
- Optional Weight: {results['evaluation_config']['weights']['optional']}
        """.strip()
        
        return summary
    
    def is_available(self) -> bool:
        """
        Check if the evaluator is properly initialized and available.
        
        Returns:
            True if evaluator is available, False otherwise
        """
        return (
            CONTEXTUAL_GATE_AVAILABLE and 
            self.similarity_model is not None
        ) or True  # Fallback is always available
