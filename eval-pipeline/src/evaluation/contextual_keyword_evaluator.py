"""
Contextual Keyword Evaluator for RAG Evaluation Pipeline

Leverages existing contextual_keyword_gate.py functionality to evaluate
RAG responses using contextual keyword matching.
"""

# Import fix applied
import sys
from pathlib import Path

# Add utils directory to Python path for local imports
current_file_dir = Path(__file__).parent
utils_dir = current_file_dir.parent / "utils"
if str(utils_dir) not in sys.path:
    sys.path.insert(0, str(utils_dir))


import logging
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import re

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

# Import offline model manager
try:
    from offline_model_manager import get_offline_manager
    OFFLINE_MANAGER_AVAILABLE = True
except ImportError:
    logging.warning("Offline model manager not available")
    OFFLINE_MANAGER_AVAILABLE = False

# Import NaN handling utilities
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "utils"))
from nan_handling import safe_mean, safe_std, safe_min_max, is_valid_score

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
        
        # Initialize models using offline manager
        self.similarity_model = None
        self.nlp = None
        
        if OFFLINE_MANAGER_AVAILABLE:
            # Use offline model manager
            offline_manager = get_offline_manager(config)
            
            # Load sentence transformer model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    self.similarity_model = offline_manager.load_sentence_transformer(
                        self.similarity_model_name
                    )
                    if self.similarity_model:
                        logger.info(f"✅ Loaded sentence transformer offline: {self.similarity_model_name}")
                    else:
                        logger.warning(f"❌ Failed to load sentence transformer offline: {self.similarity_model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load sentence transformer offline: {e}")
            
            # Load spaCy model
            if SPACY_AVAILABLE:
                try:
                    self.nlp = offline_manager.load_spacy_model(self.spacy_model_name)
                    if self.nlp:
                        logger.info(f"✅ Loaded spaCy model offline: {self.spacy_model_name}")
                    else:
                        logger.warning(f"❌ Failed to load spaCy model offline: {self.spacy_model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load spaCy model offline: {e}")
        
        else:
            # Fallback to original loading method
            logger.warning("⚠️ Offline model manager not available, using fallback loading")
            
            # Initialize sentence transformer model
            if SENTENCE_TRANSFORMERS_AVAILABLE:
                try:
                    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                    self.similarity_model = SentenceTransformer(self.similarity_model_name, device=device)
                    logger.info(f"✅ Loaded sentence transformer: {self.similarity_model_name} on {device}")
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
        
        # Initialize components
        self._initialize_models()
        
        logger.info("Contextual keyword evaluator initialized")
    
    def _initialize_models(self) -> None:
        """Initialize required models."""
        # Initialize sentence transformer model
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                self.similarity_model = SentenceTransformer(self.similarity_model_name, device=device)
                logger.info(f"✅ Loaded sentence transformer: {self.similarity_model_name} on {device}")
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
            'passes_threshold': bool(passes_threshold),  # Ensure it's a Python bool
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
        Enhanced for Chinese keyword matching.
        
        Args:
            response: RAG response text
            mandatory_keywords: Mandatory keywords to check
            optional_keywords: Optional keywords to check
            
        Returns:
            Evaluation results dictionary
        """
        # Enhanced keyword matching for Chinese text
        response_lower = response.lower()
        
        # Check mandatory keywords with enhanced Chinese matching
        mandatory_found = 0
        mandatory_matched = []
        mandatory_missing = []
        
        if mandatory_keywords:  # Ensure list is not empty
            for keyword in mandatory_keywords:
                if keyword and self._is_keyword_present(keyword, response_lower):
                    mandatory_found += 1
                    mandatory_matched.append(keyword)
                else:
                    mandatory_missing.append(keyword)
            mandatory_score = (mandatory_found / len(mandatory_keywords))
        else:
            mandatory_score = 1.0  # If no mandatory keywords, consider it a pass
        
        # Check optional keywords with enhanced Chinese matching
        optional_found = 0
        optional_matched = []
        optional_missing = []
        
        if optional_keywords:  # Ensure list is not empty
            for keyword in optional_keywords:
                if keyword and self._is_keyword_present(keyword, response_lower):
                    optional_found += 1
                    optional_matched.append(keyword)
                else:
                    optional_missing.append(keyword)
            optional_score = (optional_found / len(optional_keywords))
        else:
            optional_score = 1.0  # If no optional keywords, consider it a pass
        
        # Calculate weighted total score with safety checks
        mandatory_weight = self.weights.get('mandatory', 0.8)
        optional_weight = self.weights.get('optional', 0.2)
        
        total_score = (
            mandatory_score * mandatory_weight + 
            optional_score * optional_weight
        )
        
        passes_threshold = total_score >= self.threshold
        
        return {
            'total_score': float(total_score),
            'mandatory_score': float(mandatory_score),
            'optional_score': float(optional_score),
            'passes_threshold': bool(passes_threshold),  # Ensure it's a Python bool
            'threshold': self.threshold,
            'answer_segments': self._extract_segments_fallback(response),
            'mandatory_keywords': mandatory_keywords or [],
            'optional_keywords': optional_keywords or [],
            'matched_mandatory': mandatory_matched,
            'missing_mandatory': mandatory_missing,
            'matched_optional': optional_matched,
            'missing_optional': optional_missing,
            'evaluation_method': 'enhanced_fallback_keyword_presence'
        }
    
    def _is_keyword_present(self, keyword: str, response_text: str) -> bool:
        """
        Enhanced keyword presence check for Chinese and English text.
        
        Args:
            keyword: Keyword to search for
            response_text: Response text (already lowercase)
            
        Returns:
            True if keyword is present in response
        """
        keyword_lower = keyword.lower().strip()
        if not keyword_lower:
            return False
        
        # Direct substring match (works for both English and Chinese)
        if keyword_lower in response_text:
            return True
        
        # For Chinese text, try character-by-character matching
        # This helps with compound words that might be split differently
        if self._contains_chinese(keyword_lower):
            # Split keyword into individual characters and check if most are present
            chars = list(keyword_lower)
            char_matches = sum(1 for char in chars if char in response_text)
            # Consider it a match if at least 70% of characters are present
            if len(chars) > 0 and char_matches / len(chars) >= 0.7:
                return True
        
        # For English keywords, try word boundary matching
        elif keyword_lower.replace(' ', '').isalpha():
            # Check for word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword_lower) + r'\b'
            if re.search(pattern, response_text):
                return True
        
        return False
    
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        for char in text:
            if '\u4e00' <= char <= '\u9fff':  # Chinese Unicode range
                return True
        return False
    
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
            
            # Ensure all result values are JSON-serializable
            json_safe_result = {}
            for key, value in result.items():
                if isinstance(value, (np.bool_, bool)):
                    json_safe_result[key] = bool(value)
                elif isinstance(value, (np.integer, np.floating)):
                    json_safe_result[key] = float(value)
                elif hasattr(value, 'item'):  # numpy scalar
                    json_safe_result[key] = value.item()
                else:
                    json_safe_result[key] = value
            
            individual_results.append(json_safe_result)
        
        # Calculate aggregate statistics
        total_scores = [r['total_score'] for r in individual_results]
        mandatory_scores = [r['mandatory_score'] for r in individual_results]
        optional_scores = [r['optional_score'] for r in individual_results]
        pass_rates = [r['passes_threshold'] for r in individual_results]
        
        # Handle empty lists safely
        if not individual_results:
            return {
                'individual_results': [],
                'aggregate_stats': {
                    'mean_total_score': 0.0,
                    'mean_mandatory_score': 0.0,
                    'mean_optional_score': 0.0,
                    'pass_rate': 0.0,
                    'total_evaluations': 0,
                    'passed_evaluations': 0,
                    'failed_evaluations': 0
                },
                'evaluation_config': {
                    'threshold': self.threshold,
                    'weights': self.weights,
                    'similarity_model': self.similarity_model_name,
                    'spacy_model': self.spacy_model_name
                }
            }
        
        return {
            'individual_results': individual_results,
            'aggregate_stats': {
                'mean_total_score': float(safe_mean(total_scores)) if total_scores else 0.0,
                'mean_mandatory_score': float(safe_mean(mandatory_scores)) if mandatory_scores else 0.0,
                'mean_optional_score': float(safe_mean(optional_scores)) if optional_scores else 0.0,
                'pass_rate': float(safe_mean(pass_rates)) if pass_rates else 0.0,
                'total_evaluations': len(individual_results),
                'passed_evaluations': int(sum(pass_rates)),
                'failed_evaluations': len(individual_results) - int(sum(pass_rates))
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
    
    def evaluate(self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate RAG responses using contextual keyword matching.
        
        Args:
            testset: Test dataset containing questions and auto_keywords
            rag_responses: List of RAG system responses
            
        Returns:
            Evaluation results
        """
        logger.info("🔍 Starting contextual keyword evaluation...")
        
        evaluations = []
        questions = testset.get('questions', [])
        auto_keywords = testset.get('auto_keywords', [])
        
        # Ensure we have matching data lengths
        min_length = min(len(questions), len(rag_responses))
        if auto_keywords:
            min_length = min(min_length, len(auto_keywords))
        
        if min_length == 0:
            logger.warning("⚠️ No valid evaluations to process")
            return {
                'error': 'No valid evaluations to process',
                'pass_count': 0,
                'fail_count': 0,
                'average_score': 0.0,
                'total_evaluations': 0
            }
        
        # Prepare evaluation data
        for i in range(min_length):
            response = rag_responses[i]
            
            # Handle different response formats
            if isinstance(response, dict):
                answer_text = response.get('answer', str(response))
            else:
                answer_text = str(response)
            
            # Extract keywords for this question
            keywords = []
            if i < len(auto_keywords) and auto_keywords[i]:
                if isinstance(auto_keywords[i], str):
                    # Handle both comma and Chinese comma separators
                    raw_keywords = auto_keywords[i]
                    # Split by both English and Chinese commas
                    # Parse keywords from string or list
                    if isinstance(raw_keywords, str):
                        keywords = re.split(r'[,，、]', raw_keywords)
                        keywords = [kw.strip() for kw in keywords if kw.strip()]
                    else:
                        keywords = [str(kw).strip() for kw in raw_keywords if str(kw).strip()]

                    logger.debug(f"Extracted {len(keywords)} keywords from: {raw_keywords[:100]}...")
                elif isinstance(auto_keywords[i], list):
                    keywords = auto_keywords[i]
            
            # Only evaluate if we have keywords
            if keywords:
                evaluations.append({
                    'response': answer_text,
                    'mandatory_keywords': keywords,
                    'optional_keywords': [],
                    'question': questions[i] if i < len(questions) else '',
                    'response_index': i
                })
        
        if not evaluations:
            logger.warning("⚠️ No evaluations with keywords to process")
            return {
                'error': 'No evaluations with keywords to process',
                'pass_count': 0,
                'fail_count': 0,
                'average_score': 0.0,
                'total_evaluations': 0
            }
        
        # Run batch evaluation
        batch_results = self.evaluate_batch(evaluations)
        
        # Format results for compatibility
        aggregate_stats = batch_results.get('aggregate_stats', {})
        
        # Ensure all values are JSON-serializable
        def safe_convert(value, default=0):
            """Convert numpy/pandas types to Python native types"""
            try:
                if hasattr(value, 'item'):  # numpy scalar
                    return value.item()
                elif isinstance(value, (np.integer, np.floating)):
                    return float(value)
                elif isinstance(value, np.bool_):
                    return bool(value)
                else:
                    return value
            except:
                return default
        
        return {
            'pass_count': safe_convert(aggregate_stats.get('passed_evaluations', 0)),
            'fail_count': safe_convert(aggregate_stats.get('failed_evaluations', 0)),
            'total_evaluations': safe_convert(aggregate_stats.get('total_evaluations', 0)),
            'average_score': safe_convert(aggregate_stats.get('mean_total_score', 0.0)),
            'pass_rate': safe_convert(aggregate_stats.get('pass_rate', 0.0)),
            'individual_results': batch_results.get('individual_results', []),
            'evaluation_config': batch_results.get('evaluation_config', {}),
            'method': 'contextual_keyword_matching',
            'available': True
        }
    
    def evaluate_testset(self, testset_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate testset data directly.
        
        Args:
            testset_data: Testset data containing qa_pairs
            
        Returns:
            List of evaluation results
        """
        logger.info("🔄 Running contextual keyword evaluation on testset...")
        
        try:
            qa_pairs = testset_data.get('qa_pairs', [])
            if not qa_pairs:
                logger.warning("No QA pairs found in testset data")
                return []
            
            results = []
            for qa_pair in qa_pairs:
                # Extract expected keywords (if available)
                expected_keywords = qa_pair.get('expected_keywords', [])
                mandatory_keywords = qa_pair.get('mandatory_keywords', [])
                optional_keywords = qa_pair.get('optional_keywords', [])
                
                # Mock RAG response evaluation (this would normally come from RAG system)
                rag_response = qa_pair.get('rag_answer', qa_pair.get('reference', ''))
                
                if expected_keywords:
                    eval_result = self.evaluate_response(
                        rag_response, 
                        expected_keywords, 
                        mandatory_keywords, 
                        optional_keywords
                    )
                    results.append({
                        **qa_pair,
                        'contextual_keyword_score': eval_result.get('total_score', 0.0),
                        'passed': eval_result.get('passes_threshold', False),
                        'keyword_evaluation': eval_result
                    })
                else:
                    # Skip if no keywords to evaluate
                    results.append({
                        **qa_pair,
                        'contextual_keyword_score': 0.0,
                        'passed': False,
                        'keyword_evaluation': {'message': 'No keywords provided for evaluation'}
                    })
            
            logger.info(f"✅ Contextual keyword evaluation completed for {len(results)} items")
            return results
            
        except Exception as e:
            logger.error(f"❌ Contextual keyword evaluation failed: {e}")
            return []
    
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
