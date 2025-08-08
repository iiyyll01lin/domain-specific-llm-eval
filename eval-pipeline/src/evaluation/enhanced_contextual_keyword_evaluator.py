#!/usr/bin/env python3
"""
Enhanced Contextual Keyword Evaluator with Improved Matching

This module implements your contextual method with enhancements:
1. Semantic similarity using sentence transformers
2. Fuzzy matching for technical terminology
3. Multilingual support (English/Chinese)
4. Domain-specific keyword extraction
5. Adaptive thresholds
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
import pandas as pd
import json
import re
import time
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime

# Add parent directory to access your contextual_keyword_gate.py
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

# Import your original contextual method
try:
    from contextual_keyword_gate import weighted_keyword_score, get_contextual_segments
    CONTEXTUAL_GATE_AVAILABLE = True
except ImportError:
    logging.warning("Could not import contextual_keyword_gate functions")
    CONTEXTUAL_GATE_AVAILABLE = False

# Import enhanced libraries
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

try:
    from fuzzywuzzy import fuzz
    FUZZYWUZZY_AVAILABLE = True
except ImportError:
    logging.warning("fuzzywuzzy not available")
    FUZZYWUZZY_AVAILABLE = False

try:
    import keybert
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    logging.warning("KeyBERT not available")
    KEYBERT_AVAILABLE = False

# Import NaN handling utilities
import math
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent / "utils"))

from nan_handling import safe_mean, safe_std, safe_min_max, is_valid_score

logger = logging.getLogger(__name__)

class EnhancedContextualKeywordEvaluator:
    """
    Enhanced contextual keyword evaluator that combines:
    1. Your original semantic similarity approach (primary)
    2. Fuzzy matching for technical terms (secondary)  
    3. Multilingual support (English/Chinese)
    4. Domain-specific keyword extraction
    5. Adaptive thresholds based on language/domain
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize enhanced evaluator.
        
        Args:
            config: Configuration dictionary with evaluation settings
        """
        self.config = config
        self.rag_config = config.get('rag_system', {})
        self.keyword_config = config.get('evaluation', {}).get('contextual_keywords', {})
        
        # Core settings
        self.similarity_threshold = self.keyword_config.get('similarity_threshold', 0.7)
        self.fuzzy_threshold = self.keyword_config.get('fuzzy_threshold', 80)
        self.weights = self.keyword_config.get('weights', {'mandatory': 0.8, 'optional': 0.2})
        
        # Language models
        self.sentence_model_name = self.keyword_config.get('sentence_model', 'all-MiniLM-L6-v2')
        self.spacy_model_en = self.keyword_config.get('spacy_model_en', 'en_core_web_sm')
        self.spacy_model_zh = self.keyword_config.get('spacy_model_zh', 'zh_core_web_sm')
        
        # Initialize models
        self._initialize_models()
        
        logger.info(f"EnhancedContextualKeywordEvaluator initialized")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
        logger.info(f"Available enhancements: SentenceTransformers={SENTENCE_TRANSFORMERS_AVAILABLE}, "
                   f"spaCy={SPACY_AVAILABLE}, FuzzyWuzzy={FUZZYWUZZY_AVAILABLE}, KeyBERT={KEYBERT_AVAILABLE}")
    
    def _initialize_models(self):
        """Initialize all required models with fallbacks."""
        # Initialize sentence transformer
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
                self.sentence_model = SentenceTransformer(self.sentence_model_name, device=device)
                logger.info(f"✅ Sentence transformer loaded: {self.sentence_model_name} on {device}")
            except Exception as e:
                logger.warning(f"Failed to load sentence transformer: {e}")
                self.sentence_model = None
        else:
            self.sentence_model = None
        
        # Initialize spaCy models using offline model manager for fallbacks
        self.nlp_en = None
        self.nlp_zh = None
        
        if SPACY_AVAILABLE:
            try:
                # Try to use offline model manager if available
                try:
                    from offline_model_manager import get_offline_manager
                    offline_manager = get_offline_manager(self.config)
                    self.nlp_en = offline_manager.load_spacy_model(self.spacy_model_en)
                    if self.nlp_en:
                        logger.info(f"✅ English spaCy model loaded via offline manager: {self.spacy_model_en}")
                    else:
                        raise Exception("Offline manager failed")
                except:
                    # Fallback to direct loading
                    self.nlp_en = spacy.load(self.spacy_model_en)
                    logger.info(f"✅ English spaCy model loaded directly: {self.spacy_model_en}")
            except Exception as e:
                logger.warning(f"Failed to load English spaCy model: {e}")
            
            try:
                # Try to use offline model manager if available
                try:
                    from offline_model_manager import get_offline_manager
                    offline_manager = get_offline_manager(self.config)
                    self.nlp_zh = offline_manager.load_spacy_model(self.spacy_model_zh)
                    if self.nlp_zh:
                        logger.info(f"✅ Chinese spaCy model loaded via offline manager: {self.spacy_model_zh}")
                    else:
                        raise Exception("Offline manager failed")
                except:
                    # Fallback to direct loading
                    self.nlp_zh = spacy.load(self.spacy_model_zh)
                    logger.info(f"✅ Chinese spaCy model loaded directly: {self.spacy_model_zh}")
            except Exception as e:
                logger.warning(f"Failed to load Chinese spaCy model: {e}")
        
        # Initialize KeyBERT
        if KEYBERT_AVAILABLE:
            try:
                self.keybert_model = KeyBERT(model=self.sentence_model_name if self.sentence_model else 'all-MiniLM-L6-v2')
                logger.info("✅ KeyBERT model loaded")
            except Exception as e:
                logger.warning(f"Failed to load KeyBERT: {e}")
                self.keybert_model = None
        else:
            self.keybert_model = None
    
    def _detect_language(self, text: str) -> str:
        """
        Detect if text is primarily English or Chinese.
        
        Args:
            text: Text to analyze
            
        Returns:
            'chinese' or 'english'
        """
        if not text:
            return 'english'
        
        # Count Chinese characters (CJK Unicode ranges)
        chinese_chars = 0
        total_chars = 0
        
        for char in text:
            if char.strip():  # Ignore whitespace
                total_chars += 1
                # Check for CJK characters
                if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                    chinese_chars += 1
        
        if total_chars == 0:
            return 'english'
        
        # If more than 30% Chinese characters, consider it Chinese
        chinese_ratio = chinese_chars / total_chars
        return 'chinese' if chinese_ratio > 0.3 else 'english'
    
    def _get_adaptive_threshold(self, language: str, domain: str = "technical") -> float:
        """
        Get adaptive threshold based on language and domain.
        
        Args:
            language: Detected language
            domain: Domain type (technical, general, etc.)
            
        Returns:
            Adjusted threshold value
        """
        base_threshold = self.similarity_threshold
        
        # Language adjustments
        if language == 'chinese':
            # Chinese often needs slightly lower threshold due to character-based matching
            language_adjustment = -0.05
        else:
            language_adjustment = 0.0
        
        # Domain adjustments
        if domain == "technical":
            # Technical terms might need slightly lower threshold for flexibility
            domain_adjustment = -0.05
        else:
            domain_adjustment = 0.0
        
        adaptive_threshold = max(0.1, base_threshold + language_adjustment + domain_adjustment)
        return adaptive_threshold
    
    def enhanced_contextual_evaluation(self, expected_keywords: List[str], rag_answer: str, 
                                     language: str = "auto") -> Dict[str, Any]:
        """
        Enhanced contextual evaluation combining multiple approaches.
        
        Args:
            expected_keywords: Keywords expected to be found
            rag_answer: RAG system response
            language: Language override or "auto" for detection
            
        Returns:
            Comprehensive evaluation results
        """
        start_time = time.time()
        
        # Detect language if auto
        if language == "auto":
            detected_language = self._detect_language(rag_answer)
        else:
            detected_language = language
        
        # Get adaptive threshold
        adaptive_threshold = self._get_adaptive_threshold(detected_language, "technical")
        
        results = {
            'language': detected_language,
            'adaptive_threshold': adaptive_threshold,
            'evaluation_methods': {},
            'combined_results': {},
            'performance': {}
        }
        
        # Method 1: Your original contextual method (primary)
        if CONTEXTUAL_GATE_AVAILABLE:
            contextual_score = self._evaluate_with_original_method(
                expected_keywords, rag_answer, detected_language
            )
            results['evaluation_methods']['original_contextual'] = contextual_score
        
        # Method 2: Enhanced semantic similarity (if original fails)
        if self.sentence_model:
            semantic_score = self._evaluate_with_enhanced_semantic(
                expected_keywords, rag_answer, detected_language
            )
            results['evaluation_methods']['enhanced_semantic'] = semantic_score
        
        # Method 3: Fuzzy matching for technical terms
        if FUZZYWUZZY_AVAILABLE:
            fuzzy_score = self._evaluate_with_fuzzy_matching(
                expected_keywords, rag_answer
            )
            results['evaluation_methods']['fuzzy_matching'] = fuzzy_score
        
        # Method 4: Domain-specific keyword extraction overlap
        if self.keybert_model:
            extraction_score = self._evaluate_with_keyword_extraction(
                expected_keywords, rag_answer, detected_language
            )
            results['evaluation_methods']['keyword_extraction'] = extraction_score
        
        # Combine all methods
        combined_results = self._combine_evaluation_methods(
            results['evaluation_methods'], adaptive_threshold
        )
        results['combined_results'] = combined_results
        
        # Performance metrics
        evaluation_time = time.time() - start_time
        results['performance'] = {
            'evaluation_time': evaluation_time,
            'methods_used': len(results['evaluation_methods']),
            'timestamp': datetime.now().isoformat()
        }
        
        return results
    
    def _evaluate_with_original_method(self, expected_keywords: List[str], 
                                     rag_answer: str, language: str) -> Dict[str, Any]:
        """
        Evaluate using your original contextual method.
        
        Args:
            expected_keywords: Keywords to find
            rag_answer: Response text
            language: Detected language
            
        Returns:
            Original method results
        """
        try:
            # Use your existing weighted_keyword_score function
            total_score, mandatory_score, optional_score, answer_segments = weighted_keyword_score(
                expected_keywords, rag_answer, self.weights, []
            )
            
            # Get contextual segments using your method
            if language == 'chinese' and self.nlp_zh:
                segments = self._get_contextual_segments_enhanced(rag_answer, 'chinese')
            elif self.nlp_en:
                segments = self._get_contextual_segments_enhanced(rag_answer, 'english')
            else:
                segments = get_contextual_segments(rag_answer) if CONTEXTUAL_GATE_AVAILABLE else []
            
            return {
                'total_score': float(total_score),
                'mandatory_score': float(mandatory_score),
                'optional_score': float(optional_score),
                'answer_segments': answer_segments,
                'contextual_segments': segments,
                'method': 'original_contextual',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Original contextual method failed: {e}")
            return {
                'total_score': 0.0,
                'error': str(e),
                'method': 'original_contextual',
                'success': False
            }
    
    def _get_contextual_segments_enhanced(self, text: str, language: str) -> List[str]:
        """
        Enhanced contextual segmentation with language awareness.
        Fixed to handle Chinese language noun_chunks issue.
        
        Args:
            text: Input text
            language: Detected language ('english' or 'chinese')
            
        Returns:
            List of contextual segments
        """
        if not SPACY_AVAILABLE:
            # Fallback to simple segmentation
            return self._simple_text_segmentation(text)
        
        # Use appropriate spaCy model
        if language == "chinese" and self.nlp_zh:
            nlp = self.nlp_zh
        elif language == "english" and self.nlp_en:
            nlp = self.nlp_en
        elif self.nlp_en:
            # Default to English if no Chinese model
            nlp = self.nlp_en
        else:
            # Fallback to simple segmentation
            return self._simple_text_segmentation(text)
        
        doc = nlp(text.lower())
        segments = []
        
        # Get named entities (works for both languages)
        segments.extend([ent.text for ent in doc.ents])
        
        # Language-specific processing
        if language == "chinese":
            # For Chinese, use token-based segmentation instead of noun_chunks
            segments.extend([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"] and len(token.text) > 1])
            
            # Add sentence-level segments for Chinese
            segments.extend([sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 5])
        else:
            # For English, use noun_chunks safely
            try:
                segments.extend([chunk.text for chunk in doc.noun_chunks])
            except Exception as e:
                logger.warning(f"noun_chunks failed: {e}, using fallback")
                # Fallback if noun_chunks fails
                segments.extend([token.text for token in doc if token.pos_ in ["NOUN", "PROPN"]])
        
        # Get dependency-based phrases (works for both languages)
        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ["ROOT", "xcomp", "ccomp", "advcl"]:
                    phrase = " ".join([t.text for t in token.subtree])
                    segments.append(phrase)
        
        # Remove duplicates and very short segments
        segments = list(set([seg.strip() for seg in segments if len(seg.split()) > 1]))
        
        return segments

    def _simple_text_segmentation(self, text: str) -> List[str]:
        """Simple text segmentation fallback when spaCy fails"""
        import re
        
        # Split by punctuation and extract meaningful phrases
        sentences = re.split(r'[.!?;，。！？；：]+', text)
        segments = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 5:
                segments.append(sentence)
                
                # Extract potential noun phrases (simple heuristic)
                words = sentence.split()
                for i, word in enumerate(words):
                    if word.lower() in ['the', 'a', 'an'] and i + 1 < len(words):
                        # Extract noun phrase starting with article
                        phrase_words = [words[i]]
                        for j in range(i + 1, min(i + 4, len(words))):
                            if words[j].isalpha():
                                phrase_words.append(words[j])
                            else:
                                break
                        if len(phrase_words) > 1:
                            segments.append(' '.join(phrase_words))
        
        return segments
        
        # Remove duplicates and very short segments
        segments = list(set([seg.strip() for seg in segments if len(seg.split()) > 1]))
        
        return segments
    
    def _evaluate_with_enhanced_semantic(self, expected_keywords: List[str], 
                                       rag_answer: str, language: str) -> Dict[str, Any]:
        """
        Enhanced semantic similarity evaluation.
        
        Args:
            expected_keywords: Keywords to find
            rag_answer: Response text
            language: Detected language
            
        Returns:
            Enhanced semantic results
        """
        try:
            if not self.sentence_model:
                return {'error': 'Sentence transformer not available', 'success': False}
            
            # Get enhanced segments
            segments = self._get_contextual_segments_enhanced(rag_answer, language)
            if not segments:
                segments = [rag_answer]  # Fallback to full text
            
            # Encode all text
            keyword_embeddings = self.sentence_model.encode(expected_keywords, convert_to_tensor=True)
            segment_embeddings = self.sentence_model.encode(segments, convert_to_tensor=True)
            
            # Calculate similarities
            keyword_scores = []
            detailed_matches = []
            
            for i, keyword in enumerate(expected_keywords):
                similarities = util.pytorch_cos_sim(keyword_embeddings[i], segment_embeddings)[0]
                max_similarity = float(similarities.max())
                best_segment_idx = int(similarities.argmax())
                
                keyword_scores.append(max_similarity)
                detailed_matches.append({
                    'keyword': keyword,
                    'best_match_segment': segments[best_segment_idx],
                    'similarity': max_similarity
                })
            
            # Calculate aggregate scores
            mean_similarity = safe_mean(keyword_scores)
            min_similarity = np.min(keyword_scores)
            
            return {
                'mean_similarity': float(mean_similarity),
                'min_similarity': float(min_similarity),
                'keyword_scores': keyword_scores,
                'detailed_matches': detailed_matches,
                'segments_used': segments,
                'method': 'enhanced_semantic',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Enhanced semantic evaluation failed: {e}")
            return {
                'error': str(e),
                'method': 'enhanced_semantic',
                'success': False
            }
    
    def _evaluate_with_fuzzy_matching(self, expected_keywords: List[str], 
                                    rag_answer: str) -> Dict[str, Any]:
        """
        Fuzzy matching for technical terminology.
        
        Args:
            expected_keywords: Keywords to find
            rag_answer: Response text
            
        Returns:
            Fuzzy matching results
        """
        try:
            if not FUZZYWUZZY_AVAILABLE:
                return {'error': 'FuzzyWuzzy not available', 'success': False}
            
            # Extract potential keywords from answer
            # Simple approach: split by common delimiters
            answer_terms = re.findall(r'\b\w+\b', rag_answer.lower())
            answer_phrases = re.findall(r'[\w\s]{3,20}', rag_answer.lower())
            candidate_terms = list(set(answer_terms + answer_phrases))
            
            fuzzy_matches = []
            keyword_scores = []
            
            for keyword in expected_keywords:
                best_score = 0
                best_match = ""
                
                # Check against all candidate terms
                for candidate in candidate_terms:
                    score = fuzz.ratio(keyword.lower(), candidate.strip())
                    if score > best_score:
                        best_score = score
                        best_match = candidate.strip()
                
                # Normalize score to 0-1 range
                normalized_score = best_score / 100.0
                keyword_scores.append(normalized_score)
                
                fuzzy_matches.append({
                    'keyword': keyword,
                    'best_match': best_match,
                    'fuzzy_score': best_score,
                    'normalized_score': normalized_score,
                    'passes_threshold': best_score >= self.fuzzy_threshold
                })
            
            # Calculate aggregate metrics
            mean_fuzzy_score = safe_mean(keyword_scores)
            matches_above_threshold = sum(1 for match in fuzzy_matches if match['passes_threshold'])
            match_rate = matches_above_threshold / len(expected_keywords) if expected_keywords else 0
            
            return {
                'mean_fuzzy_score': float(mean_fuzzy_score),
                'match_rate': float(match_rate),
                'matches_above_threshold': matches_above_threshold,
                'total_keywords': len(expected_keywords),
                'detailed_matches': fuzzy_matches,
                'method': 'fuzzy_matching',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Fuzzy matching evaluation failed: {e}")
            return {
                'error': str(e),
                'method': 'fuzzy_matching',
                'success': False
            }
    
    def _evaluate_with_keyword_extraction(self, expected_keywords: List[str], 
                                        rag_answer: str, language: str) -> Dict[str, Any]:
        """
        Domain-specific keyword extraction and overlap analysis.
        
        Args:
            expected_keywords: Keywords to find
            rag_answer: Response text
            language: Detected language
            
        Returns:
            Keyword extraction results
        """
        try:
            if not self.keybert_model:
                return {'error': 'KeyBERT not available', 'success': False}
            
            # Extract keywords from response
            extracted_keywords = self.keybert_model.extract_keywords(
                rag_answer,
                keyphrase_ngram_range=(1, 3),  # Capture 1-3 word phrases
                stop_words='english' if language == 'english' else None,
                use_mmr=True,  # Maximal Marginal Relevance
                diversity=0.3  # Lower diversity for technical terms
            )
            
            # Convert to list of keywords
            extracted_keyword_list = [kw[0] for kw in extracted_keywords]
            
            # Calculate overlap with expected keywords
            exact_matches = []
            semantic_matches = []
            
            for expected_kw in expected_keywords:
                # Check for exact matches (case insensitive)
                exact_match = any(expected_kw.lower() in extracted_kw.lower() or 
                                extracted_kw.lower() in expected_kw.lower() 
                                for extracted_kw in extracted_keyword_list)
                
                if exact_match:
                    exact_matches.append(expected_kw)
                elif self.sentence_model:
                    # Check semantic similarity
                    expected_emb = self.sentence_model.encode([expected_kw])
                    extracted_embs = self.sentence_model.encode(extracted_keyword_list)
                    
                    similarities = util.pytorch_cos_sim(expected_emb, extracted_embs)[0]
                    max_sim = float(similarities.max()) if len(similarities) > 0 else 0.0
                    
                    if max_sim > 0.7:  # High semantic similarity threshold
                        best_match_idx = int(similarities.argmax())
                        semantic_matches.append({
                            'expected': expected_kw,
                            'matched': extracted_keyword_list[best_match_idx],
                            'similarity': max_sim
                        })
            
            # Calculate metrics
            exact_match_rate = len(exact_matches) / len(expected_keywords) if expected_keywords else 0
            semantic_match_rate = len(semantic_matches) / len(expected_keywords) if expected_keywords else 0
            total_match_rate = min(1.0, exact_match_rate + semantic_match_rate)
            
            return {
                'extracted_keywords': extracted_keyword_list,
                'exact_matches': exact_matches,
                'semantic_matches': semantic_matches,
                'exact_match_rate': float(exact_match_rate),
                'semantic_match_rate': float(semantic_match_rate),
                'total_match_rate': float(total_match_rate),
                'method': 'keyword_extraction',
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Keyword extraction evaluation failed: {e}")
            return {
                'error': str(e),
                'method': 'keyword_extraction',
                'success': False
            }
    
    def _combine_evaluation_methods(self, method_results: Dict[str, Any], 
                                  adaptive_threshold: float) -> Dict[str, Any]:
        """
        Combine results from all evaluation methods.
        
        Args:
            method_results: Results from all methods
            adaptive_threshold: Threshold adjusted for language/domain
            
        Returns:
            Combined evaluation results
        """
        # Method weights (can be configured)
        method_weights = {
            'original_contextual': 0.4,    # Your method is primary
            'enhanced_semantic': 0.3,      # Strong secondary
            'fuzzy_matching': 0.2,         # Technical term fallback
            'keyword_extraction': 0.1      # Overlap analysis
        }
        
        # Extract scores from each method
        method_scores = {}
        total_weight = 0
        
        for method_name, results in method_results.items():
            if not results.get('success', False):
                continue
            
            weight = method_weights.get(method_name, 0)
            total_weight += weight
            
            # Extract primary score from each method
            if method_name == 'original_contextual':
                score = results.get('total_score', 0)
            elif method_name == 'enhanced_semantic':
                score = results.get('mean_similarity', 0)
            elif method_name == 'fuzzy_matching':
                score = results.get('mean_fuzzy_score', 0)
            elif method_name == 'keyword_extraction':
                score = results.get('total_match_rate', 0)
            else:
                score = 0
            
            method_scores[method_name] = score * weight
        
        # Calculate weighted average
        if total_weight > 0:
            final_score = sum(method_scores.values()) / total_weight
        else:
            final_score = 0.0
        
        # Determine pass/fail
        passes_threshold = final_score >= adaptive_threshold
        
        # Get best performing method
        individual_scores = {}
        for method_name, results in method_results.items():
            if results.get('success', False):
                if method_name == 'original_contextual':
                    individual_scores[method_name] = results.get('total_score', 0)
                elif method_name == 'enhanced_semantic':
                    individual_scores[method_name] = results.get('mean_similarity', 0)
                elif method_name == 'fuzzy_matching':
                    individual_scores[method_name] = results.get('mean_fuzzy_score', 0)
                elif method_name == 'keyword_extraction':
                    individual_scores[method_name] = results.get('total_match_rate', 0)
        
        best_method = max(individual_scores.items(), key=lambda x: x[1]) if individual_scores else ('none', 0)
        
        return {
            'final_score': float(final_score),
            'passes_threshold': bool(passes_threshold),
            'threshold_used': float(adaptive_threshold),
            'method_scores': method_scores,
            'individual_scores': individual_scores,
            'best_method': best_method[0],
            'best_method_score': float(best_method[1]),
            'total_weight': float(total_weight),
            'methods_contributing': len(method_scores)
        }
    
    def evaluate_single_question(self, question: str, expected_keywords: List[str], 
                               rag_answer: str, question_index: int = 0) -> Dict[str, Any]:
        """
        Evaluate a single question using enhanced contextual method.
        
        Args:
            question: The question asked
            expected_keywords: Keywords expected in the answer
            rag_answer: RAG system response
            question_index: Index for tracking
            
        Returns:
            Complete evaluation results
        """
        start_time = time.time()
        
        # Prepare expected keywords (clean and split if needed)
        if isinstance(expected_keywords, str):
            clean_keywords = [kw.strip() for kw in expected_keywords.split(',') if kw.strip()]
        else:
            clean_keywords = [str(kw).strip() for kw in expected_keywords if str(kw).strip()]
        
        # Run enhanced evaluation
        evaluation_results = self.enhanced_contextual_evaluation(
            clean_keywords, rag_answer, "auto"
        )
        
        # Extract combined results
        combined = evaluation_results.get('combined_results', {})
        
        # Prepare final result
        result = {
            'success': True,
            'question_index': question_index,
            'question': question,
            'rag_answer': rag_answer,
            'expected_keywords': clean_keywords,
            'language': evaluation_results.get('language', 'unknown'),
            'adaptive_threshold': evaluation_results.get('adaptive_threshold', self.similarity_threshold),
            
            # Combined results (primary)
            'final_score': combined.get('final_score', 0.0),
            'passes_threshold': combined.get('passes_threshold', False),
            'best_method': combined.get('best_method', 'none'),
            'best_method_score': combined.get('best_method_score', 0.0),
            
            # Method breakdown
            'evaluation_methods': evaluation_results.get('evaluation_methods', {}),
            'method_scores': combined.get('method_scores', {}),
            'individual_scores': combined.get('individual_scores', {}),
            
            # Performance
            'evaluation_time': time.time() - start_time,
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def evaluate_testset(self, testset_file: Path, output_dir: Path, 
                        rag_responses: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate complete testset using enhanced contextual method.
        
        Args:
            testset_file: Path to testset CSV file
            output_dir: Directory to save results
            rag_responses: Optional pre-computed RAG responses
            
        Returns:
            Complete evaluation results
        """
        logger.info(f"🚀 Starting enhanced contextual keyword evaluation")
        logger.info(f"📄 Testset: {testset_file}")
        logger.info(f"📁 Output: {output_dir}")
        
        start_time = time.time()
        
        try:
            # Load testset
            testset_df = pd.read_csv(testset_file)
            logger.info(f"📊 Loaded {len(testset_df)} questions from testset")
            
            # Ensure output directory exists
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize results tracking
            detailed_results = []
            summary_stats = {
                'total_questions': len(testset_df),
                'successful_evaluations': 0,
                'failed_evaluations': 0,
                'scores': [],
                'pass_rates': {
                    'overall': 0,
                    'by_method': {},
                    'by_language': {}
                },
                'method_usage': {},
                'language_distribution': {}
            }
            
            # Process each question
            for idx, row in testset_df.iterrows():
                try:
                    question = row.get('question', '')
                    expected_keywords = row.get('auto_keywords', row.get('keywords', ''))
                    
                    # Get RAG response (use provided or mock)
                    if rag_responses and idx < len(rag_responses):
                        rag_answer = rag_responses[idx].get('answer', '')
                    else:
                        rag_answer = row.get('answer', row.get('ground_truth', ''))
                    
                    if not rag_answer:
                        logger.warning(f"No RAG answer for question {idx}, skipping")
                        continue
                    
                    # Evaluate single question
                    result = self.evaluate_single_question(
                        question, expected_keywords, rag_answer, idx
                    )
                    
                    detailed_results.append(result)
                    
                    # Update summary stats
                    if result['success']:
                        summary_stats['successful_evaluations'] += 1
                        summary_stats['scores'].append(result['final_score'])
                        
                        # Track language distribution
                        lang = result['language']
                        summary_stats['language_distribution'][lang] = \
                            summary_stats['language_distribution'].get(lang, 0) + 1
                        
                        # Track method usage
                        best_method = result['best_method']
                        summary_stats['method_usage'][best_method] = \
                            summary_stats['method_usage'].get(best_method, 0) + 1
                        
                        logger.info(f"✅ Question {idx}: Score={result['final_score']:.3f}, "
                                  f"Pass={result['passes_threshold']}, Method={best_method}")
                    else:
                        summary_stats['failed_evaluations'] += 1
                        logger.warning(f"❌ Question {idx} evaluation failed")
                
                except Exception as e:
                    logger.error(f"❌ Error evaluating question {idx}: {e}")
                    summary_stats['failed_evaluations'] += 1
                    continue
            
            # Calculate final summary statistics
            if summary_stats['scores']:
                summary_stats['mean_score'] = float(safe_mean(summary_stats['scores']))
                summary_stats['std_score'] = float(safe_std(summary_stats['scores']))
                summary_stats['min_score'] = float(np.min(summary_stats['scores']))
                summary_stats['max_score'] = float(np.max(summary_stats['scores']))
                
                # Calculate pass rates
                passes = [r['passes_threshold'] for r in detailed_results if r['success']]
                summary_stats['pass_rates']['overall'] = safe_mean(passes) if passes else 0
            
            # Save detailed results
            results_file = output_dir / f"enhanced_contextual_detailed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(detailed_results, f, indent=2, ensure_ascii=False)
            
            # Save summary report
            summary_file = output_dir / f"enhanced_contextual_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            evaluation_time = time.time() - start_time
            
            final_summary = {
                'evaluation_metadata': {
                    'testset_file': str(testset_file),
                    'output_directory': str(output_dir),
                    'evaluation_time': evaluation_time,
                    'timestamp': datetime.now().isoformat(),
                    'evaluator': 'EnhancedContextualKeywordEvaluator'
                },
                'summary_statistics': summary_stats,
                'configuration': {
                    'similarity_threshold': self.similarity_threshold,
                    'fuzzy_threshold': self.fuzzy_threshold,
                    'weights': self.weights,
                    'models_available': {
                        'sentence_transformers': SENTENCE_TRANSFORMERS_AVAILABLE,
                        'spacy': SPACY_AVAILABLE,
                        'fuzzywuzzy': FUZZYWUZZY_AVAILABLE,
                        'keybert': KEYBERT_AVAILABLE
                    }
                }
            }
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                json.dump(final_summary, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Enhanced contextual evaluation completed in {evaluation_time:.2f} seconds")
            logger.info(f"📊 Success rate: {summary_stats['successful_evaluations']}/{summary_stats['total_questions']}")
            logger.info(f"📈 Mean score: {summary_stats.get('mean_score', 0):.3f}")
            logger.info(f"✅ Pass rate: {summary_stats['pass_rates']['overall']:.1%}")
            
            return {
                'success': True,
                'detailed_results_file': str(results_file),
                'summary_report_file': str(summary_file),
                'summary_statistics': summary_stats,
                'summary_metrics': {
                    'total_questions': summary_stats['total_questions'],
                    'successful_evaluations': summary_stats['successful_evaluations'],
                    'avg_similarity_score': summary_stats.get('mean_score', 0.0),
                    'pass_rate': summary_stats['pass_rates']['overall'],
                    'evaluation_time': evaluation_time
                },
                'total_questions': summary_stats['total_questions'],
                'successful_evaluations': summary_stats['successful_evaluations'],
                'evaluation_time': evaluation_time
            }
            
        except Exception as e:
            logger.error(f"❌ Enhanced contextual evaluation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'evaluation_time': time.time() - start_time
            }
