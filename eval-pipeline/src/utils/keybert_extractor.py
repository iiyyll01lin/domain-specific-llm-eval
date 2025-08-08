"""
Unified KeyBERT Keyword Extractor for Testset Generation and Evaluation

This module provides consistent keyword extraction capabilities using KeyBERT
for both the testset generation stage and the evaluation stage.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import re

logger = logging.getLogger(__name__)

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    logger.warning("KeyBERT not available. Install with: pip install keybert")
    KEYBERT_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    logger.warning("YAKE not available. Install with: pip install yake")
    YAKE_AVAILABLE = False


class UnifiedKeyBERTExtractor:
    """
    Unified KeyBERT-based keyword extractor for both testset generation and evaluation.
    
    Features:
    - Multi-language support (English and Chinese)
    - Configurable extraction parameters
    - Fallback to YAKE if KeyBERT fails
    - Context-aware keyword filtering
    - Domain-specific keyword enhancement
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the KeyBERT extractor.
        
        Args:
            config: Configuration dictionary with extraction parameters
        """
        self.config = config or {}
        self.keybert_config = self.config.get('keybert', {})
        self.yake_config = self.config.get('yake', {})
        
        # Initialize KeyBERT model
        self.keybert_model = None
        if KEYBERT_AVAILABLE and self.keybert_config.get('enabled', True):
            try:
                model_name = self.keybert_config.get('model', 'all-MiniLM-L6-v2')
                self.keybert_model = KeyBERT(model=model_name)
                logger.info(f"✅ KeyBERT initialized with model: {model_name}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize KeyBERT: {e}")
                self.keybert_model = None
        
        # Initialize YAKE as fallback
        self.yake_extractor = None
        if YAKE_AVAILABLE and self.yake_config.get('enabled', True):
            try:
                self.yake_extractor = yake.KeywordExtractor(
                    lan=self.yake_config.get('language', 'en'),
                    n=self.yake_config.get('n', 3),
                    dedupLim=self.yake_config.get('dedupLim', 0.7),
                    top=self.yake_config.get('top', 20)
                )
                logger.info("✅ YAKE fallback extractor initialized")
            except Exception as e:
                logger.error(f"❌ Failed to initialize YAKE: {e}")
                self.yake_extractor = None
    
    def detect_language(self, text: str) -> str:
        """
        Detect if text is primarily English or Chinese.
        
        Args:
            text: Input text to analyze
            
        Returns:
            'chinese' or 'english'
        """
        if not text:
            return 'english'
        
        chinese_chars = 0
        total_chars = 0
        
        for char in text:
            if char.isalpha() or '\u4e00' <= char <= '\u9fff':
                total_chars += 1
                if '\u4e00' <= char <= '\u9fff':
                    chinese_chars += 1
        
        if total_chars == 0:
            return 'english'
        
        chinese_ratio = chinese_chars / total_chars
        return 'chinese' if chinese_ratio > 0.3 else 'english'
    
    def extract_keywords_keybert(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples
        """
        if not self.keybert_model or not text.strip():
            return []
        
        try:
            # Detect language and adjust parameters
            language = self.detect_language(text)
            
            # Configure KeyBERT parameters based on language
            if language == 'chinese':
                # For Chinese text, use character-based n-grams
                keyphrase_ngram_range = self.keybert_config.get('chinese_ngram_range', (1, 3))
                stop_words = None  # No Chinese stop words in default KeyBERT
            else:
                # For English text
                keyphrase_ngram_range = self.keybert_config.get('ngram_range', (1, 2))
                stop_words = 'english'
            
            # Extract keywords
            keywords = self.keybert_model.extract_keywords(
                text,
                keyphrase_ngram_range=keyphrase_ngram_range,
                stop_words=stop_words,
                use_maxsum=self.keybert_config.get('use_maxsum', True),
                nr_candidates=self.keybert_config.get('nr_candidates', 20),
                top_n=max_keywords,  # Fixed: use top_n instead of top_k
                use_mmr=self.keybert_config.get('use_mmr', True),
                diversity=self.keybert_config.get('diversity', 0.5)
            )
            
            return keywords
            
        except Exception as e:
            logger.error(f"KeyBERT extraction failed: {e}")
            return []
    
    def extract_keywords_yake(self, text: str, max_keywords: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using YAKE as fallback.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of (keyword, score) tuples (note: YAKE scores are inverted - lower is better)
        """
        if not self.yake_extractor or not text.strip():
            return []
        
        try:
            # YAKE returns (score, keyword) - lower scores are better
            yake_keywords = self.yake_extractor.extract_keywords(text)
            
            # Convert to (keyword, confidence_score) format
            # Invert YAKE scores to match KeyBERT (higher = better)
            keywords = []
            for score, keyword in yake_keywords[:max_keywords]:
                # Convert YAKE score (lower=better) to confidence score (higher=better)
                confidence_score = 1.0 / (1.0 + score)
                keywords.append((keyword, confidence_score))
            
            return keywords
            
        except Exception as e:
            logger.error(f"YAKE extraction failed: {e}")
            return []
    
    def extract_keywords(self, text: str, max_keywords: int = 10, 
                        method: str = 'auto') -> List[Dict[str, Any]]:
        """
        Extract keywords using the best available method.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            method: 'keybert', 'yake', or 'auto' (try KeyBERT first, fallback to YAKE)
            
        Returns:
            List of keyword dictionaries with 'keyword', 'score', and 'method' fields
        """
        if not text or not text.strip():
            return []
        
        keywords = []
        method_used = None
        
        # Try KeyBERT first (unless specifically requesting YAKE)
        if method in ['keybert', 'auto'] and self.keybert_model:
            keybert_keywords = self.extract_keywords_keybert(text, max_keywords)
            if keybert_keywords:
                keywords = [
                    {
                        'keyword': kw,
                        'score': score,
                        'method': 'keybert',
                        'language': self.detect_language(text)
                    }
                    for kw, score in keybert_keywords
                ]
                method_used = 'keybert'
        
        # Fallback to YAKE if KeyBERT failed or was not requested
        if not keywords and method in ['yake', 'auto'] and self.yake_extractor:
            yake_keywords = self.extract_keywords_yake(text, max_keywords)
            if yake_keywords:
                keywords = [
                    {
                        'keyword': kw,
                        'score': score,
                        'method': 'yake',
                        'language': self.detect_language(text)
                    }
                    for kw, score in yake_keywords
                ]
                method_used = 'yake'
        
        # Final fallback: simple word extraction
        if not keywords:
            simple_keywords = self._extract_simple_keywords(text, max_keywords)
            keywords = [
                {
                    'keyword': kw,
                    'score': 0.5,
                    'method': 'simple',
                    'language': self.detect_language(text)
                }
                for kw in simple_keywords
            ]
            method_used = 'simple'
        
        logger.debug(f"Extracted {len(keywords)} keywords using {method_used}")
        return keywords
    def get_keyword_strings(self, keyword_data: List[Dict[str, Any]]) -> List[str]:
        """
        Extract just the keyword strings from keyword data.
        
        Args:
            keyword_data: List of keyword dictionaries
            
        Returns:
            List of keyword strings
        """
        return [kw['keyword'] for kw in keyword_data]
    
    def _extract_simple_keywords(self, text: str, max_keywords: int = 10) -> List[str]:
        """
        Simple keyword extraction as final fallback.
        
        Args:
            text: Input text
            max_keywords: Maximum number of keywords to extract
            
        Returns:
            List of keywords
        """
        # Detect language
        language = self.detect_language(text)
        
        if language == 'chinese':
            # For Chinese, extract characters and short phrases
            try:
                import jieba
                words = list(jieba.cut(text))
                # Filter out short words and common characters
                keywords = [w for w in words if len(w) >= 2 and not w.isspace()]
            except:
                # Fallback: extract character sequences
                keywords = re.findall(r'[\u4e00-\u9fff]+', text)
        else:
            # For English, extract meaningful words
            words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            # Remove common stop words
            stop_words = {
                'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were'
            }
            keywords = [w for w in words if w not in stop_words]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        
        return unique_keywords[:max_keywords]
    
    def extract_for_testset_generation(self, document_content: str, 
                                     context: str = None) -> Dict[str, Any]:
        """
        Extract keywords specifically for testset generation.
        
        Args:
            document_content: Main document content
            context: Additional context (optional)
            
        Returns:
            Dictionary with extracted keywords and metadata
        """
        # Combine content and context
        full_text = document_content
        if context:
            full_text += " " + context
        
        # Extract keywords with higher limit for testset generation
        max_keywords = self.config.get('testset_generation', {}).get('max_keywords', 15)
        keywords = self.extract_keywords(full_text, max_keywords=max_keywords)
        
        # Categorize keywords by relevance (adjusted thresholds for typical KeyBERT scores)
        high_relevance = [kw for kw in keywords if kw['score'] >= 0.4]  # Lowered from 0.7
        medium_relevance = [kw for kw in keywords if 0.25 <= kw['score'] < 0.4]  # Adjusted range
        low_relevance = [kw for kw in keywords if kw['score'] < 0.25]  # Lowered threshold
        
        return {
            'all_keywords': keywords,
            'high_relevance': high_relevance,
            'medium_relevance': medium_relevance,
            'low_relevance': low_relevance,
            'language': self.detect_language(full_text),
            'total_count': len(keywords),
            'extraction_method': keywords[0]['method'] if keywords else 'none'
        }
    
    def extract_for_evaluation(self, text: str, context: str = None) -> Dict[str, Any]:
        """
        Extract keywords specifically for evaluation stage.
        
        Args:
            text: Input text to extract keywords from
            context: Additional context (optional)
            
        Returns:
            Dictionary with extracted keywords and metadata
        """
        # Combine text and context
        full_text = text
        if context:
            full_text = f"{text} {context}"
        
        # Extract keywords with evaluation settings
        max_keywords = self.config.get('evaluation', {}).get('max_keywords', 10)
        keywords = self.extract_keywords(full_text, max_keywords=max_keywords)
        
        # Categorize by relevance score
        high_relevance = [kw for kw in keywords if kw['score'] >= 0.7]
        medium_relevance = [kw for kw in keywords if 0.4 <= kw['score'] < 0.7]
        low_relevance = [kw for kw in keywords if kw['score'] < 0.4]
        
        return {
            'response_keywords': keywords,
            'high_relevance': high_relevance,
            'medium_relevance': medium_relevance,
            'low_relevance': low_relevance,
            'total_count': len(keywords),
            'language': self.detect_language(text),
            'extraction_method': keywords[0]['method'] if keywords else 'none'
        }
    
    def filter_keywords_by_score(self, keywords: List[Dict[str, Any]], 
                                min_score: float = 0.3) -> List[Dict[str, Any]]:
        """
        Filter keywords by minimum score threshold.
        
        Args:
            keywords: List of keyword dictionaries
            min_score: Minimum score threshold
            
        Returns:
            Filtered list of keywords
        """
        return [kw for kw in keywords if kw['score'] >= min_score]