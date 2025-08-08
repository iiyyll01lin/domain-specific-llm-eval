"""
Fixed Contextual Keyword Evaluator with RAG endpoint integration

This module implements contextual keyword evaluation using the enhanced testset
and your custom RAG endpoint.
"""
import logging
import pandas as pd
import json
import requests
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class ContextualKeywordEvaluatorFixed:
    """
    Fixed contextual keyword evaluator that:
    1. Uses the RAG endpoint to get answers
    2. Compares with keywords from enhanced testset  
    3. Provides detailed calculation tracking
    4. Uses contextual keyword gate logic
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize evaluator with RAG endpoint configuration.
        
        Args:
            config: Pipeline configuration containing RAG system settings
        """
        self.config = config
        self.rag_config = config.get('rag_system', {})
        self.keyword_config = config.get('contextual_keyword', {})
        
        # RAG endpoint configuration
        self.rag_endpoint = self.rag_config.get('endpoint', 'http://10.3.30.13:8855/app/smt_assistant_chat')
        self.rag_timeout = self.rag_config.get('timeout', 30)
        
        # Keyword evaluation settings
        self.similarity_threshold = self.keyword_config.get('similarity_threshold', 0.7)
        self.contextual_weight = self.keyword_config.get('contextual_weight', 0.8)
        
        logger.info(f"ContextualKeywordEvaluatorFixed initialized")
        logger.info(f"RAG endpoint: {self.rag_endpoint}")
        logger.info(f"Similarity threshold: {self.similarity_threshold}")
    
    def query_rag_system(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system endpoint with a question.
        
        Args:
            question: Question to ask the RAG system
            
        Returns:
            Dictionary with RAG response and metadata
        """
        try:
            # Prepare request payload
            payload = {
                'message': question,
                'conversation_id': f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}',
                'user_id': 'evaluation_pipeline'
            }
            
            # Make request to RAG endpoint
            logger.debug(f"Querying RAG system: {question[:100]}...")
            
            response = requests.post(
                self.rag_endpoint,
                json=payload,
                timeout=self.rag_timeout,
                headers={'Content-Type': 'application/json'}
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Extract answer from response (adapt based on your RAG system's response format)
                if isinstance(result, dict):
                    # Try common response formats
                    answer = (
                        result.get('answer') or 
                        result.get('response') or 
                        result.get('message') or
                        result.get('text') or
                        str(result)
                    )
                else:
                    answer = str(result)
                
                return {
                    'success': True,
                    'answer': answer,
                    'raw_response': result,
                    'response_time': response.elapsed.total_seconds(),
                    'status_code': response.status_code
                }
            else:
                logger.error(f"RAG endpoint error: {response.status_code} - {response.text}")
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'status_code': response.status_code
                }
                
        except requests.exceptions.Timeout:
            logger.error(f"RAG endpoint timeout after {self.rag_timeout}s")
            return {
                'success': False,
                'error': f'Timeout after {self.rag_timeout}s'
            }
        except Exception as e:
            logger.error(f"Error querying RAG system: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
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
            if char.isalpha() or '\u4e00' <= char <= '\u9fff':
                total_chars += 1
                if '\u4e00' <= char <= '\u9fff':  # CJK Unified Ideographs
                    chinese_chars += 1
        
        if total_chars == 0:
            return 'english'
        
        # If more than 30% Chinese characters, consider it Chinese
        chinese_ratio = chinese_chars / total_chars
        return 'chinese' if chinese_ratio > 0.3 else 'english'
    
    def _extract_keywords_english(self, text: str) -> List[str]:
        """
        Extract keywords from English text using KeyBERT or fallback methods.
        
        Args:
            text: English text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
        
        # Try using unified KeyBERT extractor first
        try:
            from utils.keybert_extractor import UnifiedKeyBERTExtractor
            
            extractor_config = {
                'keybert': self.config.get('keybert', {}),
                'yake': self.config.get('yake', {}),
                'evaluation': {'max_keywords': 10}
            }
            
            extractor = UnifiedKeyBERTExtractor(extractor_config)
            result = extractor.extract_for_evaluation(text)
            
            if result['response_keywords']:
                return extractor.get_keyword_strings(result['response_keywords'])
        
        except Exception as e:
            logger.debug(f"KeyBERT extraction failed, using fallback: {e}")
        
        import re
        
        # Fixed regex pattern (remove double backslashes)
        words = re.findall(r'\b\w{3,}\b', text.lower())
        
        # Remove common English stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'can', 'may', 'might', 'must', 'shall', 'not',
            'from', 'they', 'them', 'their', 'what', 'when', 'where', 'who',
            'how', 'why', 'which', 'than', 'more', 'most', 'some', 'any',
            'very', 'much', 'many', 'about', 'into', 'through', 'over', 'under'
        }
        
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        
        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        
        return unique_keywords[:10]  # Return top 10 keywords
    
    def _extract_keywords_chinese(self, text: str) -> List[str]:
        """
        Extract keywords from Chinese text using KeyBERT with jieba segmentation.
        
        Args:
            text: Chinese text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
        
        # Try using unified KeyBERT extractor first
        try:
            from utils.keybert_extractor import UnifiedKeyBERTExtractor
            
            extractor_config = {
                'keybert': self.config.get('keybert', {}),
                'yake': self.config.get('yake', {}),
                'evaluation': {'max_keywords': 10}
            }
            
            extractor = UnifiedKeyBERTExtractor(extractor_config)
            result = extractor.extract_for_evaluation(text)
            
            if result['response_keywords']:
                return extractor.get_keyword_strings(result['response_keywords'])
        
        except Exception as e:
            logger.debug(f"KeyBERT extraction failed, using fallback: {e}")
        
        try:
            # Try to use jieba for Chinese text segmentation
            import jieba
            import re
            
            # Segment Chinese text
            words = list(jieba.cut(text))
            
            # Filter out common Chinese stop words and short words
            chinese_stop_words = {
                '是', '的', '在', '和', '有', '了', '为', '与', '及', '或', '但',
                '这', '那', '我', '你', '他', '她', '它', '们', '我们', '你们',
                '他们', '她们', '它们', '什么', '怎么', '为什么', '哪里', '哪个',
                '如何', '因为', '所以', '但是', '然后', '现在', '以前', '以后',
                '可以', '能够', '应该', '必须', '需要', '想要', '希望', '认为',
                '知道', '看到', '听到', '感到', '觉得', '发现', '找到', '得到',
                '一个', '一些', '很多', '非常', '特别', '尤其', '包括', '例如',
                '比如', '如果', '虽然', '尽管', '除了', '另外', '此外', '而且'
            }
            
            # Clean and filter keywords
            keywords = []
            for word in words:
                word = word.strip()
                # Keep words that are:
                # 1. At least 2 characters long
                # 2. Not in stop words
                # 3. Contain meaningful content (not just punctuation)
                if (len(word) >= 2 and 
                    word not in chinese_stop_words and
                    re.search(r'[\u4e00-\u9fff]', word)):  # Contains Chinese characters
                    keywords.append(word)
            
            # Remove duplicates while preserving order
            seen = set()
            unique_keywords = []
            for kw in keywords:
                if kw not in seen:
                    unique_keywords.append(kw)
                    seen.add(kw)
            
            return unique_keywords[:10]  # Return top 10 keywords
            
        except ImportError:
            # Fallback: Simple character-based segmentation
            logger.warning("jieba not available, using simple Chinese keyword extraction")
            import re
            
            # Extract Chinese words (2-4 characters)
            chinese_words = re.findall(r'[\u4e00-\u9fff]{2,4}', text)
            
            # Remove duplicates and limit
            seen = set()
            unique_keywords = []
            for word in chinese_words:
                if word not in seen and len(word) >= 2:
                    unique_keywords.append(word)
                    seen.add(word)
            
            return unique_keywords[:10]
    
    def extract_keywords_from_text(self, text: str) -> List[str]:
        """
        Extract keywords from text with automatic language detection.
        
        Args:
            text: Text to extract keywords from
            
        Returns:
            List of extracted keywords
        """
        if not text:
            return []
        
        # Detect language
        language = self._detect_language(text)
        
        # Use appropriate extraction method
        if language == 'chinese':
            return self._extract_keywords_chinese(text)
        else:
            return self._extract_keywords_english(text)
    
    def calculate_keyword_similarity(self, expected_keywords: List[str], 
                                   actual_keywords: List[str]) -> Dict[str, Any]:
        """
        Calculate similarity between expected and actual keywords.
        Enhanced for Chinese compound word matching.
        
        Args:
            expected_keywords: Keywords from the enhanced testset
            actual_keywords: Keywords extracted from RAG response
            
        Returns:
            Dictionary with similarity metrics
        """
        if not expected_keywords or not actual_keywords:
            return {
                'similarity_score': 0.0,
                'matched_keywords': [],
                'missing_keywords': expected_keywords if expected_keywords else [],
                'extra_keywords': actual_keywords if actual_keywords else [],
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'expected_count': len(expected_keywords) if expected_keywords else 0,
                'actual_count': len(actual_keywords) if actual_keywords else 0,
                'matched_count': 0
            }
        
        # Enhanced keyword matching for Chinese compound words
        matched_keywords = []
        missing_keywords = []
        
        # Convert to lowercase for comparison
        expected_lower = [kw.lower().strip() for kw in expected_keywords]
        actual_lower = [kw.lower().strip() for kw in actual_keywords]
        
        for expected_kw in expected_lower:
            match_found = False
            
            # Method 1: Exact match
            if expected_kw in actual_lower:
                matched_keywords.append(expected_kw)
                match_found = True
            
            # Method 2: Substring match for compound words
            elif not match_found:
                # Check if expected keyword is contained in any actual keyword
                for actual_kw in actual_lower:
                    if expected_kw in actual_kw or actual_kw in expected_kw:
                        matched_keywords.append(expected_kw)
                        match_found = True
                        break
                
                # Method 3: For Chinese compound words, check if components are present
                if not match_found and self._contains_chinese(expected_kw):
                    # Check if most characters of the expected keyword appear in actual keywords
                    expected_chars = set(expected_kw)
                    actual_chars = set(''.join(actual_lower))
                    
                    # If at least 70% of characters match, consider it a partial match
                    char_overlap = len(expected_chars.intersection(actual_chars))
                    if len(expected_chars) > 0 and char_overlap / len(expected_chars) >= 0.7:
                        matched_keywords.append(expected_kw)
                        match_found = True
            
            if not match_found:
                missing_keywords.append(expected_kw)
        
        # Find extra keywords (actual keywords not matched to any expected)
        matched_actual = set()
        for expected_kw in expected_lower:
            for actual_kw in actual_lower:
                if (expected_kw == actual_kw or 
                    expected_kw in actual_kw or 
                    actual_kw in expected_kw):
                    matched_actual.add(actual_kw)
        
        extra_keywords = [kw for kw in actual_lower if kw not in matched_actual]
        
        # Calculate metrics
        matched_count = len(matched_keywords)
        expected_count = len(expected_keywords)
        actual_count = len(actual_keywords)
        
        precision = matched_count / actual_count if actual_count > 0 else 0.0
        recall = matched_count / expected_count if expected_count > 0 else 0.0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        # Overall similarity score (weighted F1)
        similarity_score = f1_score * self.contextual_weight + precision * (1 - self.contextual_weight)
        
        return {
            'similarity_score': similarity_score,
            'matched_keywords': matched_keywords,
            'missing_keywords': missing_keywords,
            'extra_keywords': extra_keywords,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'expected_count': expected_count,
            'actual_count': actual_count,
            'matched_count': matched_count
        }
    
    def _contains_chinese(self, text: str) -> bool:
        """Check if text contains Chinese characters."""
        return any('\u4e00' <= char <= '\u9fff' for char in text)
    
    def evaluate_single_question(self, question: str, expected_keywords: List[str], 
                               question_index: int, rag_response: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Evaluate a single question using contextual keyword metrics.
        
        Args:
            question: Question to evaluate
            expected_keywords: Expected keywords from testset
            question_index: Index of question for tracking
            rag_response: Optional pre-computed RAG response to avoid re-querying
            
        Returns:
            Dictionary with evaluation results
        """
        logger.debug(f"Evaluating question {question_index}: {question[:50]}...")
        
        try:
            # Use pre-computed RAG response or query RAG system
            if rag_response:
                rag_result = {
                    'success': rag_response.get('success', True),
                    'answer': rag_response.get('answer', ''),
                    'contexts': rag_response.get('contexts', []),
                    'confidence': rag_response.get('confidence'),
                    'response_time': rag_response.get('response_time', 0.0),
                    'source': rag_response.get('source', 'pre_computed')
                }
            else:
                rag_result = self.query_rag_system(question)
            
            if not rag_result['success']:
                return {
                    'success': False,
                    'question_index': question_index,
                    'question': question,
                    'error': f"RAG query failed: {rag_result.get('error', 'Unknown error')}",
                    'rag_response': rag_result
                }
            
            rag_answer = rag_result['answer']
            
            # Extract keywords from RAG answer
            actual_keywords = self.extract_keywords_from_text(rag_answer)
            
            # Calculate keyword similarity
            similarity_metrics = self.calculate_keyword_similarity(expected_keywords, actual_keywords)
            
            # Determine if evaluation passes threshold
            passes_threshold = similarity_metrics['similarity_score'] >= self.similarity_threshold
            
            # Compile detailed results
            result = {
                'success': True,
                'question_index': question_index,
                'question': question,
                'rag_answer': rag_answer,
                'expected_keywords': expected_keywords,
                'actual_keywords': actual_keywords,
                'similarity_metrics': similarity_metrics,
                'passes_threshold': passes_threshold,
                'threshold': self.similarity_threshold,
                'rag_response_time': rag_result.get('response_time', 0),
                'timestamp': datetime.now().isoformat()
            }
            
            # Log result
            score = similarity_metrics['similarity_score']
            status = "✅ PASS" if passes_threshold else "❌ FAIL"
            logger.info(f"Q{question_index}: {status} Score: {score:.3f} "
                       f"(matched: {similarity_metrics['matched_count']}/{similarity_metrics['expected_count']})")
            
            return result
            
        except Exception as e:
            logger.error(f"Error evaluating question {question_index}: {e}")
            return {
                'success': False,
                'question_index': question_index,
                'question': question,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def evaluate_testset(self, testset_file: Path, output_dir: Path, rag_responses: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Evaluate entire testset using contextual keyword metrics.
        
        Args:
            testset_file: Path to enhanced testset CSV
            output_dir: Directory to save evaluation results
            rag_responses: Optional pre-computed RAG responses to avoid re-querying
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"🎯 Starting contextual keyword evaluation of testset: {testset_file}")
        
        try:
            # Load enhanced testset
            df = pd.read_csv(testset_file)
            logger.info(f"📊 Loaded testset with {len(df)} rows")
            
            # Validate required columns
            required_columns = ['user_input', 'auto_keywords']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Evaluate each question
            detailed_results = []
            successful_evaluations = 0
            failed_evaluations = 0
            total_score = 0.0
            passed_threshold = 0
            
            logger.info(f"🚀 Starting evaluation of {len(df)} questions...")
            
            for idx, row in df.iterrows():
                question = str(row['user_input']).strip()
                keywords_str = str(row['auto_keywords']).strip()
                
                # Parse keywords
                if keywords_str and keywords_str != 'nan':
                    # Handle both English and Chinese comma separators
                    import re
                    expected_keywords = re.split(r'[,，、]', keywords_str)
                    expected_keywords = [kw.strip() for kw in expected_keywords if kw.strip()]
                    logger.debug(f"Parsed {len(expected_keywords)} keywords from: {keywords_str[:100]}...")
                else:
                    expected_keywords = []
                
                if not question or not expected_keywords:
                    logger.warning(f"Skipping row {idx}: empty question or keywords")
                    failed_evaluations += 1
                    continue
                
                # Get pre-computed RAG response if available
                rag_response = None
                if rag_responses and idx < len(rag_responses):
                    rag_response = rag_responses[idx]
                
                # Evaluate question
                result = self.evaluate_single_question(question, expected_keywords, idx, rag_response)
                detailed_results.append(result)
                
                if result['success']:
                    successful_evaluations += 1
                    score = result['similarity_metrics']['similarity_score']
                    total_score += score
                    if result['passes_threshold']:
                        passed_threshold += 1
                else:
                    failed_evaluations += 1
                    logger.warning(f"Question {idx} evaluation failed: {result.get('error', 'Unknown error')}")
            
            # Calculate summary statistics
            avg_score = total_score / successful_evaluations if successful_evaluations > 0 else 0.0
            pass_rate = passed_threshold / successful_evaluations if successful_evaluations > 0 else 0.0
            success_rate = successful_evaluations / len(df) if len(df) > 0 else 0.0
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save individual calculations
            detailed_file = output_dir / f"contextual_keyword_detailed_{timestamp}.json"
            with open(detailed_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            logger.info(f"💾 Detailed calculations saved to: {detailed_file}")
            
            # Compile summary results
            summary_results = {
                'testset_file': str(testset_file),
                'timestamp': timestamp,
                'total_questions': len(df),
                'successful_evaluations': successful_evaluations,
                'failed_evaluations': failed_evaluations,
                'passed_threshold': passed_threshold,
                'summary_metrics': {
                    'avg_similarity_score': avg_score,
                    'pass_rate': pass_rate,
                    'success_rate': success_rate,
                    'threshold_used': self.similarity_threshold
                },
                'config': {
                    'rag_endpoint': self.rag_endpoint,
                    'similarity_threshold': self.similarity_threshold,
                    'contextual_weight': self.contextual_weight
                },
                'detailed_results_file': str(detailed_file)
            }
            
            # Save summary results
            summary_file = output_dir / f"contextual_keyword_summary_{timestamp}.json"
            with open(summary_file, 'w') as f:
                json.dump(summary_results, f, indent=2)
            
            logger.info(f"✅ Contextual keyword evaluation completed!")
            logger.info(f"📊 Results: {successful_evaluations}/{len(df)} successful, "
                       f"avg score: {avg_score:.3f}, pass rate: {pass_rate:.1%}")
            logger.info(f"💾 Summary saved to: {summary_file}")
            
            return {
                'success': True,
                'summary_metrics': summary_results['summary_metrics'],
                'total_questions': len(df),
                'successful_evaluations': successful_evaluations,
                'detailed_results_file': str(detailed_file),
                'summary_file': str(summary_file),
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"❌ Contextual keyword evaluation failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
