"""
Keyword Extractor for RAG Evaluation Pipeline

This module extracts keywords from RAGAS-generated testsets using custom LLM endpoint.
The keywords will be used later for contextual keyword evaluation.

Input: RAGAS testset CSV (user_input,reference_contexts,reference,synthesizer_name)
Output: Enhanced testset CSV with auto_keywords column added
"""
import pandas as pd
import json
import requests
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class KeywordExtractor:
    """Extracts keywords from testset answers using custom LLM endpoint."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize keyword extractor with configuration.
        
        Args:
            config: Pipeline configuration containing LLM settings
        """
        self.config = config
        self.llm_config = config.get('llm', {})
        self.keyword_config = config.get('keyword_extraction', {})
        
        # Setup LLM client
        self.endpoint_url = self.llm_config.get('endpoint_url')
        self.api_key = self.llm_config.get('api_key')
        self.model = self.llm_config.get('model', 'gpt-4o')
        
        if not self.endpoint_url or not self.api_key:
            raise ValueError("LLM endpoint URL and API key must be configured")
        
        logger.info(f"KeywordExtractor initialized with model: {self.model}")
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection for English vs Chinese.
        
        Args:
            text: Text to analyze
            
        Returns:
            'chinese' or 'english'
        """
        # Count Chinese characters (CJK Unicode ranges)
        chinese_chars = 0
        total_chars = 0
        
        for char in text:
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':  # Chinese characters
                chinese_chars += 1
            if char.isalnum():
                total_chars += 1
        
        if total_chars == 0:
            return 'english'  # Default to English
            
        chinese_ratio = chinese_chars / total_chars
        return 'chinese' if chinese_ratio > 0.3 else 'english'
    
    def extract_keywords_with_llm(self, text: str, num_keywords: int = 8) -> List[str]:
        """
        Extract keywords from text using custom LLM endpoint with language-specific prompts.
        
        Args:
            text: Text to extract keywords from
            num_keywords: Number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        try:
            # Detect language and use appropriate prompt
            language = self.detect_language(text)
            
            if language == 'chinese':
                prompt = f"""從以下文本中提取{num_keywords}個相關關鍵詞。
關鍵詞應該是最重要的術語，能夠捕捉主要概念、技術術語和關鍵信息。

文本：{text}

請只提供關鍵詞，用逗號分隔，不要解釋或額外的文字。

關鍵詞："""
            else:
                prompt = f"""Extract {num_keywords} relevant keywords from the following text. 
The keywords should be the most important terms that capture the main concepts, technical terms, and key information.

Text: {text}

Please provide ONLY the keywords as a comma-separated list, no explanations or additional text.

Keywords:"""

            # Prepare request payload
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system", 
                        "content": "You are a keyword extraction specialist. Extract the most relevant and important keywords from the given text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,  # Low temperature for consistent extraction
                "max_tokens": 200
            }
            
            # Make API call
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            response = requests.post(
                self.endpoint_url,
                json=payload,
                headers=headers,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result['choices'][0]['message']['content'].strip()
                
                # Parse keywords from response
                keywords = [kw.strip() for kw in content.split(',')]
                keywords = [kw for kw in keywords if kw]  # Remove empty strings
                
                logger.debug(f"Extracted {len(keywords)} keywords: {keywords}")
                return keywords[:num_keywords]  # Ensure we don't exceed requested count
                
            else:
                logger.error(f"LLM API error: {response.status_code} - {response.text}")
                return self._fallback_keyword_extraction(text, num_keywords)
                
        except Exception as e:
            logger.error(f"Error in LLM keyword extraction: {e}")
            return self._fallback_keyword_extraction(text, num_keywords)
    
    def _fallback_keyword_extraction(self, text: str, num_keywords: int) -> List[str]:
        """
        Fallback keyword extraction using language-specific text processing.
        
        Args:
            text: Text to extract keywords from
            num_keywords: Number of keywords to extract
            
        Returns:
            List of extracted keywords
        """
        logger.warning("Using fallback keyword extraction")
        
        language = self.detect_language(text)
        
        if language == 'chinese':
            return self._extract_chinese_keywords(text, num_keywords)
        else:
            return self._extract_english_keywords(text, num_keywords)
    
    def _extract_chinese_keywords(self, text: str, num_keywords: int) -> List[str]:
        """Extract keywords from Chinese text"""
        try:
            import jieba
            # Use jieba for Chinese word segmentation
            words = jieba.lcut(text)
            
            # Filter meaningful Chinese words
            keywords = []
            for word in words:
                # Keep words that are 2+ characters and not pure punctuation/numbers
                if len(word) >= 2 and any('\u4e00' <= char <= '\u9fff' for char in word):
                    keywords.append(word)
            
            # Remove duplicates while preserving order
            unique_keywords = []
            seen = set()
            for kw in keywords:
                if kw not in seen:
                    unique_keywords.append(kw)
                    seen.add(kw)
            
            return unique_keywords[:num_keywords]
            
        except ImportError:
            logger.warning("jieba not available, using character-based extraction")
            # Fallback: extract Chinese phrases by character combinations
            import re
            # Extract sequences of Chinese characters
            chinese_words = re.findall(r'[\u4e00-\u9fff]{2,}', text)
            return list(set(chinese_words))[:num_keywords]
    
    def _extract_english_keywords(self, text: str, num_keywords: int) -> List[str]:
        """Extract keywords from English text"""
        # Simple keyword extraction
        words = text.lower().split()
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 
            'could', 'should', 'if', 'then', 'than', 'when', 'where', 'why', 
            'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 
            'than', 'too', 'very', 'can', 'just', 'should', 'now'
        }
        
        # Filter meaningful words
        keywords = []
        for word in words:
            # Remove punctuation and check length
            word = ''.join(char for char in word if char.isalnum())
            if len(word) > 3 and word not in stop_words:
                keywords.append(word)
        
        # Remove duplicates while preserving order
        unique_keywords = []
        seen = set()
        for kw in keywords:
            if kw not in seen:
                unique_keywords.append(kw)
                seen.add(kw)
        
        return unique_keywords[:num_keywords]
    
    def enhance_testset_with_keywords(self, testset_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Enhance RAGAS testset with keyword extraction.
        
        Args:
            testset_file: Path to RAGAS testset CSV file
            output_dir: Directory to save enhanced testset
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Enhancing testset with keywords: {testset_file}")
        
        try:
            # Load RAGAS testset
            df = pd.read_csv(testset_file)
            logger.info(f"Loaded testset with {len(df)} rows")
            
            # Validate required columns
            required_columns = ['user_input', 'reference_contexts', 'reference', 'synthesizer_name']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Extract keywords from the 'reference' column (this will be the "answer" for evaluation)
            logger.info("Extracting keywords from reference answers...")
            keywords_list = []
            
            for idx, row in df.iterrows():
                reference_text = str(row['reference'])
                logger.info(f"Processing row {idx + 1}/{len(df)}: Extracting keywords...")
                
                keywords = self.extract_keywords_with_llm(
                    reference_text,
                    num_keywords=self.keyword_config.get('num_keywords', 3)
                )
                
                # Convert to comma-separated string for CSV storage
                keywords_str = ','.join(keywords)
                keywords_list.append(keywords_str)
                
                logger.debug(f"Row {idx + 1} keywords: {keywords_str}")
            
            # Add keywords column
            df['auto_keywords'] = keywords_list
            
            # Generate output filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = output_dir / f"enhanced_testset_{timestamp}.csv"
            
            # Save enhanced testset
            df.to_csv(output_file, index=False)
            logger.info(f"Enhanced testset saved to: {output_file}")
            
            # Generate summary statistics
            total_keywords = sum(len(kw.split(',')) for kw in keywords_list if kw)
            avg_keywords = total_keywords / len(df) if len(df) > 0 else 0
            
            results = {
                'success': True,
                'input_file': str(testset_file),
                'output_file': str(output_file),
                'total_rows': len(df),
                'total_keywords_extracted': total_keywords,
                'avg_keywords_per_row': round(avg_keywords, 2),
                'columns': list(df.columns),
                'timestamp': timestamp
            }
            
            logger.info(f"Keyword extraction completed successfully:")
            logger.info(f"  - Processed {len(df)} testset entries")
            logger.info(f"  - Extracted {total_keywords} total keywords")
            logger.info(f"  - Average {avg_keywords:.2f} keywords per entry")
            
            return results
            
        except Exception as e:
            logger.error(f"Error enhancing testset with keywords: {e}")
            return {
                'success': False,
                'error': str(e),
                'input_file': str(testset_file)
            }
    
    def process_latest_testset(self, base_output_dir: Path) -> Dict[str, Any]:
        """
        Find and process the latest RAGAS testset in the outputs directory.
        
        Args:
            base_output_dir: Base outputs directory
            
        Returns:
            Dictionary with processing results
        """
        logger.info(f"Looking for latest RAGAS testset in: {base_output_dir}")
        
        try:
            # Find all run directories
            run_dirs = [d for d in base_output_dir.iterdir() if d.is_dir() and d.name.startswith('run_')]
            
            if not run_dirs:
                raise FileNotFoundError("No run directories found")
            
            # Sort by creation time (most recent first)
            run_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Look for RAGAS testset files
            for run_dir in run_dirs:
                testsets_dir = run_dir / 'testsets'
                if testsets_dir.exists():
                    # Look for pure RAGAS testset files
                    ragas_files = list(testsets_dir.glob('pure_ragas_testset_*.csv'))
                    if ragas_files:
                        # Use the most recent file
                        latest_testset = max(ragas_files, key=lambda x: x.stat().st_mtime)
                        logger.info(f"Found latest RAGAS testset: {latest_testset}")
                        
                        # Process the testset
                        return self.enhance_testset_with_keywords(latest_testset, testsets_dir)
            
            raise FileNotFoundError("No RAGAS testset files found in any run directory")
            
        except Exception as e:
            logger.error(f"Error processing latest testset: {e}")
            return {
                'success': False,
                'error': str(e)
            }
