"""
CSV Testset Generator for Domain-Specific LLM Evaluation Pipeline

This generator creates Q&A pairs from CSV input where each row represents
a pre-chunked piece of content. Each CSV row should generate exactly one
testset question-answer pair.
"""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

class CSVTestsetGenerator:
    """
    Generates testset Q&A pairs from CSV-processed documents.
    Each input content chunk generates one Q&A pair.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV Testset Generator.
        
        Args:
            config: Configuration dictionary containing generation settings
        """
        self.config = config
        self.testset_config = config.get('testset_generation', {})
        self.csv_config = self.testset_config.get('csv_generation', {})
        
        # Initialize question templates
        self.question_templates = self._load_question_templates()
        
        # Initialize keyword extraction
        self.keyword_extractor = self._initialize_keyword_extractor()
        
    def _load_question_templates(self) -> Dict[str, List[str]]:
        """Load question templates for different question types."""
        default_templates = {
            'factual': [
                "根據文件內容，{topic}是什麼？",
                "文件中提到的{topic}有什麼特點？", 
                "According to the document, what is {topic}?",
                "What are the characteristics of {topic} mentioned in the document?",
                "文件中關於{topic}的描述是什麼？",
                "How is {topic} described in the document?"
            ],
            'procedural': [
                "根據文件，{topic}的操作步驟是什麼？",
                "文件中說明的{topic}流程如何進行？",
                "According to the document, what are the steps for {topic}?",
                "How should {topic} be performed according to the document?",
                "文件中{topic}的執行方法是什麼？",
                "What is the procedure for {topic} as described in the document?"
            ],
            'explanatory': [
                "為什麼文件中提到{topic}？",
                "文件中{topic}的原因是什麼？",
                "Why is {topic} mentioned in the document?",
                "What is the reason for {topic} according to the document?",
                "文件中{topic}的目的是什麼？",
                "What is the purpose of {topic} in the document?"
            ],
            'requirement': [
                "根據文件，{topic}需要符合什麼要求？",
                "文件中{topic}的規範是什麼？",
                "What requirements must {topic} meet according to the document?",
                "What are the specifications for {topic} in the document?",
                "文件中{topic}的標準是什麼？",
                "What standards apply to {topic} as per the document?"
            ]
        }
        
        # Allow custom templates from config
        custom_templates = self.csv_config.get('question_templates', {})
        for category, templates in custom_templates.items():
            if category in default_templates:
                default_templates[category].extend(templates)
            else:
                default_templates[category] = templates
                
        return default_templates
    
    def _initialize_keyword_extractor(self):
        """Initialize keyword extraction capability."""
        try:
            # Try to use KeyBERT if available
            from keybert import KeyBERT
            return KeyBERT()
        except ImportError:
            logger.warning("⚠️ KeyBERT not available, using fallback keyword extraction")
            return None
    
    def generate_testset_from_csv_docs(self, csv_docs: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Generate testset from CSV-processed documents.
        
        Args:
            csv_docs: List of processed CSV document dictionaries
            
        Returns:
            DataFrame containing generated testset
        """
        logger.info(f"🚀 Generating testset from {len(csv_docs)} CSV content chunks")
        
        testset_data = []
        
        for i, doc in enumerate(csv_docs):
            try:
                qa_pair = self._generate_qa_pair_from_doc(doc)
                if qa_pair:
                    qa_pair['sample_id'] = i + 1
                    testset_data.append(qa_pair)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"  📊 Generated {i + 1}/{len(csv_docs)} Q&A pairs")
                        
            except Exception as e:
                logger.warning(f"⚠️ Failed to generate Q&A pair for doc {doc.get('id', 'unknown')}: {e}")
                continue
        
        if not testset_data:
            logger.error("❌ No Q&A pairs were generated")
            return pd.DataFrame()
            
        # Convert to DataFrame
        testset_df = pd.DataFrame(testset_data)
        
        # Add generation metadata
        testset_df['generation_method'] = 'csv_input'
        testset_df['generation_timestamp'] = datetime.now().isoformat()
        
        logger.info(f"✅ Generated {len(testset_df)} Q&A pairs from CSV input")
        
        return testset_df
    
    def _generate_qa_pair_from_doc(self, doc: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Generate a single Q&A pair from a document.
        
        Args:
            doc: Document dictionary from CSV processing
            
        Returns:
            Q&A pair dictionary or None if generation fails
        """
        content = doc.get('content', '')
        if not content:
            return None
            
        try:
            # Extract keywords and topics
            keywords = self._extract_keywords(content)
            main_topic = self._identify_main_topic(content, keywords)
            
            # Determine question type based on content
            question_type = self._classify_question_type(content)
            
            # Generate question
            question = self._generate_question(content, main_topic, question_type)
            
            # Generate answer (use the content as context for answer)
            answer = self._generate_answer(content, question)
            
            # Create Q&A pair
            qa_pair = {
                'question': question,
                'answer': answer,
                'contexts': content,  # The CSV content serves as context
                'ground_truth': answer,  # For consistency with RAGAS format
                'source_id': doc.get('id', ''),
                'source_file': doc.get('source_file', ''),
                'content_title': doc.get('content_title', ''),
                'content_source': doc.get('content_source', ''),
                'content_language': doc.get('content_language', 'EN'),
                'keywords': keywords,
                'main_topic': main_topic,
                'question_type': question_type,
                'word_count': doc.get('word_count', 0),
                'metadata': doc.get('metadata', {})
            }
            
            # Add RAGAS-style scores (mock values for compatibility)
            qa_pair.update(self._generate_mock_ragas_scores())
            
            return qa_pair
            
        except Exception as e:
            logger.error(f"❌ Error generating Q&A pair: {e}")
            return None
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content."""
        try:
            if self.keyword_extractor:
                # Use KeyBERT
                keywords = self.keyword_extractor.extract_keywords(
                    content, 
                    keyphrase_ngram_range=(1, 2),
                    stop_words='english',
                    top_k=8
                )
                return [kw[0] for kw in keywords]
            else:
                # Fallback: simple keyword extraction
                return self._simple_keyword_extraction(content)
        except Exception as e:
            logger.warning(f"⚠️ Keyword extraction failed: {e}")
            return self._simple_keyword_extraction(content)
    
    def _simple_keyword_extraction(self, content: str) -> List[str]:
        """Simple fallback keyword extraction."""
        import re
        from collections import Counter

        # Extract words (3+ characters)
        words = re.findall(r'\b[a-zA-Z\u4e00-\u9fff]{3,}\b', content.lower())
        
        # Filter common stop words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by',
            'is', 'are', 'was', 'were', 'been', 'be', 'have', 'has', 'had', 'will', 'would',
            'should', 'could', 'may', 'might', 'can', 'shall', 'must', 'this', 'that', 'these',
            'those', 'they', 'them', 'their', 'there', 'where', 'when', 'what', 'who', 'how',
            '的', '是', '在', '有', '和', '與', '或', '但', '如果', '如', '因為', '所以', '這', '那'
        }
        
        words = [w for w in words if w not in stop_words and len(w) > 2]
        
        # Return top keywords
        word_counts = Counter(words)
        return [word for word, count in word_counts.most_common(8)]
    
    def _identify_main_topic(self, content: str, keywords: List[str]) -> str:
        """Identify the main topic from content and keywords."""
        # Look for title or key phrases
        title_indicators = ['標題', 'title', '主題', 'topic', '關於', 'about']
        
        # Try to find a title or main concept
        lines = content.split('\n')
        for line in lines[:3]:  # Check first few lines
            line = line.strip()
            if line and len(line) < 100:  # Likely a title
                return line
                
        # Fallback: use the most common keyword
        if keywords:
            return keywords[0]
            
        # Last resort: extract from first sentence
        sentences = re.split(r'[.。!！?？]', content)
        if sentences:
            first_sentence = sentences[0].strip()
            if len(first_sentence) < 200:
                # Extract key phrases
                words = first_sentence.split()
                if len(words) > 3:
                    return ' '.join(words[:5])  # First 5 words
                    
        return "相關內容"  # Generic fallback
    
    def _classify_question_type(self, content: str) -> str:
        """Classify the type of question based on content."""
        content_lower = content.lower()
        
        # Look for procedural indicators
        procedural_indicators = [
            '步驟', 'step', '流程', 'process', '操作', 'operation', 
            '方法', 'method', '如何', 'how to', '執行', 'execute',
            '進行', 'perform', '實施', 'implement'
        ]
        
        # Look for requirement indicators  
        requirement_indicators = [
            '要求', 'requirement', '規範', 'specification', '標準', 'standard',
            '必須', 'must', '需要', 'need', '應該', 'should', '規定', 'regulation'
        ]
        
        # Look for explanatory indicators
        explanatory_indicators = [
            '為什麼', 'why', '原因', 'reason', '因為', 'because', '目的', 'purpose',
            '解釋', 'explain', '說明', 'description'
        ]
        
        # Check content for indicators
        if any(indicator in content_lower for indicator in procedural_indicators):
            return 'procedural'
        elif any(indicator in content_lower for indicator in requirement_indicators):
            return 'requirement'
        elif any(indicator in content_lower for indicator in explanatory_indicators):
            return 'explanatory'
        else:
            return 'factual'  # Default
    
    def _generate_question(self, content: str, main_topic: str, question_type: str) -> str:
        """Generate a question based on content and topic."""
        templates = self.question_templates.get(question_type, self.question_templates['factual'])
        
        # Choose template based on content language
        language = self._detect_language(content)
        if language == 'chinese':
            chinese_templates = [t for t in templates if any('\u4e00' <= c <= '\u9fff' for c in t)]
            if chinese_templates:
                templates = chinese_templates
        else:
            english_templates = [t for t in templates if not any('\u4e00' <= c <= '\u9fff' for c in t)]
            if english_templates:
                templates = english_templates
        
        # Select a template
        template = np.random.choice(templates)
        
        # Format with topic
        try:
            question = template.format(topic=main_topic)
        except:
            # Fallback if formatting fails
            if language == 'chinese':
                question = f"根據文件內容，{main_topic}是什麼？"
            else:
                question = f"What is {main_topic} according to the document?"
                
        return question
    
    def _generate_answer(self, content: str, question: str) -> str:
        """Generate an answer based on content and question."""
        # For now, we'll create a summary-style answer from the content
        # In a production system, you might use an LLM here
        
        # Simple approach: use the content as a base and create a focused answer
        sentences = re.split(r'[.。!！?？]', content)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Try to find the most relevant sentences
        if len(sentences) <= 3:
            # Short content: use all
            answer = '. '.join(sentences) + '.'
        else:
            # Longer content: use first few sentences as answer
            answer = '. '.join(sentences[:3]) + '.'
            
        # Clean up the answer
        answer = re.sub(r'\s+', ' ', answer)
        answer = answer.strip()
        
        # Ensure minimum answer length
        if len(answer) < 20:
            answer = content[:200] + '...' if len(content) > 200 else content
            
        return answer
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Chinese or English."""
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        total_chars = len([c for c in text if c.isalpha()])
        
        if total_chars == 0:
            return 'english'
            
        chinese_ratio = chinese_chars / total_chars
        return 'chinese' if chinese_ratio > 0.3 else 'english'
    
    def _generate_mock_ragas_scores(self) -> Dict[str, float]:
        """Generate mock RAGAS scores for compatibility."""
        return {
            'context_precision': round(np.random.uniform(0.7, 0.95), 3),
            'context_recall': round(np.random.uniform(0.7, 0.95), 3), 
            'faithfulness': round(np.random.uniform(0.7, 0.95), 3),
            'answer_relevancy': round(np.random.uniform(0.7, 0.95), 3),
            'keyword_score': round(np.random.uniform(0.6, 0.9), 3)
        }
    
    def save_testset(self, testset_df: pd.DataFrame, output_dir: str) -> str:
        """
        Save the generated testset to files.
        
        Args:
            testset_df: Generated testset DataFrame
            output_dir: Directory to save the testset
            
        Returns:
            Path to the saved Excel file
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as Excel (primary format)
        excel_file = output_path / f"csv_testset_{timestamp}.xlsx"
        testset_df.to_excel(excel_file, index=False, engine='openpyxl')
        
        # Save as CSV (backup format)
        csv_file = output_path / f"csv_testset_{timestamp}.csv"
        testset_df.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Save metadata
        metadata = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_samples': len(testset_df),
            'generation_method': 'csv_input',
            'question_types': testset_df['question_type'].value_counts().to_dict() if 'question_type' in testset_df.columns else {},
            'languages': testset_df['content_language'].value_counts().to_dict() if 'content_language' in testset_df.columns else {},
            'sources': testset_df['content_source'].value_counts().to_dict() if 'content_source' in testset_df.columns else {}
        }
        
        metadata_file = output_path / f"csv_testset_metadata_{timestamp}.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved testset to: {excel_file}")
        logger.info(f"💾 Saved metadata to: {metadata_file}")
        
        return str(excel_file)
