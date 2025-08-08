"""
KeyBERT-Enhanced Testset Generator for Pipeline Integration
"""
import logging
from typing import Dict, List, Any
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class KeyBERTEnhancedTestsetGenerator:
    """
    Testset generator that uses KeyBERT for consistent keyword extraction
    across both generation and evaluation stages.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the KeyBERT-enhanced testset generator.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize unified KeyBERT extractor
        try:
            from utils.keybert_extractor import UnifiedKeyBERTExtractor
            
            extractor_config = {
                'keybert': self.config.get('testset_generation', {}).get('keyword_extraction', {}).get('keybert', {}),
                'yake': self.config.get('testset_generation', {}).get('keyword_extraction', {}).get('yake', {}),
                'testset_generation': self.config.get('testset_generation', {}).get('keyword_extraction', {}).get('testset_generation', {}),
                'evaluation': self.config.get('testset_generation', {}).get('keyword_extraction', {}).get('evaluation', {})
            }
            
            self.keyword_extractor = UnifiedKeyBERTExtractor(extractor_config)
            logger.info("✅ KeyBERT-enhanced testset generator initialized")
            
        except ImportError as e:
            logger.error(f"❌ Failed to initialize KeyBERT extractor: {e}")
            self.keyword_extractor = None
    
    def enhance_testset_with_keybert(self, testset_data: List[Dict[str, Any]], 
                                   output_dir: Path) -> Dict[str, Any]:
        """
        Enhance existing testset data with KeyBERT-extracted keywords.
        
        Args:
            testset_data: List of QA pairs to enhance
            output_dir: Directory to save enhanced testset
            
        Returns:
            Dictionary with enhancement results
        """
        if not self.keyword_extractor:
            logger.warning("KeyBERT extractor not available, skipping enhancement")
            return {'success': False, 'error': 'KeyBERT extractor not available'}
        
        enhanced_data = []
        total_keywords_extracted = 0
        
        logger.info(f"🔍 Enhancing {len(testset_data)} QA pairs with KeyBERT keywords...")
        
        for i, qa_pair in enumerate(testset_data):
            try:
                # Extract keywords from question and answer
                question_text = qa_pair.get('question', '')
                answer_text = qa_pair.get('answer', '')
                combined_text = f"{question_text} {answer_text}"
                
                # Use KeyBERT for keyword extraction
                keyword_result = self.keyword_extractor.extract_for_testset_generation(
                    document_content=combined_text,
                    context=qa_pair.get('contexts', '')
                )
                
                # Get keyword strings
                all_keywords = keyword_result.get('all_keywords', [])
                keyword_strings = [kw['keyword'] for kw in all_keywords]
                
                # Update the QA pair with enhanced keywords
                enhanced_qa = qa_pair.copy()
                enhanced_qa['auto_keywords'] = ', '.join(keyword_strings[:8])  # Top 8 keywords
                enhanced_qa['keybert_method'] = all_keywords[0]['method'] if all_keywords else 'none'
                enhanced_qa['keybert_language'] = keyword_result.get('language', 'english')
                enhanced_qa['keybert_total_count'] = len(all_keywords)
                
                # Categorize keywords by relevance
                enhanced_qa['high_relevance_keywords'] = ', '.join([kw['keyword'] for kw in keyword_result.get('high_relevance', [])])
                enhanced_qa['medium_relevance_keywords'] = ', '.join([kw['keyword'] for kw in keyword_result.get('medium_relevance', [])])
                
                enhanced_data.append(enhanced_qa)
                total_keywords_extracted += len(keyword_strings)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"  Processed {i + 1}/{len(testset_data)} QA pairs...")
                    
            except Exception as e:
                logger.warning(f"Failed to enhance QA pair {i}: {e}")
                # Add original QA pair without enhancement
                enhanced_data.append(qa_pair)
        
        # Save enhanced testset
        try:
            df = pd.DataFrame(enhanced_data)
            
            # Save as both CSV and Excel for compatibility
            timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
            
            csv_file = output_dir / f"keybert_enhanced_testset_{timestamp}.csv"
            excel_file = output_dir / f"keybert_enhanced_testset_{timestamp}.xlsx"
            
            df.to_csv(csv_file, index=False, encoding='utf-8')
            df.to_excel(excel_file, index=False, engine='openpyxl')
            
            logger.info(f"✅ Enhanced testset saved to:")
            logger.info(f"  📄 CSV: {csv_file}")
            logger.info(f"  📊 Excel: {excel_file}")
            
            return {
                'success': True,
                'enhanced_data': enhanced_data,
                'csv_file': str(csv_file),
                'excel_file': str(excel_file),
                'total_keywords_extracted': total_keywords_extracted,
                'avg_keywords_per_qa': total_keywords_extracted / len(enhanced_data) if enhanced_data else 0,
                'processing_summary': {
                    'total_qa_pairs': len(testset_data),
                    'successfully_enhanced': len(enhanced_data),
                    'failed_enhancements': len(testset_data) - len(enhanced_data),
                    'keybert_available': True
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to save enhanced testset: {e}")
            return {
                'success': False,
                'error': f"Failed to save enhanced testset: {e}",
                'enhanced_data': enhanced_data
            }
    
    def generate_keybert_testset_from_documents(self, documents: List[Dict[str, Any]], 
                                              output_dir: Path) -> Dict[str, Any]:
        """
        Generate a new testset from documents using KeyBERT for keyword extraction.
        
        Args:
            documents: List of processed documents
            output_dir: Directory to save generated testset
            
        Returns:
            Dictionary with generation results
        """
        if not self.keyword_extractor:
            logger.warning("KeyBERT extractor not available")
            return {'success': False, 'error': 'KeyBERT extractor not available'}
        
        logger.info(f"🎯 Generating KeyBERT-enhanced testset from {len(documents)} documents...")
        
        testset_data = []
        
        for doc in documents:
            try:
                # Extract keywords from document content
                doc_content = doc.get('content', '')
                doc_title = doc.get('title', 'Unknown')
                
                keyword_result = self.keyword_extractor.extract_for_testset_generation(
                    document_content=doc_content
                )
                
                # Generate questions based on extracted keywords
                all_keywords = keyword_result.get('all_keywords', [])
                high_relevance = keyword_result.get('high_relevance', [])
                
                if high_relevance:
                    # Generate questions using high-relevance keywords
                    for i, keyword_data in enumerate(high_relevance[:3]):  # Top 3 keywords
                        keyword = keyword_data['keyword']
                        
                        # Generate question variations
                        questions = [
                            f"What is {keyword}?",
                            f"How does {keyword} work?",
                            f"Explain {keyword} in detail.",
                        ]
                        
                        for question in questions:
                            qa_pair = {
                                'question': question,
                                'answer': f"Based on the document, {keyword} is an important concept that requires detailed explanation.",
                                'contexts': doc_content[:500],  # First 500 chars as context
                                'source_file': doc.get('source_file', 'unknown'),
                                'auto_keywords': keyword,
                                'keybert_method': keyword_data['method'],
                                'keybert_score': keyword_data['score'],
                                'keybert_language': keyword_data['language'],
                                'question_type': 'keybert_generated',
                                'generation_method': 'keybert_enhanced'
                            }
                            testset_data.append(qa_pair)
                            
                            # Limit questions per document
                            if len(testset_data) >= self.config.get('testset_generation', {}).get('max_total_samples', 50):
                                break
                        
                        if len(testset_data) >= self.config.get('testset_generation', {}).get('max_total_samples', 50):
                            break
                    
            except Exception as e:
                logger.warning(f"Failed to process document {doc.get('source_file', 'unknown')}: {e}")
        
        # Save generated testset
        if testset_data:
            try:
                df = pd.DataFrame(testset_data)
                timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
                
                csv_file = output_dir / f"keybert_generated_testset_{timestamp}.csv"
                excel_file = output_dir / f"keybert_generated_testset_{timestamp}.xlsx"
                
                df.to_csv(csv_file, index=False, encoding='utf-8')
                df.to_excel(excel_file, index=False, engine='openpyxl')
                
                logger.info(f"✅ Generated KeyBERT testset saved to:")
                logger.info(f"  📄 CSV: {csv_file}")
                logger.info(f"  📊 Excel: {excel_file}")
                
                return {
                    'success': True,
                    'testset_data': testset_data,
                    'csv_file': str(csv_file),
                    'excel_file': str(excel_file),
                    'total_samples': len(testset_data),
                    'generation_method': 'keybert_enhanced'
                }
                
            except Exception as e:
                logger.error(f"❌ Failed to save generated testset: {e}")
                return {
                    'success': False,
                    'error': f"Failed to save generated testset: {e}",
                    'testset_data': testset_data
                }
        else:
            logger.warning("No testset data generated")
            return {
                'success': False,
                'error': 'No testset data generated',
                'testset_data': []
            }