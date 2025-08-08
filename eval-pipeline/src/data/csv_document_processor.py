"""
CSV Document Processor for Domain-Specific LLM Evaluation Pipeline

This processor handles CSV files with the format:
- id: unique identifier for each content chunk
- content: JSON string containing the actual text content and metadata
- Other fields: metadata that can be used for filtering and organization

Each CSV row represents a pre-chunked piece of content that will generate one testset Q&A pair.
"""

import json
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class CSVDocumentProcessor:
    """
    Processes CSV files where each row contains a pre-chunked piece of content
    that should generate one testset question-answer pair.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize CSV Document Processor.
        
        Args:
            config: Configuration dictionary containing CSV processing settings
        """
        self.config = config
        self.csv_config = config.get('data_sources', {}).get('csv_input', {})
        
    def process_csv_files(self, csv_file_paths: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple CSV files and extract content for testset generation.
        
        Args:
            csv_file_paths: List of paths to CSV files
            
        Returns:
            List of processed document dictionaries, one per CSV row
        """
        processed_docs = []
        
        for csv_path in csv_file_paths:
            try:
                csv_docs = self._process_single_csv(csv_path)
                processed_docs.extend(csv_docs)
                logger.info(f"✅ Processed {len(csv_docs)} content chunks from {Path(csv_path).name}")
            except Exception as e:
                logger.error(f"❌ Failed to process CSV file {csv_path}: {e}")
                
        logger.info(f"📊 Total processed content chunks: {len(processed_docs)}")
        return processed_docs
    
    def _process_single_csv(self, csv_path: str) -> List[Dict[str, Any]]:
        """
        Process a single CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            List of processed document dictionaries
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
            
        logger.info(f"📄 Processing CSV file: {csv_file.name}")
        
        # Read CSV file
        try:
            df = pd.read_csv(csv_file, encoding='utf-8')
            logger.info(f"📋 CSV loaded with {len(df)} rows and columns: {list(df.columns)}")
        except Exception as e:
            logger.error(f"❌ Failed to read CSV file: {e}")
            raise
            
        # Validate required columns
        required_columns = self.csv_config.get('required_columns', ['id', 'content'])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns in CSV: {missing_columns}")
            
        processed_docs = []
        
        for idx, row in df.iterrows():
            try:
                doc = self._process_csv_row(row, csv_file.name)
                if doc:
                    processed_docs.append(doc)
            except Exception as e:
                logger.warning(f"⚠️ Failed to process row {idx}: {e}")
                continue
                
        return processed_docs
    
    def _process_csv_row(self, row: pd.Series, source_file: str) -> Optional[Dict[str, Any]]:
        """
        Process a single CSV row into a document dictionary.
        
        Args:
            row: Pandas Series representing a CSV row
            source_file: Name of the source CSV file
            
        Returns:
            Processed document dictionary or None if processing fails
        """
        try:
            # Extract basic information
            doc_id = str(row.get('id', ''))
            content_field = row.get('content', '')
            
            if not doc_id or not content_field:
                logger.warning(f"⚠️ Skipping row with missing id or content")
                return None
                
            # Parse content field (expected to be JSON)
            try:
                if isinstance(content_field, str) and content_field.startswith('{'):
                    content_data = json.loads(content_field)
                else:
                    # If not JSON, treat as plain text
                    content_data = {'text': str(content_field)}
            except json.JSONDecodeError:
                # Fallback: treat as plain text
                content_data = {'text': str(content_field)}
                
            # Extract actual text content
            text_content = content_data.get('text', content_data.get('content', ''))
            if not text_content or not isinstance(text_content, str):
                logger.warning(f"⚠️ No valid text content found in row {doc_id}")
                return None
                
            # Clean and validate content
            text_content = self._clean_content(text_content)
            if not self._validate_content(text_content):
                logger.warning(f"⚠️ Content validation failed for row {doc_id}")
                return None
                
            # Extract metadata
            metadata = {
                'csv_row_id': doc_id,
                'source_file': source_file,
                'template_key': str(row.get('template_key', '')),
                'source': str(row.get('source', '')),
                'label': content_data.get('label', ''),
                'language': content_data.get('language', 'EN'),
                'title': content_data.get('title', ''),
                'author': str(row.get('author', '')),
                'created_at': str(row.get('created_at', '')),
                'processing_timestamp': datetime.now().isoformat()
            }
            
            # Add any additional fields as metadata
            for col in row.index:
                if col not in ['id', 'content'] and pd.notna(row[col]):
                    metadata[f'csv_{col}'] = str(row[col])
                    
            # Create document dictionary
            doc = {
                'id': doc_id,
                'source_file': source_file,
                'filename': source_file,
                'file_type': '.csv',
                'content': text_content,
                'word_count': len(text_content.split()),
                'char_count': len(text_content),
                'metadata': metadata,
                'chunk_type': 'csv_row',  # Indicates this is a pre-chunked CSV row
                'content_title': content_data.get('title', ''),
                'content_source': content_data.get('source', ''),
                'content_language': content_data.get('language', 'EN')
            }
            
            return doc
            
        except Exception as e:
            logger.error(f"❌ Error processing CSV row: {e}")
            return None
    
    def _clean_content(self, content: str) -> str:
        """
        Clean and normalize content text.
        
        Args:
            content: Raw content text
            
        Returns:
            Cleaned content text
        """
        if not content:
            return ""
            
        # Remove excessive whitespace
        import re
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Remove HTML tags if present
        content = re.sub(r'<[^>]+>', '', content)
        
        # Clean up common artifacts
        content = content.replace('\\"', '"')  # Unescape quotes
        content = content.strip()
        
        return content
    
    def _validate_content(self, content: str) -> bool:
        """
        Validate that content meets minimum requirements.
        
        Args:
            content: Text content to validate
            
        Returns:
            True if content is valid, False otherwise
        """
        min_length = self.csv_config.get('min_content_length', 10)
        min_words = self.csv_config.get('min_word_count', 3)
        
        if len(content) < min_length:
            return False
            
        word_count = len(content.split())
        if word_count < min_words:
            return False
            
        return True
    
    def get_processing_stats(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics about processed CSV documents.
        
        Args:
            processed_docs: List of processed document dictionaries
            
        Returns:
            Statistics dictionary
        """
        if not processed_docs:
            return {
                'total_chunks': 0,
                'total_words': 0,
                'total_chars': 0,
                'languages': {},
                'sources': {},
                'avg_words_per_chunk': 0
            }
            
        total_words = sum(doc['word_count'] for doc in processed_docs)
        total_chars = sum(doc['char_count'] for doc in processed_docs)
        
        # Count languages
        languages = {}
        for doc in processed_docs:
            lang = doc.get('content_language', 'Unknown')
            languages[lang] = languages.get(lang, 0) + 1
            
        # Count sources
        sources = {}
        for doc in processed_docs:
            source = doc.get('content_source', 'Unknown')
            sources[source] = sources.get(source, 0) + 1
            
        return {
            'total_chunks': len(processed_docs),
            'total_words': total_words,
            'total_chars': total_chars,
            'languages': languages,
            'sources': sources,
            'avg_words_per_chunk': total_words // len(processed_docs) if processed_docs else 0,
            'content_titles': [doc.get('content_title', '') for doc in processed_docs[:10]],  # Sample titles
            'sample_chunks': [doc['content'][:100] + '...' for doc in processed_docs[:3]]  # Sample content
        }
    
    def export_processed_docs(self, processed_docs: List[Dict[str, Any]], 
                            output_path: str) -> None:
        """
        Export processed documents to a file for review.
        
        Args:
            processed_docs: List of processed document dictionaries
            output_path: Path to save the exported data
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Create a simplified view for export
            export_data = []
            for doc in processed_docs:
                export_data.append({
                    'id': doc['id'],
                    'source_file': doc['source_file'], 
                    'content_title': doc.get('content_title', ''),
                    'content_source': doc.get('content_source', ''),
                    'content_language': doc.get('content_language', ''),
                    'word_count': doc['word_count'],
                    'content_preview': doc['content'][:200] + '...' if len(doc['content']) > 200 else doc['content']
                })
            
            # Export as Excel for easy review
            df = pd.DataFrame(export_data)
            df.to_excel(output_path, index=False, engine='openpyxl')
            
            logger.info(f"📄 Exported processed documents to: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Failed to export processed documents: {e}")
