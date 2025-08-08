#!/usr/bin/env python3
"""
CSV Data Processor for RAG Evaluation Pipeline

Processes CSV files and converts them to document format for RAGAS testset generation.
Handles the JSON content parsing and template application as configured.
"""

import logging
import sys
import json
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional
import hashlib

# Add parent directories to path to import existing code
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

logger = logging.getLogger(__name__)

class CSVDataProcessor:
    """Processes CSV files for testset generation."""
    
    def __init__(self, config: Dict[str, Any], output_dir):
        """
        Initialize CSV data processor.
        
        Args:
            config: Data sources configuration
            output_dir: Output directory for processed documents
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract CSV-specific configuration
        self.csv_config = config.get('csv', {})
        self.format_config = self.csv_config.get('format', {})
        self.processing_config = self.format_config.get('processing', {})
        
        logger.info(f"🔧 CSV Data Processor initialized")
        logger.info(f"   CSV files: {self.csv_config.get('csv_files', [])}")
        logger.info(f"   Output dir: {self.output_dir}")
    
    def process_documents(self) -> List[Dict[str, Any]]:
        """
        Process all configured CSV files.
        
        Returns:
            List of processed document dictionaries
        """
        processed_docs = []
        csv_files = self.csv_config.get('csv_files', [])
        
        if not csv_files:
            logger.warning("⚠️ No CSV files configured for processing")
            return processed_docs
        
        for csv_file in csv_files:
            try:
                logger.info(f"📊 Processing CSV file: {csv_file}")
                docs = self._load_csv_file(csv_file)
                processed_docs.extend(docs)
                logger.info(f"✅ Loaded {len(docs)} documents from {csv_file}")
            except Exception as e:
                logger.error(f"❌ Failed to process CSV file {csv_file}: {e}")
                continue
        
        logger.info(f"🎯 Total documents processed: {len(processed_docs)}")
        return processed_docs
    
    def _load_csv_file(self, csv_file: str) -> List[Dict[str, Any]]:
        """
        Load CSV file and convert rows to document format.
        
        Args:
            csv_file: Path to CSV file
            
        Returns:
            List of document dictionaries
        """
        csv_path = Path(csv_file)
        
        # Convert to absolute path if relative
        if not csv_path.is_absolute():
            csv_path = Path.cwd() / csv_path
        
        if not csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        # Load CSV with configuration
        encoding = self.format_config.get('encoding', 'utf-8')
        delimiter = self.format_config.get('delimiter', ',')
        
        try:
            df = pd.read_csv(
                csv_path,
                encoding=encoding,
                delimiter=delimiter,
                quotechar=self.format_config.get('quote_char', '"'),
                skipinitialspace=True,
                skip_blank_lines=self.format_config.get('skip_blank_lines', True)
            )
            logger.info(f"📈 Loaded CSV with {len(df)} rows and {len(df.columns)} columns")
            logger.info(f"   Columns: {list(df.columns)}")
            
        except Exception as e:
            raise Exception(f"Error reading CSV file: {e}")
        
        # Validate required columns
        required_columns = self.format_config.get('validation', {}).get('required_columns', [])
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise Exception(f"Missing required columns: {missing_columns}")
        
        # Process rows to documents
        documents = []
        column_mapping = self.format_config.get('column_mapping', {})
        content_json_fields = self.format_config.get('content_json_fields', {})
        json_text_template = self.processing_config.get('json_text_template', '')
        
        for idx, row in df.iterrows():
            try:
                doc = self._process_csv_row(
                    row, idx, csv_path,
                    column_mapping, content_json_fields, json_text_template
                )
                if doc:
                    documents.append(doc)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing row {idx}: {e}")
                continue
        
        logger.info(f"🎯 Successfully processed {len(documents)} documents from CSV")
        return documents
    
    def _process_csv_row(self, row: pd.Series, idx: int, csv_path: Path,
                        column_mapping: Dict[str, str], 
                        content_json_fields: Dict[str, str],
                        json_text_template: str) -> Optional[Dict[str, Any]]:
        """
        Process a single CSV row into document format.
        
        Args:
            row: Pandas Series representing the CSV row
            idx: Row index
            csv_path: Path to the CSV file
            column_mapping: Column name mappings
            content_json_fields: JSON field mappings for content
            json_text_template: Template for formatting JSON content
            
        Returns:
            Document dictionary or None if processing fails
        """
        try:
            # Extract content field (usually contains JSON)
            content_field = column_mapping.get('content', 'content')
            raw_content = row.get(content_field, '')
            
            if pd.isna(raw_content) or not raw_content:
                logger.debug(f"Skipping row {idx}: empty content")
                return None
            
            # Parse JSON content if configured
            if content_json_fields and json_text_template:
                try:
                    content_data = json.loads(raw_content)
                    formatted_content = self._apply_json_template(content_data, content_json_fields, json_text_template)
                except (json.JSONDecodeError, Exception) as e:
                    logger.debug(f"Row {idx}: JSON parsing failed, using raw content: {e}")
                    formatted_content = str(raw_content)
            else:
                formatted_content = str(raw_content)
            
            # Apply content length validation
            validation_config = self.format_config.get('validation', {})
            min_length = validation_config.get('min_content_length', 20)
            max_length = validation_config.get('max_content_length', 10000)
            
            if len(formatted_content) < min_length:
                logger.debug(f"Skipping row {idx}: content too short ({len(formatted_content)} < {min_length})")
                return None
            
            if len(formatted_content) > max_length:
                logger.debug(f"Truncating row {idx}: content too long ({len(formatted_content)} > {max_length})")
                formatted_content = formatted_content[:max_length] + "..."
            
            # Clean text if configured
            if self.processing_config.get('clean_text', False):
                formatted_content = self._clean_text(formatted_content)
            
            # Create document dictionary
            document = {
                'source_file': str(csv_path),
                'filename': f"csv_row_{idx}",
                'content': formatted_content,
                'word_count': len(formatted_content.split()),
                'metadata': {
                    'csv_row_index': idx,
                    'source_type': 'csv',
                    'template_key': row.get(column_mapping.get('template_key', 'template_key')),
                    'id': row.get(column_mapping.get('id', 'id')),
                    'author': row.get(column_mapping.get('author', 'author')),
                    'created_at': row.get(column_mapping.get('created_at', 'created_at')),
                    'content_hash': hashlib.md5(formatted_content.encode()).hexdigest()
                }
            }
            
            return document
            
        except Exception as e:
            logger.warning(f"⚠️ Error processing CSV row {idx}: {e}")
            return None
    
    def _apply_json_template(self, content_data: Dict[str, Any], 
                           content_json_fields: Dict[str, str],
                           json_text_template: str) -> str:
        """
        Apply JSON template to format content data.
        
        Args:
            content_data: Parsed JSON data from content field
            content_json_fields: Field mappings for template
            json_text_template: Template string with placeholders
            
        Returns:
            Formatted content string
        """
        try:
            # Extract values for template placeholders
            template_values = {}
            for placeholder, field_name in content_json_fields.items():
                value = content_data.get(field_name, '')
                template_values[placeholder] = str(value) if value is not None else ''
            
            # Apply template
            formatted_content = json_text_template.format(**template_values)
            return formatted_content
            
        except Exception as e:
            logger.debug(f"Template application failed: {e}, using raw JSON")
            return json.dumps(content_data, ensure_ascii=False, indent=2)
    
    def _clean_text(self, text: str) -> str:
        """
        Clean text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        # Basic text cleaning
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Remove empty lines
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        text = '\n'.join(lines)
        
        return text
    
    def get_processing_stats(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get processing statistics.
        
        Args:
            processed_docs: List of processed documents
            
        Returns:
            Statistics dictionary
        """
        if not processed_docs:
            return {
                'total_documents': 0,
                'total_words': 0,
                'avg_words_per_doc': 0,
                'unique_sources': 0
            }
        
        total_words = sum(doc.get('word_count', 0) for doc in processed_docs)
        unique_sources = len(set(doc.get('source_file', '') for doc in processed_docs))
        
        return {
            'total_documents': len(processed_docs),
            'total_words': total_words,
            'avg_words_per_doc': total_words / len(processed_docs) if processed_docs else 0,
            'unique_sources': unique_sources,
            'source_breakdown': self._get_source_breakdown(processed_docs)
        }
    
    def _get_source_breakdown(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of documents by source file."""
        breakdown = {}
        for doc in processed_docs:
            source = doc.get('source_file', 'unknown')
            breakdown[source] = breakdown.get(source, 0) + 1
        return breakdown
