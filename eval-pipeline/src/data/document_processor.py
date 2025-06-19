"""
Document Processor for RAG Evaluation Pipeline

Leverages existing document_loader.py functionality to process documents
for testset generation.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import pandas as pd

# Add parent directories to path to import existing code
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from document_loader import DocumentLoader
except ImportError:
    logging.warning("Could not import existing DocumentLoader, using fallback")
    DocumentLoader = None

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Processes documents for testset generation using existing DocumentLoader."""
    
    def __init__(self, config: Dict[str, Any], output_dir: Path):
        """
        Initialize document processor.
        
        Args:
            config: Data sources configuration
            output_dir: Output directory for processed documents
        """
        self.config = config
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize document loader if available
        if DocumentLoader:
            self.document_loader = DocumentLoader()
        else:
            self.document_loader = None
            logger.warning("DocumentLoader not available, using basic processing")
    
    def process_documents(self) -> List[Dict[str, Any]]:
        """
        Process all configured documents.
        
        Returns:
            List of processed document dictionaries
        """
        processed_docs = []
        
        # Process primary documents
        primary_docs = self.config.get('documents', {}).get('primary_docs', [])
        for doc_path in primary_docs:
            try:
                doc_result = self._process_single_document(doc_path)
                if doc_result:
                    processed_docs.append(doc_result)
            except Exception as e:
                logger.error(f"Failed to process document {doc_path}: {e}")
        
        # Process additional directories
        additional_dirs = self.config.get('documents', {}).get('additional_dirs', [])
        for dir_path in additional_dirs:
            try:
                dir_docs = self._process_directory(dir_path)
                processed_docs.extend(dir_docs)
            except Exception as e:
                logger.error(f"Failed to process directory {dir_path}: {e}")
        
        logger.info(f"Successfully processed {len(processed_docs)} documents")
        return processed_docs
    
    def _process_single_document(self, doc_path: str) -> Optional[Dict[str, Any]]:
        """
        Process a single document.
        
        Args:
            doc_path: Path to document
            
        Returns:
            Processed document dictionary or None if failed
        """
        doc_file = Path(doc_path)
        
        # Convert to absolute path if relative
        if not doc_file.is_absolute():
            # Try relative to config file or current directory
            doc_file = Path.cwd() / doc_file
        
        if not doc_file.exists():
            logger.warning(f"Document not found: {doc_path}")
            return None
        
        logger.info(f"Processing document: {doc_file.name}")
        
        try:
            # Use existing DocumentLoader if available
            if self.document_loader:
                content = self._load_with_document_loader(doc_file)
            else:
                content = self._load_with_fallback(doc_file)
            
            if not content:
                logger.warning(f"No content extracted from: {doc_file.name}")
                return None
            
            # Process content
            processed_content = self._process_content(content)
            
            return {
                'source_file': str(doc_file),
                'filename': doc_file.name,
                'file_type': doc_file.suffix.lower(),
                'content': processed_content,
                'word_count': len(processed_content.split()),
                'char_count': len(processed_content)
            }
            
        except Exception as e:
            logger.error(f"Error processing {doc_file.name}: {e}")
            return None
    
    def _process_directory(self, dir_path: str) -> List[Dict[str, Any]]:
        """
        Process all documents in a directory.
        
        Args:
            dir_path: Path to directory
            
        Returns:
            List of processed document dictionaries
        """
        dir_path = Path(dir_path)
        
        if not dir_path.exists() or not dir_path.is_dir():
            logger.warning(f"Directory not found: {dir_path}")
            return []
        
        processed_docs = []
        file_types = self.config.get('documents', {}).get('file_types', ['pdf', 'docx', 'txt', 'md'])
        
        # Find files with supported extensions
        for file_type in file_types:
            pattern = f"*.{file_type}"
            for doc_file in dir_path.rglob(pattern):
                doc_result = self._process_single_document(str(doc_file))
                if doc_result:
                    processed_docs.append(doc_result)
        
        logger.info(f"Processed {len(processed_docs)} documents from directory: {dir_path}")
        return processed_docs
    
    def _load_with_document_loader(self, doc_file: Path) -> str:
        """
        Load document using existing DocumentLoader.
        
        Args:
            doc_file: Path to document file
            
        Returns:
            Extracted text content
        """
        file_type = doc_file.suffix.lower()
        
        try:
            if file_type == '.pdf':
                content = self.document_loader.load_pdf_file(str(doc_file))
            elif file_type in ['.docx', '.doc']:
                content = self.document_loader.load_docx_file(str(doc_file))
            elif file_type in ['.txt', '.md']:
                content = self.document_loader.load_text_file(str(doc_file))
            else:
                logger.warning(f"Unsupported file type: {file_type}")
                return ""
            
            # Clean content if method available
            if hasattr(self.document_loader, 'clean_text'):
                content = self.document_loader.clean_text(content)
            
            return content
            
        except Exception as e:
            logger.error(f"DocumentLoader failed for {doc_file.name}: {e}")
            return ""
    
    def _load_with_fallback(self, doc_file: Path) -> str:
        """
        Fallback document loading without DocumentLoader.
        
        Args:
            doc_file: Path to document file
            
        Returns:
            Extracted text content
        """
        file_type = doc_file.suffix.lower()
        
        try:
            if file_type in ['.txt', '.md']:
                # Simple text file loading
                with open(doc_file, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            
            elif file_type == '.pdf':
                # Try PyPDF2 if available
                try:
                    import PyPDF2
                    with open(doc_file, 'rb') as f:
                        reader = PyPDF2.PdfReader(f)
                        text = ""
                        for page in reader.pages:
                            text += page.extract_text() + "\n"
                        return text
                except ImportError:
                    logger.warning("PyPDF2 not available for PDF processing")
                    return ""
            
            elif file_type in ['.docx']:
                # Try python-docx if available
                try:
                    from docx import Document
                    doc = Document(doc_file)
                    text = ""
                    for paragraph in doc.paragraphs:
                        text += paragraph.text + "\n"
                    return text
                except ImportError:
                    logger.warning("python-docx not available for DOCX processing")
                    return ""
            
            else:
                logger.warning(f"Unsupported file type in fallback: {file_type}")
                return ""
                
        except Exception as e:
            logger.error(f"Fallback loading failed for {doc_file.name}: {e}")
            return ""
    
    def _process_content(self, content: str) -> str:
        """
        Process and clean document content.
        
        Args:
            content: Raw document content
            
        Returns:
            Processed content
        """
        processing_config = self.config.get('documents', {}).get('processing', {})
        min_length = processing_config.get('min_doc_length', 100)
        
        # Basic cleaning
        content = content.strip()
        
        # Remove excessive whitespace
        import re
        content = re.sub(r'\s+', ' ', content)
        content = re.sub(r'\n\s*\n', '\n\n', content)
        
        # Check minimum length
        if len(content) < min_length:
            logger.warning(f"Document content too short: {len(content)} < {min_length}")
            return ""
        
        return content
    
    def get_processing_stats(self, processed_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get statistics about processed documents.
        
        Args:
            processed_docs: List of processed documents
            
        Returns:
            Statistics dictionary
        """
        if not processed_docs:
            return {
                'total_documents': 0,
                'total_words': 0,
                'total_chars': 0,
                'file_types': {},
                'avg_words_per_doc': 0
            }
        
        total_words = sum(doc['word_count'] for doc in processed_docs)
        total_chars = sum(doc['char_count'] for doc in processed_docs)
        
        # Count file types
        file_types = {}
        for doc in processed_docs:
            file_type = doc['file_type']
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        return {
            'total_documents': len(processed_docs),
            'total_words': total_words,
            'total_chars': total_chars,
            'file_types': file_types,
            'avg_words_per_doc': total_words // len(processed_docs) if processed_docs else 0
        }
