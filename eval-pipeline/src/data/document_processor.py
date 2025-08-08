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
    
    def __init__(self, config: Dict[str, Any], output_dir):
        """
        Initialize document processor.
        
        Args:
            config: Data sources configuration
            output_dir: Output directory for processed documents
        """
        self.config = config
        self.output_dir = Path(output_dir)  # Convert to Path object
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize document loader if available
        if DocumentLoader:
            self.document_loader = DocumentLoader(config)
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
        
        logger.info(f"Document processor config: {self.config}")
        
        # Process CSV files
        csv_files = self.config.get('csv_files', [])
        if csv_files:
            logger.info(f"Processing {len(csv_files)} CSV files: {csv_files}")
            for csv_file in csv_files:
                try:
                    doc_result = self._process_single_document(csv_file)
                    if doc_result:
                        processed_docs.append(doc_result)
                except Exception as e:
                    logger.error(f"Failed to process CSV file {csv_file}: {e}")
        
        # Process PDF files
        pdf_files = self.config.get('pdf_files', [])
        if pdf_files:
            logger.info(f"Processing {len(pdf_files)} PDF files: {pdf_files}")
            for pdf_file in pdf_files:
                try:
                    doc_result = self._process_single_document(pdf_file)
                    if doc_result:
                        processed_docs.append(doc_result)
                except Exception as e:
                    logger.error(f"Failed to process PDF file {pdf_file}: {e}")
        
        # Process text files
        text_files = self.config.get('text_files', [])
        if text_files:
            logger.info(f"Processing {len(text_files)} text files: {text_files}")
            for text_file in text_files:
                try:
                    doc_result = self._process_single_document(text_file)
                    if doc_result:
                        processed_docs.append(doc_result)
                except Exception as e:
                    logger.error(f"Failed to process text file {text_file}: {e}")
        
        # Process directories
        directories = self.config.get('directories', [])
        if directories:
            logger.info(f"Processing {len(directories)} directories: {directories}")
            for dir_path in directories:
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
        logger.info(f"Processing document path: {doc_path}")
        doc_file = Path(doc_path)
        
        # Convert to absolute path if relative
        if not doc_file.is_absolute():
            # Try relative to eval-pipeline directory first
            pipeline_root = Path(__file__).parent.parent.parent
            doc_file = pipeline_root / doc_file
            logger.info(f"Trying relative to pipeline root: {doc_file}")
            
            # If that doesn't exist, try current directory
            if not doc_file.exists():
                doc_file = Path.cwd() / doc_path
                logger.info(f"Trying relative to current directory: {doc_file}")
        
        logger.info(f"Final resolved path: {doc_file}")
        logger.info(f"File exists: {doc_file.exists()}")
        
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
        
        # Handle different config structures - check if we have full config or just data_sources
        documents_config = self.config.get('documents', {})
        if not documents_config and 'data_sources' in self.config:
            # We have full config, extract documents section
            documents_config = self.config.get('data_sources', {}).get('documents', {})
        
        file_types = documents_config.get('file_types', ['pdf', 'docx', 'txt', 'md'])
        
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
                documents, metadata = self.document_loader.load_pdf(str(doc_file))
                content = " ".join(documents) if documents else ""
            elif file_type in ['.docx', '.doc']:
                documents, metadata = self.document_loader.load_docx(str(doc_file))
                content = " ".join(documents) if documents else ""
            elif file_type in ['.txt', '.md']:
                documents, metadata = self.document_loader.load_text_file(str(doc_file))
                content = " ".join(documents) if documents else ""
            elif file_type == '.csv':
                # Handle CSV files by reading and concatenating text content
                try:
                    import pandas as pd
                    df = pd.read_csv(doc_file)
                    # Concatenate all text columns into a single string
                    text_content = []
                    for column in df.columns:
                        # Skip numeric columns, focus on text content
                        if df[column].dtype == 'object':  # String columns
                            text_content.extend(df[column].dropna().astype(str).tolist())
                    content = " ".join(text_content)
                except Exception as e:
                    logger.warning(f"Failed to process CSV file {doc_file.name}: {e}")
                    content = ""
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
            
            elif file_type == '.csv':
                # Process CSV files by concatenating all text content
                try:
                    import pandas as pd
                    df = pd.read_csv(doc_file)
                    # Concatenate all text columns into a single string
                    text_content = []
                    for column in df.columns:
                        # Skip numeric columns, focus on text content
                        if df[column].dtype == 'object':  # String columns
                            text_content.extend(df[column].dropna().astype(str).tolist())
                    return " ".join(text_content)
                except Exception as e:
                    logger.warning(f"Failed to process CSV file {doc_file.name}: {e}")
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
                logger.warning(f"Unsupported file type: {file_type}")
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
    
    def save_processed_documents(self, processed_docs: List[Dict[str, Any]], output_path: Path) -> None:
        """
        Save processed documents to JSON file.
        
        Args:
            processed_docs: List of processed documents
            output_path: Path where to save the JSON file
        """
        import json
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert Path objects to strings for JSON serialization
        serializable_docs = []
        for doc in processed_docs:
            serializable_doc = dict(doc)
            if 'source_file' in serializable_doc:
                serializable_doc['source_file'] = str(serializable_doc['source_file'])
            serializable_docs.append(serializable_doc)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_docs, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved {len(processed_docs)} processed documents to {output_path}")
    
    
    def _create_document_loader_config(self, pipeline_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert pipeline configuration to DocumentLoader configuration format.
        
        Args:
            pipeline_config: Pipeline configuration
            
        Returns:
            DocumentLoader compatible configuration
        """
        # Get pipeline root for resolving relative paths
        pipeline_root = Path(__file__).parent.parent.parent
        
        # Extract document configuration
        documents_config = pipeline_config.get('documents', {})
        primary_docs = documents_config.get('primary_docs', [])
        additional_dirs = documents_config.get('additional_dirs', [])
        processing_config = documents_config.get('processing', {})
        
        # Convert to absolute paths
        pdf_files = []
        directories = []
        
        for doc_path in primary_docs:
            doc_file = Path(doc_path)
            if not doc_file.is_absolute():
                doc_file = pipeline_root / doc_file
            if doc_file.exists():
                pdf_files.append(str(doc_file))
            else:
                logger.warning(f"Primary document not found: {doc_path} (resolved to {doc_file})")
        
        for dir_path in additional_dirs:
            dir_file = Path(dir_path)
            if not dir_file.is_absolute():
                dir_file = pipeline_root / dir_path
            if dir_file.exists():
                directories.append(str(dir_file))
            else:
                logger.warning(f"Additional directory not found: {dir_path} (resolved to {dir_file})")
        
        # Create DocumentLoader compatible config
        return {
            'custom_data': {
                'enabled': True,
                'data_sources': {
                    'pdf_files': pdf_files,
                    'directories': directories
                },
                'processing': processing_config
            }
        }
