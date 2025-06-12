"""
Document Loader for Custom Data Integration
Supports PDF, DOCX, TXT, CSV, Excel and directory-based loading
"""

import os
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# PDF processing
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    print("⚠️  PDF support not available. Install with: pip install PyPDF2 pdfplumber")
    PDF_AVAILABLE = False

# Word document processing
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    print("⚠️  DOCX support not available. Install with: pip install python-docx")
    DOCX_AVAILABLE = False

# Advanced text processing
try:
    from langdetect import detect, detect_langs
    import langdetect
    LANGDETECT_AVAILABLE = True
except ImportError:
    print("⚠️  Language detection not available. Install with: pip install langdetect")
    LANGDETECT_AVAILABLE = False

# Topic extraction
try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    KEYBERT_AVAILABLE = False

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import LatentDirichletAllocation
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

class DocumentLoader:
    """Comprehensive document loader supporting multiple formats"""
    
    def __init__(self, config: Dict):
        """Initialize document loader with configuration"""
        self.config = config
        self.custom_config = config.get('custom_data', {})
        self.processing_config = self.custom_config.get('processing', {})
        self.topic_config = self.custom_config.get('topic_extraction', {})
        
        # Initialize topic extraction models
        self.keybert_model = KeyBERT() if KEYBERT_AVAILABLE else None
        self.yake_extractor = yake.KeywordExtractor(
            lan="en", n=3, dedupLim=0.7, top=10
        ) if YAKE_AVAILABLE else None
        
        # Document storage
        self.documents = []
        self.document_metadata = []
        
    def load_all_documents(self) -> Tuple[List[str], List[Dict]]:
        """Load all documents from configured sources"""
        print("📂 Loading documents from configured sources...")
        
        data_sources = self.custom_config.get('data_sources', {})
        
        # Load PDF files
        pdf_files = data_sources.get('pdf_files', [])
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                print(f"  📄 Loading PDF: {pdf_file}")
                docs, metadata = self.load_pdf(pdf_file)
                self.documents.extend(docs)
                self.document_metadata.extend(metadata)
            else:
                print(f"  ⚠️  PDF not found: {pdf_file}")
        
        # Load text files
        text_files = data_sources.get('text_files', [])
        for text_file in text_files:
            if os.path.exists(text_file):
                print(f"  📝 Loading text: {text_file}")
                docs, metadata = self.load_text_file(text_file)
                self.documents.extend(docs)
                self.document_metadata.extend(metadata)
            else:
                print(f"  ⚠️  Text file not found: {text_file}")
        
        # Load Word documents
        word_files = data_sources.get('word_files', [])
        for word_file in word_files:
            if os.path.exists(word_file):
                print(f"  📄 Loading DOCX: {word_file}")
                docs, metadata = self.load_docx(word_file)
                self.documents.extend(docs)
                self.document_metadata.extend(metadata)
            else:
                print(f"  ⚠️  DOCX file not found: {word_file}")
        
        # Load from directories
        directories = data_sources.get('directories', [])
        for directory in directories:
            if os.path.exists(directory):
                print(f"  📁 Loading from directory: {directory}")
                docs, metadata = self.load_directory(directory)
                self.documents.extend(docs)
                self.document_metadata.extend(metadata)
            else:
                print(f"  ⚠️  Directory not found: {directory}")
        
        # Load structured files
        structured_files = data_sources.get('structured_files', [])
        for struct_file in structured_files:
            file_path = struct_file.get('file')
            if file_path and os.path.exists(file_path):
                print(f"  📊 Loading structured file: {file_path}")
                docs, metadata = self.load_structured_file(struct_file)
                self.documents.extend(docs)
                self.document_metadata.extend(metadata)
            else:
                print(f"  ⚠️  Structured file not found: {file_path}")
        
        # Process documents
        if self.documents:
            print(f"📋 Processing {len(self.documents)} raw documents...")
            self.documents, self.document_metadata = self.process_documents(
                self.documents, self.document_metadata
            )
            print(f"✅ Final document count: {len(self.documents)}")
        else:
            print("⚠️  No documents loaded. Using default document corpus.")
            return self.get_default_documents()
        
        return self.documents, self.document_metadata
    
    def load_pdf(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Load and extract text from PDF file"""
        documents = []
        metadata = []
        
        if not PDF_AVAILABLE:
            print(f"  ❌ PDF processing not available for {file_path}")
            return documents, metadata
        
        try:
            # Try pdfplumber first (better text extraction)
            with pdfplumber.open(file_path) as pdf:
                full_text = ""
                for page_num, page in enumerate(pdf.pages):
                    page_text = page.extract_text()
                    if page_text:
                        full_text += page_text + "\n\n"
                
                # Chunk the document
                chunks = self.chunk_text(full_text)
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        'source_file': file_path,
                        'file_type': 'pdf',
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    })
        
        except Exception as e:
            print(f"  ❌ Error loading PDF {file_path}: {e}")
            
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    full_text = ""
                    
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text:
                            full_text += page_text + "\n\n"
                    
                    chunks = self.chunk_text(full_text)
                    for i, chunk in enumerate(chunks):
                        documents.append(chunk)
                        metadata.append({
                            'source_file': file_path,
                            'file_type': 'pdf',
                            'chunk_id': i,
                            'total_chunks': len(chunks)
                        })
                        
            except Exception as e2:
                print(f"  ❌ PyPDF2 fallback failed for {file_path}: {e2}")
        
        return documents, metadata
    
    def load_docx(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Load and extract text from Word document"""
        documents = []
        metadata = []
        
        if not DOCX_AVAILABLE:
            print(f"  ❌ DOCX processing not available for {file_path}")
            return documents, metadata
        
        try:
            doc = Document(file_path)
            full_text = ""
            
            for paragraph in doc.paragraphs:
                if paragraph.text.strip():
                    full_text += paragraph.text + "\n"
            
            chunks = self.chunk_text(full_text)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    'source_file': file_path,
                    'file_type': 'docx',
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
                
        except Exception as e:
            print(f"  ❌ Error loading DOCX {file_path}: {e}")
        
        return documents, metadata
    
    def load_text_file(self, file_path: str) -> Tuple[List[str], List[Dict]]:
        """Load text from plain text file"""
        documents = []
        metadata = []
        
        try:
            # Try different encodings
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                print(f"  ❌ Could not decode {file_path} with any encoding")
                return documents, metadata
            
            # Filter by language if enabled
            if self.processing_config.get('filter_language'):
                if not self.is_target_language(content):
                    print(f"  ⚠️  Skipping {file_path} - not target language")
                    return documents, metadata
            
            chunks = self.chunk_text(content)
            for i, chunk in enumerate(chunks):
                documents.append(chunk)
                metadata.append({
                    'source_file': file_path,
                    'file_type': 'txt',
                    'chunk_id': i,
                    'total_chunks': len(chunks)
                })
                
        except Exception as e:
            print(f"  ❌ Error loading text file {file_path}: {e}")
        
        return documents, metadata
    
    def load_structured_file(self, file_config: Dict) -> Tuple[List[str], List[Dict]]:
        """Load documents from structured files (CSV/Excel)"""
        documents = []
        metadata = []
        
        file_path = file_config.get('file')
        document_column = file_config.get('document_column', 'content')
        topic_column = file_config.get('topic_column', 'topic')
        
        try:
            # Load based on file extension
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                print(f"  ❌ Unsupported structured file format: {file_path}")
                return documents, metadata
            
            if document_column not in df.columns:
                print(f"  ❌ Column '{document_column}' not found in {file_path}")
                return documents, metadata
            
            for idx, row in df.iterrows():
                content = str(row[document_column])
                if pd.isna(content) or len(content.strip()) < self.processing_config.get('min_document_length', 50):
                    continue
                
                chunks = self.chunk_text(content)
                for i, chunk in enumerate(chunks):
                    documents.append(chunk)
                    metadata.append({
                        'source_file': file_path,
                        'file_type': 'structured',
                        'chunk_id': i,
                        'total_chunks': len(chunks),
                        'row_index': idx,
                        'topic': row.get(topic_column, 'unknown') if topic_column in df.columns else 'unknown'
                    })
                    
        except Exception as e:
            print(f"  ❌ Error loading structured file {file_path}: {e}")
        
        return documents, metadata
    
    def load_directory(self, directory_path: str) -> Tuple[List[str], List[Dict]]:
        """Load all supported files from directory recursively"""
        all_documents = []
        all_metadata = []
        
        supported_extensions = {
            '.pdf': self.load_pdf,
            '.txt': self.load_text_file,
            '.docx': self.load_docx,
            '.doc': self.load_docx,
        }
        
        print(f"    🔍 Scanning directory: {directory_path}")
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                if file_ext in supported_extensions:
                    print(f"      📄 Processing: {file}")
                    try:
                        load_function = supported_extensions[file_ext]
                        docs, metadata = load_function(file_path)
                        all_documents.extend(docs)
                        all_metadata.extend(metadata)
                    except Exception as e:
                        print(f"      ❌ Error processing {file}: {e}")
        
        print(f"    ✅ Loaded {len(all_documents)} documents from directory")
        return all_documents, all_metadata
    
    def chunk_text(self, text: str) -> List[str]:
        """Split text into manageable chunks"""
        chunk_size = self.processing_config.get('chunk_size', 1000)
        chunk_overlap = self.processing_config.get('chunk_overlap', 200)
        min_chunk_size = self.processing_config.get('min_chunk_size', 100)
        
        if len(text) <= chunk_size:
            return [text.strip()] if text.strip() else []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            # Find a good break point (sentence end)
            if end < len(text):
                # Look for sentence breaks near the end
                for i in range(min(50, chunk_size // 4)):
                    if text[end - i] in '.!?':
                        end = end - i + 1
                        break
            
            chunk = text[start:end].strip()
            
            if len(chunk) >= min_chunk_size:
                chunks.append(chunk)
            
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return chunks
    
    def process_documents(self, documents: List[str], metadata: List[Dict]) -> Tuple[List[str], List[Dict]]:
        """Post-process loaded documents"""
        processed_docs = []
        processed_metadata = []
        
        min_length = self.processing_config.get('min_document_length', 100)
        
        for doc, meta in zip(documents, metadata):
            # Clean and validate document
            cleaned_doc = self.clean_text(doc)
            
            if len(cleaned_doc) >= min_length:
                processed_docs.append(cleaned_doc)
                processed_metadata.append(meta)
        
        print(f"  📋 Filtered documents: {len(documents)} → {len(processed_docs)}")
        return processed_docs, processed_metadata
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text content"""
        import re
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters if configured
        if self.processing_config.get('remove_special_chars', False):
            text = re.sub(r'[^\w\s\.\,\!\?\;\:]', '', text)
        
        # Strip and return
        return text.strip()
    
    def is_target_language(self, text: str) -> bool:
        """Check if text is in target language"""
        if not LANGDETECT_AVAILABLE:
            return True
        
        target_lang = self.processing_config.get('filter_language')
        if not target_lang:
            return True
        
        try:
            detected_lang = detect(text)
            return detected_lang == target_lang
        except:
            return True  # Default to including if detection fails
    
    def extract_topics_from_text(self, text: str, n_topics: int = 5) -> List[str]:
        """Extract topics from text using available methods"""
        topics = []
        
        # Try KeyBERT first
        if KEYBERT_AVAILABLE and hasattr(self, 'keybert_model'):
            try:
                keywords = self.keybert_model.extract_keywords(
                    text, 
                    keyphrase_ngram_range=(1, 2), 
                    stop_words='english'
                )
                topics.extend([kw[0] for kw in keywords[:n_topics]])
            except Exception as e:
                print(f"  ⚠️  KeyBERT topic extraction failed: {e}")
        
        # Try YAKE if KeyBERT failed
        if not topics and YAKE_AVAILABLE and hasattr(self, 'yake_extractor'):
            try:
                keywords = self.yake_extractor.extract_keywords(text)
                topics.extend([kw[1] for kw in keywords[:n_topics]])
            except Exception as e:
                print(f"  ⚠️  YAKE topic extraction failed: {e}")
        
        # Fallback to simple extraction
        if not topics:
            import re
            words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
            
            # Get most frequent words as topics
            sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
            topics = [word for word, freq in sorted_words[:n_topics] if freq > 1]
        
        return topics[:n_topics]

    def get_document_stats(self) -> Dict:
        """Get statistics about loaded documents"""
        if not self.documents:
            return {}
        
        # Group by source file
        source_files = {}
        for meta in self.document_metadata:
            source = meta.get('source_file', 'unknown')
            if source not in source_files:
                source_files[source] = {'chunks': 0, 'file_type': meta.get('file_type', 'unknown')}
            source_files[source]['chunks'] += 1
        
        stats = {
            'total_documents': len(self.documents),
            'total_source_files': len(source_files),
            'files_by_type': {},
            'chunks_per_file': {},
            'avg_document_length': sum(len(doc) for doc in self.documents) / len(self.documents)
        }
        
        # Calculate file type statistics
        for source, info in source_files.items():
            file_type = info['file_type']
            if file_type not in stats['files_by_type']:
                stats['files_by_type'][file_type] = {'count': 0, 'total_chunks': 0}
            
            stats['files_by_type'][file_type]['count'] += 1
            stats['files_by_type'][file_type]['total_chunks'] += info['chunks']
            stats['chunks_per_file'][source] = info['chunks']
        
        return stats
    
    def get_default_documents(self) -> Tuple[List[str], List[Dict]]:
        """Return default documents when no custom data is loaded"""
        default_docs = [
            "Type 2 diabetes is a chronic condition affecting glucose metabolism. Insulin resistance and beta-cell dysfunction are key pathophysiological mechanisms.",
            "Hypertension diagnosis requires multiple blood pressure measurements. Systolic >140 mmHg or diastolic >90 mmHg indicates hypertension.",
            "Pneumonia involves lung inflammation caused by bacterial, viral, or fungal pathogens. Streptococcus pneumoniae is the most common bacterial cause.",
            "Cardiovascular disease risk factors include modifiable elements like smoking, diet, exercise, and non-modifiable factors like age and genetics.",
            "Asthma management involves bronchodilators for acute symptoms and anti-inflammatory medications for long-term control.",
            "Machine learning algorithms learn patterns from data without explicit programming. Supervised, unsupervised, and reinforcement learning are main paradigms.",
            "Cloud computing provides scalable computing resources through virtualization and distributed systems across multiple data centers.",
            "Cybersecurity protects digital assets through threat detection, access controls, encryption, and incident response protocols.",
            "Blockchain technology uses cryptographic hashing and distributed consensus mechanisms to create immutable transaction records.",
            "API integration enables software interoperability through standardized communication protocols and data exchange formats."
        ]
        
        default_metadata = [
            {'source_file': 'default', 'file_type': 'builtin', 'chunk_id': i, 'total_chunks': len(default_docs)}
            for i in range(len(default_docs))
        ]
        
        return default_docs, default_metadata
    
    def get_topics_from_metadata(self) -> List[str]:
        """Extract unique topics from loaded document metadata"""
        topics = set()
        
        for meta in self.document_metadata:
            # From structured file topics
            if 'topic' in meta:
                topics.add(meta['topic'])
            
            # From extracted topics
            if 'extracted_topics' in meta:
                topics.update(meta['extracted_topics'])
        
        # Convert to list and add some generic topics if none found
        topic_list = list(topics)
        if not topic_list:
            topic_list = [
                "concept", "method", "approach", "technique", "strategy",
                "system", "process", "framework", "model", "solution"
            ]
        
        return topic_list
    
    def print_loading_summary(self):
        """Print summary of loaded documents"""
        if not self.documents:
            print("📋 No custom documents loaded - using default corpus")
            return
        
        print(f"\n📋 Document Loading Summary:")
        print("=" * 40)
        print(f"Total documents loaded: {len(self.documents)}")
        
        # Group by file type
        file_types = {}
        for meta in self.document_metadata:
            file_type = meta.get('file_type', 'unknown')
            file_types[file_type] = file_types.get(file_type, 0) + 1
        
        for file_type, count in file_types.items():
            print(f"  {file_type.upper()}: {count} chunks")
        
        # Show topics if available
        topics = self.get_topics_from_metadata()
        if topics:
            print(f"\nExtracted topics: {', '.join(topics[:5])}")
            if len(topics) > 5:
                print(f"  ... and {len(topics) - 5} more")

def main():
    """Test document loading functionality"""
    # Example configuration
    config = {
        'custom_data': {
            'enabled': True,
            'data_sources': {
                'pdf_files': ['example.pdf'],
                'text_files': ['example.txt'],
                'directories': ['documents/']
            },
            'processing': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'filter_by_language': True,
                'target_language': 'en'
            },
            'topic_extraction': {
                'enabled': True,
                'method': 'keybert',
                'max_topics_per_document': 3
            }
        }
    }
    
    loader = DocumentLoader(config)
    documents, metadata = loader.load_all_documents()
    loader.print_loading_summary()
    
    print(f"\nSample document:\n{documents[0][:200]}..." if documents else "No documents loaded")

if __name__ == "__main__":
    main()
