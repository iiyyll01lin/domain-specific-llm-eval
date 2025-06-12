# Custom Document Integration Guide

This guide explains how to use the enhanced dataset generator with your own PDF, DOCX, and text documents to create domain-specific evaluation datasets.

## 🚀 Quick Start with Your Documents

### 1. Install Dependencies

First, install the document processing dependencies:

```bash
python install_dependencies.py
# Choose option 1 (LOCAL mode) or 5 (Document processing only)
```

Or install manually:
```bash
pip install PyPDF2 pdfplumber python-docx langdetect keybert yake pandas numpy openpyxl sentence-transformers scikit-learn
```

### 2. Prepare Your Documents

Create a folder structure like this:
```
your_project/
├── documents/
│   ├── technical_manual.pdf
│   ├── research_paper.pdf
│   └── specifications.docx
├── knowledge_base/
│   ├── article1.txt
│   └── article2.txt
└── config.yaml
```

### 3. Configure the System

Copy `config_with_custom_documents.yaml` to `config.yaml` and update the paths:

```yaml
# Enable custom data processing
custom_data:
  enabled: true  # IMPORTANT: Set to true
  
  data_sources:
    # Your PDF files
    pdf_files:
      - 'documents/technical_manual.pdf'
      - 'documents/research_paper.pdf'
    
    # Your text files  
    text_files:
      - 'knowledge_base/article1.txt'
      - 'knowledge_base/article2.txt'
    
    # Your Word documents
    word_files:
      - 'documents/specifications.docx'
    
    # Directories containing multiple documents
    directories:
      - 'documents/'
      - 'knowledge_base/'

# Customize for your domain
question_templates:
  - "What does the document say about {}?"
  - "How is {} explained in the source material?"
  - "According to the document, what is {}?"
```

### 4. Generate Your Dataset

```bash
python generate_dataset_configurable.py
```

The system will:
- 📂 Load and process your documents
- 🔍 Extract topics automatically 
- 📝 Generate domain-specific Q&A pairs
- 💾 Save to Excel format ready for evaluation

### 5. Run Your Evaluation Pipeline

```bash
python contextual_keyword_gate.py
python dynamic_ragas_gate_with_human_feedback.py
```

## 📋 Supported Document Formats

| Format | Extension       | Library Used       | Notes                                 |
|--------|-----------------|--------------------|---------------------------------------|
| PDF    | `.pdf`          | pdfplumber, PyPDF2 | Automatic fallback between libraries  |
| Word   | `.docx`         | python-docx        | Modern Word documents                 |
| Text   | `.txt`          | Built-in           | Plain text files                      |
| CSV    | `.csv`          | pandas             | Structured data with document columns |
| Excel  | `.xlsx`, `.xls` | pandas             | Structured data with document columns |

## ⚙️ Configuration Options

### Document Processing Settings

```yaml
processing:
  chunk_size: 1000              # Characters per chunk
  chunk_overlap: 200            # Overlap between chunks  
  min_chunk_size: 100           # Minimum chunk size to keep
  remove_headers_footers: true  # Clean headers/footers
  remove_page_numbers: true     # Remove page numbers
  filter_by_language: true      # Language filtering
  target_language: 'en'         # Target language
```

### Topic Extraction Methods

```yaml
topic_extraction:
  enabled: true
  method: 'keybert'  # Options: 'keybert', 'yake', 'tfidf', 'manual'
  max_topics_per_document: 3
  
  # Manual topics (if method is 'manual')
  manual_topics:
    - 'machine learning'
    - 'data science' 
    - 'artificial intelligence'
```

### Question Template Customization

```yaml
question_templates:
  # General templates
  - "What is {} according to the document?"
  - "How does the document explain {}?"
  
  # Domain-specific templates  
  - "What are the technical specifications for {}?"
  - "What compliance requirements apply to {}?"
  - "What are the best practices for {}?"
```

## 🧪 Testing Your Setup

Use the test script to verify everything works:

```bash
python test_custom_documents.py
```

This will:
- Create sample documents
- Test document loading
- Generate a test dataset
- Verify the complete pipeline

## 📊 Example Output

When you run the system with custom documents, you'll see:

```
🔍 Custom data enabled - loading documents...
📂 Loading documents from configured sources...
  📄 Loading PDF: documents/manual.pdf
  📝 Loading text: knowledge_base/article.txt
  📄 Loading DOCX: documents/spec.docx

📋 Document Loading Summary:
Total documents loaded: 25
  PDF: 15 chunks
  TXT: 8 chunks  
  DOCX: 2 chunks

Extracted topics: machine learning, neural networks, deep learning, data processing

🎯 Initialized in 'LOCAL' mode with 25 custom documents
🚀 Generating 20 samples using LOCAL mode...
✅ Dataset saved as: my_custom_testset.xlsx
```

## 🔧 Troubleshooting

### Common Issues

**1. PDF extraction fails**
```bash
# Install both PDF libraries
pip install PyPDF2 pdfplumber
```

**2. Language detection errors**
```bash
pip install langdetect
# Or disable language filtering in config:
filter_by_language: false
```

**3. No topics extracted**
```bash
pip install keybert
# Or use manual topics in config:
method: 'manual'
manual_topics: ['topic1', 'topic2']
```

**4. Empty document chunks**
```yaml
# Adjust chunk settings in config:
processing:
  chunk_size: 500
  min_chunk_size: 50
```

### Debugging

Enable debug logging:
```yaml
logging:
  level: 'DEBUG'
  show_progress: true
```

## 🎯 Advanced Usage

### Structured Data Integration

For CSV/Excel files with document content:

```yaml
structured_files:
  - file: 'data/knowledge_base.csv'
    document_column: 'article_text'
    topic_column: 'category'
  - file: 'data/qa_pairs.xlsx'  
    document_column: 'context'
    topic_column: 'domain'
```

### Mixed Document Sources

You can combine multiple sources:

```yaml
data_sources:
  pdf_files: ['research/paper1.pdf']
  directories: ['manuals/', 'specs/']
  structured_files:
    - file: 'data/articles.csv'
      document_column: 'content'
```

### Custom Question Generation

Override question templates for specific domains:

```yaml
# For medical domain
question_templates:
  - "What are the symptoms of {}?"
  - "How is {} diagnosed?"
  - "What is the treatment for {}?"

# For technical domain  
question_templates:
  - "What are the specifications for {}?"
  - "How do you configure {}?"
  - "What are the requirements for {}?"
```

## 🔄 Integration with Existing Pipeline

The generated dataset maintains full compatibility with your existing evaluation framework:

1. **Same Excel format** - Exact column structure as original
2. **Same file name** - Uses expected filename by default
3. **Same data types** - All RAGAS metrics included
4. **Enhanced content** - But now with YOUR domain-specific documents

Your existing evaluation scripts will work without any changes:
- `contextual_keyword_gate.py` ✅
- `dynamic_ragas_gate_with_human_feedback.py` ✅

## 📈 Benefits of Custom Document Integration

1. **Domain Relevance** - Q&A pairs from YOUR actual documents
2. **Realistic Testing** - Evaluation on content your LLM will actually see
3. **Topic Accuracy** - Automatic extraction of relevant keywords/topics
4. **Scalable Processing** - Handle hundreds of documents automatically
5. **Multiple Formats** - Support for all common document types
6. **Quality Control** - Configurable text processing and filtering

## 🔍 Next Steps

1. **Start Small** - Test with 2-3 documents first
2. **Refine Configuration** - Adjust chunk sizes and processing settings
3. **Validate Output** - Review generated Q&A pairs for quality
4. **Scale Up** - Add more documents as needed
5. **Customize Templates** - Create domain-specific question patterns
6. **Monitor Performance** - Track evaluation metrics on your custom data

Your domain-specific LLM evaluation framework is now ready to use YOUR documents! 🎉
