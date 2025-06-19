# Running RAG Evaluation Pipeline Without Docker

This guide shows you how to run the RAG evaluation pipeline directly on Windows without Docker.

## ⚡ **TL;DR - Minimal Steps**

For the absolute quickest start:
```powershell
cd d:\workspace\domain-specific-llm-eval\eval-pipeline
python -m pip install pandas numpy pyyaml openpyxl scikit-learn keybert
python generate_dataset_configurable.py
```
Check `outputs/` folder for results.

## 🚀 **Quick Start Steps**

### **1. Prerequisites**
- Python 3.10+ (you have Python 3.13.5 ✅)
- Windows PowerShell or Command Prompt
- Internet connection for initial package installation

### **2. Navigate to Pipeline Directory**
```powershell
cd d:\workspace\domain-specific-llm-eval\eval-pipeline
```

### **3. Install Required Python Packages**

Since there might be network issues, try these installation approaches in order:

#### **Option A: Install Essential Packages Only**
```powershell
# Core data processing
python -m pip install pandas numpy pyyaml openpyxl

# NLP and ML packages
python -m pip install scikit-learn spacy sentence-transformers

# Keyword extraction
python -m pip install keybert yake

# Utilities
python -m pip install requests tqdm matplotlib
```

#### **Option B: Install from Requirements File**
```powershell
python -m pip install -r requirements.minimal.txt
```

#### **Option C: If Network Issues Persist**
```powershell
# Install with different index
python -m pip install --index-url https://pypi.org/simple/ -r requirements.minimal.txt

# Or with timeout
python -m pip install --timeout 60 -r requirements.minimal.txt
```

### **4. Download Spacy Language Model**
```powershell
python -m spacy download en_core_web_sm
```

### **5. Setup Configuration**

Copy and customize the configuration file:
```powershell
# Copy the template if needed
copy config\pipeline_config.yaml config\my_config.yaml
```

Edit `config\my_config.yaml` to point to your documents:
```yaml
data_sources:
  documents:
    primary_docs:
      - "../../documents/DSP0266_1.22.0.pdf"  # Your document path
      - "../../documents/your_other_document.pdf"

rag_system:
  api_endpoint: "http://localhost:8000/api/query"  # Your RAG system endpoint

testset_generation:
  method: "configurable"         # Start with 'configurable' (no external API needed)
  samples_per_document: 20       # Adjust based on your needs
```

### **6. Run the Pipeline**

#### **Option 1: Full Pipeline (Recommended)**
```powershell
python run_pipeline.py --config config\my_config.yaml
```

#### **Option 2: Generate Testset Only**
```powershell
python generate_dataset_configurable.py
```

#### **Option 3: Test with Sample Documents**
```powershell
python test_custom_documents.py
```

## 📁 **What Gets Generated**

After running, check these directories:
- `outputs\testsets\` - Generated Q&A datasets
- `outputs\evaluations\` - Evaluation results
- `outputs\reports\` - HTML reports
- `outputs\visualizations\` - Charts and graphs

## 🔧 **Configuration Options**

### **For Local-Only Operation (No API Keys)**
```yaml
testset_generation:
  method: "configurable"  # Uses local models only
  
evaluation:
  enable_ragas: false     # Skip RAGAS if no API keys
  enable_contextual: true # Use contextual keyword matching
```

### **For RAGAS Integration (Requires API Keys)**
```yaml
testset_generation:
  method: "ragas"         # Or "hybrid"
  
ragas:
  llm_provider: "openai"  # or "custom"
  api_base: "https://api.openai.com/v1"
```

## 🔍 **Troubleshooting**

### **Import Errors**
```powershell
# Test essential imports
python -c "import pandas, numpy, yaml; print('Core packages OK')"
python -c "import sklearn, spacy; print('ML packages OK')"
python -c "import keybert, yake; print('Keyword extraction OK')"
```

### **Document Loading Issues**
```powershell
# Test document loading
python -c "from document_loader import DocumentLoader; print('Document loader OK')"
```

### **Missing Dependencies**
```powershell
# Install additional packages as needed
python -m pip install PyPDF2 python-docx beautifulsoup4
```

## 🎯 **Simple Test Run**

Create a minimal test:
```powershell
# Create test documents directory
mkdir sample_documents

# Run the test script
python test_custom_documents.py
```

## 📊 **Expected Output Structure**

```
outputs/
├── testsets/
│   ├── testset_20250619_143022.xlsx
│   └── testset_configurable.xlsx
├── evaluations/
│   ├── evaluation_results_20250619_143022.xlsx
│   └── detailed_analysis.json
├── reports/
│   ├── executive_summary.html
│   └── technical_report.html
└── visualizations/
    ├── keyword_performance.png
    └── evaluation_metrics.png
```

## ⚡ **Performance Tips**

1. **Start Small**: Use `samples_per_document: 10` for initial testing
2. **Local Mode**: Use `method: "configurable"` to avoid API calls
3. **Document Size**: Smaller documents (~1-10MB) process faster
4. **Chunk Settings**: Adjust `chunk_size: 256` based on your content

## 🆘 **Common Issues & Solutions**

### **Issue: "No module named 'ragas'"**
**Solution**: Use `method: "configurable"` in config or install RAGAS:
```powershell
python -m pip install ragas
```

### **Issue: "Document loading failed"**
**Solution**: Check document paths and install document processors:
```powershell
python -m pip install PyPDF2 python-docx
```

### **Issue: "Spacy model not found"**
**Solution**: Download the English model:
```powershell
python -m spacy download en_core_web_sm
```

### **Issue: Permission denied or access errors**
**Solution**: Run PowerShell as Administrator:
```powershell
# Right-click PowerShell and select "Run as Administrator"
```

### **Issue: Python not found**
**Solution**: Check Python installation:
```powershell
python --version
# Should show Python 3.10+ 
# If not, add Python to PATH or use full path like C:\Python313\python.exe
```

## 🚀 **Next Steps**

1. Run the pipeline with your documents
2. Check the generated reports in `outputs/reports/`
3. Customize the configuration based on results
4. Integrate with your RAG system endpoint
5. Add custom evaluation metrics as needed

The pipeline is designed to work incrementally - you can start with basic document processing and add more advanced features (RAGAS, custom LLMs) as needed.
