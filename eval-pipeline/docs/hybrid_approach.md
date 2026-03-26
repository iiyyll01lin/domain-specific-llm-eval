# Hybrid Evaluation Approach Documentation

## 🎯 **Overview**

The Domain-Specific RAG Evaluation Pipeline implements a comprehensive **hybrid approach** that combines the strengths of your existing contextual keyword system with RAGAS metrics and advanced human feedback integration. This document explains how all components work together to provide comprehensive RAG system evaluation.

## 🏗️ **Architecture Overview**

```
📊 Hybrid Evaluation Pipeline Architecture
==========================================

Input Documents → Document Processing → Testset Generation → RAG Evaluation → Reports
     │                    │                    │                   │            │
     ▼                    ▼                    ▼                   ▼            ▼
┌─────────────┐  ┌─────────────────┐  ┌────────────────┐  ┌─────────────┐  ┌────────────┐
│ PDF, DOCX,  │  │ Your Document   │  │ Hybrid Testset │  │ Your Hybrid │  │ Executive  │
│ TXT, etc.   │  │ Loader +        │  │ Generator:     │  │ Evaluator:  │  │ Summary +  │
│             │  │ RAGAS Processor │  │ • Configurable │  │ • Contextual│  │ • Visualiz.│
│             │  │                 │  │ • RAGAS        │  │   Keywords  │  │ • RAGAS    │
│             │  │                 │  │ • Auto-keyword │  │ • RAGAS     │  │ • Analysis │
└─────────────┘  └─────────────────┘  └────────────────┘  └─────────────┘  └────────────┘
```

## 🔄 **Hybrid Components Integration**

### **1. Document Processing Layer**
```python
# Leverages your existing document_loader.py
DocumentLoader → Enhanced with RAGAS document processing
              → Maintains your chunking and metadata extraction
              → Adds LangChain document format support
```

### **2. Testset Generation Layer**
```python
# Combines your generate_dataset_configurable.py with RAGAS
HybridTestsetGenerator:
├── ConfigurableDatasetGenerator (Your existing system)
│   ├── Auto-keyword extraction (KeyBERT, YAKE, spaCy)
│   ├── Domain-adaptive question generation
│   └── Multi-format document support
│
└── RAGAS TestsetGenerator (Advanced question types)
    ├── Simple questions (factual)
    ├── Multi-context questions (cross-document)
    ├── Conditional questions (reasoning)
    └── Complex reasoning questions
```

### **3. Evaluation Layer**
```python
# Integrates your contextual_keyword_gate.py + dynamic_ragas_gate_with_human_feedback.py
HybridEvaluator:
├── Contextual Keywords (Your system)
│   ├── weighted_keyword_score()
│   ├── get_contextual_segments()
│   ├── Semantic similarity with sentence transformers
│   └── NLP-based segmentation with spaCy
│
├── RAGAS Metrics (Standard evaluation)
│   ├── Faithfulness
│   ├── Answer Relevancy
│   ├── Context Precision/Recall
│   └── Answer Similarity/Correctness
│
└── Human Feedback (Your dynamic system)
    ├── needs_human_feedback_dynamic()
    ├── adaptive_exponential_smoothing()
    ├── Uncertainty-based sampling
    └── Dynamic threshold adjustment
```

## 📊 **Evaluation Flow**

### **Step 1: Input Processing**
```yaml
Document Paths → Document Loader → Processed Documents
    │
    ├── Extract text content
    ├── Identify document structure  
    ├── Generate metadata
    └── Prepare for testset generation
```

### **Step 2: Hybrid Testset Generation**
```yaml
Method Selection (Configuration Parameter):
├── method: "configurable" - Uses your existing system only
├── method: "ragas"        - Uses RAGAS TestsetGenerator only  
└── method: "hybrid"       - Combines both approaches (RECOMMENDED)

Configuration Example:
testset_generation:
  method: "hybrid"                    # Choose generation approach
  samples_per_document: 100
  max_total_samples: 1000
  
  # RAGAS configuration (for 'ragas' and 'hybrid' modes)
  ragas_config:
    use_custom_llm: true             # Enable custom LLM endpoint
    custom_llm:
      endpoint: "http://localhost:8000/v1/chat/completions"
      api_key_file: "config/secrets.yaml"
      api_key_path: "llm.api_key"
      model: "llama2-7b-chat"
      temperature: 0.3

Auto-Keyword Extraction:
├── KeyBERT: Transformer-based keyword extraction
├── YAKE: Statistical keyword extraction
└── spaCy NER: Named entity recognition

Output Format:
├── Excel: testset_hybrid_20240101_120000.xlsx
├── CSV: testset_hybrid_20240101_120000.csv
└── JSON: testset_metadata_20240101_120000.json
```

### **Step 3: RAG System Evaluation**
```yaml
For each Q&A pair:
1. Query RAG System → Get response + contexts + confidence
2. Apply Contextual Keywords → Semantic matching score
3. Apply RAGAS Metrics → Faithfulness + relevancy scores  
4. Calculate Semantic Similarity → Sentence transformer comparison
5. Assess Human Feedback Need → Uncertainty-based sampling
6. Combine Scores → Weighted overall assessment
```

### **Step 4: Report Generation**
```yaml
Executive Summary (HTML):
├── Overall performance metrics
├── Pass rates by evaluation type
├── Performance by source document
└── Actionable recommendations

Technical Analysis (HTML):
├── Statistical analysis of scores
├── Correlation analysis between metrics
├── Threshold optimization suggestions
└── Detailed failure pattern analysis

Visualizations (PNG):
├── Score distribution histograms
├── Pass rate comparison charts
├── Correlation heatmaps
└── Performance trend analysis
```

## 🎯 **Key Advantages of Hybrid Approach**

### **1. Comprehensive Coverage**
- **Your Contextual System**: Ensures domain-specific terminology accuracy
- **RAGAS Metrics**: Provides standardized RAG evaluation benchmarks
- **Combined Approach**: Covers both domain-specific and general quality aspects

### **2. Document Adaptability**
- **Auto-keyword extraction**: No manual keyword configuration needed
- **Multi-format support**: PDF, DOCX, TXT, CSV, Excel
- **Domain agnostic**: Works for any document domain (medical, legal, technical, etc.)

### **3. Advanced Human Feedback**
- **Dynamic thresholds**: Automatically adjusts based on performance trends
- **Uncertainty sampling**: Focuses human attention on most uncertain cases
- **Active learning**: Minimizes human annotation effort while maximizing impact

### **4. Scalable Architecture**
- **Configurable pipeline**: Easy to customize for different use cases
- **Batch processing**: Handles multiple documents efficiently
- **Parallel evaluation**: Supports concurrent RAG system queries

## 🔧 **Configuration Examples**

### **Example 1: Pure Contextual Evaluation (Your System)**
```yaml
# Focus on domain-specific keyword accuracy
testset_generation:
  method: "configurable"
  samples_per_document: 100
  
evaluation:
  methods:
    contextual_keywords: true
    ragas_metrics: false
    semantic_similarity: true
    human_feedback: true
  
  score_weights:
    contextual: 0.6
    semantic: 0.4
```

### **Example 2: RAGAS-Focused Evaluation**
```yaml
# Focus on standard RAG metrics
testset_generation:
  method: "ragas"
  question_types:
    simple: 0.2
    multi_context: 0.4
    reasoning: 0.4
    
evaluation:
  methods:
    contextual_keywords: false
    ragas_metrics: true
    semantic_similarity: true
    
  score_weights:
    ragas: 0.7
    semantic: 0.3
```

### **Example 3: Full Hybrid Evaluation (RECOMMENDED)**
```yaml
# Best of both approaches
testset_generation:
  method: "hybrid"
  samples_per_document: 100
  
evaluation:
  methods:
    contextual_keywords: true
    ragas_metrics: true
    semantic_similarity: true
    human_feedback: true
    
  score_weights:
    contextual: 0.3
    ragas: 0.4
    semantic: 0.3
    
  pass_criteria: "majority"  # Pass if 2/3 methods pass
```

## 📈 **Performance Optimization**

### **For Large Documents (>1000 pages)**
```yaml
document_processing:
  chunk_size: 2000
  chunk_overlap: 100
  
testset_generation:
  samples_per_document: 50
  max_total_samples: 1000
  
performance:
  max_workers: 8
  batch_processing: true
```

### **For High-Accuracy Domains (Medical, Legal)**
```yaml
evaluation:
  thresholds:
    contextual_threshold: 0.8
    ragas_threshold: 0.8
    semantic_threshold: 0.7
    
  keyword_evaluation:
    weights:
      mandatory: 0.9
      optional: 0.1
```

### **For Development/Testing**
```yaml
testset_generation:
  samples_per_document: 10
  max_total_samples: 50
  
evaluation:
  methods:
    ragas_metrics: false  # Skip LLM-dependent evaluation
```

## 🔄 **Integration with Your Existing Code**

### **Your Code → Hybrid Pipeline Mapping**

| Your Component                              | Hybrid Pipeline Usage  | Enhancement                          |
|---------------------------------------------|------------------------|--------------------------------------|
| `contextual_keyword_gate.py`                | Core evaluation engine | ✅ Direct integration                 |
| `dynamic_ragas_gate_with_human_feedback.py` | Human feedback system  | ✅ Enhanced with uncertainty sampling |
| `generate_dataset_configurable.py`          | Testset generation     | ✅ Combined with RAGAS                |
| `document_loader.py`                        | Document processing    | ✅ Enhanced with LangChain support    |

### **Backwards Compatibility**
- ✅ **All your existing functions work unchanged**
- ✅ **Configuration format extended, not replaced**
- ✅ **Can run in pure "configurable" mode using only your code**
- ✅ **Gradual migration path available**

## 🎯 **Evaluation Criteria Mapping**

### **User Perspective (Phase 1) - IMPLEMENTED**

| Requirement                            | Implementation               | Your Code Used                              |
|----------------------------------------|------------------------------|---------------------------------------------|
| Objective evaluation with automated QA | ✅ Hybrid testset generator   | `generate_dataset_configurable.py`          |
| Subjective evaluation with user QA     | ✅ Custom testset loading     | `document_loader.py`                        |
| Black-box RAG testing                  | ✅ RAG interface layer        | New: `rag_interface.py`                     |
| Auto-keyword extraction                | ✅ Domain-adaptive extraction | `contextual_keyword_gate.py`                |
| Human feedback integration             | ✅ Dynamic adjustment         | `dynamic_ragas_gate_with_human_feedback.py` |

### **Expected Output for Your Redfish Use Case**

```
🎯 Redfish RAG Evaluation Results
=================================
📊 Overall Performance: 85.2%

🔍 Contextual Keywords (Your System):
   • Pass Rate: 88.5%
   • Avg Score: 0.847
   • Domain Keywords Found: redfish, bmc, chassis, manager, session
   • Contextual Segments: 156 relevant segments identified

📈 RAGAS Metrics:
   • Faithfulness: 0.823
   • Answer Relevancy: 0.871  
   • Context Precision: 0.794
   • Overall RAGAS Score: 0.829

🧠 Semantic Similarity:
   • Avg Similarity: 0.792
   • Pass Rate: 84.1%

👥 Human Feedback:
   • Samples Needing Review: 7/100 (7%)
   • Current Threshold: 0.735 (auto-adjusted)
   • Uncertainty Range: 0.3-0.9

📁 Generated Outputs:
   ✅ Testset: testset_redfish_20240101_120000.xlsx (100 Q&A pairs)
   ✅ Executive Summary: reports/redfish_executive_summary.html
   ✅ Technical Analysis: reports/redfish_technical_analysis.html
   ✅ Detailed Results: evaluations/redfish_detailed_results.xlsx
```

## 💻 **On-Premise Operation Guide**

The hybrid approach supports complete on-premise operation for secure environments:

### **Fully On-Premise Configuration**
```yaml
# config/pipeline_config.yaml - On-premise setup
testset_generation:
  method: "configurable"      # Pure on-premise, no external APIs
  samples_per_document: 100
  keyword_extraction:
    methods: ["keybert", "yake", "spacy_entities"]

# OR use hybrid with custom LLM endpoint
testset_generation:
  method: "hybrid"
  ragas_config:
    use_custom_llm: true
    custom_llm:
      endpoint: "http://your-internal-llm:8000/v1/chat/completions"
      api_key_file: "config/secrets.yaml"
      api_key_path: "internal_llm.api_key"
      model: "your-internal-model"

rag_system:
  api_endpoint: "http://internal-rag-system:8000/api/query"

evaluation:
  # All evaluation runs on-premise
  thresholds:
    contextual_threshold: 0.6
    ragas_threshold: 0.7
```

### **Network Isolation Benefits**
- **Data Security**: Documents and queries never leave your network
- **Compliance**: Meets strict data governance requirements
- **Performance**: No external API latency or rate limits
- **Cost Control**: No per-token charges or API costs

### **On-Premise Components**
```yaml
Local Processing:
├── Document Processing: ✅ Full local processing
├── Keyword Extraction: ✅ KeyBERT, YAKE, spaCy (local models)
├── Testset Generation: ✅ Configurable mode (100% local)
├── RAG Evaluation: ✅ Your RAG system (local endpoint)
├── Semantic Similarity: ✅ Sentence transformers (local models)
├── Report Generation: ✅ Full local processing
└── RAGAS Mode: ⚠️ Requires LLM endpoint (can be local)
```

## 🚀 **Quick Start Checklist**

1. **✅ Setup Environment**
   ```bash
   cd eval-pipeline
   python setup.py
   ```

2. **✅ Configure for Your Documents**
   ```yaml
   # config/pipeline_config.yaml
   data_sources:
     documents:
       primary_docs:
         - "../../documents/DSP0266_1.22.0.pdf"  # Your Redfish doc
   ```

3. **✅ Configure Your RAG System**
   ```yaml
   rag_system:
     api_endpoint: "http://your-rag-system:8000/query"
   ```

4. **✅ Run Hybrid Evaluation**
   ```bash
   python run_pipeline.py --config config/pipeline_config.yaml
   ```

5. **✅ Review Results**
   ```bash
   # Open in browser:
   outputs/reports/executive_summary.html
   outputs/reports/technical_analysis.html
   ```

The hybrid approach gives you the **best of both worlds**: your sophisticated domain-specific evaluation capabilities combined with standardized RAGAS metrics and advanced human feedback integration, all in a scalable, configurable pipeline! 🎉
