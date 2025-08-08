# Configurable Synthetic Dataset Generator

This enhanced version provides **both local and RAGAS approaches** with easy configuration switching. No more choosing between approaches - use what works best for your needs!

## 🚀 Quick Start

### 1. Install Dependencies
```bash
python install_dependencies.py
```
Choose your preferred mode:
- **LOCAL mode** (recommended): No API keys, fast, runs completely offline
- **RAGAS mode**: Professional-grade metrics, optional local LLM support  
- **Basic mode**: Minimal dependencies for testing

### 2. Configure Your Mode
Edit `config.yaml`:
```yaml
# Choose: 'local', 'ragas', or 'simple'
mode: 'local'  

# LOCAL mode settings (no API keys required)
local:
  sentence_model: 'all-MiniLM-L6-v2'  # Fast model
  keybert:
    enabled: true
  yake:
    enabled: true

# RAGAS mode settings  
ragas:
  use_local_llm: true  # Set false to use OpenAI

# Dataset settings
dataset:
  num_samples: 10
  output_file: 'SystemQAListallQuestion_eval_step4_final_report 1.xlsx'
```

### 3. Generate Dataset
```bash
python generate_dataset_configurable.py
```

### 4. Run Evaluation
```bash
python contextual_keyword_gate.py
python dynamic_ragas_gate_with_human_feedback.py  
```

## 🏠 LOCAL Mode (Recommended)

**Perfect for: Privacy-conscious users, offline work, no API costs**

✅ **Advantages:**
- No API keys required
- Runs completely offline
- Fast execution
- Privacy-friendly
- Cost-effective

🔧 **How it works:**
- Uses sentence transformers for semantic similarity
- KeyBERT/YAKE for keyword extraction
- Custom RAGAS-like metric implementations
- Local processing only

📦 **Dependencies:**
```bash
pip install sentence-transformers scikit-learn keybert yake pandas numpy openpyxl spacy
python -m spacy download en_core_web_sm
```

## 🔬 RAGAS Mode

**Perfect for: Research, professional evaluation, when you need "official" RAGAS metrics**

✅ **Advantages:**
- Real RAGAS metrics
- Can use local LLMs (no API needed) or OpenAI
- Professional-grade evaluation
- Research-ready results

⚙️ **Configuration Options:**

**Option A: RAGAS with Local LLM (No API Key)**
```yaml
mode: 'ragas'
ragas:
  use_local_llm: true
  local_llm:
    model_name: 'microsoft/DialoGPT-small'  # Fast
    # model_name: 'microsoft/DialoGPT-medium'  # Better quality
```

**Option B: RAGAS with OpenAI**
```yaml
mode: 'ragas' 
ragas:
  use_local_llm: false
  openai:
    api_key: 'your-api-key-here'  # or set OPENAI_API_KEY env var
```

📦 **Dependencies:**
```bash
pip install ragas datasets transformers torch langchain-community keybert yake pandas numpy openpyxl
```

## 📊 Mode Comparison

| Feature          | LOCAL Mode    | RAGAS Mode       | Simple Mode   |
|------------------|---------------|------------------|---------------|
| **API Keys**     | ❌ None needed | ⚠️ Optional      | ❌ None needed |
| **Speed**        | ⚡ Fast        | 🐌 Slower        | ⚡ Very fast   |
| **Quality**      | ✅ Good        | ✅ Excellent      | ⚠️ Basic      |
| **Privacy**      | ✅ Complete    | ✅ With local LLM | ✅ Complete    |
| **Dependencies** | 📦 Moderate   | 📦 Heavy         | 📦 Minimal    |
| **Offline**      | ✅ Yes         | ✅ With local LLM | ✅ Yes         |

## 🔧 Configuration Options

### Dataset Settings
```yaml
dataset:
  num_samples: 10  # Number of samples to generate
  output_file: 'your-file.xlsx'  # Output filename
  
  # Domains to include
  domains:
    medical: true
    technology: true
    
  # Quality distribution for synthetic answers
  answer_quality_distribution:
    high: 0.4    # 40% high quality
    medium: 0.4  # 40% medium quality  
    low: 0.2     # 20% low quality
```

### Local Mode Fine-tuning
```yaml
local:
  # Sentence transformer model
  sentence_model: 'all-MiniLM-L6-v2'      # Fast, lightweight
  # sentence_model: 'all-mpnet-base-v2'   # Better quality, slower
  
  # Keyword extraction
  keybert:
    enabled: true
    keyphrase_ngram_range: [1, 2]  # Single words and 2-word phrases
    max_keywords: 5
    
  yake:
    enabled: true
    language: 'en'
    max_keywords: 5
    
  # Similarity thresholds
  thresholds:
    relevance_threshold: 0.3  # Context relevance cutoff
    similarity_min: 0.1       # Minimum similarity score
    similarity_max: 1.0       # Maximum similarity score
```

### RAGAS Mode Fine-tuning
```yaml
ragas:
  # Local LLM settings
  local_llm:
    model_name: 'microsoft/DialoGPT-small'  # Choose your model
    max_length: 512                         # Token limit
    device: 'auto'                          # 'auto', 'cpu', or 'cuda'
```

### Logging & Debug
```yaml
logging:
  level: 'INFO'        # DEBUG, INFO, WARNING, ERROR
  show_progress: true  # Show progress messages
```

## 🎯 Use Cases

### Research & Development
- **Use RAGAS mode** for official metrics
- Local LLM for privacy + quality balance
- Full configuration control

### Production & Privacy
- **Use LOCAL mode** for complete offline processing
- No external API dependencies
- Fast, consistent results

### Quick Testing
- **Use simple mode** for rapid prototyping
- Minimal setup required
- Good for initial testing

## 🔄 Migration from Original Scripts

### From `generate_synthetic_dataset_advanced.py`:
```yaml
mode: 'ragas'
ragas:
  use_local_llm: true  # Matches your original local approach
```

### From `generate_synthetic_dataset.py`:
```yaml
mode: 'simple'  # Uses your original simple approach
```

### New Enhanced Version:
```yaml
mode: 'local'  # Best of both worlds - sophisticated but local
```

## 🐛 Troubleshooting

### Common Issues

**"Config file not found"**
- The script will create a default config automatically
- Or run: `cp config.yaml.example config.yaml`

**"Dependencies missing"**
- Run: `python install_dependencies.py`
- Choose your preferred mode for targeted installation

**"RAGAS evaluation failed"** 
- Falls back to simulated metrics automatically
- Check your local LLM configuration or API keys

**"KeyBERT extraction failed"**
- Falls back to simple keyword extraction
- Check if KeyBERT is properly installed

### Performance Tips

**Speed up LOCAL mode:**
- Use `all-MiniLM-L6-v2` (default) instead of larger models
- Set `keybert.enabled: false` if keyword extraction is slow

**Speed up RAGAS mode:**
- Use `microsoft/DialoGPT-small` for local LLM
- Set `max_length: 256` for faster processing

**Reduce memory usage:**
- Use smaller sentence transformer models
- Set `device: 'cpu'` if CUDA memory issues

## 📈 Next Steps

1. **Generate your dataset** using your preferred mode
2. **Run the evaluation pipeline**:
   ```bash
   python contextual_keyword_gate.py
   python dynamic_ragas_gate_with_human_feedback.py
   ```
3. **Analyze results** from the generated reports
4. **Fine-tune configuration** based on your needs

## 🤝 Contributing

The configurable system makes it easy to add new modes:
1. Create a new generator class (e.g., `MyCustomGenerator`)
2. Add mode to `config.yaml` 
3. Update `generate_dataset_configurable.py` to use your generator

## 📝 License

Same as the original project - use responsibly for research and evaluation purposes.

---

**🎉 You now have both local AND RAGAS approaches available with easy configuration switching!**