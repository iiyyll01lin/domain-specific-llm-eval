"""
RAGAS-based Synthetic Dataset Generator
Uses RAGAS library with local or OpenAI models based on configuration
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# RAGAS and related imports
try:
    from ragas import evaluate
    from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    print("⚠️  RAGAS not installed. Install with: pip install ragas")
    RAGAS_AVAILABLE = False

try:
    from keybert import KeyBERT
    KEYBERT_AVAILABLE = True
except ImportError:
    print("⚠️  KeyBERT not installed. Install with: pip install keybert")
    KEYBERT_AVAILABLE = False

try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    print("⚠️  YAKE not installed. Install with: pip install yake")
    YAKE_AVAILABLE = False

# For local LLM support
try:
    from langchain_community.llms import HuggingFacePipeline
    from transformers import pipeline
    import torch
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    print("⚠️  Local LLM support not available. Install with: pip install transformers torch langchain-community")
    LOCAL_LLM_AVAILABLE = False

class RAGASSyntheticDatasetGenerator:
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        
        # Initialize keyword extractors
        if KEYBERT_AVAILABLE and config.get('local', {}).get('keybert', {}).get('enabled', True):
            self.kw_model = KeyBERT()
        else:
            self.kw_model = None
            
        if YAKE_AVAILABLE and config.get('local', {}).get('yake', {}).get('enabled', True):
            yake_config = config.get('local', {}).get('yake', {})
            self.yake_extractor = yake.KeywordExtractor(
                lan=yake_config.get('language', 'en'),
                n=yake_config.get('n', 3),
                dedupLim=yake_config.get('dedupLim', 0.7),
                top=yake_config.get('top', 10)
            )
        else:
            self.yake_extractor = None
        
        # Initialize LLM for RAGAS if needed
        self.local_llm = self._setup_local_llm() if (
            RAGAS_AVAILABLE and 
            config.get('ragas', {}).get('use_local_llm', True) and
            LOCAL_LLM_AVAILABLE
        ) else None
        
        # Domain-specific document corpus
        self.documents = [
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

    def _setup_local_llm(self):
        """Setup local LLM for RAGAS evaluation"""
        if not LOCAL_LLM_AVAILABLE:
            return None
            
        try:
            local_llm_config = self.config.get('ragas', {}).get('local_llm', {})
            model_name = local_llm_config.get('model_name', 'microsoft/DialoGPT-small')
            max_length = local_llm_config.get('max_length', 512)
            device = local_llm_config.get('device', 'auto')
            
            if self.config.get('logging', {}).get('show_progress', True):
                print(f"🤖 Setting up local LLM: {model_name}")
            
            # Determine device
            if device == 'auto':
                device_map = "auto" if torch.cuda.is_available() else None
            else:
                device_map = device if device != 'cpu' else None
            
            # Create HuggingFace pipeline
            hf_pipeline = pipeline(
                "text-generation",
                model=model_name,
                tokenizer=model_name,
                max_length=max_length,
                device_map=device_map,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            
            # Wrap for LangChain/RAGAS
            llm = HuggingFacePipeline(pipeline=hf_pipeline)
            
            if self.config.get('logging', {}).get('show_progress', True):
                print("✅ Local LLM setup successful")
            
            return llm
            
        except Exception as e:
            print(f"⚠️  Failed to setup local LLM: {e}")
            return None

    def extract_keywords_keybert(self, text: str, n_keywords: int = 5) -> List[str]:
        """Extract keywords using KeyBERT"""
        if not self.kw_model:
            return ["keyword1", "keyword2", "keyword3"]  # Fallback
        
        try:
            keybert_config = self.config.get('local', {}).get('keybert', {})
            keywords = self.kw_model.extract_keywords(
                text, 
                keyphrase_ngram_range=tuple(keybert_config.get('keyphrase_ngram_range', [1, 2])),
                stop_words=keybert_config.get('stop_words', 'english')
            )
            max_keywords = keybert_config.get('max_keywords', 5)
            return [kw[0] for kw in keywords[:max_keywords]]
        except Exception as e:
            print(f"⚠️  KeyBERT extraction failed: {e}")
            # Fallback to simple keyword extraction
            words = text.lower().split()
            return [w for w in words if len(w) > 4][:n_keywords]

    def extract_keywords_yake(self, text: str) -> List[str]:
        """Extract keywords using YAKE"""
        if not self.yake_extractor:
            return ["keyword1", "keyword2", "keyword3"]  # Fallback
        
        try:
            keywords = self.yake_extractor.extract_keywords(text)
            max_keywords = self.config.get('local', {}).get('yake', {}).get('max_keywords', 5)
            return [kw[1] for kw in keywords[:max_keywords]]
        except Exception as e:
            print(f"⚠️  YAKE extraction failed: {e}")
            # Fallback to simple keyword extraction
            words = text.lower().split()
            return [w for w in words if len(w) > 4][:5]

    def generate_qa_pairs_from_documents(self, num_pairs: int = 10) -> List[Dict]:
        """Generate QA pairs from document corpus"""
        qa_pairs = []
        question_templates = [
            "What is {}?", "How is {} diagnosed?", "What causes {}?",
            "What are the symptoms of {}?", "How is {} treated?",
            "What are the risk factors for {}?", "How does {} work?",
            "What is the mechanism of {}?", "How do you manage {}?",
            "What are complications of {}?"
        ]
        
        topics = [
            "diabetes", "hypertension", "pneumonia", "heart disease", "asthma",
            "machine learning", "cloud computing", "cybersecurity", "blockchain", "API integration"
        ]
        
        # Get quality distribution from config
        quality_dist = self.config.get('dataset', {}).get('answer_quality_distribution', {})
        high_prob = quality_dist.get('high', 0.4)
        medium_prob = quality_dist.get('medium', 0.4)
        low_prob = quality_dist.get('low', 0.2)
        
        for i in range(num_pairs):
            topic = topics[i % len(topics)]
            question = question_templates[i % len(question_templates)].format(topic)
            context = self.documents[i % len(self.documents)]
            
            ground_truth = f"This relates to {topic} and involves multiple factors including clinical presentation, diagnostic criteria, and treatment protocols."
            
            # Generate LLM answer with varying quality
            answer_quality = np.random.choice(['high', 'medium', 'low'], p=[high_prob, medium_prob, low_prob])
            
            if answer_quality == 'high':
                answer = f"Regarding {topic}, the key aspects include {context[:100]}... This involves comprehensive evaluation and evidence-based approaches."
            elif answer_quality == 'medium':
                answer = f"For {topic}, there are several important considerations. {context[:60]}... Additional factors may apply."
            else:
                answer = f"This is about {topic}. There are various aspects to consider."
            
            qa_pairs.append({
                'question': question, 'contexts': context,
                'answer': answer, 'ground_truth': ground_truth
            })
        
        return qa_pairs

    def calculate_ragas_metrics_real(self, qa_data: List[Dict]) -> pd.DataFrame:
        """Calculate real RAGAS metrics using RAGAS library"""
        if not RAGAS_AVAILABLE:
            if self.config.get('logging', {}).get('show_progress', True):
                print("🔄 RAGAS not available, using simulated metrics...")
            return self.simulate_ragas_metrics(qa_data)
        
        try:
            if self.config.get('logging', {}).get('show_progress', True):
                print("📊 Calculating RAGAS metrics...")
            
            # Convert to RAGAS format - ensure correct field names
            dataset_dict = {
                'question': [item['question'] for item in qa_data],
                'contexts': [[item['contexts']] for item in qa_data],  # RAGAS expects list of contexts
                'answer': [item['answer'] for item in qa_data],
                'ground_truth': [item['ground_truth'] for item in qa_data]
            }
            
            # Create dataset with validation
            dataset = Dataset.from_dict(dataset_dict)
            
            # Configure metrics to use local LLM if available
            metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
            
            if self.local_llm:
                if self.config.get('logging', {}).get('show_progress', True):
                    print("🤖 Using local LLM for RAGAS evaluation...")
                # Configure RAGAS metrics to use local LLM
                for metric in metrics:
                    if hasattr(metric, 'llm'):
                        metric.llm = self.local_llm
            else:
                if self.config.get('logging', {}).get('show_progress', True):
                    print("🌐 Using default RAGAS configuration...")
            
            # Evaluate with RAGAS
            result = evaluate(dataset, metrics=metrics)
            
            return result.to_pandas()
            
        except Exception as e:
            print(f"⚠️  RAGAS evaluation failed: {e}")
            if self.config.get('logging', {}).get('debug', False):
                import traceback
                print(f"Debug traceback: {traceback.format_exc()}")
            return self.simulate_ragas_metrics(qa_data)

    def simulate_ragas_metrics(self, qa_data: List[Dict]) -> pd.DataFrame:
        """Fallback: simulate RAGAS metrics"""
        if self.config.get('logging', {}).get('show_progress', True):
            print("🎲 Using simulated RAGAS metrics...")
            
        metrics_data = []
        for item in qa_data:
            # Simulate correlated metrics based on answer quality
            base_quality = 0.6 + 0.3 * (len(item['answer']) / max(100, len(item['answer'])))
            noise = 0.1
            
            fallback_config = self.config.get('fallback', {}).get('score_ranges', {})
            
            metrics = {
                'context_precision': max(0.0, min(1.0, base_quality + np.random.uniform(-noise, noise))),
                'context_recall': max(0.0, min(1.0, base_quality + np.random.uniform(-noise, noise))),
                'faithfulness': max(0.0, min(1.0, base_quality + np.random.uniform(-noise, noise))),
                'answer_relevancy': max(0.0, min(1.0, base_quality + np.random.uniform(-noise, noise)))
            }
            metrics_data.append(metrics)
        
        return pd.DataFrame(metrics_data)

    def generate_ragas_dataset(self, num_samples: int = 10) -> pd.DataFrame:
        """Generate dataset using RAGAS methods"""
        if self.config.get('logging', {}).get('show_progress', True):
            print("🔬 Generating dataset using RAGAS methods...")
        
        # Step 1: Generate QA pairs
        if self.config.get('logging', {}).get('show_progress', True):
            print("📝 Generating QA pairs from document corpus...")
        qa_data = self.generate_qa_pairs_from_documents(num_samples)
        
        # Step 2: Extract keywords
        if self.config.get('logging', {}).get('show_progress', True):
            print("🔍 Extracting keywords using advanced methods...")
        for item in qa_data:
            if KEYBERT_AVAILABLE and self.kw_model:
                keywords = self.extract_keywords_keybert(item['answer'])
                item['kw'] = str(keywords)
            elif YAKE_AVAILABLE and self.yake_extractor:
                keywords = self.extract_keywords_yake(item['answer'])
                item['kw'] = str(keywords)
            else:
                # Fallback to simple extraction
                words = item['answer'].lower().split()
                keywords = [w for w in words if len(w) > 4][:5]
                item['kw'] = str(keywords)
        
        # Step 3: Calculate RAGAS metrics
        ragas_df = self.calculate_ragas_metrics_real(qa_data)
        
        # Step 4: Combine everything
        if self.config.get('logging', {}).get('show_progress', True):
            print("🔧 Combining data into final format...")
        final_data = []
        
        for i, item in enumerate(qa_data):
            kw_score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('kw_metric', [0.5, 0.9])
            
            row = {
                'question': item['question'],
                'contexts': item['contexts'],
                'answer': item['answer'], 
                'ground_truth': item['ground_truth'],
                'context_precision': round(ragas_df.iloc[i]['context_precision'], 3),
                'context_recall': round(ragas_df.iloc[i]['context_recall'], 3),
                'faithfulness': round(ragas_df.iloc[i]['faithfulness'], 3),
                'answer_relevancy': round(ragas_df.iloc[i]['answer_relevancy'], 3),
                'kw': item['kw'],
                'kw_metric': round(np.random.uniform(kw_score_range[0], kw_score_range[1]), 3),
                'weighted_average_score': round(ragas_df.iloc[i][['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']].mean(), 3)
            }
            final_data.append(row)
        
        return pd.DataFrame(final_data)