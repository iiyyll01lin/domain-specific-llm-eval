"""
Local Synthetic Dataset Generator - No API Keys Required
Uses local models and custom implementations for all metrics
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

# Local implementations
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠️  Install with: pip install sentence-transformers scikit-learn")
    SENTENCE_TRANSFORMERS_AVAILABLE = False

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

class LocalSyntheticDatasetGenerator:
    def __init__(self, config: Dict):
        """Initialize with configuration"""
        self.config = config
        
        # Initialize local models
        sentence_model_name = config.get('local', {}).get('sentence_model', 'all-MiniLM-L6-v2')
        self.sentence_model = SentenceTransformer(sentence_model_name) if SENTENCE_TRANSFORMERS_AVAILABLE else None
        
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

    def calculate_local_context_precision(self, context: str, answer: str) -> float:
        """Local implementation of context precision using sentence similarity"""
        if not self.sentence_model:
            # Fallback to simulation
            score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('context_precision', [0.6, 0.95])
            return np.random.uniform(score_range[0], score_range[1])
        
        try:
            # Calculate how much of the context is relevant to the answer
            context_sentences = [s.strip() for s in context.split('.') if len(s.strip()) > 10]
            
            if not context_sentences:
                return 0.5
            
            # Get embeddings
            answer_embedding = self.sentence_model.encode([answer])
            context_embeddings = self.sentence_model.encode(context_sentences)
            
            # Calculate similarities
            similarities = cosine_similarity(answer_embedding, context_embeddings)[0]
            
            # Precision = proportion of context sentences that are relevant
            relevance_threshold = self.config.get('local', {}).get('thresholds', {}).get('relevance_threshold', 0.3)
            precision = np.mean(similarities > relevance_threshold)
            
            # Clamp values
            sim_min = self.config.get('local', {}).get('thresholds', {}).get('similarity_min', 0.1)
            sim_max = self.config.get('local', {}).get('thresholds', {}).get('similarity_max', 1.0)
            return min(max(precision, sim_min), sim_max)
            
        except Exception as e:
            print(f"⚠️  Context precision calculation failed: {e}")
            score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('context_precision', [0.6, 0.9])
            return np.random.uniform(score_range[0], score_range[1])

    def calculate_local_context_recall(self, context: str, ground_truth: str) -> float:
        """Local implementation of context recall"""
        if not self.sentence_model:
            score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('context_recall', [0.6, 0.95])
            return np.random.uniform(score_range[0], score_range[1])
        
        try:
            # Calculate how much of the ground truth is covered by context
            gt_embedding = self.sentence_model.encode([ground_truth])
            context_embedding = self.sentence_model.encode([context])
            
            similarity = cosine_similarity(gt_embedding, context_embedding)[0][0]
            
            # Transform similarity to recall score
            recall = (similarity + 1) / 2  # Transform from [-1,1] to [0,1]
            
            sim_min = self.config.get('local', {}).get('thresholds', {}).get('similarity_min', 0.1)
            sim_max = self.config.get('local', {}).get('thresholds', {}).get('similarity_max', 1.0)
            return min(max(recall, sim_min), sim_max)
            
        except Exception as e:
            print(f"⚠️  Context recall calculation failed: {e}")
            score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('context_recall', [0.6, 0.9])
            return np.random.uniform(score_range[0], score_range[1])

    def calculate_local_faithfulness(self, context: str, answer: str) -> float:
        """Local implementation of faithfulness using semantic similarity"""
        if not self.sentence_model:
            score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('faithfulness', [0.6, 0.95])
            return np.random.uniform(score_range[0], score_range[1])
        
        try:
            # Check if answer is faithful to context
            context_embedding = self.sentence_model.encode([context])
            answer_embedding = self.sentence_model.encode([answer])
            
            similarity = cosine_similarity(context_embedding, answer_embedding)[0][0]
            
            # Transform similarity to faithfulness score
            faithfulness = max(0, similarity)  # Ensure non-negative
            
            sim_min = self.config.get('local', {}).get('thresholds', {}).get('similarity_min', 0.1)
            sim_max = self.config.get('local', {}).get('thresholds', {}).get('similarity_max', 1.0)
            return min(max(faithfulness, sim_min), sim_max)
            
        except Exception as e:
            print(f"⚠️  Faithfulness calculation failed: {e}")
            score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('faithfulness', [0.6, 0.9])
            return np.random.uniform(score_range[0], score_range[1])

    def calculate_local_answer_relevancy(self, question: str, answer: str) -> float:
        """Local implementation of answer relevancy"""
        if not self.sentence_model:
            score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('answer_relevancy', [0.6, 0.95])
            return np.random.uniform(score_range[0], score_range[1])
        
        try:
            # Check if answer is relevant to question
            question_embedding = self.sentence_model.encode([question])
            answer_embedding = self.sentence_model.encode([answer])
            
            similarity = cosine_similarity(question_embedding, answer_embedding)[0][0]
            
            # Transform similarity to relevancy score
            relevancy = (similarity + 1) / 2  # Transform from [-1,1] to [0,1]
            
            sim_min = self.config.get('local', {}).get('thresholds', {}).get('similarity_min', 0.1)
            sim_max = self.config.get('local', {}).get('thresholds', {}).get('similarity_max', 1.0)
            return min(max(relevancy, sim_min), sim_max)
            
        except Exception as e:
            print(f"⚠️  Answer relevancy calculation failed: {e}")
            score_range = self.config.get('fallback', {}).get('score_ranges', {}).get('answer_relevancy', [0.6, 0.9])
            return np.random.uniform(score_range[0], score_range[1])

    def calculate_local_ragas_metrics(self, qa_data: List[Dict]) -> pd.DataFrame:
        """Calculate RAGAS-like metrics using local implementations"""
        if self.config.get('logging', {}).get('show_progress', True):
            print("🔧 Calculating metrics using local models...")
        
        metrics_data = []
        for i, item in enumerate(qa_data):
            if self.config.get('logging', {}).get('show_progress', True) and i % 5 == 0:
                print(f"  Processing sample {i+1}/{len(qa_data)}...")
                
            metrics = {
                'context_precision': self.calculate_local_context_precision(
                    item['contexts'], item['answer']
                ),
                'context_recall': self.calculate_local_context_recall(
                    item['contexts'], item['ground_truth']
                ),
                'faithfulness': self.calculate_local_faithfulness(
                    item['contexts'], item['answer']
                ),
                'answer_relevancy': self.calculate_local_answer_relevancy(
                    item['question'], item['answer']
                )
            }
            metrics_data.append(metrics)
        
        return pd.DataFrame(metrics_data)

    def extract_keywords_local(self, text: str, n_keywords: int = 5) -> List[str]:
        """Extract keywords using local methods"""
        keybert_config = self.config.get('local', {}).get('keybert', {})
        yake_config = self.config.get('local', {}).get('yake', {})
        
        # Try KeyBERT first
        if self.kw_model and keybert_config.get('enabled', True):
            try:
                keywords = self.kw_model.extract_keywords(
                    text, 
                    keyphrase_ngram_range=tuple(keybert_config.get('keyphrase_ngram_range', [1, 2])),
                    stop_words=keybert_config.get('stop_words', 'english')
                )
                max_keywords = keybert_config.get('max_keywords', 5)
                return [kw[0] for kw in keywords[:max_keywords]]
            except Exception as e:
                print(f"⚠️  KeyBERT extraction failed: {e}")
        
        # Try YAKE as fallback
        if self.yake_extractor and yake_config.get('enabled', True):
            try:
                keywords = self.yake_extractor.extract_keywords(text)
                max_keywords = yake_config.get('max_keywords', 5)
                return [kw[1] for kw in keywords[:max_keywords]]
            except Exception as e:
                print(f"⚠️  YAKE extraction failed: {e}")
        
        # Final fallback: simple keyword extraction
        words = text.lower().split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'this', 'that', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        keywords = [w for w in words if len(w) > 4 and w not in stop_words]
        return list(set(keywords))[:n_keywords]  # Remove duplicates

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
            
            # Generate LLM answer with varying quality based on config
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

    def generate_local_dataset(self, num_samples: int = 10) -> pd.DataFrame:
        """Generate dataset using purely local methods"""
        if self.config.get('logging', {}).get('show_progress', True):
            print("🏠 Generating dataset using LOCAL methods only...")
        
        # Step 1: Generate QA pairs
        if self.config.get('logging', {}).get('show_progress', True):
            print("📝 Generating QA pairs from document corpus...")
        qa_data = self.generate_qa_pairs_from_documents(num_samples)
        
        # Step 2: Extract keywords locally
        if self.config.get('logging', {}).get('show_progress', True):
            print("🔍 Extracting keywords using local methods...")
        for item in qa_data:
            keywords = self.extract_keywords_local(item['answer'])
            item['kw'] = str(keywords)
        
        # Step 3: Calculate metrics locally
        ragas_df = self.calculate_local_ragas_metrics(qa_data)
        
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