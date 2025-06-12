"""
Unified Synthetic Dataset Generator with Configuration Support
Supports both local and RAGAS modes based on config.yaml settings
Enhanced with custom document loading from PDFs and other formats
"""

import pandas as pd
import numpy as np
import yaml
import os
import logging
from typing import Dict, Optional

# Import custom document loader
from document_loader import DocumentLoader

# Import generators
from local_dataset_generator import LocalSyntheticDatasetGenerator
from ragas_dataset_generator import RAGASSyntheticDatasetGenerator

# Simple fallback generator
try:
    from generate_synthetic_dataset import SyntheticDatasetGenerator
except ImportError:
    try:
        # Use the simple generator file that exists  
        exec(open('generate_synthetic_dataset-simple.py').read())
        # Create a fallback class
        class SimpleFallbackGenerator:
            def generate_dataset(self, num_samples=10):
                # Simple fallback implementation
                import random
                data = []
                for i in range(num_samples):
                    data.append({
                        'question': f"Sample question {i+1}",
                        'contexts': f"Sample context {i+1}",  
                        'answer': f"Sample answer {i+1}",
                        'ground_truth': f"Sample ground truth {i+1}",
                        'context_precision': round(random.uniform(0.6, 0.9), 3),
                        'context_recall': round(random.uniform(0.6, 0.9), 3),
                        'faithfulness': round(random.uniform(0.6, 0.9), 3),
                        'answer_relevancy': round(random.uniform(0.6, 0.9), 3),
                        'kw': "['keyword1', 'keyword2']",
                        'kw_metric': round(random.uniform(0.5, 0.9), 3),
                        'weighted_average_score': round(random.uniform(0.6, 0.9), 3)
                    })
                return pd.DataFrame(data)
        SyntheticDatasetGenerator = SimpleFallbackGenerator
    except Exception:
        # Final fallback
        SyntheticDatasetGenerator = None

class ConfigurableDatasetGenerator:
    """Unified dataset generator that switches between local and RAGAS modes"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize with configuration file"""
        self.config = self.load_config(config_path)
        self.mode = self.config.get('mode', 'local').lower()
        
        # Initialize document loader for custom data
        self.document_loader = None
        self.custom_documents = []
        self.custom_topics = []
        
        # Load custom documents if enabled
        if self.config.get('custom_data', {}).get('enabled', False):
            print("🔍 Custom data enabled - loading documents...")
            try:
                self.document_loader = DocumentLoader(self.config)
                self.custom_documents, self.custom_metadata = self.document_loader.load_all_documents()
                self.custom_topics = self.document_loader.get_topics_from_metadata()
                self.document_loader.print_loading_summary()
            except Exception as e:
                print(f"⚠️  Failed to load custom documents: {e}")
                print("🔄 Continuing with default documents...")
        
        # Initialize appropriate generator based on mode
        if self.mode == 'local':
            self.generator = LocalSyntheticDatasetGenerator(self.config)
            # Inject custom data if available
            if self.custom_documents:
                self.generator.documents = self.custom_documents
        elif self.mode == 'ragas':
            self.generator = RAGASSyntheticDatasetGenerator(self.config)
            # Inject custom data if available
            if self.custom_documents:
                self.generator.documents = self.custom_documents
        elif self.mode == 'simple':
            if SyntheticDatasetGenerator is None:
                print("⚠️  Simple generator not available, falling back to local mode")
                self.mode = 'local'
                self.generator = LocalSyntheticDatasetGenerator(self.config)
            else:
                self.generator = SyntheticDatasetGenerator()
        else:
            raise ValueError(f"Unknown mode: {self.mode}. Use 'local', 'ragas', or 'simple'")
        
        if self.config.get('logging', {}).get('show_progress', True):
            custom_status = f" with {len(self.custom_documents)} custom documents" if self.custom_documents else ""
            print(f"🎯 Initialized in '{self.mode.upper()}' mode{custom_status}")
    
    def generate_qa_pairs_with_custom_data(self, num_pairs: int = 10):
        """Generate QA pairs using custom documents and topics"""
        qa_pairs = []
        
        # Use custom question templates from config
        custom_config = self.config.get('custom_data', {})
        question_templates = custom_config.get('question_templates', [
            "What is {} according to the document?",
            "How does the document explain {}?",
            "What are the key aspects of {} mentioned?",
            "Can you summarize the information about {}?",
            "What does the document say about {}?"
        ])
        
        # Use custom topics or fallback
        topics = self.custom_topics if self.custom_topics else [
            "concept", "method", "approach", "technique", "strategy"
        ]
        
        # Get quality distribution from config
        quality_dist = self.config.get('dataset', {}).get('answer_quality_distribution', {})
        high_prob = quality_dist.get('high', 0.4)
        medium_prob = quality_dist.get('medium', 0.4)
        low_prob = quality_dist.get('low', 0.2)
        
        for i in range(num_pairs):
            topic = topics[i % len(topics)]
            question = question_templates[i % len(question_templates)].format(topic)
            context = self.custom_documents[i % len(self.custom_documents)]
            
            # Generate ground truth based on the document
            ground_truth = f"Based on the document, {topic} involves the following aspects: {context[:150]}... This information provides comprehensive understanding of the concept."
            
            # Generate answer with varying quality
            answer_quality = np.random.choice(
                ['high', 'medium', 'low'], 
                p=[high_prob, medium_prob, low_prob]
            )
            
            if answer_quality == 'high':
                answer = f"According to the document, {topic} is comprehensively explained as: {context[:200]}... This detailed explanation covers all key aspects mentioned in the source material."
            elif answer_quality == 'medium':
                answer = f"The document mentions that {topic} involves {context[:120]}... Additional considerations may apply based on the context."
            else:
                answer = f"Regarding {topic}, the document provides some information. There are various aspects to consider."
            
            qa_pairs.append({
                'question': question,
                'contexts': context,
                'answer': answer,
                'ground_truth': ground_truth
            })
        
        return qa_pairs

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        if not os.path.exists(config_path):
            print(f"⚠️  Config file {config_path} not found. Using default configuration.")
            return self.get_default_config()
        
        try:
            with open(config_path, 'r') as file:
                config = yaml.safe_load(file)
                if self.get_logging_level(config) in ['DEBUG', 'INFO']:
                    print(f"✅ Loaded configuration from {config_path}")
                return config
        except Exception as e:
            print(f"⚠️  Failed to load config: {e}. Using default configuration.")
            return self.get_default_config()

    def get_default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'mode': 'local',
            'dataset': {
                'num_samples': 10,
                'output_file': 'SystemQAListallQuestion_eval_step4_final_report 1.xlsx'
            },
            'local': {
                'sentence_model': 'all-MiniLM-L6-v2',
                'keybert': {'enabled': True},
                'yake': {'enabled': True}
            },
            'ragas': {
                'use_local_llm': True
            },
            'fallback': {
                'use_simulation': True
            },            'logging': {
                'level': 'INFO',
                'show_progress': True
            }
        }

    def get_logging_level(self, config: Dict) -> str:
        """Get logging level from config"""
        return config.get('logging', {}).get('level', 'INFO')

    def generate_dataset(self, num_samples: Optional[int] = None) -> pd.DataFrame:
        """Generate dataset using the configured mode"""
        # Use config value if num_samples not provided
        if num_samples is None:
            num_samples = self.config.get('dataset', {}).get('num_samples', 10)
        
        if self.config.get('logging', {}).get('show_progress', True):
            print(f"🚀 Generating {num_samples} samples using {self.mode.upper()} mode...")
        
        # If we have custom documents, use them to override generator's QA generation
        if self.custom_documents and hasattr(self.generator, 'generate_qa_pairs_from_documents'):
            # Replace the generator's QA generation with our custom version
            original_method = self.generator.generate_qa_pairs_from_documents
            self.generator.generate_qa_pairs_from_documents = lambda n: self.generate_qa_pairs_with_custom_data(n)
        
        # Generate dataset based on mode
        if self.mode == 'local':
            return self.generator.generate_local_dataset(num_samples)
        elif self.mode == 'ragas':
            return self.generator.generate_ragas_dataset(num_samples)
        elif self.mode == 'simple':
            # For simple mode, generate directly if we have custom data
            if self.custom_documents:
                return self.generate_simple_dataset_with_custom_data(num_samples)
            else:
                return self.generator.generate_dataset(num_samples)
        else:
            raise ValueError(f"Unknown mode: {self.mode}")
    
    def generate_simple_dataset_with_custom_data(self, num_samples: int) -> pd.DataFrame:
        """Generate simple dataset using custom documents"""
        qa_data = self.generate_qa_pairs_with_custom_data(num_samples)
        
        # Convert to DataFrame with required columns
        data = []
        for qa in qa_data:
            # Simple keyword extraction from answer
            words = qa['answer'].lower().split()
            keywords = [w for w in words if len(w) > 4][:5]
            
            data.append({
                'question': qa['question'],
                'contexts': qa['contexts'],
                'answer': qa['answer'],
                'ground_truth': qa['ground_truth'],
                'context_precision': round(np.random.uniform(0.6, 0.9), 3),
                'context_recall': round(np.random.uniform(0.6, 0.9), 3),
                'faithfulness': round(np.random.uniform(0.6, 0.9), 3),
                'answer_relevancy': round(np.random.uniform(0.6, 0.9), 3),
                'kw': str(keywords),
                'kw_metric': round(np.random.uniform(0.5, 0.9), 3),
                'weighted_average_score': round(np.random.uniform(0.6, 0.9), 3)
            })
        
        return pd.DataFrame(data)

    def save_dataset(self, dataset: pd.DataFrame, output_file: Optional[str] = None) -> str:
        """Save dataset to Excel file"""
        if output_file is None:
            output_file = self.config.get('dataset', {}).get('output_file', 
                                                           'SystemQAListallQuestion_eval_step4_final_report 1.xlsx')
        
        # Ensure the file has .xlsx extension
        if not output_file.endswith('.xlsx'):
            output_file += '.xlsx'
        
        dataset.to_excel(output_file, index=False)
        
        if self.config.get('logging', {}).get('show_progress', True):
            print(f"✅ Dataset saved as: {output_file}")
        
        return output_file

    def print_summary(self, dataset: pd.DataFrame):
        """Print dataset summary"""
        if not self.config.get('logging', {}).get('show_progress', True):
            return
            
        print(f"\n📊 Dataset Summary:")
        print("=" * 50)
        print(f"Total samples: {len(dataset)}")
        print(f"Mode used: {self.mode.upper()}")
        
        # Show average RAGAS scores
        ragas_cols = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        available_cols = [col for col in ragas_cols if col in dataset.columns]
        
        if available_cols:
            print("\n📈 Average RAGAS Scores:")
            for col in available_cols:
                avg_score = dataset[col].mean()
                print(f"  {col}: {avg_score:.3f}")
        
        # Show sample questions
        print(f"\n📝 Sample Questions:")
        for i, row in dataset.head(3).iterrows():
            print(f"  {i+1}. {row.get('question', 'N/A')[:60]}...")
        
        print(f"\n🎯 Ready for evaluation pipeline!")

    def print_usage_info(self):
        """Print usage information"""
        if not self.config.get('logging', {}).get('show_progress', True):
            return
            
        print(f"\n🔧 Usage Information:")
        print("=" * 50)
        
        if self.mode == 'local':
            print("📍 LOCAL MODE:")
            print("  ✅ Uses sentence transformers for similarity")
            print("  ✅ Uses KeyBERT/YAKE for keyword extraction")  
            print("  ✅ No API keys required")
            print("  ✅ All processing done locally")
            
        elif self.mode == 'ragas':
            print("📍 RAGAS MODE:")
            if self.config.get('ragas', {}).get('use_local_llm', True):
                print("  ✅ Uses RAGAS with local LLM")
                print("  ✅ No API keys required")
            else:
                print("  🌐 Uses RAGAS with OpenAI")
                print("  ⚠️  Requires OpenAI API key")
            print("  ✅ Professional-grade metrics")
            
        elif self.mode == 'simple':
            print("📍 SIMPLE MODE:")
            print("  ✅ Basic simulation approach")
            print("  ✅ No external dependencies")
            print("  ✅ Fast generation")
        
        print(f"\nNext steps:")
        print("  1. python contextual_keyword_gate.py")
        print("  2. python dynamic_ragas_gate_with_human_feedback.py")

def main():
    """Main function to generate synthetic dataset"""
    print("🔬 Configurable Synthetic Dataset Generator")
    print("=" * 60)
    
    try:
        # Initialize generator with config
        generator = ConfigurableDatasetGenerator()
        
        # Generate dataset
        dataset = generator.generate_dataset()
        
        # Save dataset
        output_file = generator.save_dataset(dataset)
        
        # Print summary
        generator.print_summary(dataset)
        
        # Print usage info
        generator.print_usage_info()
        
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("  1. Check config.yaml file exists and is valid")
        print("  2. Install required dependencies based on chosen mode")
        print("  3. Check your Python environment")
        
        # Print dependency info
        print(f"\n📦 Dependencies by mode:")
        print("  LOCAL mode: pip install sentence-transformers scikit-learn keybert yake")
        print("  RAGAS mode: pip install ragas transformers torch langchain-community")
        print("  SIMPLE mode: No additional dependencies")

if __name__ == "__main__":
    main()