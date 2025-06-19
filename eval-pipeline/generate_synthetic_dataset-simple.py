"""
Synthetic Dataset Generator for Domain-Specific LLM Evaluation Pipeline
Generates test data that matches the exact structure required by the evaluation framework.
"""

import pandas as pd
import numpy as np
import random
from typing import List, Dict

# Set random seed for reproducibility
random.seed(42)
np.random.seed(42)

class SyntheticDatasetGenerator:
    def __init__(self):
        # Sample domain-specific questions and contexts
        self.medical_qa_templates = [
            {
                "question": "What are the symptoms of type 2 diabetes?",
                "context": "Type 2 diabetes is a chronic condition that affects the way your body metabolizes sugar (glucose). It's characterized by insulin resistance and relative insulin deficiency.",
                "ground_truth": "Type 2 diabetes symptoms include increased thirst, frequent urination, increased hunger, unintended weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
                "keywords": ["diabetes", "symptoms", "insulin", "glucose", "blood sugar"]
            },
            {
                "question": "How is hypertension diagnosed?",
                "context": "Hypertension, or high blood pressure, is diagnosed when blood pressure readings consistently exceed normal levels over multiple measurements.",
                "ground_truth": "Hypertension is diagnosed when systolic pressure is 140 mmHg or higher, or diastolic pressure is 90 mmHg or higher, measured on at least two separate occasions.",
                "keywords": ["hypertension", "blood pressure", "systolic", "diastolic", "diagnosis"]
            },
            {
                "question": "What causes pneumonia?",
                "context": "Pneumonia is an infection that inflames air sacs in one or both lungs, which may fill with fluid or pus.",
                "ground_truth": "Pneumonia can be caused by bacteria (most commonly Streptococcus pneumoniae), viruses, fungi, or other microorganisms that enter the lungs.",
                "keywords": ["pneumonia", "infection", "bacteria", "virus", "lungs"]
            },
            {
                "question": "What are the risk factors for heart disease?",
                "context": "Heart disease encompasses various conditions affecting the heart and blood vessels, with multiple contributing factors.",
                "ground_truth": "Risk factors for heart disease include high blood pressure, high cholesterol, smoking, diabetes, obesity, physical inactivity, family history, and age.",
                "keywords": ["heart disease", "risk factors", "cholesterol", "smoking", "obesity"]
            },
            {
                "question": "How is asthma treated?",
                "context": "Asthma is a chronic respiratory condition characterized by inflammation and narrowing of the airways.",
                "ground_truth": "Asthma treatment includes quick-relief medications (bronchodilators) for immediate symptoms and long-term control medications (corticosteroids) to prevent attacks.",
                "keywords": ["asthma", "treatment", "bronchodilators", "corticosteroids", "airways"]
            }
        ]
        
        self.tech_qa_templates = [
            {
                "question": "What is machine learning?",
                "context": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed.",
                "ground_truth": "Machine learning is a method of data analysis that automates analytical model building using algorithms that iteratively learn from data.",
                "keywords": ["machine learning", "artificial intelligence", "algorithms", "data analysis", "automation"]
            },
            {
                "question": "How does cloud computing work?",
                "context": "Cloud computing delivers computing services over the internet, including servers, storage, databases, networking, and software.",
                "ground_truth": "Cloud computing works by providing on-demand access to computing resources through virtualization and distributed systems across multiple data centers.",
                "keywords": ["cloud computing", "virtualization", "data centers", "distributed systems", "servers"]
            },
            {
                "question": "What is cybersecurity?",
                "context": "Cybersecurity involves protecting digital systems, networks, and data from digital attacks and unauthorized access.",
                "ground_truth": "Cybersecurity is the practice of defending computers, servers, mobile devices, networks, and data from malicious attacks and breaches.",
                "keywords": ["cybersecurity", "networks", "data protection", "malicious attacks", "security"]
            },
            {
                "question": "How does blockchain technology work?",
                "context": "Blockchain is a distributed ledger technology that maintains a continuously growing list of records linked and secured using cryptography.",
                "ground_truth": "Blockchain works by creating immutable blocks of data that are cryptographically linked and distributed across a network of computers.",
                "keywords": ["blockchain", "distributed ledger", "cryptography", "immutable", "decentralized"]
            },
            {
                "question": "What is API integration?",
                "context": "API integration involves connecting different software applications through their application programming interfaces to share data and functionality.",
                "ground_truth": "API integration enables different software systems to communicate and exchange data through standardized protocols and endpoints.",
                "keywords": ["API", "integration", "software applications", "data exchange", "endpoints"]
            }
        ]

    def generate_llm_response(self, question: str, ground_truth: str, keywords: List[str]) -> str:
        """Generate a realistic LLM response with varying quality and keyword inclusion"""
        
        # Create different response quality levels
        response_variants = [
            # High quality - includes most keywords and accurate info
            f"Based on the question about {keywords[0]}, {ground_truth.lower()} Additionally, {keywords[1]} and {keywords[2]} are important factors to consider.",
            
            # Medium quality - partial keywords, some accuracy
            f"Regarding {keywords[0]}, it involves {keywords[1]} and related to {keywords[-1]}. {ground_truth[:50]}...",
            
            # Lower quality - fewer keywords, less complete
            f"This relates to {keywords[0]} and {keywords[1]}. There are several factors involved in this condition.",
            
            # Variable quality with different keyword inclusion
            f"The answer involves {keywords[0]}, {keywords[2]}, and {keywords[3]}. {ground_truth[:80]}... This is important for understanding the overall concept."
        ]
        
        return random.choice(response_variants)

    def generate_ragas_scores(self) -> Dict[str, float]:
        """Generate realistic RAGAS scores with some variation"""
        # Generate correlated scores (good answers tend to score well across metrics)
        base_quality = random.uniform(0.4, 0.95)
        noise = 0.1
        
        scores = {
            "context_precision": max(0.0, min(1.0, base_quality + random.uniform(-noise, noise))),
            "context_recall": max(0.0, min(1.0, base_quality + random.uniform(-noise, noise))),
            "faithfulness": max(0.0, min(1.0, base_quality + random.uniform(-noise, noise))),
            "answer_relevancy": max(0.0, min(1.0, base_quality + random.uniform(-noise, noise)))
        }
        
        return scores

    def format_keywords_for_dataset(self, keywords: List[str]) -> str:
        """Format keywords as string list to match expected format"""
        return str(keywords)

    def generate_dataset(self, num_samples: int = 10) -> pd.DataFrame:
        """Generate complete synthetic dataset"""
        
        # Combine all templates
        all_templates = self.medical_qa_templates + self.tech_qa_templates
        
        data = []
        
        for i in range(num_samples):
            # Select template (cycle through if needed)
            template = all_templates[i % len(all_templates)]
            
            # Generate LLM response
            llm_answer = self.generate_llm_response(
                template["question"], 
                template["ground_truth"], 
                template["keywords"]
            )
            
            # Generate RAGAS scores
            ragas_scores = self.generate_ragas_scores()
            
            # Calculate weighted average score
            weighted_avg = np.mean(list(ragas_scores.values()))
            
            # Create row data
            row = {
                "question": template["question"],
                "contexts": template["context"],
                "answer": llm_answer,
                "ground_truth": template["ground_truth"],
                "context_precision": round(ragas_scores["context_precision"], 3),
                "context_recall": round(ragas_scores["context_recall"], 3),
                "faithfulness": round(ragas_scores["faithfulness"], 3),
                "answer_relevancy": round(ragas_scores["answer_relevancy"], 3),
                "kw": self.format_keywords_for_dataset(template["keywords"]),
                "kw_metric": round(random.uniform(0.5, 0.9), 3),  # Placeholder keyword metric
                "weighted_average_score": round(weighted_avg, 3)
            }
            
            data.append(row)
        
        return pd.DataFrame(data)

def main():
    """Generate and save synthetic dataset"""
    generator = SyntheticDatasetGenerator()
    
    print("Generating 10-line synthetic dataset for LLM evaluation pipeline...")
    
    # Generate dataset
    dataset = generator.generate_dataset(10)
    
    # Save to Excel file with exact name expected by the pipeline
    output_file = "testset-gen-simple.xlsx"
    # output_file = "SystemQAListallQuestion_eval_step4_final_report 1.xlsx"
    dataset.to_excel(output_file, index=False)
    
    print(f"✅ Dataset saved as: {output_file}")
    print(f"📊 Generated {len(dataset)} samples")
    print("\nDataset Preview:")
    print("=" * 80)
    
    # Display summary
    for i, row in dataset.head(3).iterrows():
        print(f"\nSample {i+1}:")
        print(f"Question: {row['question'][:60]}...")
        print(f"Keywords: {row['kw']}")
        print(f"RAGAS Scores: P={row['context_precision']:.2f}, R={row['context_recall']:.2f}, F={row['faithfulness']:.2f}, AR={row['answer_relevancy']:.2f}")
        print("-" * 60)
    
    print(f"\n🎯 Dataset is ready for evaluation pipeline!")
    print("You can now run:")
    print("  1. python contextual_keyword_gate.py")
    print("  2. python dynamic_ragas_gate_with_human_feedback.py")

if __name__ == "__main__":
    main()
