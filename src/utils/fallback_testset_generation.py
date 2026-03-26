#!/usr/bin/env python3
"""
Fallback Testset Generation Strategies
======================================

This module provides fallback strategies when RAGAS testset generation fails.
"""

import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class FallbackTestsetGenerator:
    """Fallback testset generation when RAGAS fails."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize fallback generator."""
        self.config = config
        
    def generate_from_csv_templates(self, csv_files: List[str], output_dir: Path, target_size: int = 100) -> Dict[str, Any]:
        """Generate testset using CSV data and templates."""
        try:
            logger.info(f"🔄 Generating fallback testset from CSV templates ({target_size} samples)")
            
            # Load all CSV data
            all_data = []
            for csv_file in csv_files:
                if Path(csv_file).exists():
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
            
            if not all_data:
                return {'success': False, 'error': 'No valid CSV files found'}
            
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Question templates based on CSV content structure
            templates = [
                "What is {field_name}?",
                "Explain the {field_name}.",
                "How does {field_name} work?",
                "What are the details about {field_name}?",
                "Can you describe {field_name}?",
                "What information is available about {field_name}?",
                "How is {field_name} defined?",
                "What should I know about {field_name}?"
            ]
            
            testset_data = []
            template_idx = 0
            
            for idx, row in combined_df.iterrows():
                if len(testset_data) >= target_size:
                    break
                
                # Extract content from different possible columns
                content_fields = ['content', 'display', 'description', 'text', 'message']
                content = ""
                
                for field in content_fields:
                    if field in row and pd.notna(row[field]):
                        content = str(row[field])
                        break
                
                if not content:
                    # Use first non-null column as content
                    for col in row.index:
                        if pd.notna(row[col]) and row[col] != '':
                            content = str(row[col])
                            break
                
                if content:
                    # Generate question using template
                    template = templates[template_idx % len(templates)]
                    question = template.format(field_name=f"item {idx + 1}")
                    
                    # Create answer from content
                    answer = content[:300] + "..." if len(content) > 300 else content
                    
                    testset_data.append({
                        'user_input': question,
                        'reference_contexts': [content],
                        'reference': answer,
                        'auto_keywords': self._extract_simple_keywords(content)
                    })
                    
                    template_idx += 1
            
            # Save testset
            if testset_data:
                df_testset = pd.DataFrame(testset_data)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                testset_file = output_dir / f"fallback_template_testset_{timestamp}.csv"
                df_testset.to_csv(testset_file, index=False)
                
                return {
                    'success': True,
                    'testset_file': str(testset_file),
                    'samples_generated': len(df_testset),
                    'method': 'csv_template_fallback'
                }
            else:
                return {'success': False, 'error': 'No testset data generated'}
                
        except Exception as e:
            return {'success': False, 'error': f"Template generation failed: {str(e)}"}
    
    def generate_minimal_sample(self, output_dir: Path) -> Dict[str, Any]:
        """Generate a minimal 1-sample testset for testing."""
        try:
            logger.info("🔄 Generating minimal 1-sample testset")
            
            minimal_data = [{
                'user_input': 'What is the purpose of this system?',
                'reference_contexts': ['This system is designed to evaluate RAG (Retrieval-Augmented Generation) applications using various metrics and approaches.'],
                'reference': 'This system evaluates RAG applications using comprehensive metrics including contextual keyword matching and RAGAS evaluation frameworks.',
                'auto_keywords': 'system, evaluation, RAG, metrics'
            }]
            
            df_minimal = pd.DataFrame(minimal_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            testset_file = output_dir / f"minimal_testset_{timestamp}.csv"
            df_minimal.to_csv(testset_file, index=False)
            
            return {
                'success': True,
                'testset_file': str(testset_file),
                'samples_generated': 1,
                'method': 'minimal_fallback'
            }
            
        except Exception as e:
            return {'success': False, 'error': f"Minimal generation failed: {str(e)}"}
    
    def _extract_simple_keywords(self, text: str, max_keywords: int = 3) -> str:
        """Extract simple keywords from text."""
        try:
            # Simple keyword extraction using common words
            words = text.lower().split()
            
            # Filter out common stop words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
            
            keywords = []
            for word in words:
                word_clean = ''.join(c for c in word if c.isalnum())
                if len(word_clean) > 3 and word_clean not in stop_words:
                    keywords.append(word_clean)
                if len(keywords) >= max_keywords:
                    break
            
            return ', '.join(keywords[:max_keywords])
            
        except Exception:
            return ''

def create_fallback_generator(config: Dict[str, Any]) -> FallbackTestsetGenerator:
    """Create a fallback testset generator."""
    return FallbackTestsetGenerator(config)

if __name__ == "__main__":
    logger.info("🔄 Fallback Testset Generation module loaded")
