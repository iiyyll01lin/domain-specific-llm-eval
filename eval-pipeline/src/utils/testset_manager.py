#!/usr/bin/env python3
"""
Testset Manager - Save and Load Testsets for RAG Evaluation
This utility provides functions to save generated testsets and reload them for evaluation.
"""

import pandas as pd
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TestsetManager:
    """Manager for saving and loading testsets"""
    
    def __init__(self, base_output_dir: str = "outputs"):
        self.base_output_dir = Path(base_output_dir)
        self.testsets_dir = self.base_output_dir / "testsets"
        self.testsets_dir.mkdir(parents=True, exist_ok=True)
    
    def save_testset(self, testset_data: pd.DataFrame, 
                    name: str = None, 
                    metadata: Dict[str, Any] = None) -> str:
        """
        Save a testset to CSV and JSON formats
        
        Args:
            testset_data: DataFrame containing the testset
            name: Optional name for the testset file
            metadata: Optional metadata about the testset
            
        Returns:
            str: Path to the saved testset file
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if name is None:
            name = f"testset_{timestamp}"
        else:
            name = f"{name}_{timestamp}"
        
        # Save CSV
        csv_file = self.testsets_dir / f"{name}.csv"
        testset_data.to_csv(csv_file, index=False, encoding='utf-8')
        
        # Save Excel with metadata
        excel_file = self.testsets_dir / f"{name}.xlsx"
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            testset_data.to_excel(writer, sheet_name='Testset', index=False)
            
            # Create metadata sheet
            if metadata is None:
                metadata = {}
            
            metadata.update({
                'save_timestamp': timestamp,
                'total_questions': len(testset_data),
                'file_format': 'CSV + Excel',
                'columns': list(testset_data.columns)
            })
            
            metadata_df = pd.DataFrame([
                {'key': k, 'value': str(v)} for k, v in metadata.items()
            ])
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False)
        
        # Save JSON metadata
        json_file = self.testsets_dir / f"{name}_metadata.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Testset saved: {csv_file}")
        logger.info(f"Metadata saved: {json_file}")
        
        return str(csv_file)
    
    def load_testset(self, testset_path: str) -> tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Load a testset from file
        
        Args:
            testset_path: Path to the testset CSV file
            
        Returns:
            tuple: (testset_dataframe, metadata_dict)
        """
        testset_path = Path(testset_path)
        
        if not testset_path.exists():
            raise FileNotFoundError(f"Testset file not found: {testset_path}")
        
        # Load CSV
        testset_df = pd.read_csv(testset_path, encoding='utf-8')
        
        # Try to load metadata
        metadata = {}
        json_file = testset_path.parent / f"{testset_path.stem}_metadata.json"
        if json_file.exists():
            with open(json_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
        
        logger.info(f"Testset loaded: {testset_path}")
        logger.info(f"Questions loaded: {len(testset_df)}")
        
        return testset_df, metadata
    
    def list_available_testsets(self) -> List[Dict[str, Any]]:
        """
        List all available testsets
        
        Returns:
            List of dictionaries with testset information
        """
        testsets = []
        
        for csv_file in self.testsets_dir.glob("*.csv"):
            if csv_file.name.endswith("_metadata.csv"):
                continue
                
            info = {
                'name': csv_file.stem,
                'path': str(csv_file),
                'created': datetime.fromtimestamp(csv_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S'),
                'size_mb': round(csv_file.stat().st_size / 1024 / 1024, 2)
            }
            
            # Try to load metadata
            json_file = csv_file.parent / f"{csv_file.stem}_metadata.json"
            if json_file.exists():
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    info['questions'] = metadata.get('total_questions', 'unknown')
                    info['generation_method'] = metadata.get('generation_method', 'unknown')
                except Exception as e:
                    logger.warning(f"Failed to load metadata for {csv_file}: {e}")
            
            # Quick load to get question count if metadata unavailable
            if 'questions' not in info:
                try:
                    df = pd.read_csv(csv_file)
                    info['questions'] = len(df)
                except Exception as e:
                    info['questions'] = 'error'
                    logger.warning(f"Failed to load testset for counting: {e}")
            
            testsets.append(info)
        
        # Sort by creation time (newest first)
        testsets.sort(key=lambda x: x['created'], reverse=True)
        
        return testsets
    
    def validate_testset(self, testset_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate a testset format for RAG evaluation
        
        Args:
            testset_df: DataFrame to validate
            
        Returns:
            Dictionary with validation results
        """
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'stats': {}
        }
        
        required_columns = ['question', 'answer', 'contexts']
        optional_columns = ['ground_truth', 'auto_keywords', 'source_file']
        
        # Check required columns
        missing_required = [col for col in required_columns if col not in testset_df.columns]
        if missing_required:
            results['valid'] = False
            results['errors'].append(f"Missing required columns: {missing_required}")
        
        # Check for empty values
        if results['valid']:
            for col in required_columns:
                if col in testset_df.columns:
                    empty_count = testset_df[col].isna().sum()
                    if empty_count > 0:
                        results['warnings'].append(f"Column '{col}' has {empty_count} empty values")
        
        # Calculate stats
        results['stats'] = {
            'total_questions': len(testset_df),
            'columns': list(testset_df.columns),
            'avg_question_length': testset_df['question'].str.len().mean() if 'question' in testset_df.columns else 0,
            'avg_answer_length': testset_df['answer'].str.len().mean() if 'answer' in testset_df.columns else 0
        }
        
        return results

def main():
    """CLI interface for testset management"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Testset Manager CLI")
    parser.add_argument('action', choices=['list', 'validate'], 
                       help='Action to perform')
    parser.add_argument('--testset', help='Path to testset file (for validate)')
    parser.add_argument('--output-dir', default='outputs', 
                       help='Base output directory')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    manager = TestsetManager(args.output_dir)
    
    if args.action == 'list':
        testsets = manager.list_available_testsets()
        print(f"\n📋 Available Testsets ({len(testsets)} found):")
        print("=" * 80)
        
        for i, testset in enumerate(testsets, 1):
            print(f"{i}. {testset['name']}")
            print(f"   📄 Questions: {testset['questions']}")
            print(f"   📅 Created: {testset['created']}")
            print(f"   💾 Size: {testset['size_mb']} MB")
            print(f"   📁 Path: {testset['path']}")
            if 'generation_method' in testset:
                print(f"   🔧 Method: {testset['generation_method']}")
            print()
    
    elif args.action == 'validate':
        if not args.testset:
            print("❌ Error: --testset path required for validate action")
            return
        
        try:
            testset_df, metadata = manager.load_testset(args.testset)
            results = manager.validate_testset(testset_df)
            
            print(f"\n🔍 Validation Results for: {args.testset}")
            print("=" * 80)
            
            if results['valid']:
                print("✅ Testset is valid for RAG evaluation")
            else:
                print("❌ Testset has validation errors")
                for error in results['errors']:
                    print(f"   ❌ {error}")
            
            if results['warnings']:
                print("\n⚠️ Warnings:")
                for warning in results['warnings']:
                    print(f"   ⚠️ {warning}")
            
            print(f"\n📊 Statistics:")
            for key, value in results['stats'].items():
                print(f"   {key}: {value}")
            
            if metadata:
                print(f"\n📋 Metadata:")
                for key, value in metadata.items():
                    print(f"   {key}: {value}")
        
        except Exception as e:
            print(f"❌ Error validating testset: {e}")

if __name__ == "__main__":
    main()
