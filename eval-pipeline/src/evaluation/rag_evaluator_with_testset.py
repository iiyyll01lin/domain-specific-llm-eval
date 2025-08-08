#!/usr/bin/env python3
"""
RAG Evaluation with Saved Testsets
This script evaluates RAG systems using pre-generated and saved testsets.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import json
import requests
import time
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import yaml

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from utils.testset_manager import TestsetManager

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RAGEvaluatorWithSavedTestset:
    """RAG evaluator that uses saved testsets"""
    
    def __init__(self, config_path: str = None):
        """Initialize with configuration"""
        if config_path is None:
            config_path = project_root / 'config' / 'pipeline_config.yaml'
        
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.testset_manager = TestsetManager()
        
    def load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from: {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def call_rag_endpoint(self, question: str, 
                         endpoint_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Call RAG endpoint with a question
        
        Args:
            question: Question to ask the RAG system
            endpoint_config: Optional endpoint configuration override
            
        Returns:
            Dict containing response and metadata
        """
        if endpoint_config is None:
            endpoint_config = self.config.get('rag_system', {})
        
        # Build request based on your SMT endpoint format
        url = endpoint_config.get('endpoint', 'http://10.3.30.13:8855/app/smt_assistant_chat')
        
        payload = {
            "content": question,
            "session_id": "",
            "command": "",
            "context": {},
            "app": "smt_assistant_chat",
            "user": "",
            "site": "tao",
            # "language": "tw"
            "language": "en"
        }
        
        headers = {
            "Content-Type": "application/json"
        }
        
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, headers=headers, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("message", [{}])[0].get('content', '')
                
                return {
                    'question': question,
                    'answer': answer,
                    'response_time': response_time,
                    'status': 'success',
                    'raw_response': result
                }
            else:
                return {
                    'question': question,
                    'answer': f"Error: HTTP {response.status_code}",
                    'response_time': response_time,
                    'status': 'error',
                    'error': f"HTTP {response.status_code}: {response.text}"
                }
                
        except requests.exceptions.Timeout:
            return {
                'question': question,
                'answer': "Error: Request timeout",
                'response_time': 30.0,
                'status': 'timeout',
                'error': 'Request timeout after 30 seconds'
            }
        except Exception as e:
            return {
                'question': question,
                'answer': f"Error: {str(e)}",
                'response_time': 0.0,
                'status': 'error',
                'error': str(e)
            }
    
    def evaluate_with_testset(self, testset_path: str, 
                            max_questions: int = None,
                            save_results: bool = True) -> Dict[str, Any]:
        """
        Evaluate RAG system using a saved testset
        
        Args:
            testset_path: Path to the saved testset
            max_questions: Optional limit on number of questions to evaluate
            save_results: Whether to save evaluation results
            
        Returns:
            Dictionary with evaluation results
        """
        logger.info(f"Starting RAG evaluation with testset: {testset_path}")
        
        # Load testset
        testset_df, metadata = self.testset_manager.load_testset(testset_path)
        
        # Validate testset
        validation = self.testset_manager.validate_testset(testset_df)
        if not validation['valid']:
            raise ValueError(f"Invalid testset: {validation['errors']}")
        
        # Limit questions if specified
        if max_questions and len(testset_df) > max_questions:
            testset_df = testset_df.head(max_questions)
            logger.info(f"Limited evaluation to {max_questions} questions")
        
        logger.info(f"Evaluating {len(testset_df)} questions...")
        
        # Store results
        results = []
        start_time = datetime.now()
        
        for i, row in testset_df.iterrows():
            question = row['question']
            expected_answer = row.get('answer', '')
            contexts = row.get('contexts', '')
            
            logger.info(f"Evaluating question {i+1}/{len(testset_df)}: {question[:100]}...")
            
            # Call RAG endpoint
            rag_result = self.call_rag_endpoint(question)
            
            # Store result
            result = {
                'question_id': i,
                'question': question,
                'expected_answer': expected_answer,
                'rag_answer': rag_result['answer'],
                'contexts': contexts,
                'response_time': rag_result['response_time'],
                'status': rag_result['status'],
                'timestamp': datetime.now().isoformat()
            }
            
            if 'error' in rag_result:
                result['error'] = rag_result['error']
            
            results.append(result)
            
            # Brief delay to avoid overwhelming the endpoint
            time.sleep(0.5)
        
        end_time = datetime.now()
        evaluation_duration = (end_time - start_time).total_seconds()
        
        # Calculate basic metrics
        successful_responses = sum(1 for r in results if r['status'] == 'success')
        error_responses = sum(1 for r in results if r['status'] == 'error')
        timeout_responses = sum(1 for r in results if r['status'] == 'timeout')
        
        avg_response_time = sum(r['response_time'] for r in results) / len(results)
        
        evaluation_summary = {
            'testset_path': testset_path,
            'testset_metadata': metadata,
            'total_questions': len(results),
            'successful_responses': successful_responses,
            'error_responses': error_responses,
            'timeout_responses': timeout_responses,
            'success_rate': successful_responses / len(results) * 100,
            'avg_response_time': avg_response_time,
            'total_evaluation_time': evaluation_duration,
            'evaluation_start': start_time.isoformat(),
            'evaluation_end': end_time.isoformat(),
            'results': results
        }
        
        logger.info(f"Evaluation completed!")
        logger.info(f"Success rate: {evaluation_summary['success_rate']:.1f}%")
        logger.info(f"Average response time: {avg_response_time:.2f}s")
        
        # Save results if requested
        if save_results:
            self.save_evaluation_results(evaluation_summary)
        
        return evaluation_summary
    
    def save_evaluation_results(self, evaluation_summary: Dict[str, Any]) -> str:
        """Save evaluation results to files"""
        output_dir = Path("outputs") / "evaluations"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save JSON results
        json_file = output_dir / f"rag_evaluation_{timestamp}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_summary, f, indent=2, ensure_ascii=False)
        
        # Save Excel report
        excel_file = output_dir / f"rag_evaluation_{timestamp}.xlsx"
        
        # Create DataFrame from results
        results_df = pd.DataFrame(evaluation_summary['results'])
        
        with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
            # Results sheet
            results_df.to_excel(writer, sheet_name='Results', index=False)
            
            # Summary sheet
            summary_data = []
            for key, value in evaluation_summary.items():
                if key != 'results':
                    summary_data.append({'Metric': key, 'Value': str(value)})
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Statistics sheet
            if len(results_df) > 0:
                stats_data = [
                    {'Metric': 'Total Questions', 'Value': len(results_df)},
                    {'Metric': 'Successful Responses', 'Value': evaluation_summary['successful_responses']},
                    {'Metric': 'Error Responses', 'Value': evaluation_summary['error_responses']},
                    {'Metric': 'Timeout Responses', 'Value': evaluation_summary['timeout_responses']},
                    {'Metric': 'Success Rate (%)', 'Value': f"{evaluation_summary['success_rate']:.1f}%"},
                    {'Metric': 'Avg Response Time (s)', 'Value': f"{evaluation_summary['avg_response_time']:.2f}"},
                    {'Metric': 'Total Evaluation Time (s)', 'Value': f"{evaluation_summary['total_evaluation_time']:.1f}"},
                ]
                
                stats_df = pd.DataFrame(stats_data)
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
        
        logger.info(f"Evaluation results saved:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Excel: {excel_file}")
        
        return str(excel_file)
    
    def print_evaluation_report(self, evaluation_summary: Dict[str, Any]):
        """Print a formatted evaluation report"""
        print("\n" + "="*80)
        print("🎯 RAG EVALUATION REPORT")
        print("="*80)
        
        print(f"📁 Testset: {evaluation_summary['testset_path']}")
        print(f"⏰ Evaluation Time: {evaluation_summary['evaluation_start']} to {evaluation_summary['evaluation_end']}")
        print(f"⏱️  Total Duration: {evaluation_summary['total_evaluation_time']:.1f} seconds")
        
        print(f"\n📊 RESULTS SUMMARY:")
        print(f"   Total Questions: {evaluation_summary['total_questions']}")
        print(f"   ✅ Successful: {evaluation_summary['successful_responses']}")
        print(f"   ❌ Errors: {evaluation_summary['error_responses']}")
        print(f"   ⏰ Timeouts: {evaluation_summary['timeout_responses']}")
        print(f"   📈 Success Rate: {evaluation_summary['success_rate']:.1f}%")
        print(f"   ⚡ Avg Response Time: {evaluation_summary['avg_response_time']:.2f}s")
        
        print(f"\n📝 SAMPLE RESULTS:")
        print("-" * 80)
        
        for i, result in enumerate(evaluation_summary['results'][:3], 1):
            print(f"\n{i}. Question: {result['question'][:100]}...")
            print(f"   Expected: {result['expected_answer'][:100]}...")
            print(f"   RAG Answer: {result['rag_answer'][:100]}...")
            print(f"   Status: {result['status']} ({result['response_time']:.2f}s)")
        
        if len(evaluation_summary['results']) > 3:
            print(f"\n   ... and {len(evaluation_summary['results']) - 3} more results")

def main():
    """CLI interface for RAG evaluation with saved testsets"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Evaluator with Saved Testsets")
    parser.add_argument('action', choices=['list', 'evaluate'], 
                       help='Action to perform')
    parser.add_argument('--testset', help='Path to testset file (for evaluate)')
    parser.add_argument('--max-questions', type=int, help='Maximum number of questions to evaluate')
    parser.add_argument('--config', help='Path to pipeline configuration file')
    parser.add_argument('--no-save', action='store_true', help='Do not save evaluation results')
    
    args = parser.parse_args()
    
    try:
        evaluator = RAGEvaluatorWithSavedTestset(args.config)
        
        if args.action == 'list':
            testsets = evaluator.testset_manager.list_available_testsets()
            print(f"\n📋 Available Testsets ({len(testsets)} found):")
            print("=" * 80)
            
            for i, testset in enumerate(testsets, 1):
                print(f"{i}. {testset['name']}")
                print(f"   📄 Questions: {testset['questions']}")
                print(f"   📅 Created: {testset['created']}")
                print(f"   📁 Path: {testset['path']}")
                print()
        
        elif args.action == 'evaluate':
            if not args.testset:
                # Use the most recent testset if none specified
                testsets = evaluator.testset_manager.list_available_testsets()
                if not testsets:
                    print("❌ No testsets found. Please generate a testset first.")
                    return
                
                args.testset = testsets[0]['path']
                print(f"🔄 Using most recent testset: {args.testset}")
            
            # Run evaluation
            results = evaluator.evaluate_with_testset(
                testset_path=args.testset,
                max_questions=args.max_questions,
                save_results=not args.no_save
            )
            
            # Print report
            evaluator.print_evaluation_report(results)
            
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

if __name__ == "__main__":
    main()
