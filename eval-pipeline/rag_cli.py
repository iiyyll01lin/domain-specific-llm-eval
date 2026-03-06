#!/usr/bin/env python3
"""
RAG Testset CLI - Command-line utility for managing and evaluating testsets
"""

import argparse
import sys
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / 'src'))

from src.utils.testset_manager import TestsetManager
from src.evaluation.rag_evaluator_with_testset import RAGEvaluatorWithSavedTestset

def cmd_list_testsets():
    """List all available testsets"""
    manager = TestsetManager()
    testsets = manager.list_available_testsets()
    
    if not testsets:
        print("📋 No testsets found.")
        return
    
    print(f"📋 Available Testsets ({len(testsets)} found):")
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

def cmd_validate_testset(testset_path: str):
    """Validate a testset"""
    manager = TestsetManager()
    
    try:
        testset_df, metadata = manager.load_testset(testset_path)
        results = manager.validate_testset(testset_df)
        
        print(f"🔍 Validation Results for: {testset_path}")
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
            if isinstance(value, float):
                print(f"   {key}: {value:.1f}")
            else:
                print(f"   {key}: {value}")
        
        if metadata:
            print(f"\n📋 Metadata:")
            for key, value in metadata.items():
                print(f"   {key}: {value}")
    
    except Exception as e:
        print(f"❌ Error validating testset: {e}")

def cmd_evaluate_rag(testset_path: str = None, max_questions: int = None, 
                    config_path: str = None, no_save: bool = False):
    """Evaluate RAG system with testset"""
    try:
        evaluator = RAGEvaluatorWithSavedTestset(config_path)
        
        if testset_path is None:
            # Use the most recent testset
            testsets = evaluator.testset_manager.list_available_testsets()
            if not testsets:
                print("❌ No testsets found. Please generate a testset first.")
                return
            
            testset_path = testsets[0]['path']
            print(f"🔄 Using most recent testset: {testset_path}")
        
        # Run evaluation
        results = evaluator.evaluate_with_testset(
            testset_path=testset_path,
            max_questions=max_questions,
            save_results=not no_save
        )
        
        # Print report
        evaluator.print_evaluation_report(results)
        
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return None

def cmd_quick_eval(questions: int = 3):
    """Quick evaluation with the most recent testset"""
    print(f"🚀 Quick RAG evaluation with {questions} questions")
    print("=" * 60)
    
    return cmd_evaluate_rag(max_questions=questions)

def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="RAG Testset CLI - Manage and evaluate testsets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s list                    # List all available testsets
  %(prog)s validate testset.csv    # Validate a testset
  %(prog)s evaluate               # Evaluate with most recent testset
  %(prog)s evaluate --testset path/to/testset.csv --max-questions 5
  %(prog)s quick --questions 3    # Quick evaluation with 3 questions
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List available testsets')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate a testset')
    validate_parser.add_argument('testset', help='Path to testset file')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate RAG system')
    eval_parser.add_argument('--testset', help='Path to testset file')
    eval_parser.add_argument('--max-questions', type=int, help='Maximum number of questions')
    eval_parser.add_argument('--config', help='Path to configuration file')
    eval_parser.add_argument('--no-save', action='store_true', help='Do not save results')
    
    # Quick command
    quick_parser = subparsers.add_parser('quick', help='Quick evaluation')
    quick_parser.add_argument('--questions', type=int, default=3, 
                             help='Number of questions to evaluate (default: 3)')
    
    args = parser.parse_args()
    
    if args.command == 'list':
        cmd_list_testsets()
    
    elif args.command == 'validate':
        cmd_validate_testset(args.testset)
    
    elif args.command == 'evaluate':
        cmd_evaluate_rag(
            testset_path=args.testset,
            max_questions=args.max_questions,
            config_path=args.config,
            no_save=args.no_save
        )
    
    elif args.command == 'quick':
        cmd_quick_eval(args.questions)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
