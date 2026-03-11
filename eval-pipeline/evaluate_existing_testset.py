#!/usr/bin/env python3
"""
SMT RAG Evaluation Script - Reuse Existing Testset
This script evaluates your SMT RAG endpoint using an existing testset to save time.
"""

import argparse
import os
import sys
import yaml
import pandas as pd
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import requests
import time

EVAL_PIPELINE_ROOT = Path(__file__).resolve().parent

# Add eval-pipeline to Python path
if str(EVAL_PIPELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(EVAL_PIPELINE_ROOT))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SMTRAGEvaluator:
    """Evaluate SMT RAG endpoint using existing testset"""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """Initialize evaluator with configuration"""
        raw_config_path = Path(config_path)
        if raw_config_path.is_absolute():
            resolved_config_path = raw_config_path
        else:
            resolved_config_path = (EVAL_PIPELINE_ROOT / raw_config_path).resolve()
        self.config_path = str(resolved_config_path)
        self.config = self._load_config()
        self.config_dir = Path(self.config_path).resolve().parent
        self.rag_config = self.config.get('rag_system', {})
        self.eval_config = self.config.get('evaluation', {})
        
        # Initialize evaluators
        self._setup_evaluators()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f) or {}
        config['__config_dir__'] = str(Path(self.config_path).resolve().parent)
        return config

    def _resolve_path(self, path_value: str) -> Path:
        """Resolve paths relative to the loaded config file when possible."""
        path = Path(path_value)
        if path.is_absolute():
            return path
        config_dir = getattr(self, 'config_dir', EVAL_PIPELINE_ROOT)
        config_relative = (config_dir / path).resolve()
        if config_relative.exists():
            return config_relative
        pipeline_relative = (EVAL_PIPELINE_ROOT / path).resolve()
        if pipeline_relative.exists():
            return pipeline_relative
        return path.resolve()

    @staticmethod
    def _normalize_value(value: Any, default: Any = "") -> Any:
        if pd.isna(value):
            return default
        return value

    @staticmethod
    def _decode_json_like_value(value: Any) -> Any:
        if not isinstance(value, str):
            return value
        stripped = value.strip()
        if not stripped or stripped[0] not in "[{":
            return value
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return value
    
    def _setup_evaluators(self):
        """Setup evaluation components"""
        logger.info("🔧 Setting up evaluators...")
        
        # Setup contextual keyword evaluator (bypass human feedback)
        self.contextual_evaluator = None
        if self.eval_config.get('contextual_keywords', {}).get('enabled', False):
            try:
                from src.evaluation.contextual_keyword_evaluator import ContextualKeywordEvaluator
                # Create a mock config without human feedback
                eval_config = self.eval_config.get('contextual_keywords', {}).copy()
                eval_config['human_feedback'] = {'enabled': False}
                self.contextual_evaluator = ContextualKeywordEvaluator(eval_config)
                logger.info("✅ Contextual keyword evaluator initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize contextual evaluator: {e}")
        
        # Setup RAGAS evaluator (if enabled)
        self.ragas_evaluator = None
        if self.eval_config.get('ragas_metrics', {}).get('enabled', False):
            try:
                from src.evaluation.ragas_evaluator import RagasEvaluator
                self.ragas_evaluator = RagasEvaluator(self.eval_config)
                logger.info("✅ RAGAS evaluator initialized")
            except Exception as e:
                logger.warning(f"⚠️ Could not initialize RAGAS evaluator: {e}")
    
    def call_smt_assistant(self, question: str) -> Dict[str, Any]:
        """Call SMT RAG assistant with a question"""
        url = self.rag_config.get('endpoint')
        if not url:
            raise ValueError("RAG endpoint not configured")
        
        # Build payload using template
        payload = self.rag_config.get('request_format', {}).get('payload_template', {}).copy()
        payload['content'] = question
        
        headers = {"Content-Type": "application/json"}
        
        start_time = time.time()
        try:
            response = requests.post(
                url, 
                json=payload, 
                headers=headers,
                timeout=self.rag_config.get('timeout', 30)
            )
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                answer = result.get("message", [{}])[0].get('content', 'No response')
                return {
                    'success': True,
                    'answer': answer,
                    'response_time': response_time,
                    'full_response': result
                }
            else:
                return {
                    'success': False,
                    'error': f"HTTP {response.status_code}: {response.text}",
                    'response_time': response_time
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }

    def call_mock_assistant(self, question: str, ground_truth: str, contexts: str) -> Dict[str, Any]:
        """Provide a deterministic local response for evaluation smoke tests."""
        answer = ground_truth or contexts or f"Mock answer for: {question}"
        return {
            'success': True,
            'answer': answer,
            'response_time': 0.0,
            'full_response': {'mode': 'mock', 'question': question},
        }
    
    def load_testset(self, testset_path: str) -> pd.DataFrame:
        """Load existing testset"""
        resolved_testset_path = self._resolve_path(testset_path)
        logger.info(f"📂 Loading testset from: {resolved_testset_path}")
        
        if not resolved_testset_path.exists():
            raise FileNotFoundError(f"Testset file not found: {resolved_testset_path}")
        
        # Load CSV testset
        df = pd.read_csv(resolved_testset_path)
        rename_map = {}
        if 'question' not in df.columns and 'user_input' in df.columns:
            rename_map['user_input'] = 'question'
        if 'contexts' not in df.columns and 'reference_contexts' in df.columns:
            rename_map['reference_contexts'] = 'contexts'
        if 'answer' not in df.columns and 'reference' in df.columns:
            rename_map['reference'] = 'answer'
        if rename_map:
            df = df.rename(columns=rename_map)
        if 'ground_truth' not in df.columns and 'answer' in df.columns:
            df['ground_truth'] = df['answer']
        if 'contexts' in df.columns:
            df['contexts'] = df['contexts'].apply(self._decode_json_like_value)
        logger.info(f"📊 Loaded {len(df)} test questions")
        
        # Display testset overview
        logger.info("📋 Testset Overview:")
        for i, row in df.iterrows():
            question = self._normalize_value(row.get('question', 'N/A'), 'N/A')
            logger.info(f"  {i+1}. {question[:80]}{'...' if len(question) > 80 else ''}")
        
        return df
    
    def evaluate_testset(self, testset_df: pd.DataFrame) -> Dict[str, Any]:
        """Evaluate RAG system using testset"""
        logger.info("🔍 Starting RAG evaluation...")
        
        results = {
            'metadata': {
                'evaluator': 'SMTRAGEvaluator',
                'testset_size': len(testset_df),
                'evaluation_start': datetime.now().isoformat(),
                'rag_endpoint': self.rag_config.get('endpoint'),
                'config_file': self.config_path
            },
            'questions': [],
            'summary': {},
            'errors': []
        }
        
        successful_responses = 0
        total_response_time = 0
        
        # Process each question
        for i, row in testset_df.iterrows():
            question = self._normalize_value(row.get('question', ''), '')
            expected_answer = self._normalize_value(row.get('answer', ''), '')
            ground_truth = self._normalize_value(row.get('ground_truth', expected_answer), expected_answer)
            contexts = self._normalize_value(row.get('contexts', ''), '')
            
            logger.info(f"🤖 Processing question {i+1}/{len(testset_df)}: {question[:60]}...")
            
            # Call RAG system
            if str(self.rag_config.get('endpoint', '')).startswith('mock://'):
                rag_response = self.call_mock_assistant(question, ground_truth, contexts)
            else:
                rag_response = self.call_smt_assistant(question)
            
            question_result = {
                'question_id': i,
                'question': question,
                'expected_answer': expected_answer,
                'ground_truth': ground_truth,
                'contexts': contexts,
                'rag_response': rag_response,
                'timestamp': datetime.now().isoformat()
            }
            
            if rag_response['success']:
                successful_responses += 1
                total_response_time += rag_response['response_time']
                
                # Evaluate with contextual keywords (if available)
                if self.contextual_evaluator:
                    try:
                        keyword_score = self._evaluate_with_contextual_keywords(
                            question, rag_response['answer'], ground_truth, contexts
                        )
                        question_result['keyword_evaluation'] = keyword_score
                        logger.info(f"  📊 Keyword Score: {keyword_score.get('score', 'N/A'):.3f}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ Keyword evaluation failed: {e}")
                
                # Evaluate with RAGAS (if available)
                if self.ragas_evaluator:
                    try:
                        ragas_scores = self._evaluate_with_ragas(
                            question, rag_response['answer'], ground_truth, contexts
                        )
                        question_result['ragas_evaluation'] = ragas_scores
                        logger.info(f"  📈 RAGAS Scores: {ragas_scores}")
                    except Exception as e:
                        logger.warning(f"  ⚠️ RAGAS evaluation failed: {e}")
                
                logger.info(f"  ✅ Response time: {rag_response['response_time']:.2f}s")
            else:
                logger.error(f"  ❌ RAG call failed: {rag_response['error']}")
                results['errors'].append({
                    'question_id': i,
                    'error': rag_response['error']
                })
            
            results['questions'].append(question_result)
        
        # Calculate summary statistics
        results['summary'] = {
            'total_questions': len(testset_df),
            'successful_responses': successful_responses,
            'error_responses': len(testset_df) - successful_responses,
            'success_rate': successful_responses / len(testset_df) if len(testset_df) > 0 else 0,
            'avg_response_time': total_response_time / successful_responses if successful_responses > 0 else 0,
            'evaluation_end': datetime.now().isoformat()
        }
        
        logger.info("📊 Evaluation Summary:")
        logger.info(f"  Total Questions: {results['summary']['total_questions']}")
        logger.info(f"  Successful Responses: {results['summary']['successful_responses']}")
        logger.info(f"  Success Rate: {results['summary']['success_rate']:.1%}")
        logger.info(f"  Average Response Time: {results['summary']['avg_response_time']:.2f}s")
        
        return results
    
    def _evaluate_with_contextual_keywords(self, question: str, answer: str, ground_truth: str, contexts: str) -> Dict[str, Any]:
        """Evaluate using contextual keyword matcher"""
        # Mock evaluation data in expected format
        testset_data = [{
            'question': question,
            'answer': answer,
            'ground_truth': ground_truth,
            'contexts': contexts
        }]
        
        rag_responses = [answer]
        
        # Call the evaluate method we added
        if hasattr(self.contextual_evaluator, 'evaluate'):
            return self.contextual_evaluator.evaluate(testset_data, rag_responses)
        else:
            # Fallback to evaluate_response
            return self.contextual_evaluator.evaluate_response(answer, ground_truth, question)
    
    def _evaluate_with_ragas(self, question: str, answer: str, ground_truth: str, contexts: str) -> Dict[str, Any]:
        """Evaluate using RAGAS metrics"""
        # Create RAGAS-compatible data
        testset_data = {
            "questions": [question],
            "ground_truths": [ground_truth],
            "contexts": [[contexts]] if contexts else [[]],
        }
        rag_responses = [{"answer": answer}]
        
        try:
            return self.ragas_evaluator.evaluate(testset_data, rag_responses)
        except Exception as e:
            logger.warning(f"RAGAS evaluation failed: {e}")
            return {"error": str(e)}

    def save_results(self, results: Dict[str, Any], output_dir: str = "outputs/evaluations") -> str:
        """Save evaluation results"""
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"smt_rag_evaluation_{timestamp}.json")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Results saved to: {output_file}")
        return output_file
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate human-readable evaluation report"""
        report = []
        report.append("=" * 80)
        report.append("SMT RAG EVALUATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        summary = results['summary']
        report.append("📊 EVALUATION SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Questions: {summary['total_questions']}")
        report.append(f"Successful Responses: {summary['successful_responses']}")
        report.append(f"Success Rate: {summary['success_rate']:.1%}")
        report.append(f"Average Response Time: {summary['avg_response_time']:.2f}s")
        report.append("")
        
        # Detailed results
        report.append("🔍 DETAILED RESULTS")
        report.append("-" * 40)
        
        for i, q_result in enumerate(results['questions'], 1):
            report.append(f"\n{i}. Question: {q_result['question']}")
            
            if q_result['rag_response']['success']:
                report.append(f"   ✅ RAG Answer: {q_result['rag_response']['answer'][:200]}...")
                report.append(f"   ⏱️ Response Time: {q_result['rag_response']['response_time']:.2f}s")
                
                # Keyword evaluation
                if 'keyword_evaluation' in q_result:
                    kw_eval = q_result['keyword_evaluation']
                    if isinstance(kw_eval, dict) and 'score' in kw_eval:
                        report.append(f"   📊 Keyword Score: {kw_eval['score']:.3f}")
                
                # RAGAS evaluation
                if 'ragas_evaluation' in q_result:
                    ragas_eval = q_result['ragas_evaluation']
                    if isinstance(ragas_eval, dict):
                        report.append(f"   📈 RAGAS Scores: {ragas_eval}")
            else:
                report.append(f"   ❌ Error: {q_result['rag_response']['error']}")
        
        # Errors section
        if results['errors']:
            report.append("\n❌ ERRORS")
            report.append("-" * 40)
            for error in results['errors']:
                report.append(f"Question {error['question_id'] + 1}: {error['error']}")
        
        report.append("\n" + "=" * 80)
        return "\n".join(report)


def main():
    """Main evaluation function"""
    parser = argparse.ArgumentParser(description="Evaluate an existing testset against a RAG endpoint")
    parser.add_argument('--config', default='config/evaluation_smoke_config.yaml', help='Evaluation config file path')
    parser.add_argument('--testset', help='Optional existing testset path; otherwise the latest output is used')
    args = parser.parse_args()

    print("🚀 SMT RAG Evaluation with Existing Testset")
    print("=" * 60)
    
    try:
        # Initialize evaluator
        evaluator = SMTRAGEvaluator(args.config)

        if args.testset:
            testset_path = args.testset
        else:
            testset_files = sorted(
                EVAL_PIPELINE_ROOT.glob('outputs/*/testsets/*.csv'),
                key=lambda path: path.stat().st_ctime,
            )
            if not testset_files:
                print("❌ No testset found! Please run testset generation first.")
                return
            testset_path = str(testset_files[-1])
            print(f"📂 Using most recent testset: {testset_path}")
        
        # Load testset
        testset_df = evaluator.load_testset(testset_path)
        
        # Run evaluation
        results = evaluator.evaluate_testset(testset_df)
        
        # Save results
        output_file = evaluator.save_results(results)
        
        # Generate and display report
        report = evaluator.generate_report(results)
        print("\n" + report)
        
        # Save report to file
        report_file = output_file.replace('.json', '_report.txt')
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n📝 Report saved to: {report_file}")
        
        print(f"\n🎉 Evaluation completed successfully!")
        print(f"📊 Results: {output_file}")
        print(f"📝 Report: {report_file}")
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
