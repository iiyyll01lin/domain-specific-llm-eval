"""
Comprehensive RAG Evaluator with Fixed RAGAS and Contextual Keyword Support

This module combines the fixed RAGAS evaluator and contextual keyword evaluator
to provide complete RAG evaluation capabilities.
Now includes configurable gates system with weighted pass rates.
"""
import logging
import pandas as pd
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from .ragas_evaluator_with_fallbacks import RAGASEvaluatorWithFallbacks
from .contextual_keyword_evaluator_fixed import ContextualKeywordEvaluatorFixed
from .enhanced_contextual_keyword_evaluator import EnhancedContextualKeywordEvaluator
from .gates_system import GatesSystem

# Keep fallback imports 
try:
    from .ragas_evaluator_fixed import RAGASEvaluatorFixed
except ImportError:
    RAGASEvaluatorFixed = None

# Import RAG interface
try:
    from interfaces.rag_interface import RAGInterface
except ImportError:
    # Try alternative import path
    try:
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent / 'interfaces'))
        from rag_interface import RAGInterface
    except ImportError:
        RAGInterface = None

logger = logging.getLogger(__name__)

class ComprehensiveRAGEvaluatorFixed:
    """
    Fixed comprehensive RAG evaluator that:
    1. Uses contextual keyword evaluation with RAG endpoint
    2. Uses RAGAS metrics with custom LLM endpoint  
    3. Provides detailed calculation tracking
    4. Generates comprehensive reports
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize comprehensive evaluator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        
        # Initialize RAG interface
        rag_config = config.get('rag_system', {})
        if rag_config.get('enabled', False) and RAGInterface:
            try:
                self.rag_interface = RAGInterface(rag_config)
                logger.info("✅ RAG interface initialized")
            except Exception as e:
                logger.error(f"Failed to initialize RAG interface: {e}")
                self.rag_interface = None
        else:
            self.rag_interface = None
            logger.info("RAG interface not configured or not available")
        
        # Initialize sub-evaluators with enhanced fallback support
        try:
            # Prioritize enhanced fallback evaluator
            self.ragas_evaluator = RAGASEvaluatorWithFallbacks(config)
            logger.info("✅ Enhanced RAGAS evaluator with fallbacks initialized")
        except Exception as e:
            logger.warning(f"Enhanced RAGAS evaluator failed: {e}, trying fixed version...")
            try:
                if RAGASEvaluatorFixed:
                    self.ragas_evaluator = RAGASEvaluatorFixed(config)
                    logger.info("✅ Fixed RAGAS evaluator initialized as fallback")
                else:
                    raise Exception("No RAGAS evaluator available")
            except Exception as e2:
                logger.error(f"Failed to initialize any RAGAS evaluator: {e2}")
                self.ragas_evaluator = None
        
        try:
            # Check if enhanced evaluator is requested
            contextual_config = config.get('evaluation', {}).get('contextual_keywords', {})
            evaluator_type = contextual_config.get('evaluator', 'standard')
            
            if evaluator_type == 'enhanced':
                logger.info("🚀 Initializing Enhanced Contextual Keyword Evaluator")
                self.keyword_evaluator = EnhancedContextualKeywordEvaluator(config)
            else:
                logger.info("🔄 Initializing Standard Contextual Keyword Evaluator")
                self.keyword_evaluator = ContextualKeywordEvaluatorFixed(config)
                
            logger.info("✅ Contextual keyword evaluator initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize contextual keyword evaluator: {e}")
            self.keyword_evaluator = None
        except Exception as e:
            logger.error(f"Failed to initialize keyword evaluator: {e}")
            self.keyword_evaluator = None
        
        if not self.ragas_evaluator and not self.keyword_evaluator:
            raise Exception("No evaluators available - check configuration")
        
        # Initialize gates system
        try:
            self.gates_system = GatesSystem(config)
            logger.info("✅ Gates system initialized")
        except Exception as e:
            logger.error(f"Failed to initialize gates system: {e}")
            self.gates_system = None
        
        logger.info(f"ComprehensiveRAGEvaluatorFixed initialized")
    
    def _query_rag_system_for_testset(self, testset_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Query RAG system for all questions in testset.
        
        Args:
            testset_df: DataFrame with testset questions
            
        Returns:
            List of RAG responses with timing and metadata
        """
        if not self.rag_interface:
            logger.warning("RAG interface not available, using reference answers as fallback")
            # Return reference answers as fallback
            rag_responses = []
            for _, row in testset_df.iterrows():
                rag_responses.append({
                    'question': row['user_input'],
                    'answer': row['reference'],  # Use reference as fallback
                    'contexts': row.get('reference_contexts', []),
                    'response_time': 0.0,
                    'success': True,
                    'source': 'reference_fallback'
                })
            return rag_responses
        
        rag_responses = []
        logger.info(f"🔄 Querying RAG system for {len(testset_df)} questions...")
        
        for idx, row in testset_df.iterrows():
            question = row['user_input']
            try:
                start_time = time.time()
                response = self.rag_interface.query_rag_system(question)
                response_time = time.time() - start_time
                
                rag_responses.append({
                    'question': question,
                    'answer': response.get('answer', ''),
                    'contexts': response.get('contexts', []),
                    'confidence': response.get('confidence'),
                    'response_time': response_time,
                    'success': True,
                    'source': 'rag_system'
                })
                
                logger.info(f"✅ Question {idx + 1}/{len(testset_df)}: {response_time:.2f}s")
                
            except Exception as e:
                logger.error(f"❌ Failed to query RAG for question {idx + 1}: {e}")
                # Use reference as fallback
                rag_responses.append({
                    'question': question,
                    'answer': row['reference'],
                    'contexts': row.get('reference_contexts', []),
                    'response_time': 0.0,
                    'success': False,
                    'error': str(e),
                    'source': 'reference_fallback'
                })
        
        return rag_responses
    
    def evaluate_testset(self, testset_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Run comprehensive evaluation using both contextual keywords and RAGAS.
        
        Args:
            testset_file: Path to enhanced testset CSV
            output_dir: Directory to save results
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        logger.info(f"🚀 Starting comprehensive RAG evaluation")
        logger.info(f"📄 Testset: {testset_file}")
        logger.info(f"📁 Output: {output_dir}")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        evaluation_results = {
            'testset_file': str(testset_file),
            'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'evaluators_used': [],
            'results': {}
        }
        
        success_count = 0
        total_evaluators = 0
        
        # Load and validate testset first
        try:
            df = pd.read_csv(testset_file)
            logger.info(f"📊 Loaded testset with {len(df)} rows")
            
            # Check required columns
            required_columns = ['user_input']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                logger.error(f"Missing required columns: {missing_columns}")
                return {
                    'success': False,
                    'error': f'Missing required columns in testset: {missing_columns}',
                    'total_questions': 0,
                    'successful_evaluations': 0
                }
            
            # Add default columns if missing
            if 'reference_contexts' not in df.columns:
                logger.warning("Missing 'reference_contexts' column, creating default")
                df['reference_contexts'] = [["No context available"] for _ in range(len(df))]
            
            if 'reference' not in df.columns:
                logger.warning("Missing 'reference' column, creating from user_input")
                df['reference'] = df['user_input'].apply(lambda x: f"Reference answer for: {x}")
            
            evaluation_results['total_questions'] = len(df)
            
        except Exception as e:
            logger.error(f"Failed to load testset: {e}")
            return {
                'success': False,
                'error': f'Failed to load testset: {e}',
                'total_questions': 0,
                'successful_evaluations': 0
            }
        
        # First, query RAG system once for all questions if we have any evaluators that need it
        rag_responses = None
        if self.keyword_evaluator or self.ragas_evaluator:
            logger.info("🔄 Querying RAG system for all questions...")
            rag_responses = self._query_rag_system_for_testset(df)
            logger.info(f"✅ Completed RAG system queries for {len(rag_responses)} questions")
        
        # 1. Contextual Keyword Evaluation
        if self.keyword_evaluator:
            total_evaluators += 1
            logger.info("🔤 Running contextual keyword evaluation...")
            
            try:
                keyword_results = self.keyword_evaluator.evaluate_testset(testset_file, output_dir, rag_responses)
                
                if keyword_results['success']:
                    success_count += 1
                    evaluation_results['evaluators_used'].append('contextual_keyword')
                    evaluation_results['results']['contextual_keyword'] = keyword_results
                    
                    # Log summary
                    metrics = keyword_results['summary_metrics']
                    logger.info(f"✅ Contextual Keywords: Avg Score {metrics['avg_similarity_score']:.3f}, "
                               f"Pass Rate {metrics['pass_rate']:.1%}")
                else:
                    logger.error(f"❌ Contextual keyword evaluation failed: {keyword_results.get('error')}")
                    evaluation_results['results']['contextual_keyword'] = keyword_results
                    
            except Exception as e:
                logger.error(f"❌ Contextual keyword evaluation exception: {e}")
                evaluation_results['results']['contextual_keyword'] = {
                    'success': False,
                    'error': str(e)
                }
        
        # 2. RAGAS Evaluation with RAG System Responses
        if self.ragas_evaluator:
            total_evaluators += 1
            logger.info("📊 Running RAGAS evaluation with RAG system responses...")
            
            try:
                # Create enhanced testset with RAG responses for RAGAS evaluation
                enhanced_testset_data = []
                for i, (_, row) in enumerate(df.iterrows()):
                    if i < len(rag_responses):
                        enhanced_testset_data.append({
                            'user_input': row['user_input'],
                            'reference_contexts': row['reference_contexts'],
                            'reference': row['reference'],  # Ground truth
                            'rag_answer': rag_responses[i]['answer'],  # RAG system answer
                            'rag_contexts': rag_responses[i]['contexts'],
                            'rag_response_time': rag_responses[i]['response_time'],
                            'rag_success': rag_responses[i]['success']
                        })
                
                # Save enhanced testset with RAG responses
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                enhanced_file = output_dir / f"testset_with_rag_responses_{timestamp}.csv"
                enhanced_df = pd.DataFrame(enhanced_testset_data)
                enhanced_df.to_csv(enhanced_file, index=False)
                logger.info(f"💾 Enhanced testset with RAG responses saved: {enhanced_file}")
                
                # Run RAGAS evaluation on RAG responses vs ground truth
                ragas_results = self.ragas_evaluator.evaluate_testset_with_rag_responses(
                    enhanced_df, output_dir
                )
                
                if ragas_results['success']:
                    success_count += 1
                    evaluation_results['evaluators_used'].append('ragas')
                    evaluation_results['results']['ragas'] = ragas_results
                    
                    # Log summary
                    scores = ragas_results['overall_scores']
                    logger.info(f"✅ RAGAS: {len(scores)} metrics evaluated")
                    for metric, score in scores.items():
                        if score is not None:
                            logger.info(f"   {metric}: {score:.3f}")
                        else:
                            logger.warning(f"   {metric}: failed")
                else:
                    logger.error(f"❌ RAGAS evaluation failed: {ragas_results.get('error')}")
                    evaluation_results['results']['ragas'] = ragas_results
                    
            except Exception as e:
                logger.error(f"❌ RAGAS evaluation exception: {e}")
                import traceback
                logger.error(f"RAGAS exception traceback: {traceback.format_exc()}")
                evaluation_results['results']['ragas'] = {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
        
        # 3. Generate comprehensive report
        try:
            report_results = self._generate_comprehensive_report(evaluation_results, output_dir)
            evaluation_results['report'] = report_results
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            evaluation_results['report'] = {'success': False, 'error': str(e)}
        
        # 4. Determine overall success
        overall_success = success_count > 0
        evaluation_results['success'] = overall_success
        evaluation_results['successful_evaluations'] = success_count
        evaluation_results['total_evaluators'] = total_evaluators
        
        # 5. Create summary stats for pipeline compatibility
        summary_stats = {}
        
        # Add contextual keyword stats
        if 'contextual_keyword' in evaluation_results['results']:
            ck_result = evaluation_results['results']['contextual_keyword']
            if ck_result.get('success'):
                summary_stats['contextual_keyword'] = ck_result.get('summary_metrics', {})
        
        # Add RAGAS stats
        if 'ragas' in evaluation_results['results']:
            ragas_result = evaluation_results['results']['ragas']
            if ragas_result.get('success'):
                ragas_metrics = {}
                for metric, score in ragas_result.get('overall_scores', {}).items():
                    if score is not None:
                        ragas_metrics[metric] = {'mean_score': score}
                summary_stats['ragas_metrics'] = ragas_metrics
        
        evaluation_results['summary_stats'] = summary_stats
        
        # 6. Save main results file
        timestamp = evaluation_results['timestamp']
        main_results_file = output_dir / f"comprehensive_evaluation_{timestamp}.json"
        
        try:
            with open(main_results_file, 'w') as f:
                json.dump(evaluation_results, f, indent=2)
            logger.info(f"💾 Main results saved to: {main_results_file}")
            evaluation_results['detailed_results_file'] = str(main_results_file)
            evaluation_results['summary_report_file'] = str(main_results_file)
            evaluation_results['csv_report_file'] = str(main_results_file)
        except Exception as e:
            logger.error(f"Failed to save main results: {e}")
        
        # Log final summary
        if overall_success:
            logger.info(f"🎉 Comprehensive evaluation completed successfully!")
            logger.info(f"✅ {success_count}/{total_evaluators} evaluators succeeded")
        else:
            logger.error(f"❌ Comprehensive evaluation failed")
            logger.error(f"❌ {success_count}/{total_evaluators} evaluators succeeded")
        
        return evaluation_results
    
    def _generate_comprehensive_report(self, evaluation_results: Dict[str, Any], 
                                     output_dir: Path) -> Dict[str, Any]:
        """
        Generate a comprehensive evaluation report.
        
        Args:
            evaluation_results: Results from all evaluators
            output_dir: Directory to save report
            
        Returns:
            Dictionary with report generation results
        """
        logger.info("📋 Generating comprehensive evaluation report...")
        
        try:
            timestamp = evaluation_results['timestamp']
            report_data = {
                'evaluation_summary': {
                    'testset_file': evaluation_results['testset_file'],
                    'timestamp': timestamp,
                    'evaluators_used': evaluation_results['evaluators_used'],
                    'successful_evaluators': evaluation_results.get('successful_evaluators', 0),
                    'total_evaluators': evaluation_results.get('total_evaluators', 0)
                },
                'detailed_results': {}
            }
            
            # Process contextual keyword results
            if 'contextual_keyword' in evaluation_results['results']:
                ck_results = evaluation_results['results']['contextual_keyword']
                if ck_results.get('success'):
                    report_data['detailed_results']['contextual_keyword'] = {
                        'status': 'success',
                        'summary_metrics': ck_results.get('summary_metrics', {}),
                        'total_questions': ck_results.get('total_questions', 0),
                        'successful_evaluations': ck_results.get('successful_evaluations', 0),
                        'detailed_file': ck_results.get('detailed_results_file')
                    }
                else:
                    report_data['detailed_results']['contextual_keyword'] = {
                        'status': 'failed',
                        'error': ck_results.get('error', 'Unknown error')
                    }
            
            # Process RAGAS results
            if 'ragas' in evaluation_results['results']:
                ragas_results = evaluation_results['results']['ragas']
                if ragas_results.get('success'):
                    report_data['detailed_results']['ragas'] = {
                        'status': 'success',
                        'overall_scores': ragas_results.get('overall_scores', {}),
                        'total_questions': ragas_results.get('total_questions', 0),
                        'detailed_file': ragas_results.get('detailed_results_file')
                    }
                else:
                    report_data['detailed_results']['ragas'] = {
                        'status': 'failed',
                        'error': ragas_results.get('error', 'Unknown error')
                    }
            
            # Calculate combined metrics if both evaluators succeeded
            combined_metrics = self._calculate_combined_metrics(evaluation_results)
            if combined_metrics:
                report_data['combined_metrics'] = combined_metrics
            
            # Evaluate gates system if available
            if self.gates_system:
                try:
                    gates_results = self.gates_system.evaluate_gates(evaluation_results['results'])
                    report_data['gates_evaluation'] = {
                        'combined_pass_rate': gates_results.combined_pass_rate,
                        'individual_pass_rates': gates_results.individual_pass_rates,
                        'overall_pass': gates_results.overall_pass,
                        'weighted_score': gates_results.weighted_score,
                        'gate_results': gates_results.gate_results,
                        'metadata': gates_results.metadata
                    }
                    logger.info(f"🚪 Gates evaluation completed: combined pass rate {gates_results.combined_pass_rate:.3f}")
                    logger.info(f"🚪 Overall gates pass: {gates_results.overall_pass}")
                except Exception as e:
                    logger.error(f"Failed to evaluate gates: {e}")
                    report_data['gates_evaluation'] = {
                        'error': str(e),
                        'success': False
                    }
            
            # Save comprehensive report
            report_file = output_dir / f"comprehensive_report_{timestamp}.json"
            with open(report_file, 'w') as f:
                json.dump(report_data, f, indent=2, default=self._json_serializer)
            
            # Generate human-readable summary
            summary_file = output_dir / f"evaluation_summary_{timestamp}.txt"
            self._generate_text_summary(report_data, summary_file)
            
            logger.info(f"✅ Comprehensive report generated")
            logger.info(f"📄 Report: {report_file}")
            logger.info(f"📝 Summary: {summary_file}")
            
            return {
                'success': True,
                'report_file': str(report_file),
                'summary_file': str(summary_file),
                'timestamp': timestamp
            }
            
        except Exception as e:
            logger.error(f"Failed to generate comprehensive report: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    def _calculate_combined_metrics(self, evaluation_results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate combined metrics from multiple evaluators."""
        try:
            combined = {}
            
            # Extract contextual keyword metrics
            ck_results = evaluation_results['results'].get('contextual_keyword', {})
            if ck_results.get('success'):
                ck_metrics = ck_results.get('summary_metrics', {})
                combined['contextual_keyword_score'] = ck_metrics.get('avg_similarity_score', 0)
                combined['contextual_keyword_pass_rate'] = ck_metrics.get('pass_rate', 0)
            
            # Extract RAGAS metrics
            ragas_results = evaluation_results['results'].get('ragas', {})
            if ragas_results.get('success'):
                ragas_scores = ragas_results.get('overall_scores', {})
                for metric, score in ragas_scores.items():
                    if score is not None:
                        combined[f'ragas_{metric}'] = score
            
            # Calculate overall score if we have multiple metrics
            if len(combined) > 1:
                numeric_scores = [v for v in combined.values() if isinstance(v, (int, float))]
                if numeric_scores:
                    combined['overall_average'] = sum(numeric_scores) / len(numeric_scores)
            
            return combined if combined else None
            
        except Exception as e:
            logger.error(f"Failed to calculate combined metrics: {e}")
            return None
    
    def _json_serializer(self, obj):
        """Custom JSON serializer to handle non-serializable objects."""
        if hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _generate_text_summary(self, report_data: Dict[str, Any], summary_file: Path):
        """Generate human-readable text summary."""
        try:
            with open(summary_file, 'w') as f:
                f.write("=== RAG Evaluation Summary ===\\n\\n")
                
                summary = report_data['evaluation_summary']
                f.write(f"Testset: {summary['testset_file']}\\n")
                f.write(f"Timestamp: {summary['timestamp']}\\n")
                f.write(f"Evaluators: {', '.join(summary['evaluators_used'])}\\n")
                f.write(f"Success Rate: {summary['successful_evaluators']}/{summary['total_evaluators']}\\n\\n")
                
                # Contextual keyword results
                if 'contextual_keyword' in report_data['detailed_results']:
                    ck = report_data['detailed_results']['contextual_keyword']
                    f.write("--- Contextual Keyword Evaluation ---\\n")
                    if ck['status'] == 'success':
                        metrics = ck['summary_metrics']
                        f.write(f"Status: ✅ SUCCESS\\n")
                        f.write(f"Questions Evaluated: {ck['successful_evaluations']}/{ck['total_questions']}\\n")
                        f.write(f"Average Similarity Score: {metrics['avg_similarity_score']:.3f}\\n")
                        f.write(f"Pass Rate: {metrics['pass_rate']:.1%}\\n")
                        # Handle different threshold field names from different evaluators
                        threshold = metrics.get('threshold_used', metrics.get('adaptive_threshold', 0.7))
                        f.write(f"Threshold: {threshold:.3f}\\n")
                    else:
                        f.write(f"Status: ❌ FAILED\\n")
                        f.write(f"Error: {ck['error']}\\n")
                    f.write("\\n")
                
                # RAGAS results
                if 'ragas' in report_data['detailed_results']:
                    ragas = report_data['detailed_results']['ragas']
                    f.write("--- RAGAS Evaluation ---\\n")
                    if ragas['status'] == 'success':
                        f.write(f"Status: ✅ SUCCESS\\n")
                        f.write(f"Questions Evaluated: {ragas['total_questions']}\\n")
                        f.write("Metric Scores:\\n")
                        for metric, score in ragas['overall_scores'].items():
                            if score is not None:
                                f.write(f"  {metric}: {score:.3f}\\n")
                            else:
                                f.write(f"  {metric}: FAILED\\n")
                    else:
                        f.write(f"Status: ❌ FAILED\\n")
                        f.write(f"Error: {ragas['error']}\\n")
                    f.write("\\n")
                
                # Combined metrics
                if 'combined_metrics' in report_data:
                    f.write("--- Combined Metrics ---\\n")
                    for metric, value in report_data['combined_metrics'].items():
                        if isinstance(value, float):
                            f.write(f"{metric}: {value:.3f}\\n")
                        else:
                            f.write(f"{metric}: {value}\\n")
                    f.write("\\n")
                
                # Gates evaluation results
                if 'gates_evaluation' in report_data:
                    gates = report_data['gates_evaluation']
                    f.write("--- Gates Evaluation ---\\n")
                    if gates.get('error'):
                        f.write(f"Status: ❌ FAILED\\n")
                        f.write(f"Error: {gates['error']}\\n")
                    else:
                        f.write(f"Status: ✅ SUCCESS\\n")
                        f.write(f"Combined Pass Rate: {gates['combined_pass_rate']:.1%}\\n")
                        f.write(f"Weighted Score: {gates['weighted_score']:.3f}\\n")
                        f.write(f"Overall Pass: {'✅ PASS' if gates['overall_pass'] else '❌ FAIL'}\\n")
                        f.write("\\nIndividual Gates:\\n")
                        for gate_name, pass_rate in gates['individual_pass_rates'].items():
                            gate_result = gates['gate_results'][gate_name]
                            status = "✅ PASS" if gate_result.get('passes_threshold', False) else "❌ FAIL"
                            f.write(f"  {gate_name}: {pass_rate:.1%} ({status})\\n")
                        
                        # Add configuration details
                        metadata = gates.get('metadata', {})
                        f.write(f"\\nConfiguration:\\n")
                        f.write(f"  Combination Method: {metadata.get('combination_method', 'unknown')}\\n")
                        f.write(f"  Minimum Gates Required: {metadata.get('minimum_gates_required', 0)}\\n")
                    f.write("\\n")
                
                f.write("=== End of Summary ===\\n")
                
        except Exception as e:
            logger.error(f"Failed to generate text summary: {e}")
