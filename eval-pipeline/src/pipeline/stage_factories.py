"""
Stage Factories for Pipeline Execution

This module provides factories that create standardized stage executors,
separating core execution logic from validation and enhancement features.
"""

import json
import logging
import sys
import pandas as pd
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import core stage components
from data.document_processor import DocumentProcessor
from data.hybrid_testset_generator import HybridTestsetGenerator
from evaluation.rag_evaluator import RAGEvaluator
from evaluation.contextual_keyword_evaluator import ContextualKeywordEvaluator
from evaluation.ragas_evaluator import RagasEvaluator
from evaluation.human_feedback_manager import HumanFeedbackManager
from reports.report_generator import ReportGenerator
from interfaces.rag_interface import RAGInterface

logger = logging.getLogger(__name__)

class StageExecutor(ABC):
    """Abstract base class for stage executors."""
    
    def __init__(self, config: Dict[str, Any], run_id: str, output_dirs: Dict[str, Path]):
        self.config = config
        self.run_id = run_id
        self.output_dirs = output_dirs
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
    
    @abstractmethod
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute the stage and return results."""
        pass
    
    def log_stage_start(self, stage_name: str):
        """Log stage start with consistent formatting."""
        self.logger.info(f"🎬 Starting {stage_name} stage...")
        
    def log_stage_end(self, stage_name: str, success: bool, duration: float = None):
        """Log stage end with consistent formatting."""
        status = "✅ completed" if success else "❌ failed"
        duration_str = f" ({duration:.2f}s)" if duration else ""
        self.logger.info(f"🏁 {stage_name} stage {status}{duration_str}")

class DocumentProcessingStageExecutor(StageExecutor):
    """Core document processing stage executor."""
    
    def __init__(self, config: Dict[str, Any], run_id: str, output_dirs: Dict[str, Path]):
        super().__init__(config, run_id, output_dirs)
        self.document_processor = DocumentProcessor(config, output_dirs.get('temp', output_dirs['base']))
    
    def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute document processing stage."""
        self.log_stage_start("Document Processing")
        start_time = datetime.now()
        
        try:
            # Core document processing logic
            processed_documents = self.document_processor.process_documents()
            
            # Save processed documents
            output_path = self.output_dirs['documents'] / f"processed_documents_{self.run_id}.json"
            self.document_processor.save_processed_documents(processed_documents, output_path)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_stage_end("Document Processing", True, duration)
            
            return {
                'success': True,
                'documents_processed': len(processed_documents),
                'output_path': str(output_path),
                'documents': processed_documents,
                'duration': duration
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_stage_end("Document Processing", False, duration)
            self.logger.error(f"Document processing failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }

class TestsetGenerationStageExecutor(StageExecutor):
    """Core testset generation stage executor."""
    
    def __init__(self, config: Dict[str, Any], run_id: str, output_dirs: Dict[str, Path]):
        super().__init__(config, run_id, output_dirs)
        self.testset_generator = HybridTestsetGenerator(config)
    
    def execute(self, documents: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Execute testset generation stage."""
        self.log_stage_start("Testset Generation")
        start_time = datetime.now()
        
        try:
            # If no documents provided, load them
            if documents is None:
                # Extract document configuration for DocumentProcessor
                doc_config = self._extract_document_config(self.config)
                document_processor = DocumentProcessor(doc_config, self.output_dirs.get('temp', self.output_dirs['base']))
                documents = document_processor.process_documents()
            
            # Core testset generation logic
            testsets = self.testset_generator.generate_testsets(documents)
            
            # Save testsets
            output_path = self.output_dirs['testsets'] / f"testset_{self.run_id}.json"
            self.testset_generator.save_testsets(testsets, output_path)
            
            # Calculate total QA pairs
            total_qa_pairs = sum(len(ts.get('qa_pairs', [])) for ts in testsets)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_stage_end("Testset Generation", True, duration)
            
            return {
                'success': True,
                'testsets_generated': len(testsets),
                'total_qa_pairs': total_qa_pairs,
                'output_path': str(output_path),
                'testsets': testsets,
                'duration': duration
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_stage_end("Testset Generation", False, duration)
            self.logger.error(f"Testset generation failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    def _extract_document_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract document configuration for DocumentProcessor."""
        doc_config = {}
        
        # Extract data sources
        data_sources = config.get('data_sources', {})
        
        # CSV files
        csv_config = data_sources.get('csv', {})
        if csv_config.get('csv_files'):
            doc_config['csv_files'] = csv_config['csv_files']
        
        # Documents
        documents_config = data_sources.get('documents', {})
        if documents_config.get('primary_docs'):
            doc_config['primary_docs'] = documents_config['primary_docs']
        if documents_config.get('pdf_files'):
            doc_config['pdf_files'] = documents_config['pdf_files']
        if documents_config.get('text_files'):
            doc_config['text_files'] = documents_config['text_files']
        if documents_config.get('directories'):
            doc_config['directories'] = documents_config['directories']
        
        # Processing configuration
        if documents_config.get('processing'):
            doc_config['processing'] = documents_config['processing']
        
        return doc_config

class EvaluationStageExecutor(StageExecutor):
    """Core evaluation stage executor."""
    
    def __init__(self, config: Dict[str, Any], run_id: str, output_dirs: Dict[str, Path]):
        super().__init__(config, run_id, output_dirs)
        self.rag_evaluator = RAGEvaluator(config)
        self.contextual_keyword_evaluator = ContextualKeywordEvaluator(config)
        self.ragas_evaluator = RagasEvaluator(config)
        self.human_feedback_manager = HumanFeedbackManager(config)
    
    def execute(self, testsets: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """Execute evaluation stage."""
        self.log_stage_start("Evaluation")
        start_time = datetime.now()
        
        try:
            # If no testsets provided, load them
            if testsets is None:
                testset_path = self.output_dirs['testsets'] / f"testset_{self.run_id}.json"
                if not testset_path.exists():
                    raise FileNotFoundError(f"Testset file not found: {testset_path}")
                
                # Load testsets with dtype restoration for numeric columns
                testsets = self._load_testsets_with_dtypes(testset_path)
            
            # Initialize evaluation results
            evaluation_results = {
                'rag_results': [],
                'contextual_keyword_results': [],
                'ragas_results': [],
                'human_feedback_results': []
            }
            
            # Process each testset
            queries_executed = 0
            for testset in testsets:
                qa_pairs = testset.get('qa_pairs', [])
                queries_executed += len(qa_pairs)
                
                # RAG evaluation
                rag_results = self.rag_evaluator.evaluate_testset(testset)
                evaluation_results['rag_results'].extend(rag_results)
                
                # Contextual keyword evaluation
                keyword_results = self.contextual_keyword_evaluator.evaluate_testset(testset)
                evaluation_results['contextual_keyword_results'].extend(keyword_results)
                
                # RAGAS evaluation
                ragas_results = self.ragas_evaluator.evaluate_testset(testset)
                evaluation_results['ragas_results'].extend(ragas_results)
                
                # Human feedback (if enabled)
                if self.config.get('evaluation', {}).get('human_feedback', {}).get('enabled', False):
                    feedback_results = self.human_feedback_manager.process_testset(testset)
                    evaluation_results['human_feedback_results'].extend(feedback_results)
            
            # Calculate aggregate metrics
            keyword_pass_rate = self._calculate_keyword_pass_rate(evaluation_results['contextual_keyword_results'])
            avg_ragas_score = self._calculate_average_ragas_score(evaluation_results['ragas_results'])
            feedback_requests = len(evaluation_results['human_feedback_results'])
            
            # Save evaluation results
            output_path = self.output_dirs.get('evaluations', self.output_dirs['base']) / f"evaluation_results_{self.run_id}.json"
            with open(output_path, 'w') as f:
                json.dump(evaluation_results, f, indent=2, default=str)
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_stage_end("Evaluation", True, duration)
            
            return {
                'success': True,
                'queries_executed': queries_executed,
                'keyword_pass_rate': keyword_pass_rate,
                'avg_ragas_score': avg_ragas_score,
                'feedback_requests': feedback_requests,
                'output_path': str(output_path),
                'evaluation_results': evaluation_results,
                'duration': duration
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_stage_end("Evaluation", False, duration)
            self.logger.error(f"Evaluation failed: {e}")
            self.logger.error(f"Exception type: {type(e).__name__}")
            self.logger.error(f"Exception details: {str(e)}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }
    
    def _calculate_keyword_pass_rate(self, keyword_results: List[Dict[str, Any]]) -> float:
        """Calculate keyword matching pass rate."""
        if not keyword_results:
            return 0.0
        
        passed = sum(1 for result in keyword_results if result.get('passed', False))
        return passed / len(keyword_results)
    
    def _calculate_average_ragas_score(self, ragas_results: List[Dict[str, Any]]) -> float:
        """Calculate average RAGAS score."""
        if not ragas_results:
            return 0.0
        
        total_score = 0.0
        valid_scores = 0
        
        for result in ragas_results:
            # Try different score keys that might be present
            score = None
            if 'ragas_score' in result:
                score = result['ragas_score']
            elif 'score' in result:
                score = result['score']
            elif 'overall_score' in result:
                score = result['overall_score']
            elif 'ragas_composite_score' in result:
                score = result['ragas_composite_score']
            
            if score is not None and isinstance(score, (int, float)) and not pd.isna(score):
                total_score += float(score)
                valid_scores += 1
        
        return total_score / valid_scores if valid_scores > 0 else 0.0

    def _load_testsets_with_dtypes(self, testset_path: Path) -> List[Dict[str, Any]]:
        """
        Load testsets with numeric dtype restoration.
        
        Args:
            testset_path: Path to the JSON testset file
            
        Returns:
            List of testsets with numeric columns properly typed
        """
        import pandas as pd
        
        try:
            with open(testset_path, 'r') as f:
                testsets = json.load(f)
            
            # Process each testset to restore numeric dtypes
            processed_testsets = []
            for testset in testsets:
                if 'qa_pairs' in testset:
                    # Convert qa_pairs to DataFrame for dtype restoration
                    qa_pairs = testset['qa_pairs']
                    if qa_pairs:
                        df = pd.DataFrame(qa_pairs)
                        
                        # Restore numeric columns
                        df = self._restore_numeric_dtypes(df)
                        
                        # Convert back to dict records
                        testset['qa_pairs'] = df.to_dict('records')
                
                processed_testsets.append(testset)
            
            return processed_testsets
            
        except Exception as e:
            self.logger.warning(f"Failed to restore dtypes for testsets, using raw JSON: {e}")
            # Fallback to raw JSON loading
            with open(testset_path, 'r') as f:
                return json.load(f)
    
    def _restore_numeric_dtypes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restore numeric data types for columns that should be numeric.
        
        Args:
            df: DataFrame to process
            
        Returns:
            DataFrame with restored numeric dtypes
        """
        # List of columns that should be numeric based on our knowledge
        numeric_columns = [
            'question_score', 'answer_score', 'context_score', 'overall_score',
            'relevance_score', 'coherence_score', 'groundedness_score',
            'ragas_faithfulness', 'ragas_answer_relevancy', 'ragas_context_precision',
            'ragas_context_recall', 'ragas_answer_correctness', 'ragas_answer_similarity',
            'ragas_composite_score'
        ]
        
        for col in numeric_columns:
            if col in df.columns:
                try:
                    # Convert to numeric, handling any conversion errors gracefully
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except Exception as e:
                    self.logger.debug(f"Could not convert {col} to numeric: {e}")
        
        return df

class ReportingStageExecutor(StageExecutor):
    """Core reporting stage executor."""
    
    def __init__(self, config: Dict[str, Any], run_id: str, output_dirs: Dict[str, Path]):
        super().__init__(config, run_id, output_dirs)
        self.report_generator = ReportGenerator(config)
    
    def execute(self, evaluation_results: Dict[str, Any] = None, **kwargs) -> Dict[str, Any]:
        """Execute reporting stage."""
        self.log_stage_start("Reporting")
        start_time = datetime.now()
        
        try:
            # If no evaluation results provided, load them
            if evaluation_results is None:
                eval_path = self.output_dirs.get('evaluations', self.output_dirs['base']) / f"evaluation_results_{self.run_id}.json"
                if not eval_path.exists():
                    raise FileNotFoundError(f"Evaluation results file not found: {eval_path}")
                
                with open(eval_path, 'r') as f:
                    evaluation_results = json.load(f)
            
            # Generate reports using the comprehensive report method
            # Convert evaluation results to DataFrame if it's not already
            if isinstance(evaluation_results, dict) and 'evaluation_results' in evaluation_results:
                results_data = evaluation_results['evaluation_results']
                # Flatten all results into a single list
                all_results = []
                for key in ['rag_results', 'contextual_keyword_results', 'ragas_results', 'human_feedback_results']:
                    if key in results_data and isinstance(results_data[key], list):
                        all_results.extend(results_data[key])
                
                results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()
            else:
                results_df = pd.DataFrame()
            
            reports_generated = self.report_generator.generate_comprehensive_report(
                evaluation_results=results_df,
                evaluation_summary={
                    'run_id': self.run_id,
                    'timestamp': datetime.now().isoformat(),
                    'data_summary': evaluation_results
                },
                output_dir=self.output_dirs['reports']
            )
            
            duration = (datetime.now() - start_time).total_seconds()
            self.log_stage_end("Reporting", True, duration)
            
            return {
                'success': True,
                'reports_generated': reports_generated,
                'report_directory': str(self.output_dirs['reports']),
                'duration': duration
            }
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log_stage_end("Reporting", False, duration)
            self.logger.error(f"Report generation failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'duration': duration
            }

class StageFactory:
    """Factory for creating stage executors."""
    
    @staticmethod
    def create_executor(stage_name: str, config: Dict[str, Any], run_id: str, output_dirs: Dict[str, Path]) -> StageExecutor:
        """
        Create a stage executor for the given stage.
        
        Args:
            stage_name: Name of the stage to create executor for
            config: Pipeline configuration
            run_id: Unique run identifier
            output_dirs: Output directory structure
            
        Returns:
            StageExecutor instance
            
        Raises:
            ValueError: If stage_name is not recognized
        """
        stage_map = {
            'document-processing': DocumentProcessingStageExecutor,
            'testset-generation': TestsetGenerationStageExecutor,
            'evaluation': EvaluationStageExecutor,
            'reporting': ReportingStageExecutor
        }
        
        if stage_name not in stage_map:
            raise ValueError(f"Unknown stage: {stage_name}. Available stages: {list(stage_map.keys())}")
        
        executor_class = stage_map[stage_name]
        return executor_class(config, run_id, output_dirs)

class StageComposer:
    """Composes multiple stages for multi-stage pipeline runs."""
    
    def __init__(self, config: Dict[str, Any], run_id: str, output_dirs: Dict[str, Path]):
        self.config = config
        self.run_id = run_id
        self.output_dirs = output_dirs
        self.logger = logging.getLogger(__name__)
    
    def compose_stages(self, stage_names: List[str]) -> Dict[str, Any]:
        """
        Execute multiple stages in sequence, passing data between them.
        
        Args:
            stage_names: List of stage names to execute in order
            
        Returns:
            Combined results from all stages
        """
        self.logger.info(f"🎼 Composing pipeline with stages: {stage_names}")
        
        results = {
            'success': True,
            'stages_executed': [],
            'total_duration': 0.0
        }
        
        # Data to pass between stages
        stage_data = {}
        
        try:
            for stage_name in stage_names:
                self.logger.info(f"🎯 Executing stage: {stage_name}")
                
                # Create executor for this stage
                executor = StageFactory.create_executor(stage_name, self.config, self.run_id, self.output_dirs)
                
                # Execute stage with data from previous stages
                stage_result = executor.execute(**stage_data)
                
                # Store result
                results[stage_name.replace('-', '_')] = stage_result
                results['stages_executed'].append(stage_name)
                results['total_duration'] += stage_result.get('duration', 0.0)
                
                # Check if stage succeeded
                if not stage_result.get('success', False):
                    results['success'] = False
                    results['error'] = f"Stage {stage_name} failed: {stage_result.get('error', 'Unknown error')}"
                    break
                
                # Pass data to next stage
                if stage_name == 'document-processing':
                    stage_data['documents'] = stage_result.get('documents', [])
                elif stage_name == 'testset-generation':
                    stage_data['testsets'] = stage_result.get('testsets', [])
                elif stage_name == 'evaluation':
                    stage_data['evaluation_results'] = stage_result.get('evaluation_results', {})
            
            self.logger.info(f"🎉 Pipeline composition completed. Total duration: {results['total_duration']:.2f}s")
            
        except Exception as e:
            self.logger.error(f"💥 Pipeline composition failed: {e}")
            results['success'] = False
            results['error'] = str(e)
        
        return results
