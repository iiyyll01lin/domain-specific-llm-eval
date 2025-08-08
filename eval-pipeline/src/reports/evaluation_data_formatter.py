"""
Evaluation Data Formatter

Converts evaluation results from different evaluators into a standardized 
DataFrame format for report generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)

class EvaluationDataFormatter:
    """
    Formats evaluation results into standardized DataFrame for reporting.
    """
    
    def __init__(self):
        """Initialize the formatter."""
        pass
    
    def format_comprehensive_results(self, evaluation_results: Dict[str, Any], 
                                   testset_file: Optional[str] = None) -> pd.DataFrame:
        """
        Convert comprehensive evaluation results to standardized DataFrame.
        
        Args:
            evaluation_results: Results from comprehensive evaluation
            testset_file: Path to the testset file for question data
            
        Returns:
            Formatted DataFrame with standardized columns
        """
        try:
            # Load testset data if available
            testset_data = None
            if testset_file and Path(testset_file).exists():
                try:
                    testset_data = pd.read_csv(testset_file)
                    logger.info(f"Loaded testset data: {len(testset_data)} rows")
                except Exception as e:
                    logger.warning(f"Could not load testset file {testset_file}: {e}")
            
            # Extract results from different evaluators
            contextual_results = evaluation_results.get('results', {}).get('contextual_keyword', {})
            ragas_results = evaluation_results.get('results', {}).get('ragas', {})
            
            # Load detailed results
            detailed_data = []
            
            # Load contextual keyword detailed results
            contextual_detailed = self._load_detailed_results(
                contextual_results.get('detailed_file') or contextual_results.get('detailed_results_file')
            )
            logger.info(f"Loaded contextual detailed results: {len(contextual_detailed) if contextual_detailed else 'None'}")
            
            # Load RAGAS detailed results  
            ragas_detailed = self._load_detailed_results(
                ragas_results.get('detailed_file') or ragas_results.get('detailed_results_file')
            )
            logger.info(f"Loaded RAGAS detailed results: {len(ragas_detailed) if ragas_detailed else 'None'}")
            
            # Debug: print the file paths being used
            logger.info(f"Contextual file path: {contextual_results.get('detailed_file') or contextual_results.get('detailed_results_file')}")
            logger.info(f"RAGAS file path: {ragas_results.get('detailed_file') or ragas_results.get('detailed_results_file')}")
            
            # Determine the number of questions
            total_questions = evaluation_results.get('total_questions', 0)
            if total_questions == 0:
                if contextual_detailed:
                    total_questions = len(contextual_detailed)
                elif ragas_detailed:
                    total_questions = len(ragas_detailed)
                elif testset_data is not None:
                    total_questions = len(testset_data)
            
            # Build standardized records
            records = []
            for i in range(total_questions):
                record = self._create_standardized_record(
                    i, contextual_detailed, ragas_detailed, testset_data
                )
                records.append(record)
            
            # Create DataFrame
            df = pd.DataFrame(records)
            
            # Add overall calculations
            df = self._add_overall_metrics(df)
            
            logger.info(f"Formatted {len(df)} evaluation records")
            return df
            
        except Exception as e:
            logger.error(f"Error formatting evaluation results: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=self._get_expected_columns())
    
    def _load_detailed_results(self, file_path: Optional[str]) -> Optional[List[Dict]]:
        """Load detailed results from JSON file."""
        if not file_path:
            return None
            
        try:
            path = Path(file_path)
            
            # If the path is relative and doesn't exist, try relative to eval-pipeline directory
            if not path.is_absolute() and not path.exists():
                # Try relative to eval-pipeline directory
                eval_pipeline_dir = Path(__file__).parent.parent.parent  # Go up from src/reports/ to eval-pipeline/
                alternative_path = eval_pipeline_dir / file_path
                if alternative_path.exists():
                    path = alternative_path
                    logger.info(f"Found detailed results file at: {path}")
                else:
                    logger.warning(f"Detailed results file not found at: {file_path} or {alternative_path}")
                    return None
            elif not path.exists():
                logger.warning(f"Detailed results file not found: {file_path}")
                return None
                
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and 'detailed_results' in data:
                return data['detailed_results']
            else:
                logger.warning(f"Unexpected format in {file_path}")
                return None
                
        except Exception as e:
            logger.warning(f"Could not load detailed results from {file_path}: {e}")
            return None
    
    def _create_standardized_record(self, index: int, 
                                  contextual_detailed: Optional[List[Dict]],
                                  ragas_detailed: Optional[List[Dict]],
                                  testset_data: Optional[pd.DataFrame]) -> Dict[str, Any]:
        """Create a standardized record for one question."""
        record = {
            'question_index': index,
            'question': '',
            'rag_answer': '',
            'expected_answer': '',
            'source_file': 'unknown',
            
            # Contextual keyword metrics
            'contextual_total_score': 0.0,
            'contextual_keyword_pass': False,  # Fixed column name
            'contextual_pass': False,
            'contextual_similarity_score': 0.0,
            
            # RAGAS metrics
            'ragas_composite_score': 0.0,
            'ragas_pass': False,  # Added missing column
            'ragas_faithfulness': 0.0,
            'ragas_context_precision': 0.0,
            'ragas_context_recall': 0.0,
            'ragas_answer_relevancy': 0.0,
            
            # Semantic similarity
            'semantic_similarity': 0.0,  # Added missing column
            'semantic_pass': False,  # Added missing column
            
            # Overall metrics (calculated later)
            'overall_score': 0.0,
            'overall_pass': False,
        }
        
        # Fill from testset data
        if testset_data is not None and index < len(testset_data):
            row = testset_data.iloc[index]
            record['question'] = str(row.get('user_input', ''))
            record['expected_answer'] = str(row.get('reference', ''))
            record['source_file'] = 'testset'
        
        # Fill contextual keyword results
        if contextual_detailed and index < len(contextual_detailed):
            ctx_result = contextual_detailed[index]
            record['rag_answer'] = str(ctx_result.get('rag_answer', ''))
            record['question'] = record['question'] or str(ctx_result.get('question', ''))
            
            # Extract contextual metrics - use actual field names
            record['contextual_total_score'] = float(ctx_result.get('final_score', 0.0))
            record['contextual_similarity_score'] = float(ctx_result.get('best_method_score', 0.0))
            record['contextual_pass'] = bool(ctx_result.get('passes_threshold', False))
            record['contextual_keyword_pass'] = bool(ctx_result.get('passes_threshold', False))
        
        # Fill RAGAS results - Handle flat list structure
        if ragas_detailed:
            # RAGAS data is a flat list, need to group by question_index
            ragas_metrics = {}
            for item in ragas_detailed:
                if isinstance(item, dict) and item.get('question_index') == index:
                    metric_name = item.get('metric_name', '')
                    score = float(item.get('score', 0.0))
                    ragas_metrics[metric_name] = score
            
            # Extract individual RAGAS metrics
            record['ragas_faithfulness'] = ragas_metrics.get('Faithfulness', 0.0)
            record['ragas_context_precision'] = ragas_metrics.get('ContextPrecision', 0.0)
            record['ragas_context_recall'] = ragas_metrics.get('ContextRecall', 0.0)
            record['ragas_answer_relevancy'] = ragas_metrics.get('AnswerRelevancy', 0.0)
            
            # Calculate composite RAGAS score
            ragas_scores = [
                record['ragas_faithfulness'],
                record['ragas_context_precision'], 
                record['ragas_context_recall'],
                record['ragas_answer_relevancy']
            ]
            valid_scores = [s for s in ragas_scores if not np.isnan(s) and s > 0]
            record['ragas_composite_score'] = float(np.mean(valid_scores)) if valid_scores else 0.0
            
            # Set RAGAS pass based on composite score and threshold
            ragas_threshold = 0.7
            record['ragas_pass'] = record['ragas_composite_score'] >= ragas_threshold
        
        # Add semantic similarity (placeholder - can be enhanced later)
        # For now, use a simple average of available scores
        available_scores = []
        if record['contextual_total_score'] > 0:
            available_scores.append(record['contextual_total_score'])
        if record['ragas_composite_score'] > 0:
            available_scores.append(record['ragas_composite_score'])
        
        if available_scores:
            record['semantic_similarity'] = float(np.mean(available_scores))
            semantic_threshold = 0.6
            record['semantic_pass'] = record['semantic_similarity'] >= semantic_threshold
        else:
            record['semantic_similarity'] = 0.0
            record['semantic_pass'] = False
        
        return record
    
    def _add_overall_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add overall score and pass metrics."""
        # Ensure all required columns exist with default values
        required_columns = {
            'contextual_total_score': 0.0,
            'ragas_composite_score': 0.0,
            'semantic_similarity': 0.0,
            'contextual_keyword_pass': False,
            'contextual_pass': False,
            'ragas_pass': False,
            'semantic_pass': False,
            'overall_score': 0.0,
            'overall_pass': False
        }
        
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
                logger.warning(f"Added missing column '{col}' with default value {default_val}")
        
        # Calculate overall score as weighted average
        contextual_weight = 0.4
        ragas_weight = 0.6
        
        df['overall_score'] = (
            df['contextual_total_score'] * contextual_weight + 
            df['ragas_composite_score'] * ragas_weight
        )
        
        # Overall pass if both pass their thresholds
        contextual_threshold = 0.6
        ragas_threshold = 0.7
        
        df['overall_pass'] = (
            (df['contextual_total_score'] >= contextual_threshold) & 
            (df['ragas_composite_score'] >= ragas_threshold)
        )
        
        # Ensure consistent boolean types
        boolean_columns = ['contextual_keyword_pass', 'contextual_pass', 'ragas_pass', 'semantic_pass', 'overall_pass']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        logger.info(f"Added overall metrics. DataFrame shape: {df.shape}")
        return df
    
    def _get_expected_columns(self) -> List[str]:
        """Get the list of expected columns in the formatted DataFrame."""
        return [
            'question_index', 'question', 'rag_answer', 'expected_answer', 'source_file',
            'contextual_total_score', 'contextual_keyword_pass', 'contextual_pass', 'contextual_similarity_score',
            'ragas_composite_score', 'ragas_pass', 'ragas_faithfulness', 'ragas_context_precision',
            'ragas_context_recall', 'ragas_answer_relevancy',
            'semantic_similarity', 'semantic_pass',
            'overall_score', 'overall_pass'
        ]
