"""
RAG Evaluator for RAG Evaluation Pipeline

Coordinates evaluation of RAG systems using multiple evaluation approaches.
This implements the CORRECT evaluation flow:
1. Load testsets (questions + ground truth)
2. Query RAG system with questions
3. Extract keywords from RAG responses (not from testset!)
4. Calculate evaluation metrics on RAG responses
"""

import csv
import json
import logging
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Main RAG system evaluator that coordinates different evaluation approaches."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG evaluator.
        
        Args:
            config: Pipeline configuration
        """
        self.config = config
        self.rag_config = config.get('rag_system', {})
        self.eval_config = config.get('evaluation', {})
        
        # Initialize RAG interface
        try:
            from interfaces.rag_interface import RAGInterface
            self.rag_interface = RAGInterface(self.rag_config)
        except Exception as e:
            logger.error(f"Failed to initialize RAG interface: {e}")
            self.rag_interface = None
        
        # Initialize keyword evaluator
        if self.eval_config.get('contextual_keywords', {}).get('enabled', False):
            try:
                from evaluation.contextual_keyword_evaluator import ContextualKeywordEvaluator
                self.keyword_evaluator = ContextualKeywordEvaluator(
                    self.eval_config.get('contextual_keywords', {})
                )
            except Exception as e:
                logger.warning(f"Failed to initialize keyword evaluator: {e}")
                self.keyword_evaluator = None
        else:
            self.keyword_evaluator = None
        
        # Initialize RAGAS evaluator
        if self.eval_config.get('ragas_metrics', {}).get('enabled', False):
            try:
                from evaluation.ragas_evaluator import RagasEvaluator
                self.ragas_evaluator = RagasEvaluator(
                    self.eval_config.get('ragas_metrics', {})
                )
            except Exception as e:
                logger.warning(f"Failed to initialize RAGAS evaluator: {e}")
                self.ragas_evaluator = None
        else:
            self.ragas_evaluator = None
        
        logger.info("🔧 RAG Evaluator initialized")
    
    def load_testset(self, testset_path: str) -> List[Dict]:
        """
        Load testset from file.
        
        Args:
            testset_path: Path to testset file
            
        Returns:
            List of test samples
        """
        logger.info(f"� Loading testset from: {testset_path}")
        
        try:
            if testset_path.endswith('.csv'):
                df = pd.read_csv(testset_path)
                # Try to restore dtypes after CSV loading
                df = self._restore_testset_dtypes(df, testset_path)
                testset = df.to_dict('records')
            elif testset_path.endswith('.xlsx'):
                df = pd.read_excel(testset_path)
                # Try to restore dtypes after Excel loading  
                df = self._restore_testset_dtypes(df, testset_path)
                testset = df.to_dict('records')
            elif testset_path.endswith('.json'):
                with open(testset_path, 'r') as f:
                    testset = json.load(f)
                # For JSON, we assume the data is already correctly typed
            else:
                raise ValueError(f"Unsupported testset format: {testset_path}")
            
            logger.info(f"✅ Loaded {len(testset)} test samples")
            return testset
            
        except Exception as e:
            logger.error(f"❌ Failed to load testset: {e}")
            raise
    
    def _restore_testset_dtypes(self, df: pd.DataFrame, testset_path: str) -> pd.DataFrame:
        """
        Restore data types for testset DataFrame after loading from CSV/Excel.
        
        Args:
            df: DataFrame loaded from CSV/Excel
            testset_path: Original testset file path
            
        Returns:
            DataFrame with restored data types
        """
        try:
            # Look for dtype info file in the same directory
            testset_dir = Path(testset_path).parent
            
            # Look for dtype files
            dtype_files = list(testset_dir.glob("testset_dtypes_*.json"))
            metadata_files = list(testset_dir.glob("testset_metadata_*.json"))
            
            dtype_info = {}
            
            # Try to load dtype info from dedicated dtype file
            if dtype_files:
                latest_dtype_file = max(dtype_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_dtype_file, 'r') as f:
                        dtype_info = json.load(f)
                    logger.info(f"🔧 Loaded dtype info from: {latest_dtype_file}")
                except Exception as e:
                    logger.warning(f"Could not load dtype file {latest_dtype_file}: {e}")
            
            # Fallback: try to get dtype info from metadata files
            if not dtype_info and metadata_files:
                latest_metadata_file = max(metadata_files, key=lambda x: x.stat().st_mtime)
                try:
                    with open(latest_metadata_file, 'r') as f:
                        metadata = json.load(f)
                        dtype_info = metadata.get('column_dtypes', {})
                    logger.info(f"🔧 Loaded dtype info from metadata: {latest_metadata_file}")
                except Exception as e:
                    logger.warning(f"Could not load metadata file {latest_metadata_file}: {e}")
            
            # Apply dtype restoration
            if dtype_info:
                for col, dtype_type in dtype_info.items():
                    if col in df.columns:
                        try:
                            if dtype_type == 'numeric':
                                df[col] = pd.to_numeric(df[col], errors='coerce')
                                logger.debug(f"   Restored numeric type for: {col}")
                            elif dtype_type == 'datetime':
                                df[col] = pd.to_datetime(df[col], errors='coerce')
                                logger.debug(f"   Restored datetime type for: {col}")
                        except Exception as e:
                            logger.warning(f"   Could not restore dtype for {col}: {e}")
                
                # Log success
                numeric_cols_restored = df.select_dtypes(include=['number']).columns
                logger.info(f"✅ Restored {len(numeric_cols_restored)} numeric columns: {list(numeric_cols_restored)}")
            else:
                # Fallback: try to infer numeric columns based on common RAGAS metric names
                logger.info("🔧 No dtype info found, applying heuristic type restoration...")
                numeric_column_patterns = [
                    'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy',
                    'kw_metric', 'keyword_score', 'weighted_average_score', 
                    'overall_score', 'contextual_total_score', 'ragas_composite_score',
                    'semantic_similarity', 'rag_response_time'
                ]
                
                for col in df.columns:
                    if any(pattern in col.lower() for pattern in numeric_column_patterns):
                        try:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                            logger.debug(f"   Heuristically restored numeric type for: {col}")
                        except Exception as e:
                            logger.warning(f"   Could not restore numeric type for {col}: {e}")
                
                heuristic_numeric_cols = df.select_dtypes(include=['number']).columns
                logger.info(f"✅ Heuristically restored {len(heuristic_numeric_cols)} numeric columns")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error restoring testset dtypes: {e}")
            return df  # Return original DataFrame on error
    
    def query_rag_system(self, question: str) -> Dict[str, Any]:
        """
        Query the RAG system with a question.
        
        Args:
            question: Question to ask the RAG system
            
        Returns:
            Dictionary with RAG response
        """
        if not self.rag_interface:
            logger.error("RAG interface not available")
            return {
                'question': question,
                'rag_answer': '',
                'rag_contexts': [],
                'rag_confidence': None,
                'rag_metadata': {'error': 'RAG interface not available'},
                'response_time': 0
            }
        
        try:
            response = self.rag_interface.query_rag_system(question)
            
            return {
                'question': question,
                'rag_answer': response.get('answer', ''),
                'rag_contexts': response.get('contexts', []),
                'rag_confidence': response.get('confidence', None),
                'rag_metadata': response.get('metadata', {}),
                'response_time': response.get('response_time', 0)
            }
            
        except Exception as e:
            logger.error(f"❌ RAG query failed for question: {question[:50]}... Error: {e}")
            return {
                'question': question,
                'rag_answer': '',
                'rag_contexts': [],
                'rag_confidence': None,
                'rag_metadata': {'error': str(e)},
                'response_time': 0
            }
    
    def extract_keywords_from_response(self, rag_answer: str) -> List[str]:
        """
        Extract keywords from RAG response.
        
        Args:
            rag_answer: RAG system's answer
            
        Returns:
            List of extracted keywords
        """
        if not self.keyword_evaluator:
            return []
        
        try:
            keywords = self.keyword_evaluator.extract_keywords(rag_answer)
            return keywords
        except Exception as e:
            logger.warning(f"Keyword extraction failed: {e}")
            return []
    
    def evaluate_single_testset(self, testset_path: str) -> Dict[str, Any]:
        """
        Evaluate RAG system with a single testset.
        
        Args:
            testset_path: Path to testset file
            
        Returns:
            Evaluation results
        """
        logger.info(f"🔄 Evaluating RAG system with testset: {testset_path}")
        
        try:
            # Step 1: Load testset
            testset = self.load_testset(testset_path)
            
            # Step 2: Query RAG system for each question
            logger.info("🚀 Querying RAG system with testset questions...")
            rag_responses = []
            
            for i, test_sample in enumerate(testset):
                question = test_sample.get('question', '')
                if not question:
                    logger.warning(f"Empty question in test sample {i}, skipping")
                    continue
                
                logger.info(f"  Query {i+1}/{len(testset)}: {question[:60]}...")
                rag_response = self.query_rag_system(question)
                
                # Combine with original test data
                combined_data = {
                    **test_sample,  # Original testset data (question, ground_truth, etc.)
                    **rag_response,  # RAG system response
                    'test_index': i
                }
                
                # Step 3: Extract keywords from RAG response (NOT from testset!)
                if rag_response['rag_answer']:
                    extracted_keywords = self.extract_keywords_from_response(rag_response['rag_answer'])
                    combined_data['extracted_keywords'] = extracted_keywords
                    combined_data['keyword_count'] = len(extracted_keywords)
                else:
                    combined_data['extracted_keywords'] = []
                    combined_data['keyword_count'] = 0
                
                rag_responses.append(combined_data)
            
            logger.info(f"✅ Completed {len(rag_responses)} RAG queries")
            
            # Step 4: Calculate keyword evaluation metrics
            keyword_metrics = {}
            if self.keyword_evaluator and rag_responses:
                logger.info("� Calculating keyword evaluation metrics...")
                try:
                    keyword_metrics = self.keyword_evaluator.evaluate_responses(rag_responses)
                except Exception as e:
                    logger.error(f"Keyword evaluation failed: {e}")
                    keyword_metrics = {'error': str(e)}
            
            # Step 5: Calculate RAGAS metrics on RAG responses
            ragas_metrics = {}
            if self.ragas_evaluator and rag_responses:
                logger.info("📊 Calculating RAGAS metrics on RAG responses...")
                try:
                    ragas_metrics = self.ragas_evaluator.evaluate_responses(rag_responses)
                except Exception as e:
                    logger.error(f"RAGAS evaluation failed: {e}")
                    ragas_metrics = {'error': str(e)}
            
            # Step 6: Compile results
            results = {
                'testset_path': testset_path,
                'total_questions': len(testset),
                'successful_queries': len(rag_responses),
                'failed_queries': len(testset) - len(rag_responses),
                'keyword_metrics': keyword_metrics,
                'ragas_metrics': ragas_metrics,
                'detailed_responses': rag_responses,
                'evaluation_timestamp': datetime.now().isoformat()
            }
            
            logger.info("✅ RAG evaluation completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"❌ RAG evaluation failed: {e}")
            return {
                'testset_path': testset_path,
                'error': str(e),
                'evaluation_timestamp': datetime.now().isoformat()
            }
    
    def evaluate_testsets(self, testset_files: List[str]) -> Dict[str, Any]:
        """
        Evaluate RAG system with multiple testsets.
        
        Args:
            testset_files: List of testset file paths
            
        Returns:
            Combined evaluation results
        """
        logger.info(f"🔄 Evaluating RAG system with {len(testset_files)} testsets...")
        
        all_results = []
        total_queries = 0
        total_successful = 0
        
        for testset_file in testset_files:
            logger.info(f"📋 Processing testset: {Path(testset_file).name}")
            
            result = self.evaluate_single_testset(str(testset_file))
            all_results.append(result)
            
            total_queries += result.get('total_questions', 0)
            total_successful += result.get('successful_queries', 0)
        
        # Combine metrics from all testsets
        combined_results = {
            'total_testsets': len(testset_files),
            'total_queries': total_queries,
            'total_successful': total_successful,
            'success_rate': total_successful / total_queries if total_queries > 0 else 0,
            'individual_results': all_results,
            'evaluation_timestamp': datetime.now().isoformat()
        }
        
        # Save combined results
        output_file = f"rag_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(combined_results, f, indent=2, default=str)
        
        combined_results['output_file'] = output_file
        
        logger.info(f"✅ RAG evaluation completed. Results saved to: {output_file}")
        return combined_results

    def evaluate_testset(self, testset_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Evaluate RAG system with testset data directly.
        
        Args:
            testset_data: Testset data containing qa_pairs
            
        Returns:
            List of evaluation results
        """
        logger.info("🔄 Evaluating RAG system with testset data...")
        
        try:
            qa_pairs = testset_data.get('qa_pairs', [])
            if not qa_pairs:
                logger.warning("No QA pairs found in testset data")
                return []
            
            rag_responses = []
            
            for i, qa_pair in enumerate(qa_pairs):
                question = qa_pair.get('user_input', qa_pair.get('question', ''))
                if not question:
                    logger.warning(f"Empty question in QA pair {i}, skipping")
                    continue
                
                logger.info(f"  Query {i+1}/{len(qa_pairs)}: {question[:60]}...")
                rag_response = self.query_rag_system(question)
                
                # Combine with original test data
                combined_data = {
                    **qa_pair,
                    'rag_answer': rag_response.get('answer', ''),
                    'rag_confidence': rag_response.get('confidence'),
                    'rag_contexts': rag_response.get('contexts', []),
                    'rag_response_time': rag_response.get('response_time', 0),
                    'query_timestamp': datetime.now().isoformat()
                }
                
                rag_responses.append(combined_data)
            
            logger.info(f"✅ RAG evaluation completed for {len(rag_responses)} questions")
            return rag_responses
            
        except Exception as e:
            logger.error(f"❌ RAG evaluation failed: {e}")
            return []
