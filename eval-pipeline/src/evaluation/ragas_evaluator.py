"""
RAGAS Evaluator for RAG Evaluation Pipeline

Handles RAGAS-based evaluation of RAG systems.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class RagasEvaluator:
    """RAGAS-based evaluator for RAG systems."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAGAS evaluator with configuration."""
        self.config = config
        self.ragas_config = config.get('evaluation', {}).get('ragas', {})
        self.ragas_available = False
        
        self._setup_ragas()
        
    def _setup_ragas(self):
        """Setup RAGAS evaluation components."""
        try:
            from ragas import evaluate
            from ragas.metrics import context_precision, context_recall, faithfulness, answer_relevancy
            from datasets import Dataset
            
            self.evaluate_func = evaluate
            self.metrics = [context_precision, context_recall, faithfulness, answer_relevancy]
            self.Dataset = Dataset
            self.ragas_available = True
            
            logger.info("✅ RAGAS evaluation components loaded successfully")
            
        except ImportError as e:
            logger.warning(f"⚠️ RAGAS not available: {e}")
            self.ragas_available = False
    
    def evaluate(self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate RAG responses using RAGAS metrics.
        
        Args:
            testset: The test dataset
            rag_responses: RAG system responses
            
        Returns:
            Dictionary containing RAGAS evaluation results
        """
        if not self.ragas_available:
            logger.warning("⚠️ RAGAS evaluation skipped - RAGAS not available")
            return {
                'error': 'RAGAS not available',
                'available': False,
                'message': 'RAGAS evaluation requires ragas library installation'
            }
        
        logger.info("🔍 Starting RAGAS evaluation...")
        
        try:
            # Prepare data for RAGAS evaluation
            evaluation_data = self._prepare_evaluation_data(testset, rag_responses)
            
            if not evaluation_data:
                return {
                    'error': 'No valid data for evaluation',
                    'available': True,
                    'message': 'Could not prepare data for RAGAS evaluation'
                }
            
            # Create dataset
            dataset = self.Dataset.from_dict(evaluation_data)
            
            # Run RAGAS evaluation
            results = self.evaluate_func(
                dataset=dataset,
                metrics=self.metrics
            )
            
            # Format results
            formatted_results = self._format_results(results)
            
            logger.info("✅ RAGAS evaluation completed successfully")
            return formatted_results
            
        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            return {
                'error': str(e),
                'available': True,
                'message': 'RAGAS evaluation encountered an error'
            }
    
    def _prepare_evaluation_data(self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]) -> Dict[str, List]:
        """Prepare data for RAGAS evaluation."""
        questions = testset.get('questions', [])
        ground_truths = testset.get('ground_truths', [])
        contexts = testset.get('contexts', [])
        
        # Ensure we have matching data
        min_length = min(len(questions), len(rag_responses))
        
        if min_length == 0:
            logger.warning("⚠️ No matching data for RAGAS evaluation")
            return {}
        
        evaluation_data = {
            'question': questions[:min_length],
            'answer': [resp.get('answer', '') for resp in rag_responses[:min_length]],
            'contexts': [],
            'ground_truth': []
        }
        
        # Handle contexts
        if contexts and len(contexts) >= min_length:
            evaluation_data['contexts'] = [[ctx] if isinstance(ctx, str) else ctx for ctx in contexts[:min_length]]
        else:
            # Use empty contexts if not available
            evaluation_data['contexts'] = [[]] * min_length
        
        # Handle ground truths
        if ground_truths and len(ground_truths) >= min_length:
            evaluation_data['ground_truth'] = ground_truths[:min_length]
        else:
            # Use empty ground truths if not available
            evaluation_data['ground_truth'] = [''] * min_length
        
        logger.info(f"📊 Prepared {min_length} samples for RAGAS evaluation")
        return evaluation_data
    
    def _format_results(self, results) -> Dict[str, Any]:
        """Format RAGAS results for consistent output."""
        formatted = {
            'available': True,
            'metrics': {},
            'summary': {}
        }
        
        # Extract metric scores
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            for column in df.columns:
                if column not in ['question', 'answer', 'contexts', 'ground_truth']:
                    formatted['metrics'][column] = {
                        'mean': float(df[column].mean()),
                        'std': float(df[column].std()),
                        'min': float(df[column].min()),
                        'max': float(df[column].max()),
                        'scores': df[column].tolist()
                    }
        
        # Calculate overall summary
        if formatted['metrics']:
            formatted['summary'] = {
                'total_metrics': len(formatted['metrics']),
                'average_score': sum(m['mean'] for m in formatted['metrics'].values()) / len(formatted['metrics']),
                'metric_names': list(formatted['metrics'].keys())
            }
        
        return formatted
    
    def is_available(self) -> bool:
        """Check if RAGAS evaluation is available."""
        return self.ragas_available
