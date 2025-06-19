"""
RAG Evaluator for RAG Evaluation Pipeline

Coordinates evaluation of RAG systems using multiple evaluation approaches.
"""
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

class RAGEvaluator:
    """Main RAG system evaluator that coordinates different evaluation approaches."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the RAG evaluator with configuration."""
        self.config = config
        self.evaluation_config = config.get('evaluation', {})
        
        # Initialize evaluation components
        self.contextual_evaluator = None
        self.ragas_evaluator = None
        self.human_feedback_manager = None
        
        self._setup_evaluators()
        
    def _setup_evaluators(self):
        """Setup evaluation components based on configuration."""
        try:
            from .contextual_keyword_evaluator import ContextualKeywordEvaluator
            self.contextual_evaluator = ContextualKeywordEvaluator(self.config)
            logger.info("✅ Contextual keyword evaluator initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Could not initialize contextual evaluator: {e}")
            
        try:
            from .ragas_evaluator import RagasEvaluator
            self.ragas_evaluator = RagasEvaluator(self.config)
            logger.info("✅ RAGAS evaluator initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Could not initialize RAGAS evaluator: {e}")
            
        try:
            from .human_feedback_manager import HumanFeedbackManager
            self.human_feedback_manager = HumanFeedbackManager(self.config)
            logger.info("✅ Human feedback manager initialized")
        except ImportError as e:
            logger.warning(f"⚠️ Could not initialize human feedback manager: {e}")
    
    def evaluate_testset(self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate RAG responses using available evaluation methods.
        
        Args:
            testset: The test dataset
            rag_responses: RAG system responses
            
        Returns:
            Dictionary containing evaluation results
        """
        logger.info("🔍 Starting RAG evaluation...")
        
        results = {
            'metadata': {
                'evaluator': 'RAGEvaluator',
                'testset_size': len(testset.get('questions', [])),
                'response_count': len(rag_responses)
            },
            'contextual_keyword_results': {},
            'ragas_results': {},
            'human_feedback_results': {},
            'summary': {}
        }
        
        # Contextual keyword evaluation
        if self.contextual_evaluator:
            try:
                logger.info("📊 Running contextual keyword evaluation...")
                contextual_results = self.contextual_evaluator.evaluate(testset, rag_responses)
                results['contextual_keyword_results'] = contextual_results
                logger.info("✅ Contextual keyword evaluation completed")
            except Exception as e:
                logger.error(f"❌ Contextual keyword evaluation failed: {e}")
                results['contextual_keyword_results'] = {'error': str(e)}
        
        # RAGAS evaluation
        if self.ragas_evaluator:
            try:
                logger.info("📊 Running RAGAS evaluation...")
                ragas_results = self.ragas_evaluator.evaluate(testset, rag_responses)
                results['ragas_results'] = ragas_results
                logger.info("✅ RAGAS evaluation completed")
            except Exception as e:
                logger.error(f"❌ RAGAS evaluation failed: {e}")
                results['ragas_results'] = {'error': str(e)}
        
        # Human feedback evaluation
        if self.human_feedback_manager:
            try:
                logger.info("📊 Processing human feedback...")
                feedback_results = self.human_feedback_manager.process_feedback(testset, rag_responses)
                results['human_feedback_results'] = feedback_results
                logger.info("✅ Human feedback processing completed")
            except Exception as e:
                logger.error(f"❌ Human feedback processing failed: {e}")
                results['human_feedback_results'] = {'error': str(e)}
        
        # Generate summary
        results['summary'] = self._generate_summary(results)
        
        logger.info("✅ RAG evaluation completed")
        return results
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate evaluation summary from all results."""
        summary = {
            'total_evaluations': 0,
            'successful_evaluations': 0,
            'failed_evaluations': 0,
            'evaluation_methods': []
        }
        
        # Count successful evaluations
        for method, result in results.items():
            if method in ['contextual_keyword_results', 'ragas_results', 'human_feedback_results']:
                if result and 'error' not in result:
                    summary['successful_evaluations'] += 1
                    summary['evaluation_methods'].append(method.replace('_results', ''))
                else:
                    summary['failed_evaluations'] += 1
                summary['total_evaluations'] += 1
        
        return summary
    
    def get_available_evaluators(self) -> List[str]:
        """Get list of available evaluators."""
        evaluators = []
        if self.contextual_evaluator:
            evaluators.append('contextual_keyword')
        if self.ragas_evaluator:
            evaluators.append('ragas')
        if self.human_feedback_manager:
            evaluators.append('human_feedback')
        return evaluators
