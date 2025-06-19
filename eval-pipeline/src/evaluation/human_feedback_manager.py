"""
Human Feedback Manager for RAG Evaluation Pipeline

Handles human feedback integration and processing.
"""
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class HumanFeedbackManager:
    """Manages human feedback collection and processing."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the human feedback manager with configuration."""
        self.config = config
        self.feedback_config = config.get('evaluation', {}).get('human_feedback', {})
        
        # Configuration settings
        self.feedback_enabled = self.feedback_config.get('enabled', False)
        self.feedback_threshold = self.feedback_config.get('threshold', 0.7)
        self.sampling_rate = self.feedback_config.get('sampling_rate', 0.2)
        
        logger.info(f"🤖 Human feedback manager initialized (enabled: {self.feedback_enabled})")
    
    def process_feedback(self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Process human feedback for RAG responses.
        
        Args:
            testset: The test dataset
            rag_responses: RAG system responses
            
        Returns:
            Dictionary containing human feedback results
        """
        if not self.feedback_enabled:
            logger.info("⚠️ Human feedback processing skipped - disabled in configuration")
            return {
                'enabled': False,
                'message': 'Human feedback processing is disabled',
                'samples_processed': 0
            }
        
        logger.info("👥 Processing human feedback...")
        
        try:
            # Identify samples that need human feedback
            feedback_candidates = self._identify_feedback_candidates(testset, rag_responses)
            
            # Process existing feedback if available
            feedback_results = self._process_existing_feedback(feedback_candidates)
            
            # Generate feedback recommendations
            recommendations = self._generate_feedback_recommendations(feedback_candidates)
            
            results = {
                'enabled': True,
                'samples_processed': len(feedback_candidates),
                'feedback_candidates': len(feedback_candidates),
                'existing_feedback': feedback_results,
                'recommendations': recommendations,
                'summary': self._generate_feedback_summary(feedback_results, recommendations)
            }
            
            logger.info(f"✅ Human feedback processing completed ({len(feedback_candidates)} candidates)")
            return results
            
        except Exception as e:
            logger.error(f"❌ Human feedback processing failed: {e}")
            return {
                'enabled': True,
                'error': str(e),
                'message': 'Human feedback processing encountered an error'
            }
    
    def _identify_feedback_candidates(self, testset: Dict[str, Any], rag_responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify samples that need human feedback based on uncertainty or other criteria."""
        candidates = []
        
        questions = testset.get('questions', [])
        
        for i, response in enumerate(rag_responses):
            # Check if this response needs human feedback
            needs_feedback = self._needs_human_feedback(response, i)
            
            if needs_feedback:
                candidate = {
                    'index': i,
                    'question': questions[i] if i < len(questions) else f"Question {i+1}",
                    'response': response,
                    'reason': self._get_feedback_reason(response)
                }
                candidates.append(candidate)
        
        # Apply sampling if too many candidates
        if len(candidates) > int(len(rag_responses) * self.sampling_rate):
            import random
            candidates = random.sample(candidates, int(len(rag_responses) * self.sampling_rate))
        
        return candidates
    
    def _needs_human_feedback(self, response: Dict[str, Any], index: int) -> bool:
        """Determine if a response needs human feedback."""
        # Check confidence/uncertainty scores
        confidence = response.get('confidence', 1.0)
        if confidence < self.feedback_threshold:
            return True
        
        # Check for conflicting metrics
        if 'ragas_score' in response and 'keyword_score' in response:
            ragas_score = response.get('ragas_score', 0.5)
            keyword_score = response.get('keyword_score', 0.5)
            
            # If scores differ significantly, request feedback
            if abs(ragas_score - keyword_score) > 0.3:
                return True
        
        # Random sampling for diverse feedback
        import random
        if random.random() < 0.1:  # 10% random sampling
            return True
        
        return False
    
    def _get_feedback_reason(self, response: Dict[str, Any]) -> str:
        """Get the reason why this response needs human feedback."""
        confidence = response.get('confidence', 1.0)
        if confidence < self.feedback_threshold:
            return f"Low confidence score: {confidence:.2f}"
        
        if 'ragas_score' in response and 'keyword_score' in response:
            ragas_score = response.get('ragas_score', 0.5)
            keyword_score = response.get('keyword_score', 0.5)
            
            if abs(ragas_score - keyword_score) > 0.3:
                return f"Conflicting metrics - RAGAS: {ragas_score:.2f}, Keyword: {keyword_score:.2f}"
        
        return "Random sampling for quality assurance"
    
    def _process_existing_feedback(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process any existing human feedback."""
        # In a real implementation, this would check for existing feedback
        # For now, return empty results
        return {
            'total_feedback': 0,
            'positive_feedback': 0,
            'negative_feedback': 0,
            'feedback_items': []
        }
    
    def _generate_feedback_recommendations(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate recommendations for human feedback collection."""
        if not candidates:
            return {
                'total_recommendations': 0,
                'high_priority': 0,
                'medium_priority': 0,
                'low_priority': 0,
                'recommendations': []
            }
        
        recommendations = []
        high_priority = 0
        medium_priority = 0
        low_priority = 0
        
        for candidate in candidates:
            priority = self._determine_priority(candidate)
            
            rec = {
                'index': candidate['index'],
                'question': candidate['question'],
                'reason': candidate['reason'],
                'priority': priority,
                'suggested_action': self._suggest_action(candidate, priority)
            }
            
            recommendations.append(rec)
            
            if priority == 'high':
                high_priority += 1
            elif priority == 'medium':
                medium_priority += 1
            else:
                low_priority += 1
        
        return {
            'total_recommendations': len(recommendations),
            'high_priority': high_priority,
            'medium_priority': medium_priority,
            'low_priority': low_priority,
            'recommendations': recommendations
        }
    
    def _determine_priority(self, candidate: Dict[str, Any]) -> str:
        """Determine the priority level for feedback collection."""
        reason = candidate['reason']
        
        if 'Low confidence' in reason:
            return 'high'
        elif 'Conflicting metrics' in reason:
            return 'medium'
        else:
            return 'low'
    
    def _suggest_action(self, candidate: Dict[str, Any], priority: str) -> str:
        """Suggest action for feedback collection."""
        if priority == 'high':
            return "Immediate review recommended - low confidence in response quality"
        elif priority == 'medium':
            return "Review recommended - conflicting evaluation metrics detected"
        else:
            return "Optional review for quality assurance"
    
    def _generate_feedback_summary(self, feedback_results: Dict[str, Any], recommendations: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of feedback processing."""
        return {
            'existing_feedback_count': feedback_results.get('total_feedback', 0),
            'new_recommendations': recommendations.get('total_recommendations', 0),
            'high_priority_items': recommendations.get('high_priority', 0),
            'feedback_coverage': f"{feedback_results.get('total_feedback', 0)} existing + {recommendations.get('total_recommendations', 0)} recommended"
        }
    
    def is_enabled(self) -> bool:
        """Check if human feedback processing is enabled."""
        return self.feedback_enabled
