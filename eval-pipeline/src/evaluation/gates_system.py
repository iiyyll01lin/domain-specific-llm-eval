#!/usr/bin/env python3
"""
Gates System - Configurable Pass Rate Evaluation with Weighted Combination

This module implements a configurable gates system that:
1. Calculates pass rates for contextual keywords and RAGAS metrics
2. Applies configurable thresholds for each gate
3. Combines gates with configurable weights to determine overall pass rate
4. Supports dynamic threshold adjustment based on human feedback
"""

import logging
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

logger = logging.getLogger(__name__)

@dataclass
class GateConfig:
    """Configuration for individual gates"""
    name: str
    threshold: float
    weight: float
    enabled: bool = True
    description: str = ""

@dataclass
class GatesResult:
    """Results from gates evaluation"""
    gate_results: Dict[str, Dict[str, Any]]
    combined_pass_rate: float
    individual_pass_rates: Dict[str, float]
    overall_pass: bool
    weighted_score: float
    metadata: Dict[str, Any]

class GatesSystem:
    """
    Configurable gates system for RAG evaluation with weighted pass rates
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize gates system with configuration
        
        Args:
            config: Configuration dictionary containing gates settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Load gates configuration
        gates_config = config.get('evaluation', {}).get('gates', {})
        
        # Initialize individual gates
        self.gates = self._initialize_gates(gates_config)
        
        # Load combination settings
        self.combination_config = gates_config.get('combination', {})
        self.combination_method = self.combination_config.get('method', 'weighted_average')
        self.minimum_gates_required = self.combination_config.get('minimum_gates_required', 1)
        
        # Human feedback integration
        self.human_feedback_config = gates_config.get('human_feedback', {})
        self.enable_human_feedback = self.human_feedback_config.get('enabled', False)
        
        self.logger.info(f"GatesSystem initialized with {len(self.gates)} gates")
        for gate_name, gate in self.gates.items():
            self.logger.info(f"  {gate_name}: threshold={gate.threshold}, weight={gate.weight}")
    
    def _initialize_gates(self, gates_config: Dict[str, Any]) -> Dict[str, GateConfig]:
        """Initialize gate configurations"""
        gates = {}
        
        # Contextual keyword gate
        ck_config = gates_config.get('contextual_keywords', {})
        gates['contextual_keywords'] = GateConfig(
            name='contextual_keywords',
            threshold=ck_config.get('threshold', 0.6),
            weight=ck_config.get('weight', 0.4),
            enabled=ck_config.get('enabled', True),
            description="Contextual keyword matching gate"
        )
        
        # RAGAS metrics gate
        ragas_config = gates_config.get('ragas_metrics', {})
        gates['ragas_metrics'] = GateConfig(
            name='ragas_metrics',
            threshold=ragas_config.get('threshold', 0.7),
            weight=ragas_config.get('weight', 0.6),
            enabled=ragas_config.get('enabled', True),
            description="RAGAS metrics evaluation gate"
        )
        
        # Validate weights sum to 1.0
        total_weight = sum(gate.weight for gate in gates.values() if gate.enabled)
        if not (0.99 <= total_weight <= 1.01):  # Allow small floating point errors
            self.logger.warning(f"Gate weights sum to {total_weight:.3f}, not 1.0. Normalizing...")
            # Normalize weights
            for gate in gates.values():
                if gate.enabled:
                    gate.weight = gate.weight / total_weight
        
        return gates
    
    def evaluate_gates(self, evaluation_results: Dict[str, Any]) -> GatesResult:
        """
        Evaluate all gates and calculate combined pass rate
        
        Args:
            evaluation_results: Results from contextual keyword and RAGAS evaluations
            
        Returns:
            GatesResult containing all gate results and combined metrics
        """
        gate_results = {}
        individual_pass_rates = {}
        individual_scores = {}
        
        # Evaluate contextual keywords gate
        if self.gates['contextual_keywords'].enabled:
            ck_result = self._evaluate_contextual_keywords_gate(
                evaluation_results.get('contextual_keyword', {})
            )
            gate_results['contextual_keywords'] = ck_result
            individual_pass_rates['contextual_keywords'] = float(ck_result['pass_rate'])  # Convert to Python float
            individual_scores['contextual_keywords'] = float(ck_result['average_score'])  # Convert to Python float
        
        # Evaluate RAGAS metrics gate
        if self.gates['ragas_metrics'].enabled:
            ragas_result = self._evaluate_ragas_metrics_gate(
                evaluation_results.get('ragas', {})
            )
            gate_results['ragas_metrics'] = ragas_result
            individual_pass_rates['ragas_metrics'] = float(ragas_result['pass_rate'])  # Convert to Python float
            individual_scores['ragas_metrics'] = float(ragas_result['average_score'])  # Convert to Python float
        
        # Calculate combined pass rate and overall result
        combined_pass_rate, weighted_score, overall_pass = self._calculate_combined_result(
            gate_results, individual_pass_rates, individual_scores
        )
        
        # Create metadata
        metadata = {
            'combination_method': str(self.combination_method),
            'gates_evaluated': list(gate_results.keys()),
            'gates_enabled': [name for name, gate in self.gates.items() if gate.enabled],
            'minimum_gates_required': int(self.minimum_gates_required),
            'human_feedback_enabled': bool(self.enable_human_feedback)
        }
        
        return GatesResult(
            gate_results=gate_results,
            combined_pass_rate=combined_pass_rate,
            individual_pass_rates=individual_pass_rates,
            overall_pass=overall_pass,
            weighted_score=weighted_score,
            metadata=metadata
        )
    
    def _evaluate_contextual_keywords_gate(self, ck_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate contextual keywords gate"""
        if not ck_results.get('success', False):
            return {
                'success': False,
                'error': ck_results.get('error', 'No contextual keyword results'),
                'pass_rate': 0.0,
                'average_score': 0.0,
                'passes_threshold': False
            }
        
        # Extract metrics from contextual keyword results
        summary_metrics = ck_results.get('summary_metrics', {})
        
        # Calculate pass rate based on individual question results
        if 'detailed_results_file' in ck_results:
            detailed_results = self._load_detailed_results(ck_results['detailed_results_file'])
            if detailed_results:
                pass_count = sum(1 for result in detailed_results 
                               if result.get('passes_threshold', False))
                total_count = len(detailed_results)
                pass_rate = pass_count / total_count if total_count > 0 else 0.0
                
                # Calculate average score
                scores = [result.get('final_score', 0) for result in detailed_results]
                average_score = np.mean(scores) if scores else 0.0
            else:
                # Fallback to summary metrics
                pass_rate = summary_metrics.get('pass_rate', 0.0)
                average_score = summary_metrics.get('avg_similarity_score', 0.0)
        else:
            # Use summary metrics directly
            pass_rate = summary_metrics.get('pass_rate', 0.0)
            average_score = summary_metrics.get('avg_similarity_score', 0.0)
        
        # Check if gate passes threshold
        threshold = self.gates['contextual_keywords'].threshold
        passes_threshold = bool(pass_rate >= threshold)  # Convert to Python bool
        
        return {
            'success': True,
            'pass_rate': float(pass_rate),  # Convert to Python float
            'average_score': float(average_score),  # Convert to Python float
            'threshold': float(threshold),  # Convert to Python float
            'passes_threshold': passes_threshold,
            'total_questions': int(summary_metrics.get('total_questions', 0)),  # Convert to Python int
            'successful_evaluations': int(summary_metrics.get('successful_evaluations', 0))  # Convert to Python int
        }
    
    def _evaluate_ragas_metrics_gate(self, ragas_results: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate RAGAS metrics gate"""
        if not ragas_results.get('success', False):
            return {
                'success': False,
                'error': ragas_results.get('error', 'No RAGAS results'),
                'pass_rate': 0.0,
                'average_score': 0.0,
                'passes_threshold': False
            }
        
        # Extract overall scores
        overall_scores = ragas_results.get('overall_scores', {})
        
        # Calculate average RAGAS score across all metrics
        valid_scores = [score for score in overall_scores.values() 
                       if score is not None and isinstance(score, (int, float))]
        
        if not valid_scores:
            return {
                'success': False,
                'error': 'No valid RAGAS scores found',
                'pass_rate': 0.0,
                'average_score': 0.0,
                'passes_threshold': False
            }
        
        average_score = np.mean(valid_scores)
        
        # Calculate pass rate based on individual question results
        if 'detailed_results_file' in ragas_results:
            detailed_results = self._load_detailed_results(ragas_results['detailed_results_file'])
            if detailed_results:
                # Group by question and calculate average score per question
                question_scores = {}
                for result in detailed_results:
                    q_idx = result.get('question_index', 0)
                    score = result.get('score', 0)
                    if q_idx not in question_scores:
                        question_scores[q_idx] = []
                    question_scores[q_idx].append(score)
                
                # Calculate pass rate based on question-level averages
                threshold = self.gates['ragas_metrics'].threshold
                question_passes = []
                for q_idx, scores in question_scores.items():
                    q_avg = np.mean(scores) if scores else 0.0
                    question_passes.append(q_avg >= threshold)
                
                pass_rate = np.mean(question_passes) if question_passes else 0.0
            else:
                # Fallback: assume questions pass if average score meets threshold
                threshold = self.gates['ragas_metrics'].threshold
                pass_rate = 1.0 if average_score >= threshold else 0.0
        else:
            # Fallback: assume questions pass if average score meets threshold
            threshold = self.gates['ragas_metrics'].threshold
            pass_rate = 1.0 if average_score >= threshold else 0.0
        
        # Check if gate passes threshold
        threshold = self.gates['ragas_metrics'].threshold
        passes_threshold = bool(pass_rate >= threshold)  # Convert to Python bool
        
        return {
            'success': True,
            'pass_rate': float(pass_rate),  # Convert to Python float
            'average_score': float(average_score),  # Convert to Python float
            'threshold': float(threshold),  # Convert to Python float
            'passes_threshold': passes_threshold,
            'total_questions': int(ragas_results.get('total_questions', 0)),  # Convert to Python int
            'metric_scores': {k: float(v) if v is not None else None for k, v in overall_scores.items()}  # Convert scores to Python floats
        }
    
    def _calculate_combined_result(self, gate_results: Dict[str, Any], 
                                 individual_pass_rates: Dict[str, float],
                                 individual_scores: Dict[str, float]) -> Tuple[float, float, bool]:
        """Calculate combined pass rate and overall result"""
        
        if self.combination_method == 'weighted_average':
            # Weighted average of pass rates
            weighted_pass_rate = 0.0
            weighted_score = 0.0
            total_weight = 0.0
            
            for gate_name, pass_rate in individual_pass_rates.items():
                if gate_name in self.gates and self.gates[gate_name].enabled:
                    weight = self.gates[gate_name].weight
                    weighted_pass_rate += pass_rate * weight
                    weighted_score += individual_scores.get(gate_name, 0.0) * weight
                    total_weight += weight
            
            # Normalize if needed
            if total_weight > 0:
                combined_pass_rate = weighted_pass_rate / total_weight
                weighted_score = weighted_score / total_weight
            else:
                combined_pass_rate = 0.0
                weighted_score = 0.0
                
        elif self.combination_method == 'all_pass':
            # All gates must pass
            combined_pass_rate = min(individual_pass_rates.values()) if individual_pass_rates else 0.0
            weighted_score = np.mean(list(individual_scores.values())) if individual_scores else 0.0
            
        elif self.combination_method == 'any_pass':
            # At least one gate must pass
            combined_pass_rate = max(individual_pass_rates.values()) if individual_pass_rates else 0.0
            weighted_score = np.mean(list(individual_scores.values())) if individual_scores else 0.0
            
        elif self.combination_method == 'majority_pass':
            # Majority of gates must pass
            pass_count = sum(1 for rate in individual_pass_rates.values() if rate >= 0.5)
            total_gates = len(individual_pass_rates)
            combined_pass_rate = 1.0 if pass_count >= (total_gates / 2) else 0.0
            weighted_score = np.mean(list(individual_scores.values())) if individual_scores else 0.0
            
        else:
            # Default to weighted average
            self.logger.warning(f"Unknown combination method: {self.combination_method}, using weighted_average")
            return self._calculate_combined_result(gate_results, individual_pass_rates, individual_scores)
        
        # Determine overall pass based on minimum gates required
        gates_passing = sum(1 for result in gate_results.values() 
                          if result.get('passes_threshold', False))
        overall_pass = bool(gates_passing >= self.minimum_gates_required)  # Convert to Python bool
        
        return float(combined_pass_rate), float(weighted_score), overall_pass  # Convert to Python floats
    
    def _load_detailed_results(self, results_file: str) -> Optional[List[Dict[str, Any]]]:
        """Load detailed results from file"""
        try:
            results_path = Path(results_file)
            if results_path.exists():
                with open(results_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            self.logger.warning(f"Could not load detailed results from {results_file}: {e}")
        return None
    
    def update_thresholds(self, threshold_updates: Dict[str, float]) -> None:
        """Update gate thresholds dynamically"""
        for gate_name, new_threshold in threshold_updates.items():
            if gate_name in self.gates:
                old_threshold = self.gates[gate_name].threshold
                self.gates[gate_name].threshold = new_threshold
                self.logger.info(f"Updated {gate_name} threshold: {old_threshold:.3f} -> {new_threshold:.3f}")
            else:
                self.logger.warning(f"Unknown gate: {gate_name}")
    
    def get_gates_summary(self) -> Dict[str, Any]:
        """Get summary of gates configuration"""
        return {
            'gates': {
                name: {
                    'threshold': gate.threshold,
                    'weight': gate.weight,
                    'enabled': gate.enabled,
                    'description': gate.description
                }
                for name, gate in self.gates.items()
            },
            'combination_method': self.combination_method,
            'minimum_gates_required': self.minimum_gates_required,
            'human_feedback_enabled': self.enable_human_feedback
        }
    
    def apply_human_feedback(self, feedback_data: Dict[str, Any]) -> None:
        """Apply human feedback to adjust gate thresholds"""
        if not self.enable_human_feedback:
            return
        
        # Extract feedback scores and apply threshold adjustments
        # This could be enhanced with more sophisticated feedback integration
        for gate_name, feedback in feedback_data.items():
            if gate_name in self.gates and 'threshold_adjustment' in feedback:
                adjustment = feedback['threshold_adjustment']
                current_threshold = self.gates[gate_name].threshold
                new_threshold = max(0.0, min(1.0, current_threshold + adjustment))
                
                self.gates[gate_name].threshold = new_threshold
                self.logger.info(f"Human feedback adjusted {gate_name} threshold: "
                               f"{current_threshold:.3f} -> {new_threshold:.3f}")

    def save_gates_config(self, output_path: Path) -> None:
        """Save current gates configuration to file"""
        config_data = {
            'gates': self.get_gates_summary(),
            'evaluation_timestamp': str(Path().absolute()),
            'version': '1.0'
        }
        
        with open(output_path, 'w') as f:
            json.dump(config_data, f, indent=2)
        
        self.logger.info(f"Gates configuration saved to: {output_path}")
