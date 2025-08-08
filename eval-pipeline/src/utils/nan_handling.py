"""
Utility functions for robust NaN handling in evaluation metrics.
"""
import math
import numpy as np
from typing import List, Union, Optional, Any
import logging

logger = logging.getLogger(__name__)


def is_valid_score(value: Any) -> bool:
    """
    Check if a value is a valid numeric score.
    
    Args:
        value: Value to check
        
    Returns:
        True if value is a valid numeric score, False otherwise
    """
    if value is None:
        return False
    
    try:
        # Convert to float if possible
        float_val = float(value)
        
        # Check if it's finite (not NaN or infinity)
        if not math.isfinite(float_val):
            return False
            
        return True
    except (ValueError, TypeError, OverflowError):
        return False


def safe_mean(scores: List[Union[float, int, Any]], 
              fallback_value: Optional[float] = None,
              min_valid_scores: int = 1) -> float:
    """
    Calculate mean of scores with robust NaN handling.
    
    Args:
        scores: List of scores that may contain NaN values
        fallback_value: Value to return if no valid scores (default: 0.0)
        min_valid_scores: Minimum number of valid scores required (default: 1)
        
    Returns:
        Mean of valid scores or fallback_value if insufficient valid scores
    """
    if fallback_value is None:
        fallback_value = 0.0
    
    if not scores:
        logger.warning("Empty scores list provided to safe_mean")
        return fallback_value
    
    # Filter out invalid scores
    valid_scores = [s for s in scores if is_valid_score(s)]
    
    if len(valid_scores) < min_valid_scores:
        logger.warning(f"Insufficient valid scores: {len(valid_scores)}/{len(scores)} valid, "
                      f"minimum required: {min_valid_scores}")
        return fallback_value
    
    # Calculate mean of valid scores
    try:
        mean_score = sum(float(s) for s in valid_scores) / len(valid_scores)
        
        # Final check for valid result
        if not is_valid_score(mean_score):
            logger.warning(f"Mean calculation resulted in invalid value: {mean_score}")
            return fallback_value
            
        return float(mean_score)
    except Exception as e:
        logger.error(f"Error calculating mean: {e}")
        return fallback_value


def safe_std(scores: List[Union[float, int, Any]], 
             fallback_value: Optional[float] = None) -> float:
    """
    Calculate standard deviation of scores with robust NaN handling.
    
    Args:
        scores: List of scores that may contain NaN values
        fallback_value: Value to return if calculation fails (default: 0.0)
        
    Returns:
        Standard deviation of valid scores or fallback_value
    """
    if fallback_value is None:
        fallback_value = 0.0
    
    if not scores:
        return fallback_value
    
    # Filter out invalid scores
    valid_scores = [float(s) for s in scores if is_valid_score(s)]
    
    if len(valid_scores) < 2:  # Need at least 2 values for std
        return fallback_value
    
    try:
        return float(np.std(valid_scores))
    except Exception as e:
        logger.error(f"Error calculating standard deviation: {e}")
        return fallback_value


def safe_min_max(scores: List[Union[float, int, Any]], 
                 fallback_value: Optional[float] = None) -> tuple:
    """
    Calculate min and max of scores with robust NaN handling.
    
    Args:
        scores: List of scores that may contain NaN values
        fallback_value: Value to return if calculation fails (default: 0.0)
        
    Returns:
        Tuple of (min_value, max_value) or (fallback_value, fallback_value)
    """
    if fallback_value is None:
        fallback_value = 0.0
    
    if not scores:
        return fallback_value, fallback_value
    
    # Filter out invalid scores
    valid_scores = [float(s) for s in scores if is_valid_score(s)]
    
    if not valid_scores:
        return fallback_value, fallback_value
    
    try:
        return float(min(valid_scores)), float(max(valid_scores))
    except Exception as e:
        logger.error(f"Error calculating min/max: {e}")
        return fallback_value, fallback_value


def clean_scores_for_json(scores: List[Union[float, int, Any]]) -> List[Union[float, int, None]]:
    """
    Clean scores list for JSON serialization, replacing invalid values with None.
    
    Args:
        scores: List of scores that may contain NaN values
        
    Returns:
        List with NaN values replaced by None for JSON compatibility
    """
    cleaned_scores = []
    for score in scores:
        if is_valid_score(score):
            cleaned_scores.append(float(score))
        else:
            cleaned_scores.append(None)
    return cleaned_scores


def calculate_robust_summary_stats(scores: List[Union[float, int, Any]], 
                                  metric_name: str = "metric") -> dict:
    """
    Calculate comprehensive summary statistics with robust NaN handling.
    
    Args:
        scores: List of scores that may contain NaN values
        metric_name: Name of the metric for logging purposes
        
    Returns:
        Dictionary containing summary statistics
    """
    if not scores:
        logger.warning(f"No scores provided for {metric_name}")
        return {
            'mean_score': 0.0,
            'std_score': 0.0,
            'min_score': 0.0,
            'max_score': 0.0,
            'valid_count': 0,
            'total_count': 0,
            'individual_scores': []
        }
    
    # Clean scores for JSON compatibility
    cleaned_scores = clean_scores_for_json(scores)
    
    # Calculate statistics
    mean_score = safe_mean(scores)
    std_score = safe_std(scores)
    min_score, max_score = safe_min_max(scores)
    
    # Count valid scores
    valid_count = len([s for s in scores if is_valid_score(s)])
    total_count = len(scores)
    
    # Log statistics
    if valid_count < total_count:
        logger.info(f"{metric_name}: {valid_count}/{total_count} valid scores, "
                   f"mean: {mean_score:.3f}")
    else:
        logger.info(f"{metric_name}: {valid_count} valid scores, mean: {mean_score:.3f}")
    
    return {
        'mean_score': mean_score,
        'std_score': std_score,
        'min_score': min_score,
        'max_score': max_score,
        'valid_count': valid_count,
        'total_count': total_count,
        'individual_scores': cleaned_scores
    }


def validate_metric_results(results: dict, metric_name: str) -> dict:
    """
    Validate and fix metric results to ensure no NaN values in critical fields.
    
    Args:
        results: Dictionary containing metric results
        metric_name: Name of the metric for logging
        
    Returns:
        Validated results with NaN values fixed
    """
    if not results:
        logger.warning(f"Empty results for {metric_name}")
        return {
            'mean_score': 0.0,
            'error': 'No results available',
            'valid_count': 0,
            'total_count': 0
        }
    
    # Fix mean_score if it's NaN
    if 'mean_score' in results:
        if not is_valid_score(results['mean_score']):
            logger.warning(f"Invalid mean_score for {metric_name}: {results['mean_score']}")
            
            # Try to recalculate from individual_scores if available
            if 'individual_scores' in results:
                new_mean = safe_mean(results['individual_scores'])
                results['mean_score'] = new_mean
                logger.info(f"Recalculated mean_score for {metric_name}: {new_mean:.3f}")
            else:
                results['mean_score'] = 0.0
                logger.warning(f"Set mean_score to 0.0 for {metric_name}")
    
    # Clean individual_scores if present
    if 'individual_scores' in results:
        results['individual_scores'] = clean_scores_for_json(results['individual_scores'])
    
    # Ensure all numeric fields are valid
    for field in ['std_score', 'min_score', 'max_score']:
        if field in results and not is_valid_score(results[field]):
            results[field] = 0.0
            logger.warning(f"Fixed invalid {field} for {metric_name}")
    
    return results


def apply_nan_tolerance(evaluation_results: dict, tolerance_strategy: str = "skip") -> dict:
    """
    Apply NaN tolerance strategy to evaluation results.
    
    Args:
        evaluation_results: Dictionary containing evaluation results
        tolerance_strategy: Strategy for handling NaN values ("skip", "fallback", "interpolate")
        
    Returns:
        Updated evaluation results with NaN tolerance applied
    """
    if not evaluation_results:
        return evaluation_results
    
    logger.info(f"Applying NaN tolerance strategy: {tolerance_strategy}")
    
    # Process individual metric results
    if 'ragas_metrics' in evaluation_results:
        ragas_results = evaluation_results['ragas_metrics']
        
        for metric_name, metric_results in ragas_results.items():
            if isinstance(metric_results, dict):
                # Validate and fix each metric
                ragas_results[metric_name] = validate_metric_results(metric_results, metric_name)
    
    # Process overall scores
    if 'overall_scores' in evaluation_results:
        overall_scores = evaluation_results['overall_scores']
        
        for metric_name, score in overall_scores.items():
            if not is_valid_score(score):
                logger.warning(f"Invalid overall score for {metric_name}: {score}")
                
                # Apply tolerance strategy
                if tolerance_strategy == "skip":
                    # Remove invalid scores
                    overall_scores[metric_name] = None
                elif tolerance_strategy == "fallback":
                    # Replace with fallback value
                    overall_scores[metric_name] = 0.0
                elif tolerance_strategy == "interpolate":
                    # Try to interpolate from other metrics (simple average)
                    valid_scores = [s for s in overall_scores.values() if is_valid_score(s)]
                    if valid_scores:
                        overall_scores[metric_name] = safe_mean(valid_scores)
                    else:
                        overall_scores[metric_name] = 0.0
    
    return evaluation_results
