"""
Sample Validator for RAG Evaluation Pipeline

This module provides validation and cleaning of individual samples
during testset generation to handle NaN values and validation errors.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import logging
import json
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Import pandas with fallback
try:
    import pandas as pd
    import numpy as np
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    logger.warning("⚠️ Pandas/Numpy not available, using fallback validation")

class SampleValidator:
    """Validates individual samples during testset generation."""
    
    @staticmethod
    def validate_sample_data(sample_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Validate and clean individual sample data before creating TestsetSample.
        
        Args:
            sample_data: Raw sample data dictionary
            
        Returns:
            Cleaned sample data or None if invalid
        """
        if not sample_data or not isinstance(sample_data, dict):
            logger.debug("⚠️ Sample data is empty or not a dictionary")
            return None
            
        try:
            # Check for NaN values
            cleaned_data = SampleValidator._clean_nan_values(sample_data)
            if not cleaned_data:
                logger.debug("⚠️ Sample failed NaN cleaning")
                return None
                
            # Validate required fields
            if not SampleValidator._validate_required_fields(cleaned_data):
                logger.debug("⚠️ Sample missing required fields")
                return None
                
            # Validate eval_sample structure
            cleaned_data = SampleValidator._validate_eval_sample(cleaned_data)
            if not cleaned_data:
                logger.debug("⚠️ Sample failed eval_sample validation")
                return None
                
            # Validate sample content quality
            if not SampleValidator._validate_content_quality(cleaned_data):
                logger.debug("⚠️ Sample failed content quality check")
                return None
                
            return cleaned_data
            
        except Exception as e:
            logger.warning(f"⚠️ Sample validation error: {e}")
            return None
    
    @staticmethod
    def _clean_nan_values(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Remove or replace NaN values in sample data."""
        cleaned = {}
        
        for key, value in data.items():
            try:
                # Check for NaN using multiple methods
                is_nan = False
                
                if PANDAS_AVAILABLE:
                    is_nan = pd.isna(value)
                else:
                    # Fallback NaN detection
                    is_nan = (
                        value is None or
                        (isinstance(value, float) and str(value).lower() == 'nan') or
                        (isinstance(value, str) and value.lower() in ['nan', 'none', '', 'null'])
                    )
                
                if is_nan:
                    # Handle NaN based on expected data type and field importance
                    replacement = SampleValidator._get_nan_replacement(key, value)
                    if replacement is None:
                        # Critical field is NaN, sample is invalid
                        logger.debug(f"⚠️ Critical field '{key}' is NaN, rejecting sample")
                        return None
                    else:
                        cleaned[key] = replacement
                        logger.debug(f"✅ Replaced NaN in '{key}' with '{replacement}'")
                else:
                    cleaned[key] = value
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing field '{key}': {e}")
                # Try to keep the original value
                cleaned[key] = value
                
        return cleaned
    
    @staticmethod
    def _get_nan_replacement(field_name: str, original_value: Any) -> Optional[str]:
        """Get appropriate replacement for NaN values based on field importance."""
        field_lower = field_name.lower()
        
        # Critical fields - cannot be NaN
        critical_fields = [
            'user_input', 'reference', 'question', 'answer', 
            'ground_truth', 'contexts', 'context'
        ]
        
        if any(critical in field_lower for critical in critical_fields):
            return None  # Signal that sample should be rejected
        
        # Important fields - provide meaningful defaults
        if 'user_input' in field_lower or 'question' in field_lower:
            return "[Generated question placeholder]"
        elif 'reference' in field_lower or 'answer' in field_lower or 'ground_truth' in field_lower:
            return "[Generated answer placeholder]"
        elif 'context' in field_lower:
            return ["[Generated context placeholder]"] if 'contexts' in field_lower else "[Generated context placeholder]"
        elif 'reference_contexts' in field_lower:
            return ["[Missing context information]"]
        elif 'synthesizer_name' in field_lower:
            return "default_synthesizer"
        elif 'metadata' in field_lower:
            return {}
        elif 'source' in field_lower:
            return "unknown_source"
        elif 'id' in field_lower:
            return f"auto_id_{hash(str(original_value)) % 10000}"
        else:
            # Optional fields - provide generic defaults
            return f"[Missing {field_name.replace('_', ' ').title()}]"
    
    @staticmethod
    def _validate_required_fields(data: Dict[str, Any]) -> bool:
        """Validate that required fields are present and valid."""
        
        # Define required field groups (at least one from each group must be present)
        required_groups = [
            ['user_input', 'question'],  # Input question
            ['reference', 'answer', 'ground_truth']  # Expected answer
        ]
        
        for group in required_groups:
            group_satisfied = False
            for field in group:
                if field in data and data[field] and str(data[field]).strip():
                    # Check that it's not just a placeholder
                    value_str = str(data[field]).lower()
                    if not any(placeholder in value_str for placeholder in ['[missing', '[generated', 'placeholder']):
                        group_satisfied = True
                        break
            
            if not group_satisfied:
                logger.debug(f"⚠️ Required field group {group} not satisfied")
                return False
                
        return True
    
    @staticmethod
    def _validate_eval_sample(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and fix eval_sample structure."""
        if 'eval_sample' not in data:
            # Create eval_sample from available data
            eval_sample = SampleValidator._create_eval_sample_from_data(data)
            if eval_sample:
                data['eval_sample'] = eval_sample
                logger.debug("✅ Created eval_sample from available data")
            return data
            
        eval_sample = data['eval_sample']
        
        # Handle NaN eval_sample
        if PANDAS_AVAILABLE and pd.isna(eval_sample):
            is_nan = True
        else:
            is_nan = (
                eval_sample is None or
                (isinstance(eval_sample, float) and str(eval_sample).lower() == 'nan') or
                (isinstance(eval_sample, str) and eval_sample.lower() in ['nan', 'none', ''])
            )
        
        if is_nan:
            # Create valid eval_sample from other data
            new_eval_sample = SampleValidator._create_eval_sample_from_data(data)
            if new_eval_sample:
                data['eval_sample'] = new_eval_sample
                logger.debug("✅ Fixed NaN eval_sample")
            else:
                logger.debug("⚠️ Cannot create valid eval_sample")
                return None
                
        return data
    
    @staticmethod
    def _create_eval_sample_from_data(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Create eval_sample structure from available data fields."""
        eval_sample = {}
        
        # Map common field names to eval_sample structure
        field_mappings = {
            'user_input': ['user_input', 'question', 'query'],
            'reference': ['reference', 'answer', 'ground_truth', 'expected_answer'],
            'reference_contexts': ['reference_contexts', 'contexts', 'context', 'retrieved_contexts']
        }
        
        for target_field, source_fields in field_mappings.items():
            for source_field in source_fields:
                if source_field in data and data[source_field]:
                    value = data[source_field]
                    
                    # Handle contexts specially (should be a list)
                    if target_field == 'reference_contexts':
                        if isinstance(value, str):
                            eval_sample[target_field] = [value]
                        elif isinstance(value, list):
                            eval_sample[target_field] = value
                        else:
                            eval_sample[target_field] = [str(value)]
                    else:
                        eval_sample[target_field] = str(value)
                    break
        
        # Ensure minimum required fields
        if 'user_input' not in eval_sample:
            eval_sample['user_input'] = '[Generated question]'
        if 'reference' not in eval_sample:
            eval_sample['reference'] = '[Generated answer]'
        if 'reference_contexts' not in eval_sample:
            eval_sample['reference_contexts'] = ['[Generated context]']
        
        return eval_sample if eval_sample else None
    
    @staticmethod
    def _validate_content_quality(data: Dict[str, Any]) -> bool:
        """Validate content quality of the sample."""
        
        # Check minimum content length
        min_length = 5
        
        # Check user_input quality
        user_input = data.get('user_input', '')
        if len(str(user_input).strip()) < min_length:
            logger.debug(f"⚠️ User input too short: '{user_input}'")
            return False
        
        # Check reference quality
        reference = data.get('reference', '')
        if len(str(reference).strip()) < min_length:
            logger.debug(f"⚠️ Reference too short: '{reference}'")
            return False
        
        # Check for obvious placeholder content
        combined_content = f"{user_input} {reference}".lower()
        placeholder_indicators = [
            'placeholder', '[missing', '[generated', 'lorem ipsum',
            'test test test', 'xxx', 'todo', 'fixme'
        ]
        
        if any(indicator in combined_content for indicator in placeholder_indicators):
            # Allow some placeholders but not if they dominate the content
            placeholder_ratio = sum(1 for indicator in placeholder_indicators if indicator in combined_content)
            if placeholder_ratio / len(combined_content.split()) > 0.3:  # More than 30% placeholders
                logger.debug("⚠️ Too much placeholder content")
                return False
        
        return True


class BatchSampleValidator:
    """Validates batches of samples and provides recovery mechanisms."""
    
    def __init__(self, min_success_rate: float = 0.1, max_batch_size: int = 1000):
        self.min_success_rate = min_success_rate
        self.max_batch_size = max_batch_size
        self.validation_stats = {
            'batches_processed': 0,
            'total_input_samples': 0,
            'total_output_samples': 0,
            'total_validation_errors': 0,
            'avg_success_rate': 0.0
        }
        
    def validate_batch(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Validate a batch of samples.
        
        Args:
            samples: List of sample dictionaries
            
        Returns:
            Tuple of (valid_samples, validation_report)
        """
        if not samples:
            return [], {'error': 'Empty sample batch'}
        
        # Split large batches for better error handling
        if len(samples) > self.max_batch_size:
            return self._validate_large_batch(samples)
        
        valid_samples = []
        validation_report = {
            'batch_size': len(samples),
            'total_input': len(samples),
            'valid_output': 0,
            'invalid_samples': 0,
            'issues_fixed': 0,
            'success_rate': 0.0,
            'validation_errors': [],
            'batch_timestamp': datetime.now().isoformat(),
            'sample_issues': {}
        }
        
        for i, sample in enumerate(samples):
            try:
                cleaned_sample = SampleValidator.validate_sample_data(sample)
                if cleaned_sample:
                    valid_samples.append(cleaned_sample)
                    validation_report['valid_output'] += 1
                else:
                    validation_report['invalid_samples'] += 1
                    validation_report['sample_issues'][i] = "Failed validation"
                    logger.debug(f"Sample {i} failed validation")
                    
            except Exception as e:
                validation_report['invalid_samples'] += 1
                validation_report['validation_errors'].append(f"Sample {i}: {str(e)}")
                validation_report['sample_issues'][i] = str(e)
                logger.warning(f"Sample {i} validation error: {e}")
        
        # Calculate metrics
        validation_report['success_rate'] = validation_report['valid_output'] / validation_report['total_input'] if validation_report['total_input'] > 0 else 0.0
        validation_report['samples_preserved'] = len(valid_samples)
        validation_report['samples_rejected'] = validation_report['total_input'] - len(valid_samples)
        
        # Update global stats
        self._update_stats(validation_report)
        
        # Log results
        self._log_batch_results(validation_report)
        
        # Check if success rate is acceptable
        if validation_report['success_rate'] < self.min_success_rate:
            logger.error(f"❌ Batch validation failed: success rate {validation_report['success_rate']:.1%} < minimum {self.min_success_rate:.1%}")
            validation_report['batch_status'] = 'FAILED'
        else:
            logger.info(f"✅ Batch validation passed: {validation_report['valid_output']}/{validation_report['total_input']} samples valid ({validation_report['success_rate']:.1%})")
            validation_report['batch_status'] = 'PASSED'
        
        return valid_samples, validation_report
    
    def _validate_large_batch(self, samples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Handle validation of large batches by splitting them."""
        logger.info(f"🔄 Splitting large batch of {len(samples)} samples into smaller chunks")
        
        all_valid_samples = []
        all_reports = []
        
        # Process in chunks
        for i in range(0, len(samples), self.max_batch_size):
            chunk = samples[i:i + self.max_batch_size]
            valid_chunk, chunk_report = self.validate_batch(chunk)
            all_valid_samples.extend(valid_chunk)
            all_reports.append(chunk_report)
        
        # Combine reports
        combined_report = self._combine_reports(all_reports)
        combined_report['was_split'] = True
        combined_report['num_chunks'] = len(all_reports)
        
        return all_valid_samples, combined_report
    
    def _combine_reports(self, reports: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Combine multiple validation reports."""
        combined = {
            'total_input': sum(r.get('total_input', 0) for r in reports),
            'valid_output': sum(r.get('valid_output', 0) for r in reports),
            'invalid_samples': sum(r.get('invalid_samples', 0) for r in reports),
            'issues_fixed': sum(r.get('issues_fixed', 0) for r in reports),
            'validation_errors': [],
            'chunk_reports': reports,
            'combined_timestamp': datetime.now().isoformat()
        }
        
        # Combine errors
        for report in reports:
            combined['validation_errors'].extend(report.get('validation_errors', []))
        
        # Calculate combined metrics
        combined['success_rate'] = combined['valid_output'] / combined['total_input'] if combined['total_input'] > 0 else 0.0
        combined['samples_preserved'] = combined['valid_output']
        combined['samples_rejected'] = combined['total_input'] - combined['valid_output']
        
        return combined
    
    def _update_stats(self, report: Dict[str, Any]) -> None:
        """Update global validation statistics."""
        self.validation_stats['batches_processed'] += 1
        self.validation_stats['total_input_samples'] += report['total_input']
        self.validation_stats['total_output_samples'] += report['valid_output']
        self.validation_stats['total_validation_errors'] += len(report.get('validation_errors', []))
        
        # Update average success rate
        if self.validation_stats['total_input_samples'] > 0:
            self.validation_stats['avg_success_rate'] = self.validation_stats['total_output_samples'] / self.validation_stats['total_input_samples']
    
    def _log_batch_results(self, report: Dict[str, Any]) -> None:
        """Log batch validation results."""
        success_rate = report['success_rate']
        
        if success_rate >= 0.8:
            logger.info(f"✅ Excellent validation: {report['valid_output']}/{report['total_input']} samples ({success_rate:.1%})")
        elif success_rate >= 0.5:
            logger.info(f"⚠️ Good validation: {report['valid_output']}/{report['total_input']} samples ({success_rate:.1%})")
        elif success_rate >= self.min_success_rate:
            logger.warning(f"⚠️ Acceptable validation: {report['valid_output']}/{report['total_input']} samples ({success_rate:.1%})")
        else:
            logger.error(f"❌ Poor validation: {report['valid_output']}/{report['total_input']} samples ({success_rate:.1%})")
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation operations."""
        summary = self.validation_stats.copy()
        summary['overall_preservation_rate'] = summary['avg_success_rate']
        summary['total_samples_processed'] = summary['total_input_samples']
        summary['total_samples_preserved'] = summary['total_output_samples']
        summary['total_samples_lost'] = summary['total_input_samples'] - summary['total_output_samples']
        
        return summary
    
    def save_validation_report(self, output_dir: Path, additional_data: Dict[str, Any] = None) -> Path:
        """Save comprehensive validation report."""
        from pathlib import Path
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"sample_validation_report_{timestamp}.json"
        
        report_data = {
            'validation_summary': self.get_validation_summary(),
            'validation_config': {
                'min_success_rate': self.min_success_rate,
                'max_batch_size': self.max_batch_size
            },
            'timestamp': datetime.now().isoformat()
        }
        
        if additional_data:
            report_data.update(additional_data)
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Saved sample validation report: {report_path}")
        return report_path
