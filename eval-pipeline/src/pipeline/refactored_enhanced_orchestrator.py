"""
Refactored Enhanced Pipeline Orchestrator using Stage Factories

This refactored version separates core logic from validation/enhancement logic
and uses stage factories for consistent execution patterns.
"""

import json
import logging
import time
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime

# Import validation modules
from validation.csv_validator import CSVDataValidator
from validation.sample_validator import BatchSampleValidator
from validation.kg_validator import KnowledgeGraphValidator, KGIntegrityChecker
from validation.robust_sample_processor import RobustSampleProcessor

# Import stage factories for core execution
from pipeline.stage_factories import StageFactory, StageComposer

# Import the original orchestrator to extend it
from pipeline.orchestrator import PipelineOrchestrator

logger = logging.getLogger(__name__)

class RefactoredEnhancedPipelineOrchestrator(PipelineOrchestrator):
    """
    Refactored enhanced pipeline orchestrator with separated concerns.
    
    This version uses stage factories for core execution while providing
    enhanced validation and recovery capabilities.
    """
    
    def __init__(self, config: Dict[str, Any], run_id: str, output_dirs: Dict[str, Path], force_overwrite: bool = False):
        """
        Initialize refactored enhanced pipeline orchestrator.
        
        Args:
            config: Pipeline configuration
            run_id: Unique run identifier
            output_dirs: Output directory structure
            force_overwrite: Whether to overwrite existing outputs
        """
        # Initialize base orchestrator
        super().__init__(config, run_id, output_dirs, force_overwrite)
        
        # Initialize validation components with enhanced robust settings
        validation_config = config.get('validation', {})
        
        self.csv_validator = CSVDataValidator(config)
        
        # Use more permissive thresholds for robustness
        self.sample_validator = BatchSampleValidator(
            min_success_rate=validation_config.get('min_sample_success_rate', 0.02),
            max_batch_size=validation_config.get('max_batch_size', 1000)
        )
        
        # Create stage composer for multi-stage execution
        self.stage_composer = StageComposer(config, run_id, output_dirs)
        
        # Enhanced validation settings
        self.validation_enabled = validation_config.get('enabled', True)
        self.save_validation_reports = validation_config.get('save_validation_reports', True)
        self.enable_enhanced_recovery = validation_config.get('enable_aggressive_recovery', True)
        
        # Create validation reports accumulator
        self.validation_reports = []
        
        logger.info("🏗️ Refactored Enhanced Pipeline Orchestrator initialized with validation support")
    
    def run(self, stage: str = "all") -> Dict[str, Any]:
        """
        Run the pipeline with enhanced validation and error recovery.
        
        Args:
            stage: Stage to execute ('all', 'testset-generation', 'evaluation', 'reporting')
            
        Returns:
            Pipeline execution results with validation information
        """
        logger.info(f"🚀 Starting enhanced pipeline execution (stage: {stage})")
        pipeline_start_time = datetime.now()
        
        try:
            # Pre-execution validation
            if self.validation_enabled:
                validation_result = self._run_pre_execution_validation()
                if not validation_result['success'] and not validation_result.get('can_continue', False):
                    return {
                        'success': False,
                        'error': 'Pre-execution validation failed',
                        'validation_reports': self.validation_reports
                    }
            
            # Execute pipeline stages using stage factories
            if stage == "all":
                # Full pipeline execution with all stages
                stage_names = ['testset-generation', 'evaluation', 'reporting']
                results = self.stage_composer.compose_stages(stage_names)
            else:
                # Single stage execution
                stage_executor = StageFactory.create_executor(stage, self.config, self.run_id, self.output_dirs)
                results = stage_executor.execute()
                
                # Wrap single stage result in pipeline format
                results = {
                    'success': results.get('success', False),
                    'stages_executed': [stage],
                    'total_duration': results.get('duration', 0.0),
                    stage.replace('-', '_'): results
                }
            
            # Post-execution validation
            if self.validation_enabled and results.get('success', False):
                post_validation_result = self._run_post_execution_validation(results)
                results['post_validation'] = post_validation_result
            
            # Add validation reports to results
            if self.validation_reports:
                results['validation_reports'] = self.validation_reports
            
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            results['pipeline_duration'] = pipeline_duration
            
            logger.info(f"🎉 Enhanced pipeline execution completed in {pipeline_duration:.2f}s")
            return results
            
        except Exception as e:
            pipeline_duration = (datetime.now() - pipeline_start_time).total_seconds()
            logger.error(f"💥 Enhanced pipeline execution failed: {e}")
            
            return {
                'success': False,
                'error': str(e),
                'pipeline_duration': pipeline_duration,
                'validation_reports': self.validation_reports
            }
    
    def _run_pre_execution_validation(self) -> Dict[str, Any]:
        """
        Run comprehensive pre-execution validation.
        
        Returns:
            Validation results with success status and recommendations
        """
        logger.info("🔍 Running pre-execution validation...")
        validation_start_time = datetime.now()
        
        validation_result = {
            'success': True,
            'can_continue': True,
            'issues_found': [],
            'fixes_applied': [],
            'recommendations': []
        }
        
        try:
            # CSV data validation
            if self.config.get('data_sources', {}).get('input_type') == 'csv':
                csv_validation = self._validate_csv_data()
                validation_result['csv_validation'] = csv_validation
                
                if not csv_validation.get('success', False):
                    validation_result['issues_found'].append('CSV data validation failed')
                    
                    # Apply recovery if enabled
                    if self.enable_enhanced_recovery:
                        recovery_result = self._apply_csv_recovery(csv_validation)
                        if recovery_result.get('success', False):
                            validation_result['fixes_applied'].extend(recovery_result.get('fixes_applied', []))
                        else:
                            validation_result['can_continue'] = False
                            validation_result['success'] = False
            
            # Configuration validation
            config_validation = self._validate_configuration()
            validation_result['config_validation'] = config_validation
            
            if not config_validation.get('success', False):
                validation_result['issues_found'].append('Configuration validation failed')
                # Configuration issues are usually not recoverable automatically
                validation_result['recommendations'].extend(config_validation.get('recommendations', []))
            
            # Environment validation
            env_validation = self._validate_environment()
            validation_result['environment_validation'] = env_validation
            
            if not env_validation.get('success', False):
                validation_result['issues_found'].append('Environment validation failed')
                validation_result['recommendations'].extend(env_validation.get('recommendations', []))
            
            validation_duration = (datetime.now() - validation_start_time).total_seconds()
            validation_result['duration'] = validation_duration
            
            # Add to validation reports
            self.validation_reports.append({
                'type': 'pre_execution',
                'timestamp': datetime.now().isoformat(),
                'result': validation_result
            })
            
            logger.info(f"✅ Pre-execution validation completed in {validation_duration:.2f}s")
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Pre-execution validation failed: {e}")
            validation_result['success'] = False
            validation_result['can_continue'] = False
            validation_result['error'] = str(e)
            return validation_result
    
    def _run_post_execution_validation(self, execution_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run post-execution validation to verify results quality.
        
        Args:
            execution_results: Results from pipeline execution
            
        Returns:
            Post-execution validation results
        """
        logger.info("🔍 Running post-execution validation...")
        validation_start_time = datetime.now()
        
        validation_result = {
            'success': True,
            'quality_checks': [],
            'recommendations': []
        }
        
        try:
            # Validate testset generation results
            if 'testset_generation' in execution_results:
                testset_validation = self._validate_testset_results(
                    execution_results['testset_generation']
                )
                validation_result['testset_validation'] = testset_validation
                validation_result['quality_checks'].append('testset_generation')
            
            # Validate evaluation results
            if 'evaluation' in execution_results:
                eval_validation = self._validate_evaluation_results(
                    execution_results['evaluation']
                )
                validation_result['evaluation_validation'] = eval_validation
                validation_result['quality_checks'].append('evaluation')
            
            # Validate reporting results
            if 'reporting' in execution_results:
                report_validation = self._validate_reporting_results(
                    execution_results['reporting']
                )
                validation_result['reporting_validation'] = report_validation
                validation_result['quality_checks'].append('reporting')
            
            validation_duration = (datetime.now() - validation_start_time).total_seconds()
            validation_result['duration'] = validation_duration
            
            # Add to validation reports
            self.validation_reports.append({
                'type': 'post_execution',
                'timestamp': datetime.now().isoformat(),
                'result': validation_result
            })
            
            logger.info(f"✅ Post-execution validation completed in {validation_duration:.2f}s")
            return validation_result
            
        except Exception as e:
            logger.error(f"❌ Post-execution validation failed: {e}")
            validation_result['success'] = False
            validation_result['error'] = str(e)
            return validation_result
    
    def _validate_csv_data(self) -> Dict[str, Any]:
        """Validate CSV data sources."""
        try:
            csv_files = self.config.get('data_sources', {}).get('csv', {}).get('csv_files', [])
            if not csv_files:
                return {'success': True, 'message': 'No CSV files to validate'}
            
            validation_results = []
            overall_success = True
            
            for csv_file in csv_files:
                file_validation = self.csv_validator.validate_file(csv_file)
                validation_results.append(file_validation)
                
                if not file_validation.get('success', False):
                    overall_success = False
            
            return {
                'success': overall_success,
                'file_validations': validation_results,
                'files_validated': len(csv_files)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _apply_csv_recovery(self, csv_validation: Dict[str, Any]) -> Dict[str, Any]:
        """Apply CSV data recovery mechanisms."""
        logger.info("🔧 Applying CSV data recovery...")
        
        recovery_result = {
            'success': True,
            'fixes_applied': []
        }
        
        try:
            # Apply data cleaning and repair
            file_validations = csv_validation.get('file_validations', [])
            
            for file_validation in file_validations:
                if not file_validation.get('success', False):
                    # Apply specific fixes based on validation issues
                    fixes = self.csv_validator.apply_fixes(file_validation)
                    recovery_result['fixes_applied'].extend(fixes)
            
            return recovery_result
            
        except Exception as e:
            logger.error(f"CSV recovery failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_configuration(self) -> Dict[str, Any]:
        """Validate pipeline configuration."""
        try:
            recommendations = []
            issues = []
            
            # Check required sections
            required_sections = ['data_sources', 'testset_generation', 'evaluation', 'reporting']
            for section in required_sections:
                if section not in self.config:
                    issues.append(f"Missing required configuration section: {section}")
            
            # Check LLM configuration
            llm_config = self.config.get('llm', {})
            if not llm_config.get('api_key'):
                issues.append("Missing LLM API key")
            
            # Check testset generation targets
            tg_config = self.config.get('testset_generation', {})
            max_total_samples = tg_config.get('max_total_samples', 0)
            if max_total_samples > 1000:
                recommendations.append(f"Large testset target ({max_total_samples}) may take significant time")
            
            return {
                'success': len(issues) == 0,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_environment(self) -> Dict[str, Any]:
        """Validate execution environment."""
        try:
            recommendations = []
            issues = []
            
            # Check memory configuration
            max_memory = self.config.get('advanced', {}).get('resource_limits', {}).get('max_memory_mb', 0)
            if max_memory > 50000:  # 50GB
                recommendations.append(f"High memory configuration ({max_memory}MB) - ensure sufficient RAM available")
            
            # Check output directories
            base_dir = Path(self.config.get('output', {}).get('base_dir', './outputs'))
            if not base_dir.exists():
                try:
                    base_dir.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    issues.append(f"Cannot create output directory: {base_dir}")
            
            return {
                'success': len(issues) == 0,
                'issues': issues,
                'recommendations': recommendations
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_testset_results(self, testset_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate testset generation results quality."""
        try:
            issues = []
            recommendations = []
            
            # Check if testsets were generated
            testsets_generated = testset_results.get('testsets_generated', 0)
            if testsets_generated == 0:
                issues.append("No testsets were generated")
            
            # Check QA pairs count
            total_qa_pairs = testset_results.get('total_qa_pairs', 0)
            if total_qa_pairs == 0:
                issues.append("No QA pairs were generated")
            elif total_qa_pairs < 10:
                recommendations.append(f"Low QA pair count ({total_qa_pairs}) may limit evaluation effectiveness")
            
            return {
                'success': len(issues) == 0,
                'issues': issues,
                'recommendations': recommendations,
                'testsets_generated': testsets_generated,
                'total_qa_pairs': total_qa_pairs
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_evaluation_results(self, evaluation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate evaluation results quality."""
        try:
            issues = []
            recommendations = []
            
            # Check if queries were executed
            queries_executed = evaluation_results.get('queries_executed', 0)
            if queries_executed == 0:
                issues.append("No queries were executed")
            
            # Check pass rates
            keyword_pass_rate = evaluation_results.get('keyword_pass_rate', 0)
            if keyword_pass_rate < 0.1:  # Less than 10%
                recommendations.append(f"Low keyword pass rate ({keyword_pass_rate:.1%}) may indicate configuration issues")
            
            # Check RAGAS scores
            avg_ragas_score = evaluation_results.get('avg_ragas_score', 0)
            if avg_ragas_score == 0:
                issues.append("No RAGAS scores were computed")
            elif avg_ragas_score < 0.3:  # Less than 30%
                recommendations.append(f"Low average RAGAS score ({avg_ragas_score:.3f}) may indicate quality issues")
            
            return {
                'success': len(issues) == 0,
                'issues': issues,
                'recommendations': recommendations,
                'queries_executed': queries_executed,
                'keyword_pass_rate': keyword_pass_rate,
                'avg_ragas_score': avg_ragas_score
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def _validate_reporting_results(self, reporting_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate reporting results."""
        try:
            issues = []
            recommendations = []
            
            # Check if reports were generated
            reports_generated = reporting_results.get('reports_generated', [])
            if not reports_generated:
                issues.append("No reports were generated")
            
            # Check report directory
            report_directory = reporting_results.get('report_directory')
            if report_directory and not Path(report_directory).exists():
                issues.append(f"Report directory does not exist: {report_directory}")
            
            return {
                'success': len(issues) == 0,
                'issues': issues,
                'recommendations': recommendations,
                'reports_count': len(reports_generated)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def save_validation_reports(self) -> str:
        """
        Save validation reports to file.
        
        Returns:
            Path to saved validation report file
        """
        if not self.save_validation_reports or not self.validation_reports:
            return None
        
        try:
            # Create validation reports directory
            validation_dir = self.output_dirs['base'] / 'validation'
            validation_dir.mkdir(exist_ok=True)
            
            # Save validation reports
            report_path = validation_dir / f'validation_report_{self.run_id}.json'
            
            with open(report_path, 'w') as f:
                json.dump({
                    'run_id': self.run_id,
                    'timestamp': datetime.now().isoformat(),
                    'validation_reports': self.validation_reports
                }, f, indent=2, default=str)
            
            logger.info(f"📄 Validation reports saved to: {report_path}")
            return str(report_path)
            
        except Exception as e:
            logger.error(f"Failed to save validation reports: {e}")
            return None
