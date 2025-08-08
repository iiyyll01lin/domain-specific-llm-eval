"""
Unified Pipeline Interface - Phase 3 Implementation

This module provides a single entry point that routes to appropriate execution modes
with consistent output format and deterministic behavior across all stage combinations.

Key Features:
- Single entry point for all pipeline operations
- Consistent output format regardless of execution path  
- Deterministic behavior across all stage combinations
- Automatic orchestrator selection based on requirements
- Standardized error handling and reporting
"""

import logging
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass

# Import orchestrators
from pipeline.orchestrator import PipelineOrchestrator
from pipeline.enhanced_orchestrator import EnhancedPipelineOrchestrator
from pipeline.stage_factories import StageFactory, StageComposer

logger = logging.getLogger(__name__)

@dataclass
class PipelineRequest:
    """Standardized pipeline request structure."""
    stage: str
    mode: str
    config: Dict[str, Any]
    run_id: str
    output_dirs: Dict[str, Path]
    force_overwrite: bool = False
    enable_validation: bool = True
    enable_enhanced_features: bool = True

@dataclass 
class PipelineResponse:
    """Standardized pipeline response structure."""
    success: bool
    run_id: str
    execution_mode: str
    orchestrator_type: str
    stage: str
    stages_executed: List[str]
    duration: float
    results: Dict[str, Any]
    validation_reports: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None
    warnings: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

class ExecutionMode:
    """Execution mode constants."""
    SINGLE_STAGE_SIMPLE = "single_stage_simple"
    SINGLE_STAGE_ENHANCED = "single_stage_enhanced"  
    MULTI_STAGE_SIMPLE = "multi_stage_simple"
    MULTI_STAGE_ENHANCED = "multi_stage_enhanced"
    STAGE_FACTORY_COMPOSITION = "stage_factory_composition"

class UnifiedPipelineInterface:
    """
    Unified interface for all pipeline execution modes.
    
    This class provides a single entry point that automatically selects
    the appropriate execution strategy based on the request parameters.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.execution_history = []
    
    def execute_pipeline(self, request: PipelineRequest) -> PipelineResponse:
        """
        Execute pipeline with unified interface.
        
        Args:
            request: Standardized pipeline request
            
        Returns:
            Standardized pipeline response
        """
        self.logger.info(f"🎯 Unified Pipeline Interface: Processing request for stage '{request.stage}'")
        execution_start_time = datetime.now()
        
        try:
            # Determine execution mode
            execution_mode = self._determine_execution_mode(request)
            self.logger.info(f"📋 Selected execution mode: {execution_mode}")
            
            # Route to appropriate executor
            if execution_mode == ExecutionMode.SINGLE_STAGE_SIMPLE:
                orchestrator_type = "PipelineOrchestrator"
                orchestrator = PipelineOrchestrator(
                    config=request.config,
                    run_id=request.run_id,
                    output_dirs=request.output_dirs,
                    force_overwrite=request.force_overwrite
                )
                results = orchestrator.run(stage=request.stage)
                
            elif execution_mode == ExecutionMode.SINGLE_STAGE_ENHANCED:
                orchestrator_type = "EnhancedPipelineOrchestrator"  
                orchestrator = EnhancedPipelineOrchestrator(
                    config=request.config,
                    run_id=request.run_id,
                    output_dirs=request.output_dirs,
                    force_overwrite=request.force_overwrite
                )
                results = orchestrator.run(stage=request.stage)
                
            elif execution_mode == ExecutionMode.MULTI_STAGE_SIMPLE:
                orchestrator_type = "PipelineOrchestrator"
                orchestrator = PipelineOrchestrator(
                    config=request.config,
                    run_id=request.run_id,
                    output_dirs=request.output_dirs,
                    force_overwrite=request.force_overwrite
                )
                results = orchestrator.run(stage=request.stage)
                
            elif execution_mode == ExecutionMode.MULTI_STAGE_ENHANCED:
                orchestrator_type = "EnhancedPipelineOrchestrator"
                orchestrator = EnhancedPipelineOrchestrator(
                    config=request.config,
                    run_id=request.run_id,
                    output_dirs=request.output_dirs,
                    force_overwrite=request.force_overwrite
                )
                results = orchestrator.run(stage=request.stage)
                
            elif execution_mode == ExecutionMode.STAGE_FACTORY_COMPOSITION:
                orchestrator_type = "StageComposer"
                stage_composer = StageComposer(
                    config=request.config,
                    run_id=request.run_id,
                    output_dirs=request.output_dirs
                )
                
                # Convert stage to stage list
                if request.stage == "all":
                    stage_names = ['testset-generation', 'evaluation', 'reporting']
                else:
                    stage_names = [request.stage]
                
                results = stage_composer.compose_stages(stage_names)
                
            else:
                raise ValueError(f"Unknown execution mode: {execution_mode}")
            
            # Standardize response format
            execution_duration = (datetime.now() - execution_start_time).total_seconds()
            
            response = self._create_standardized_response(
                request=request,
                results=results,
                execution_mode=execution_mode,
                orchestrator_type=orchestrator_type,
                duration=execution_duration
            )
            
            # Record execution in history
            self._record_execution(request, response)
            
            self.logger.info(f"✅ Pipeline execution completed successfully in {execution_duration:.2f}s")
            return response
            
        except Exception as e:
            execution_duration = (datetime.now() - execution_start_time).total_seconds()
            self.logger.error(f"❌ Pipeline execution failed: {e}")
            
            # Create error response
            response = PipelineResponse(
                success=False,
                run_id=request.run_id,
                execution_mode="error",
                orchestrator_type="none",
                stage=request.stage,
                stages_executed=[],
                duration=execution_duration,
                results={},
                error=str(e)
            )
            
            self._record_execution(request, response)
            return response
    
    def _determine_execution_mode(self, request: PipelineRequest) -> str:
        """
        Determine the appropriate execution mode based on request parameters.
        
        Args:
            request: Pipeline request
            
        Returns:
            Execution mode string
        """
        # Check if validation is disabled - use simple mode
        if not request.enable_validation:
            if request.stage == "all":
                return ExecutionMode.MULTI_STAGE_SIMPLE
            else:
                return ExecutionMode.SINGLE_STAGE_SIMPLE
        
        # Check if enhanced features are disabled - use simple mode  
        if not request.enable_enhanced_features:
            if request.stage == "all":
                return ExecutionMode.MULTI_STAGE_SIMPLE
            else:
                return ExecutionMode.SINGLE_STAGE_SIMPLE
        
        # Check configuration for validation settings
        validation_config = request.config.get('validation', {})
        validation_enabled = validation_config.get('enabled', True)
        
        # Use stage factory composition for complex validation requirements
        if validation_enabled and validation_config.get('enable_aggressive_recovery', False):
            return ExecutionMode.STAGE_FACTORY_COMPOSITION
        
        # Default routing based on Phase 1 logic
        if request.stage == "all":
            return ExecutionMode.MULTI_STAGE_ENHANCED
        else:
            return ExecutionMode.SINGLE_STAGE_SIMPLE
    
    def _create_standardized_response(
        self, 
        request: PipelineRequest,
        results: Dict[str, Any],
        execution_mode: str,
        orchestrator_type: str,
        duration: float
    ) -> PipelineResponse:
        """
        Create standardized response from execution results.
        
        Args:
            request: Original pipeline request
            results: Raw execution results
            execution_mode: Execution mode used
            orchestrator_type: Type of orchestrator used
            duration: Execution duration
            
        Returns:
            Standardized pipeline response
        """
        # Extract standard fields
        success = results.get('success', False)
        stages_executed = results.get('stages_executed', [request.stage] if request.stage != "all" else [])
        validation_reports = results.get('validation_reports', [])
        error = results.get('error')
        
        # Extract warnings
        warnings = []
        if 'warnings' in results:
            warnings.extend(results['warnings'])
        
        # Create metadata
        metadata = {
            'execution_timestamp': datetime.now().isoformat(),
            'config_source': getattr(request, 'config_source', 'unknown'),
            'total_pipeline_duration': results.get('total_duration', duration),
            'pipeline_version': request.config.get('pipeline', {}).get('version', '2.0.0')
        }
        
        # Add execution-specific metadata
        if 'pipeline_duration' in results:
            metadata['orchestrator_duration'] = results['pipeline_duration']
        
        # Standardize stage results
        standardized_results = self._standardize_stage_results(results, request.stage)
        
        return PipelineResponse(
            success=success,
            run_id=request.run_id,
            execution_mode=execution_mode,
            orchestrator_type=orchestrator_type,
            stage=request.stage,
            stages_executed=stages_executed,
            duration=duration,
            results=standardized_results,
            validation_reports=validation_reports if validation_reports else None,
            error=error,
            warnings=warnings if warnings else None,
            metadata=metadata
        )
    
    def _standardize_stage_results(self, results: Dict[str, Any], requested_stage: str) -> Dict[str, Any]:
        """
        Standardize stage results to consistent format.
        
        Args:
            results: Raw results from execution
            requested_stage: Originally requested stage
            
        Returns:
            Standardized stage results
        """
        standardized = {}
        
        # Stage name mappings
        stage_mappings = {
            'testset-generation': 'testset_generation',
            'evaluation': 'evaluation',
            'reporting': 'reporting'
        }
        
        # Handle different result structures
        for stage_key in ['testset_generation', 'evaluation', 'reporting']:
            if stage_key in results:
                stage_data = results[stage_key]
                
                # Ensure consistent structure
                standardized[stage_key] = {
                    'success': stage_data.get('success', False),
                    'duration': stage_data.get('duration', 0.0),
                    'output_path': stage_data.get('output_path'),
                }
                
                # Add stage-specific fields
                if stage_key == 'testset_generation':
                    standardized[stage_key].update({
                        'documents_processed': stage_data.get('documents_processed', 0),
                        'testsets_generated': stage_data.get('testsets_generated', 0),
                        'total_qa_pairs': stage_data.get('total_qa_pairs', 0)
                    })
                
                elif stage_key == 'evaluation':
                    standardized[stage_key].update({
                        'queries_executed': stage_data.get('queries_executed', 0),
                        'keyword_pass_rate': stage_data.get('keyword_pass_rate', 0.0),
                        'avg_ragas_score': stage_data.get('avg_ragas_score', 0.0),
                        'feedback_requests': stage_data.get('feedback_requests', 0)
                    })
                
                elif stage_key == 'reporting':
                    standardized[stage_key].update({
                        'reports_generated': stage_data.get('reports_generated', []),
                        'report_directory': stage_data.get('report_directory')
                    })
        
        # If only one stage was requested, ensure it's present
        if requested_stage != "all":
            mapped_stage = stage_mappings.get(requested_stage, requested_stage.replace('-', '_'))
            if mapped_stage not in standardized and requested_stage != "all":
                # Create from root level results
                standardized[mapped_stage] = {
                    'success': results.get('success', False),
                    'duration': results.get('duration', 0.0),
                    'output_path': results.get('output_path')
                }
                
                # Add stage-specific fields from root level
                for key, value in results.items():
                    if key not in ['success', 'duration', 'output_path', 'error']:
                        standardized[mapped_stage][key] = value
        
        return standardized
    
    def _record_execution(self, request: PipelineRequest, response: PipelineResponse):
        """Record execution in history for analysis."""
        execution_record = {
            'timestamp': datetime.now().isoformat(),
            'request': {
                'stage': request.stage,
                'mode': request.mode,
                'run_id': request.run_id,
                'enable_validation': request.enable_validation,
                'enable_enhanced_features': request.enable_enhanced_features
            },
            'response': {
                'success': response.success,
                'execution_mode': response.execution_mode,
                'orchestrator_type': response.orchestrator_type,
                'stages_executed': response.stages_executed,
                'duration': response.duration,
                'has_validation_reports': response.validation_reports is not None,
                'has_error': response.error is not None
            }
        }
        
        self.execution_history.append(execution_record)
        
        # Keep only last 100 executions
        if len(self.execution_history) > 100:
            self.execution_history = self.execution_history[-100:]
    
    def get_execution_history(self) -> List[Dict[str, Any]]:
        """Get execution history for analysis."""
        return self.execution_history.copy()
    
    def save_execution_history(self, output_path: Path):
        """Save execution history to file."""
        try:
            with open(output_path, 'w') as f:
                json.dump({
                    'saved_at': datetime.now().isoformat(),
                    'total_executions': len(self.execution_history),
                    'execution_history': self.execution_history
                }, f, indent=2, default=str)
            
            self.logger.info(f"📄 Execution history saved to: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save execution history: {e}")

def create_pipeline_request(
    stage: str,
    mode: str,
    config: Dict[str, Any],
    run_id: str,
    output_dirs: Dict[str, Path],
    force_overwrite: bool = False,
    enable_validation: bool = None,
    enable_enhanced_features: bool = None
) -> PipelineRequest:
    """
    Create a standardized pipeline request.
    
    Args:
        stage: Pipeline stage to execute
        mode: Execution mode
        config: Pipeline configuration
        run_id: Unique run identifier
        output_dirs: Output directory structure
        force_overwrite: Whether to overwrite existing outputs
        enable_validation: Whether to enable validation (auto-detect if None)
        enable_enhanced_features: Whether to enable enhanced features (auto-detect if None)
        
    Returns:
        Standardized pipeline request
    """
    # Auto-detect validation and enhanced features if not specified
    if enable_validation is None:
        enable_validation = config.get('validation', {}).get('enabled', True)
    
    if enable_enhanced_features is None:
        # Enhanced features enabled for multi-stage or when validation is enabled
        enable_enhanced_features = (stage == "all") or enable_validation
    
    return PipelineRequest(
        stage=stage,
        mode=mode,
        config=config,
        run_id=run_id,
        output_dirs=output_dirs,
        force_overwrite=force_overwrite,
        enable_validation=enable_validation,
        enable_enhanced_features=enable_enhanced_features
    )

# Global unified interface instance
_unified_interface = None

def get_unified_interface() -> UnifiedPipelineInterface:
    """Get or create the global unified interface instance."""
    global _unified_interface
    if _unified_interface is None:
        _unified_interface = UnifiedPipelineInterface()
    return _unified_interface
