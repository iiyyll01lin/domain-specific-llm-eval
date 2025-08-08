"""
Enhanced Pipeline Orchestrator with Comprehensive Validation

This orchestrator integrates all validation systems to ensure robust pipeline execution:
- CSV data validation and cleaning
- Sample validation during testset generation  
- Knowledge graph validation
- Comprehensive error handling and recovery

Phase 2: Architecture Cleanup Implementation
This version now uses the refactored enhanced orchestrator with separated concerns.
"""

from pipeline.refactored_enhanced_orchestrator import RefactoredEnhancedPipelineOrchestrator

# For backward compatibility, alias the refactored version
class EnhancedPipelineOrchestrator(RefactoredEnhancedPipelineOrchestrator):
    """
    Enhanced pipeline orchestrator with comprehensive validation and error recovery.
    
    Phase 2: This class now uses the refactored architecture with separated
    core execution logic and validation/enhancement features.
    """
    pass
