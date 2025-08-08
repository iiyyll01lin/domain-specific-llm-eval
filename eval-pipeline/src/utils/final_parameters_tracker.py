"""
Final Parameters Tracker for Domain-Specific RAG Evaluation Pipeline

This module tracks the final parameters actually used in the pipeline,
including fallback usage and configuration changes that occurred during execution.
This is valuable for understanding what configuration was actually applied
vs what was requested in the config file.
"""

import logging
import copy
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ParameterChange:
    """Represents a change to a parameter during pipeline execution"""
    parameter_path: str
    original_value: Any
    final_value: Any
    change_reason: str
    timestamp: datetime
    component: str
    fallback_used: bool = False

@dataclass
class FallbackUsage:
    """Represents usage of a fallback configuration"""
    component: str
    fallback_type: str
    original_config: Any
    fallback_config: Any
    reason: str
    timestamp: datetime
    success: bool

class FinalParametersTracker:
    """
    Tracks the final parameters and configuration actually used in the pipeline
    """
    
    def __init__(self, initial_config: Dict[str, Any], config: Dict[str, Any] = None):
        """
        Initialize parameters tracker
        
        Args:
            initial_config: The original configuration loaded from file
            config: Tracking configuration
        """
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        if not self.enabled:
            return
            
        # Configuration
        self.capture_fallback_usage = self.config.get('capture_fallback_usage', True)
        self.capture_final_config = self.config.get('capture_final_config', True)
        self.include_generation_metadata = self.config.get('include_generation_metadata', True)
        self.include_evaluation_metadata = self.config.get('include_evaluation_metadata', True)
        
        # Storage
        self.initial_config = copy.deepcopy(initial_config)
        self.final_config = copy.deepcopy(initial_config)
        self.parameter_changes: List[ParameterChange] = []
        self.fallback_usages: List[FallbackUsage] = []
        
        # Metadata collection
        self.generation_metadata: Dict[str, Any] = {}
        self.evaluation_metadata: Dict[str, Any] = {}
        
        # Tracking
        self.components_tracked: set = set()
        self.tracking_start_time = datetime.now()
        
        logger.info("✅ Final parameters tracker initialized")
        
    def track_parameter_change(self, parameter_path: str, original_value: Any, 
                             final_value: Any, reason: str, component: str,
                             is_fallback: bool = False):
        """
        Track a parameter change during pipeline execution
        
        Args:
            parameter_path: Dot-separated path to the parameter (e.g., 'llm.model')
            original_value: Original parameter value
            final_value: Final parameter value used
            reason: Reason for the change
            component: Component that made the change
            is_fallback: Whether this change represents a fallback
        """
        if not self.enabled:
            return
            
        change = ParameterChange(
            parameter_path=parameter_path,
            original_value=original_value,
            final_value=final_value,
            change_reason=reason,
            timestamp=datetime.now(),
            component=component,
            fallback_used=is_fallback
        )
        
        self.parameter_changes.append(change)
        
        # Update final config
        self._update_nested_dict(self.final_config, parameter_path, final_value)
        
        # Track component
        self.components_tracked.add(component)
        
        logger.debug(f"📝 Parameter change tracked: {parameter_path} = {final_value} (reason: {reason})")
        
    def track_fallback_usage(self, component: str, fallback_type: str,
                           original_config: Any, fallback_config: Any,
                           reason: str, success: bool = True):
        """
        Track usage of a fallback configuration
        
        Args:
            component: Component using the fallback
            fallback_type: Type of fallback (e.g., 'model', 'endpoint', 'metric')
            original_config: Original configuration that failed
            fallback_config: Fallback configuration used
            reason: Reason for fallback
            success: Whether the fallback was successful
        """
        if not self.enabled or not self.capture_fallback_usage:
            return
            
        fallback = FallbackUsage(
            component=component,
            fallback_type=fallback_type,
            original_config=original_config,
            fallback_config=fallback_config,
            reason=reason,
            timestamp=datetime.now(),
            success=success
        )
        
        self.fallback_usages.append(fallback)
        self.components_tracked.add(component)
        
        logger.debug(f"🔄 Fallback usage tracked: {component} {fallback_type} - {reason}")
        
    def update_generation_metadata(self, metadata: Dict[str, Any]):
        """Update testset generation metadata"""
        if not self.enabled or not self.include_generation_metadata:
            return
            
        self.generation_metadata.update(metadata)
        logger.debug("📊 Generation metadata updated")
        
    def update_evaluation_metadata(self, metadata: Dict[str, Any]):
        """Update evaluation metadata"""
        if not self.enabled or not self.include_evaluation_metadata:
            return
            
        self.evaluation_metadata.update(metadata)
        logger.debug("📊 Evaluation metadata updated")
        
    def get_final_config(self) -> Dict[str, Any]:
        """Get the final configuration actually used"""
        if not self.enabled:
            return self.initial_config
            
        return copy.deepcopy(self.final_config)
        
    def get_parameter_changes_summary(self) -> Dict[str, Any]:
        """Get summary of parameter changes"""
        if not self.enabled:
            return {}
            
        # Count changes by component and type
        changes_by_component = defaultdict(list)
        fallback_changes = []
        manual_changes = []
        
        for change in self.parameter_changes:
            changes_by_component[change.component].append(change)
            if change.fallback_used:
                fallback_changes.append(change)
            else:
                manual_changes.append(change)
                
        return {
            'total_parameter_changes': len(self.parameter_changes),
            'fallback_parameter_changes': len(fallback_changes),
            'manual_parameter_changes': len(manual_changes),
            'components_with_changes': list(changes_by_component.keys()),
            'changes_by_component': {
                component: len(changes) 
                for component, changes in changes_by_component.items()
            }
        }
        
    def get_fallback_summary(self) -> Dict[str, Any]:
        """Get summary of fallback usage"""
        if not self.enabled:
            return {}
            
        successful_fallbacks = [f for f in self.fallback_usages if f.success]
        failed_fallbacks = [f for f in self.fallback_usages if not f.success]
        
        fallbacks_by_component = defaultdict(list)
        fallbacks_by_type = defaultdict(list)
        
        for fallback in self.fallback_usages:
            fallbacks_by_component[fallback.component].append(fallback)
            fallbacks_by_type[fallback.fallback_type].append(fallback)
            
        return {
            'total_fallbacks_attempted': len(self.fallback_usages),
            'successful_fallbacks': len(successful_fallbacks),
            'failed_fallbacks': len(failed_fallbacks),
            'fallback_success_rate': len(successful_fallbacks) / len(self.fallback_usages) if self.fallback_usages else 0,
            'fallbacks_by_component': {
                component: len(fallbacks)
                for component, fallbacks in fallbacks_by_component.items()
            },
            'fallbacks_by_type': {
                fallback_type: len(fallbacks)
                for fallback_type, fallbacks in fallbacks_by_type.items()
            }
        }
        
    def get_comprehensive_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of final parameters and metadata"""
        if not self.enabled:
            return {'final_parameters_tracking': 'disabled'}
            
        tracking_duration = (datetime.now() - self.tracking_start_time).total_seconds()
        
        summary = {
            'final_parameters_tracking': {
                'enabled': True,
                'tracking_period': {
                    'start_time': self.tracking_start_time.isoformat(),
                    'end_time': datetime.now().isoformat(),
                    'duration_seconds': tracking_duration
                },
                'components_tracked': list(self.components_tracked),
                'configuration': {
                    'capture_fallback_usage': self.capture_fallback_usage,
                    'capture_final_config': self.capture_final_config,
                    'include_generation_metadata': self.include_generation_metadata,
                    'include_evaluation_metadata': self.include_evaluation_metadata
                }
            }
        }
        
        # Add parameter changes summary
        if self.parameter_changes:
            summary['parameter_changes'] = self.get_parameter_changes_summary()
            
        # Add fallback summary
        if self.fallback_usages:
            summary['fallback_usage'] = self.get_fallback_summary()
            
        # Add final configuration if enabled
        if self.capture_final_config:
            summary['final_configuration'] = self.get_final_config()
            
        # Add metadata
        if self.generation_metadata:
            summary['testset_generation_metadata'] = self.generation_metadata
            
        if self.evaluation_metadata:
            summary['evaluation_metadata'] = self.evaluation_metadata
            
        return summary
        
    def get_detailed_changes(self) -> List[Dict[str, Any]]:
        """Get detailed list of all parameter changes"""
        if not self.enabled:
            return []
            
        return [
            {
                'parameter_path': change.parameter_path,
                'original_value': change.original_value,
                'final_value': change.final_value,
                'change_reason': change.change_reason,
                'timestamp': change.timestamp.isoformat(),
                'component': change.component,
                'fallback_used': change.fallback_used
            }
            for change in self.parameter_changes
        ]
        
    def get_detailed_fallbacks(self) -> List[Dict[str, Any]]:
        """Get detailed list of all fallback usages"""
        if not self.enabled:
            return []
            
        return [
            {
                'component': fallback.component,
                'fallback_type': fallback.fallback_type,
                'original_config': fallback.original_config,
                'fallback_config': fallback.fallback_config,
                'reason': fallback.reason,
                'timestamp': fallback.timestamp.isoformat(),
                'success': fallback.success
            }
            for fallback in self.fallback_usages
        ]
        
    def _update_nested_dict(self, config_dict: Dict[str, Any], path: str, value: Any):
        """Update a nested dictionary using a dot-separated path"""
        keys = path.split('.')
        current = config_dict
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
            
        # Set the final value
        current[keys[-1]] = value
        
    def get_config_comparison(self) -> Dict[str, Any]:
        """Get comparison between initial and final configurations"""
        if not self.enabled:
            return {}
            
        def _compare_configs(initial: Any, final: Any, path: str = "") -> List[Dict[str, Any]]:
            """Recursively compare two configuration objects"""
            differences = []
            
            if isinstance(initial, dict) and isinstance(final, dict):
                all_keys = set(initial.keys()) | set(final.keys())
                
                for key in all_keys:
                    current_path = f"{path}.{key}" if path else key
                    
                    if key not in initial:
                        differences.append({
                            'path': current_path,
                            'type': 'added',
                            'initial_value': None,
                            'final_value': final[key]
                        })
                    elif key not in final:
                        differences.append({
                            'path': current_path,
                            'type': 'removed',
                            'initial_value': initial[key],
                            'final_value': None
                        })
                    elif initial[key] != final[key]:
                        if isinstance(initial[key], dict) and isinstance(final[key], dict):
                            differences.extend(_compare_configs(initial[key], final[key], current_path))
                        else:
                            differences.append({
                                'path': current_path,
                                'type': 'modified',
                                'initial_value': initial[key],
                                'final_value': final[key]
                            })
            elif initial != final:
                differences.append({
                    'path': path,
                    'type': 'modified',
                    'initial_value': initial,
                    'final_value': final
                })
                
            return differences
            
        differences = _compare_configs(self.initial_config, self.final_config)
        
        return {
            'total_differences': len(differences),
            'differences': differences,
            'comparison_timestamp': datetime.now().isoformat()
        }
        
    def reset(self):
        """Reset all tracking data"""
        if not self.enabled:
            return
            
        self.parameter_changes.clear()
        self.fallback_usages.clear()
        self.generation_metadata.clear()
        self.evaluation_metadata.clear()
        self.components_tracked.clear()
        
        # Reset final config to initial
        self.final_config = copy.deepcopy(self.initial_config)
        self.tracking_start_time = datetime.now()
        
        logger.info("🔄 Final parameters tracker reset")
