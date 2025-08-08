"""
Enhanced Trackers for Pipeline Orchestration

This module provides advanced tracking capabilities for:
1. Performance timing at granular levels
2. Composition elements from RAGAS generation  
3. Final parameters tracking with fallback detection
4. Memory and resource usage monitoring
"""

import time
import logging
import json
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Optional psutil import for memory tracking
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    logger.warning("psutil not available - memory tracking will be limited")


class PerformanceTracker:
    """Enhanced performance tracker for pipeline components."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('evaluation', {}).get('performance_tracking', {})
        self.enabled = self.config.get('enabled', True)
        
        # Initialize tracking data
        self.timings = {}
        self.memory_usage = {}
        self.response_times = {
            'rag_responses': [],
            'llm_responses': [],
            'metric_computations': {}
        }
        self.stage_timings = {}
        self.detailed_breakdowns = {}
        
        logger.info(f"📊 Performance Tracker initialized (enabled: {self.enabled})")
    
    def start_timing(self, component: str, operation: str) -> str:
        """Start timing for a specific component and operation."""
        if not self.enabled:
            return None
            
        timing_id = f"{component}_{operation}_{int(time.time() * 1000)}"
        self.timings[timing_id] = {
            'component': component,
            'operation': operation,
            'start_time': time.time(),
            'end_time': None,
            'duration': None
        }
        return timing_id
    
    def end_timing(self, timing_id: str) -> Optional[float]:
        """End timing and return duration."""
        if not self.enabled or not timing_id or timing_id not in self.timings:
            return None
            
        timing = self.timings[timing_id]
        timing['end_time'] = time.time()
        timing['duration'] = timing['end_time'] - timing['start_time']
        
        logger.debug(f"⏱️ {timing['component']}.{timing['operation']}: {timing['duration']:.3f}s")
        return timing['duration']
    
    def track_rag_response_time(self, question: str, response_time: float):
        """Track RAG system response time."""
        if not self.enabled:
            return
            
        self.response_times['rag_responses'].append({
            'question': question[:100] + '...' if len(question) > 100 else question,
            'response_time': response_time,
            'timestamp': time.time()
        })
    
    def track_llm_response_time(self, endpoint: str, response_time: float, operation: str = "generic"):
        """Track LLM endpoint response time."""
        if not self.enabled:
            return
            
        self.response_times['llm_responses'].append({
            'endpoint': endpoint,
            'operation': operation,
            'response_time': response_time,
            'timestamp': time.time()
        })
    
    def track_metric_computation_time(self, metric_name: str, computation_time: float, question_count: int = 1):
        """Track time for specific metric computation."""
        if not self.enabled:
            return
            
        if metric_name not in self.response_times['metric_computations']:
            self.response_times['metric_computations'][metric_name] = []
            
        self.response_times['metric_computations'][metric_name].append({
            'computation_time': computation_time,
            'question_count': question_count,
            'per_question_time': computation_time / question_count if question_count > 0 else 0,
            'timestamp': time.time()
        })
    
    def track_stage_timing(self, stage_name: str, duration: float):
        """Track timing for pipeline stages."""
        if not self.enabled:
            return
            
        self.stage_timings[stage_name] = {
            'duration': duration,
            'timestamp': time.time()
        }
    
    def track_memory_usage(self, checkpoint: str):
        """Track memory usage at specific checkpoints."""
        if not self.enabled:
            return
            
        try:
            if PSUTIL_AVAILABLE:
                process = psutil.Process()
                memory_info = process.memory_info()
                
                self.memory_usage[checkpoint] = {
                    'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                    'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                    'percent': process.memory_percent(),
                    'timestamp': time.time()
                }
            else:
                # Fallback memory tracking without psutil
                import os
                self.memory_usage[checkpoint] = {
                    'rss_mb': 0,  # Not available
                    'vms_mb': 0,  # Not available
                    'percent': 0,  # Not available
                    'timestamp': time.time(),
                    'note': 'Limited tracking - psutil not available'
                }
        except Exception as e:
            logger.warning(f"Could not track memory usage: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.enabled:
            return {'performance_tracking': 'disabled'}
        
        def calculate_percentiles(values: List[float], percentiles: List[int] = [50, 95, 99]) -> Dict[str, float]:
            """Calculate percentiles for a list of values."""
            if not values:
                return {}
            
            sorted_values = sorted(values)
            n = len(sorted_values)
            result = {}
            
            for p in percentiles:
                if p == 50:  # Median
                    idx = n // 2
                    result[f'p{p}'] = sorted_values[idx] if n % 2 == 1 else (sorted_values[idx-1] + sorted_values[idx]) / 2
                else:
                    idx = int((p / 100.0) * (n - 1))
                    result[f'p{p}'] = sorted_values[min(idx, n-1)]
            
            return result
        
        summary = {
            'tracking_enabled': self.enabled,
            'collection_time': datetime.now().isoformat()
        }
        
        # Stage timings
        if self.stage_timings:
            summary['stage_timings'] = self.stage_timings
            summary['total_pipeline_time'] = sum(stage['duration'] for stage in self.stage_timings.values())
        
        # RAG response times
        if self.response_times['rag_responses']:
            rag_times = [r['response_time'] for r in self.response_times['rag_responses']]
            summary['rag_response_times'] = {
                'count': len(rag_times),
                'average': sum(rag_times) / len(rag_times),
                'min': min(rag_times),
                'max': max(rag_times),
                'percentiles': calculate_percentiles(rag_times, self.config.get('timing_percentiles', [50, 95, 99]))
            }
        
        # LLM response times
        if self.response_times['llm_responses']:
            llm_times = [r['response_time'] for r in self.response_times['llm_responses']]
            summary['llm_response_times'] = {
                'count': len(llm_times),
                'average': sum(llm_times) / len(llm_times),
                'min': min(llm_times),
                'max': max(llm_times),
                'percentiles': calculate_percentiles(llm_times, self.config.get('timing_percentiles', [50, 95, 99]))
            }
        
        # Metric computation times
        if self.response_times['metric_computations']:
            summary['metric_computation_times'] = {}
            for metric_name, computations in self.response_times['metric_computations'].items():
                comp_times = [c['computation_time'] for c in computations]
                per_q_times = [c['per_question_time'] for c in computations]
                
                summary['metric_computation_times'][metric_name] = {
                    'total_computations': len(computations),
                    'total_time': sum(comp_times),
                    'average_time': sum(comp_times) / len(comp_times),
                    'average_per_question': sum(per_q_times) / len(per_q_times),
                    'percentiles': calculate_percentiles(comp_times, self.config.get('timing_percentiles', [50, 95, 99]))
                }
        
        # Memory usage
        if self.memory_usage:
            summary['memory_usage'] = self.memory_usage
            peak_rss = max(usage['rss_mb'] for usage in self.memory_usage.values())
            summary['peak_memory_mb'] = peak_rss
        
        return summary


class CompositionTracker:
    """Tracker for RAGAS composition elements (scenarios, personas, nodes, relationships)."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('testset_generation', {}).get('composition_elements_tracking', {})
        self.enabled = self.config.get('enabled', True)
        
        # Initialize tracking data
        self.scenarios = []
        self.personas = []
        self.nodes_used = []
        self.relationships_used = []
        self.query_styles = []
        self.synthesizer_usage = {}
        
        # Generation timing and metadata
        self.generation_start_time = None
        self.generation_end_time = None
        self.kg_statistics = {}
        
        logger.info(f"🧩 Composition Tracker initialized (enabled: {self.enabled})")
    
    def start_generation(self, kg_statistics: Dict[str, Any] = None):
        """Mark the start of testset generation."""
        if not self.enabled:
            return
            
        self.generation_start_time = datetime.now()
        self.kg_statistics = kg_statistics or {}
        
        logger.info(f"🎬 Started tracking composition elements for testset generation")
        
    def end_generation(self):
        """Mark the end of testset generation."""
        if not self.enabled:
            return
            
        self.generation_end_time = datetime.now()
        duration = (self.generation_end_time - self.generation_start_time).total_seconds() if self.generation_start_time else 0
        logger.info(f"🏁 Completed tracking composition elements for testset generation (duration: {duration:.2f}s)")
    
    def track_scenario(self, scenario_data: Dict[str, Any]):
        """Track a scenario generated by RAGAS."""
        if not self.enabled:
            return
            
        scenario_info = {
            'scenario_id': scenario_data.get('id', len(self.scenarios)),
            'persona': scenario_data.get('persona', {}),
            'style': scenario_data.get('style', 'unknown'),
            'length': scenario_data.get('length', 'unknown'),
            'nodes': scenario_data.get('nodes', []),
            'relationships': scenario_data.get('relationships', []),
            'timestamp': time.time()
        }
        
        self.scenarios.append(scenario_info)
        logger.debug(f"📝 Tracked scenario: {scenario_info['scenario_id']}")
    
    def track_persona(self, persona_data: Dict[str, Any]):
        """Track personas used in generation."""
        if not self.enabled:
            return
            
        persona_info = {
            'persona_id': persona_data.get('id', len(self.personas)),
            'role': persona_data.get('role', 'unknown'),
            'background': persona_data.get('background', ''),
            'expertise_level': persona_data.get('expertise_level', 'intermediate'),
            'timestamp': time.time()
        }
        
        self.personas.append(persona_info)
        logger.debug(f"👤 Tracked persona: {persona_info['role']}")
    
    def track_node_usage(self, node_data: Union[Dict[str, Any], List, str], synthesizer_name: str):
        """Track knowledge graph nodes used in generation."""
        if not self.enabled:
            return
        
        # Handle different input types defensively
        if isinstance(node_data, dict):
            # Handle dict case (expected format)
            node_info = {
                'node_id': node_data.get('id', f'node_{len(self.nodes_used)}'),
                'node_type': node_data.get('type', 'unknown'),
                'properties': list(node_data.get('properties', {}).keys()) if isinstance(node_data.get('properties'), dict) else [],
                'synthesizer': synthesizer_name,
                'content_snippet': str(node_data.get('content', ''))[:100] + '...' if len(str(node_data.get('content', ''))) > 100 else str(node_data.get('content', '')),
                'timestamp': time.time()
            }
        elif isinstance(node_data, list):
            # Handle list case - track each item in the list
            for item in node_data:
                self.track_node_usage(item, synthesizer_name)
            return
        else:
            # Handle primitive types (string, etc.)
            node_info = {
                'node_id': str(node_data),
                'node_type': 'primitive',
                'properties': [],
                'synthesizer': synthesizer_name,
                'content_snippet': str(node_data)[:100] + '...' if len(str(node_data)) > 100 else str(node_data),
                'timestamp': time.time()
            }
        
        self.nodes_used.append(node_info)
        logger.debug(f"🔗 Tracked node usage: {node_info['node_id']} by {synthesizer_name}")
    
    def track_relationship_usage(self, relationship_data: Union[Dict[str, Any], List, str], synthesizer_name: str):
        """Track knowledge graph relationships used in multi-hop generation."""
        if not self.enabled:
            return
        
        # Handle different input types defensively
        if isinstance(relationship_data, dict):
            # Handle dict case (expected format)
            rel_info = {
                'relationship_id': relationship_data.get('id', f'rel_{len(self.relationships_used)}'),
                'source_node': relationship_data.get('source', 'unknown'),
                'target_node': relationship_data.get('target', 'unknown'),
                'relationship_type': relationship_data.get('type', 'unknown'),
                'synthesizer': synthesizer_name,
                'timestamp': time.time()
            }
        elif isinstance(relationship_data, list):
            # Handle list case - track each item in the list
            for item in relationship_data:
                self.track_relationship_usage(item, synthesizer_name)
            return
        else:
            # Handle primitive types (string, etc.)
            rel_info = {
                'relationship_id': f'rel_{len(self.relationships_used)}',
                'source_node': 'unknown',
                'target_node': 'unknown',
                'relationship_type': str(relationship_data),
                'synthesizer': synthesizer_name,
                'timestamp': time.time()
            }
        
        self.relationships_used.append(rel_info)
        logger.debug(f"🔗 Tracked relationship: {rel_info['source_node']} -> {rel_info['target_node']}")
    
    def track_query_style(self, style_data: Dict[str, Any]):
        """Track query styles applied during generation."""
        if not self.enabled:
            return
            
        style_info = {
            'style_name': style_data.get('name', 'unknown'),
            'description': style_data.get('description', ''),
            'complexity': style_data.get('complexity', 'medium'),
            'synthesizer': style_data.get('synthesizer', 'unknown'),
            'timestamp': time.time()
        }
        
        self.query_styles.append(style_info)
        logger.debug(f"🎨 Tracked query style: {style_info['style_name']}")
    
    def track_synthesizer_usage(self, synthesizer_name: str, success: bool, question_generated: str = None):
        """Track synthesizer usage and success rates."""
        if not self.enabled:
            return
            
        if synthesizer_name not in self.synthesizer_usage:
            self.synthesizer_usage[synthesizer_name] = {
                'total_attempts': 0,
                'successful_generations': 0,
                'failed_generations': 0,
                'success_rate': 0.0,
                'questions_generated': []
            }
        
        usage = self.synthesizer_usage[synthesizer_name]
        usage['total_attempts'] += 1
        
        if success:
            usage['successful_generations'] += 1
            if question_generated:
                usage['questions_generated'].append({
                    'question': question_generated[:100] + '...' if len(question_generated) > 100 else question_generated,
                    'timestamp': time.time()
                })
        else:
            usage['failed_generations'] += 1
        
        usage['success_rate'] = usage['successful_generations'] / usage['total_attempts']
        
        logger.debug(f"📊 Tracked synthesizer usage: {synthesizer_name} ({'✅' if success else '❌'})")
    
    def get_composition_summary(self) -> Dict[str, Any]:
        """Get comprehensive composition elements summary."""
        if not self.enabled:
            return {'composition_tracking': 'disabled'}
        
        summary = {
            'tracking_enabled': self.enabled,
            'collection_time': datetime.now().isoformat(),
            'scenarios_generated': len(self.scenarios),
            'personas_used': len(self.personas),
            'nodes_accessed': len(self.nodes_used),
            'relationships_traversed': len(self.relationships_used),
            'query_styles_applied': len(self.query_styles)
        }
        
        # Add generation timing information
        if self.generation_start_time:
            summary['generation_start_time'] = self.generation_start_time.isoformat()
            if self.generation_end_time:
                summary['generation_end_time'] = self.generation_end_time.isoformat()
                duration = (self.generation_end_time - self.generation_start_time).total_seconds()
                summary['generation_duration_seconds'] = duration
        
        # Add knowledge graph statistics
        if self.kg_statistics:
            summary['knowledge_graph_statistics'] = self.kg_statistics
        
        # Detailed breakdowns if requested
        if self.config.get('include_detailed_breakdown', True):
            summary.update({
                'scenarios': self.scenarios,
                'personas': self.personas,
                'nodes_used': self.nodes_used,
                'relationships_used': self.relationships_used,
                'query_styles': self.query_styles
            })
        
        # Synthesizer usage statistics
        if self.synthesizer_usage:
            summary['synthesizer_usage'] = self.synthesizer_usage
            
            # Overall statistics
            total_attempts = sum(usage['total_attempts'] for usage in self.synthesizer_usage.values())
            total_successes = sum(usage['successful_generations'] for usage in self.synthesizer_usage.values())
            
            summary['overall_synthesis_stats'] = {
                'total_synthesis_attempts': total_attempts,
                'total_successful_syntheses': total_successes,
                'overall_success_rate': total_successes / total_attempts if total_attempts > 0 else 0,
                'synthesizers_used': len(self.synthesizer_usage)
            }
        
        return summary


class ParametersTracker:
    """Tracker for final used parameters with fallback detection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config.get('reporting', {}).get('final_parameters_tracking', {})
        self.enabled = self.config.get('enabled', True)
        
        # Initialize tracking data
        self.initial_config = config.copy()
        self.final_config = {}
        self.fallback_usage = []
        self.config_modifications = []
        self.generation_metadata = {}
        self.evaluation_metadata = {}
        
        logger.info(f"⚙️ Parameters Tracker initialized (enabled: {self.enabled})")
    
    def track_config_modification(self, component: str, parameter: str, original_value: Any, 
                                new_value: Any, reason: str):
        """Track when configuration parameters are modified during execution."""
        if not self.enabled:
            return
            
        modification = {
            'component': component,
            'parameter': parameter,
            'original_value': original_value,
            'new_value': new_value,
            'reason': reason,
            'timestamp': time.time()
        }
        
        self.config_modifications.append(modification)
        logger.debug(f"📝 Config modified: {component}.{parameter} -> {new_value} ({reason})")
    
    def track_fallback_usage(self, component: str, parameter: str, intended_value: Any, 
                           fallback_value: Any, reason: str, success: bool):
        """Track when fallback values are used."""
        if not self.enabled:
            return
            
        fallback = {
            'component': component,
            'parameter': parameter,
            'intended_value': intended_value,
            'fallback_value': fallback_value,
            'reason': reason,
            'fallback_successful': success,
            'timestamp': time.time()
        }
        
        self.fallback_usage.append(fallback)
        logger.debug(f"🔄 Fallback used: {component}.{parameter} ({'✅' if success else '❌'})")
    
    def update_final_config(self, config_section: str, final_values: Dict[str, Any]):
        """Update the final configuration values for a specific section."""
        if not self.enabled:
            return
            
        self.final_config[config_section] = final_values
        logger.debug(f"✅ Final config updated: {config_section}")
    
    def track_generation_metadata(self, metadata: Dict[str, Any]):
        """Track metadata from testset generation."""
        if not self.enabled:
            return
            
        self.generation_metadata.update(metadata)
        logger.debug("📊 Generation metadata tracked")
    
    def track_evaluation_metadata(self, metadata: Dict[str, Any]):
        """Track metadata from evaluation process."""
        if not self.enabled:
            return
            
        self.evaluation_metadata.update(metadata)
        logger.debug("📊 Evaluation metadata tracked")
    
    def get_parameters_summary(self) -> Dict[str, Any]:
        """Get comprehensive parameters and fallback summary."""
        if not self.enabled:
            return {'parameters_tracking': 'disabled'}
        
        summary = {
            'tracking_enabled': self.enabled,
            'collection_time': datetime.now().isoformat(),
            'initial_config': self.initial_config,
            'final_config': self.final_config,
            'config_modifications': len(self.config_modifications),
            'fallbacks_used': len(self.fallback_usage)
        }
        
        # Detailed tracking if requested
        if self.config.get('include_detailed_tracking', True):
            summary.update({
                'detailed_modifications': self.config_modifications,
                'detailed_fallbacks': self.fallback_usage,
                'generation_metadata': self.generation_metadata,
                'evaluation_metadata': self.evaluation_metadata
            })
        
        return summary
        if self.config.get('capture_detailed_changes', True):
            summary.update({
                'detailed_modifications': self.config_modifications,
                'detailed_fallbacks': self.fallback_usage
            })
        
        # Generation metadata
        if self.config.get('include_generation_metadata', True) and self.generation_metadata:
            summary['generation_metadata'] = self.generation_metadata
        
        # Evaluation metadata
        if self.config.get('include_evaluation_metadata', True) and self.evaluation_metadata:
            summary['evaluation_metadata'] = self.evaluation_metadata
        
        # Fallback analysis
        if self.fallback_usage:
            successful_fallbacks = [f for f in self.fallback_usage if f['fallback_successful']]
            failed_fallbacks = [f for f in self.fallback_usage if not f['fallback_successful']]
            
            summary['fallback_analysis'] = {
                'total_fallbacks': len(self.fallback_usage),
                'successful_fallbacks': len(successful_fallbacks),
                'failed_fallbacks': len(failed_fallbacks),
                'fallback_success_rate': len(successful_fallbacks) / len(self.fallback_usage) if self.fallback_usage else 0,
                'components_with_fallbacks': list(set(f['component'] for f in self.fallback_usage))
            }
        
        return summary
    
    def save_final_parameters(self, output_file: Path):
        """Save final parameters to a JSON file."""
        if not self.enabled:
            return
            
        try:
            summary = self.get_parameters_summary()
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"💾 Final parameters saved to: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ Failed to save final parameters: {e}")
