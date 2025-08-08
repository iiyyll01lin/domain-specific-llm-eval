"""
Composition Elements Tracker for RAGAS Testset Generation

This module tracks and captures the composition elements used in RAGAS testset generation:
- Scenarios (persona, style, length)
- Knowledge Graph nodes
- Query styles applied
- Personas matched to queries
- Relationships used in multi-hop queries

This information is valuable for understanding and debugging testset generation.
"""

import logging
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class ScenarioElement:
    """Represents a scenario element used in testset generation"""
    scenario_id: str
    scenario_type: str  # 'single_hop' or 'multi_hop'
    persona_name: str
    persona_description: str
    query_style: str
    query_length: str
    nodes_involved: List[str]
    relationships_used: List[str] = field(default_factory=list)  # For multi-hop
    keyphrases: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompositionStatistics:
    """Statistics about composition elements usage"""
    total_scenarios: int
    persona_distribution: Dict[str, int]
    style_distribution: Dict[str, int]
    length_distribution: Dict[str, int]
    node_usage_count: Dict[str, int]
    relationship_usage_count: Dict[str, int]
    scenario_type_distribution: Dict[str, int]

class CompositionElementsTracker:
    """
    Tracks composition elements used in RAGAS testset generation
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize composition elements tracker"""
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        if not self.enabled:
            return
            
        # Configuration
        self.capture_scenarios = self.config.get('capture_scenarios', True)
        self.capture_nodes = self.config.get('capture_nodes', True)
        self.capture_query_styles = self.config.get('capture_query_styles', True)
        self.capture_personas = self.config.get('capture_personas', True)
        self.capture_relationships = self.config.get('capture_relationships', True)
        self.include_in_testset_metadata = self.config.get('include_in_testset_metadata', True)
        self.include_in_reports = self.config.get('include_in_reports', True)
        
        # Storage
        self.scenarios: List[ScenarioElement] = []
        self.personas_used: Set[str] = set()
        self.query_styles_used: Set[str] = set()
        self.query_lengths_used: Set[str] = set()
        self.nodes_used: Set[str] = set()
        self.relationships_used: Set[str] = set()
        
        # Generation metadata
        self.generation_start_time: Optional[datetime] = None
        self.generation_end_time: Optional[datetime] = None
        self.kg_statistics: Dict[str, Any] = {}
        
        logger.info("✅ Composition elements tracker initialized")
        
    def start_generation(self, kg_statistics: Dict[str, Any] = None):
        """Mark the start of testset generation"""
        if not self.enabled:
            return
            
        self.generation_start_time = datetime.now()
        self.kg_statistics = kg_statistics or {}
        
        logger.info(f"🎬 Started tracking composition elements for testset generation")
        
    def end_generation(self):
        """Mark the end of testset generation"""
        if not self.enabled:
            return
            
        self.generation_end_time = datetime.now()
        logger.info(f"🏁 Completed tracking composition elements for testset generation")
        
    def track_scenario(self, scenario_data: Dict[str, Any], question_id: str = None):
        """
        Track a scenario used in testset generation
        
        Args:
            scenario_data: Dictionary containing scenario information
            question_id: Optional question ID for linking
        """
        if not self.enabled or not self.capture_scenarios:
            return
            
        try:
            # Extract scenario information
            scenario_type = scenario_data.get('type', 'unknown')
            persona = scenario_data.get('persona', {})
            
            # Handle both Persona objects and dictionaries
            if hasattr(persona, 'name'):
                persona_name = persona.name
                persona_description = persona.role_description
            elif isinstance(persona, dict):
                persona_name = persona.get('name', 'Unknown')
                persona_description = persona.get('role_description', '')
            else:
                persona_name = str(persona)
                persona_description = ''
                
            query_style = str(scenario_data.get('style', 'unknown'))
            query_length = str(scenario_data.get('length', 'unknown'))
            
            # Extract nodes
            nodes = scenario_data.get('nodes', [])
            node_ids = []
            for node in nodes:
                if hasattr(node, 'id'):
                    node_ids.append(node.id)
                elif isinstance(node, dict):
                    node_ids.append(node.get('id', str(node)))
                else:
                    node_ids.append(str(node))
                    
            # Extract relationships (for multi-hop)
            relationships = []
            if 'relationships' in scenario_data:
                for rel in scenario_data['relationships']:
                    if hasattr(rel, 'type'):
                        relationships.append(rel.type)
                    elif isinstance(rel, dict):
                        relationships.append(rel.get('type', str(rel)))
                    else:
                        relationships.append(str(rel))
                        
            # Extract keyphrases and entities
            keyphrases = scenario_data.get('keyphrases', [])
            entities = scenario_data.get('entities', [])
            
            # Create scenario element
            scenario_element = ScenarioElement(
                scenario_id=f"scenario_{len(self.scenarios)}_{datetime.now().timestamp()}",
                scenario_type=scenario_type,
                persona_name=persona_name,
                persona_description=persona_description,
                query_style=query_style,
                query_length=query_length,
                nodes_involved=node_ids,
                relationships_used=relationships,
                keyphrases=keyphrases,
                entities=entities,
                metadata={
                    'question_id': question_id,
                    'timestamp': datetime.now().isoformat(),
                    'raw_scenario_data': scenario_data
                }
            )
            
            self.scenarios.append(scenario_element)
            
            # Update tracking sets
            if self.capture_personas:
                self.personas_used.add(persona_name)
                
            if self.capture_query_styles:
                self.query_styles_used.add(query_style)
                self.query_lengths_used.add(query_length)
                
            if self.capture_nodes:
                self.nodes_used.update(node_ids)
                
            if self.capture_relationships:
                self.relationships_used.update(relationships)
                
            logger.debug(f"📊 Tracked scenario: {scenario_type} with persona '{persona_name}'")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to track scenario: {e}")
            
    def track_persona_usage(self, persona_name: str, question_id: str = None):
        """Track persona usage"""
        if not self.enabled or not self.capture_personas:
            return
            
        self.personas_used.add(persona_name)
        logger.debug(f"👤 Tracked persona usage: {persona_name}")
        
    def track_node_usage(self, node_id: str, node_type: str = None):
        """Track knowledge graph node usage"""
        if not self.enabled or not self.capture_nodes:
            return
            
        self.nodes_used.add(node_id)
        logger.debug(f"🔗 Tracked node usage: {node_id}")
        
    def track_relationship_usage(self, relationship_type: str, source_node: str = None, target_node: str = None):
        """Track knowledge graph relationship usage"""
        if not self.enabled or not self.capture_relationships:
            return
            
        self.relationships_used.add(relationship_type)
        logger.debug(f"↔️ Tracked relationship usage: {relationship_type}")
        
    def get_composition_statistics(self) -> CompositionStatistics:
        """Calculate composition statistics"""
        if not self.enabled:
            return CompositionStatistics(0, {}, {}, {}, {}, {}, {})
            
        # Count distributions
        persona_distribution = defaultdict(int)
        style_distribution = defaultdict(int)
        length_distribution = defaultdict(int)
        node_usage_count = defaultdict(int)
        relationship_usage_count = defaultdict(int)
        scenario_type_distribution = defaultdict(int)
        
        for scenario in self.scenarios:
            persona_distribution[scenario.persona_name] += 1
            style_distribution[scenario.query_style] += 1
            length_distribution[scenario.query_length] += 1
            scenario_type_distribution[scenario.scenario_type] += 1
            
            for node_id in scenario.nodes_involved:
                node_usage_count[node_id] += 1
                
            for rel_type in scenario.relationships_used:
                relationship_usage_count[rel_type] += 1
                
        return CompositionStatistics(
            total_scenarios=len(self.scenarios),
            persona_distribution=dict(persona_distribution),
            style_distribution=dict(style_distribution),
            length_distribution=dict(length_distribution),
            node_usage_count=dict(node_usage_count),
            relationship_usage_count=dict(relationship_usage_count),
            scenario_type_distribution=dict(scenario_type_distribution)
        )
        
    def get_composition_summary(self) -> Dict[str, Any]:
        """Get comprehensive composition summary for reports"""
        if not self.enabled:
            return {'composition_tracking': 'disabled'}
            
        statistics = self.get_composition_statistics()
        
        summary = {
            'composition_tracking': {
                'enabled': True,
                'generation_period': {
                    'start_time': self.generation_start_time.isoformat() if self.generation_start_time else None,
                    'end_time': self.generation_end_time.isoformat() if self.generation_end_time else None,
                    'duration_seconds': (
                        (self.generation_end_time - self.generation_start_time).total_seconds()
                        if self.generation_start_time and self.generation_end_time else None
                    )
                },
                'knowledge_graph_statistics': self.kg_statistics,
                'configuration': {
                    'capture_scenarios': self.capture_scenarios,
                    'capture_nodes': self.capture_nodes,
                    'capture_query_styles': self.capture_query_styles,
                    'capture_personas': self.capture_personas,
                    'capture_relationships': self.capture_relationships
                }
            },
            'composition_statistics': {
                'overview': {
                    'total_scenarios_generated': statistics.total_scenarios,
                    'unique_personas_used': len(statistics.persona_distribution),
                    'unique_query_styles_used': len(statistics.style_distribution),
                    'unique_query_lengths_used': len(statistics.length_distribution),
                    'unique_nodes_involved': len(statistics.node_usage_count),
                    'unique_relationships_used': len(statistics.relationship_usage_count)
                },
                'distributions': {
                    'persona_distribution': statistics.persona_distribution,
                    'query_style_distribution': statistics.style_distribution,
                    'query_length_distribution': statistics.length_distribution,
                    'scenario_type_distribution': statistics.scenario_type_distribution
                },
                'usage_patterns': {
                    'most_used_personas': sorted(
                        statistics.persona_distribution.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5],
                    'most_used_nodes': sorted(
                        statistics.node_usage_count.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:10],
                    'most_used_relationships': sorted(
                        statistics.relationship_usage_count.items(), 
                        key=lambda x: x[1], 
                        reverse=True
                    )[:5]
                }
            }
        }
        
        return summary
        
    def get_testset_metadata(self) -> Dict[str, Any]:
        """Get metadata to include in testset files"""
        if not self.enabled or not self.include_in_testset_metadata:
            return {}
            
        statistics = self.get_composition_statistics()
        
        return {
            'composition_elements': {
                'generation_timestamp': datetime.now().isoformat(),
                'total_scenarios': statistics.total_scenarios,
                'personas_used': list(self.personas_used),
                'query_styles_used': list(self.query_styles_used),
                'query_lengths_used': list(self.query_lengths_used),
                'scenario_type_distribution': statistics.scenario_type_distribution,
                'persona_distribution': statistics.persona_distribution
            }
        }
        
    def get_detailed_scenarios(self) -> List[Dict[str, Any]]:
        """Get detailed scenario information for debugging"""
        if not self.enabled:
            return []
            
        return [
            {
                'scenario_id': scenario.scenario_id,
                'scenario_type': scenario.scenario_type,
                'persona': {
                    'name': scenario.persona_name,
                    'description': scenario.persona_description
                },
                'query_properties': {
                    'style': scenario.query_style,
                    'length': scenario.query_length
                },
                'knowledge_elements': {
                    'nodes_involved': scenario.nodes_involved,
                    'relationships_used': scenario.relationships_used,
                    'keyphrases': scenario.keyphrases,
                    'entities': scenario.entities
                },
                'metadata': scenario.metadata
            }
            for scenario in self.scenarios
        ]
        
    def reset(self):
        """Reset all tracking data"""
        if not self.enabled:
            return
            
        self.scenarios.clear()
        self.personas_used.clear()
        self.query_styles_used.clear()
        self.query_lengths_used.clear()
        self.nodes_used.clear()
        self.relationships_used.clear()
        
        self.generation_start_time = None
        self.generation_end_time = None
        self.kg_statistics.clear()
        
        logger.info("🔄 Composition elements tracker reset")
