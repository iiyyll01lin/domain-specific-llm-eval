#!/usr/bin/env python3
"""
Fixes for Pure RAGAS Testset Generator Secondary Issues

This module provides fixes for:
1. Persona matching errors (KeyError issues)
2. Query distribution float iteration errors  
3. Configuration validation errors
"""
import logging
import json
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path

logger = logging.getLogger(__name__)

class PureRAGASTestsetGeneratorFixes:
    """
    Fixes for the secondary issues in PureRAGASTestsetGenerator
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logger
    
    def create_safe_personas(self) -> List[Any]:
        """Create safe personas that don't cause KeyError"""
        try:
            # Try to import from RAGAS
            from ragas.testset.persona import Persona
            
            safe_personas = [
                Persona(
                    name="Technical Specialist", 
                    role_description="A technical specialist who asks detailed questions about industrial processes and specifications."
                ),
                Persona(
                    name="Quality Inspector", 
                    role_description="A quality inspector who focuses on measurement procedures and quality control standards."
                ),
                Persona(
                    name="System Administrator",
                    role_description="A system administrator who needs to understand error codes and troubleshooting procedures."
                ),
                Persona(
                    name="Manufacturing Engineer",
                    role_description="An engineer who works with manufacturing equipment and needs to understand technical specifications."
                ),
                Persona(
                    name="Support Technician",
                    role_description="A support technician who helps resolve technical issues and provides assistance to users."
                )
            ]
            
            self.logger.info(f"Created {len(safe_personas)} safe personas")
            return safe_personas
            
        except Exception as e:
            self.logger.warning(f"Failed to create personas: {e}")
            # Return minimal fallback
            class FallbackPersona:
                def __init__(self, name: str, role_description: str):
                    self.name = name
                    self.role_description = role_description
            
            return [FallbackPersona("User", "A general user")]
    
    def create_safe_query_distribution(self, llm_wrapper, knowledge_graph=None) -> List[Tuple[Any, float]]:
        """Create query distribution that handles float iteration errors"""
        try:
            # Try to import from RAGAS
            from ragas.testset.synthesizers import default_query_distribution
            from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
            
            # Try default first
            if knowledge_graph:
                query_dist = default_query_distribution(llm_wrapper, knowledge_graph)
            else:
                query_dist = default_query_distribution(llm_wrapper)
            
            # Validate the distribution is iterable and has proper structure
            if not isinstance(query_dist, (list, tuple)):
                raise ValueError("Query distribution is not iterable")
            
            # Ensure weights are floats and sum to 1.0
            total_weight = sum(weight for _, weight in query_dist)
            if total_weight == 0:
                raise ValueError("Total weight is zero")
            
            # Normalize weights
            normalized_dist = [(synth, float(weight)/total_weight) for synth, weight in query_dist]
            
            self.logger.info(f"Created query distribution with {len(normalized_dist)} synthesizers")
            return normalized_dist
            
        except Exception as e:
            self.logger.warning(f"Failed to create default query distribution: {e}")
            
            # Create manual fallback distribution
            try:
                from ragas.testset.synthesizers.single_hop.specific import SingleHopSpecificQuerySynthesizer
                single_hop_synth = SingleHopSpecificQuerySynthesizer(llm=llm_wrapper)
                fallback_dist = [(single_hop_synth, 1.0)]
                self.logger.info("Created fallback query distribution")
                return fallback_dist
            except Exception as e2:
                self.logger.error(f"Failed to create fallback distribution: {e2}")
                return []

    def validate_and_fix_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fix configuration issues"""
        fixed_config = config.copy()
        
        # Ensure query_distribution values are numbers, not tuples
        if 'testset_generation' in fixed_config:
            tg_config = fixed_config['testset_generation']
            
            if 'query_distribution' in tg_config:
                qd = tg_config['query_distribution']
                for key, value in qd.items():
                    if isinstance(value, (tuple, list)):
                        # Take first element if it's a tuple/list
                        tg_config['query_distribution'][key] = float(value[0]) if value else 0.6
                        self.logger.info(f"Fixed query_distribution.{key}: {value} -> {tg_config['query_distribution'][key]}")
                    elif not isinstance(value, (int, float)):
                        # Set default value
                        old_value = value
                        tg_config['query_distribution'][key] = 0.6
                        self.logger.info(f"Fixed query_distribution.{key}: {old_value} -> 0.6")
                        
                # Ensure distribution sums to 1.0
                total = sum(tg_config['query_distribution'].values())
                if total != 1.0:
                    for key in tg_config['query_distribution']:
                        tg_config['query_distribution'][key] /= total
                    self.logger.info(f"Normalized query distribution to sum to 1.0")
        
        # Fix other common configuration issues
        if 'ragas_config' in fixed_config.get('testset_generation', {}):
            ragas_config = fixed_config['testset_generation']['ragas_config']
            
            # Ensure custom_llm configuration is properly structured
            if 'custom_llm' in ragas_config:
                llm_config = ragas_config['custom_llm']
                
                # Fix endpoint URL format
                if 'endpoint' in llm_config:
                    endpoint = llm_config['endpoint']
                    if not endpoint.startswith('http'):
                        llm_config['endpoint'] = f"http://{endpoint}"
                        self.logger.info(f"Fixed endpoint URL format")
                
                # Ensure required fields exist
                required_fields = ['api_key', 'model', 'endpoint']
                for field in required_fields:
                    if field not in llm_config:
                        if field == 'model':
                            llm_config[field] = 'gpt-4o'
                        elif field == 'api_key':
                            self.logger.warning(f"Missing required field: {field}")
                        self.logger.info(f"Added missing field: {field}")
        
        return fixed_config
    
    def fix_persona_matching_errors(self, themes: List[str], personas: List[Any]) -> Dict[str, List[str]]:
        """Fix persona matching errors by creating safe mappings"""
        try:
            # Create safe mapping based on persona names and themes
            mapping = {}
            
            for persona in personas:
                persona_name = getattr(persona, 'name', 'Unknown')
                persona_role = getattr(persona, 'role_description', '')
                
                # Map personas to relevant themes based on keywords
                relevant_themes = []
                role_lower = persona_role.lower()
                
                for theme in themes:
                    theme_lower = theme.lower()
                    # Simple keyword matching
                    if any(keyword in role_lower for keyword in theme_lower.split()):
                        relevant_themes.append(theme)
                
                # If no themes matched, assign first theme or default
                if not relevant_themes and themes:
                    relevant_themes = [themes[0]]
                elif not relevant_themes:
                    relevant_themes = ["general"]
                
                mapping[persona_name] = relevant_themes
                
            self.logger.info(f"Created safe persona-theme mapping for {len(personas)} personas")
            return mapping
            
        except Exception as e:
            self.logger.error(f"Failed to create persona-theme mapping: {e}")
            # Return fallback mapping
            fallback_mapping = {}
            for i, persona in enumerate(personas):
                persona_name = getattr(persona, 'name', f'Persona_{i}')
                fallback_mapping[persona_name] = themes[:1] if themes else ["general"]
            return fallback_mapping
    
    def handle_float_iteration_error(self, value: Any) -> Any:
        """Handle cases where float values are being iterated"""
        if isinstance(value, float):
            # Convert single float to list
            return [value]
        elif isinstance(value, (int, str)):
            try:
                return [float(value)]
            except (ValueError, TypeError):
                return [0.0]
        elif hasattr(value, '__iter__') and not isinstance(value, str):
            return list(value)
        else:
            return [value]
    
    def safe_cluster_selection(self, knowledge_graph, relation_type: str = 'entities_overlap') -> List[Any]:
        """Safely select clusters from knowledge graph"""
        try:
            if not knowledge_graph or not hasattr(knowledge_graph, 'nodes'):
                self.logger.warning("Invalid knowledge graph provided")
                return []
            
            # Get clusters using the specified relation type
            clusters = knowledge_graph.get_clusters(relation_type)
            
            if not clusters:
                self.logger.warning(f"No clusters found with relation type '{relation_type}', trying fallbacks...")
                
                # Try fallback relation types
                fallback_types = ['similarity', 'overlap', 'default']
                for fallback_type in fallback_types:
                    try:
                        clusters = knowledge_graph.get_clusters(fallback_type)
                        if clusters:
                            self.logger.info(f"Using fallback relation type: {fallback_type}")
                            break
                    except:
                        continue
                
                # If still no clusters, create basic clusters from nodes
                if not clusters:
                    nodes = list(knowledge_graph.nodes)
                    if nodes:
                        # Group nodes into small clusters
                        cluster_size = 3
                        clusters = [nodes[i:i+cluster_size] for i in range(0, len(nodes), cluster_size)]
                        self.logger.info(f"Created {len(clusters)} basic clusters from nodes")
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Failed to select clusters: {e}")
            return []


def apply_fixes_to_generator(generator, config: Dict[str, Any]):
    """Apply all fixes to a PureRAGASTestsetGenerator instance"""
    fixes = PureRAGASTestsetGeneratorFixes(config)
    
    # Validate and fix configuration
    fixed_config = fixes.validate_and_fix_config(config)
    
    # Apply fixes to generator
    if hasattr(generator, 'config'):
        generator.config = fixed_config
    
    # Add safe methods to generator
    generator.create_safe_personas = fixes.create_safe_personas
    generator.create_safe_query_distribution = fixes.create_safe_query_distribution
    generator.fix_persona_matching_errors = fixes.fix_persona_matching_errors
    generator.handle_float_iteration_error = fixes.handle_float_iteration_error
    generator.safe_cluster_selection = fixes.safe_cluster_selection
    
    logger.info("Applied all fixes to PureRAGASTestsetGenerator")
    return generator, fixed_config