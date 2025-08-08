"""
Knowledge Graph Validator for RAG Evaluation Pipeline

This module provides validation and cleaning of knowledge graph data
to ensure robust pipeline execution.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeGraphValidator:
    """Validates knowledge graph data."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.kg_config = self.config.get('knowledge_graph', {})
        self.validation_stats = {
            'files_processed': 0,
            'nodes_processed': 0,
            'edges_processed': 0,
            'nodes_preserved': 0,
            'edges_preserved': 0,
            'issues_found': 0,
            'fixes_applied': 0
        }
    
    def validate_kg_file(self, kg_file: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Validate knowledge graph file.
        
        Args:
            kg_file: Path to the knowledge graph JSON file
            
        Returns:
            Tuple of (cleaned_kg_data, validation_report)
        """
        logger.info(f"🔍 Validating Knowledge Graph: {kg_file}")
        
        try:
            with open(kg_file, 'r', encoding='utf-8') as f:
                kg_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"❌ Invalid JSON in KG file {kg_file}: {e}")
            raise ValueError(f"Invalid JSON in knowledge graph file: {e}")
        except Exception as e:
            logger.error(f"❌ Failed to load KG file {kg_file}: {e}")
            raise
        
        validation_report = {
            'file_name': kg_file.name,
            'nodes_original': 0,
            'edges_original': 0,
            'nodes_final': 0,
            'edges_final': 0,
            'issues_found': [],
            'fixes_applied': [],
            'critical_issues': [],
            'warnings': [],
            'validation_timestamp': datetime.now().isoformat()
        }
        
        # Validate basic structure
        kg_data = self._validate_basic_structure(kg_data, validation_report)
        
        # Validate nodes
        if 'nodes' in kg_data:
            kg_data['nodes'], node_report = self._validate_nodes(kg_data['nodes'])
            validation_report.update(node_report)
        
        # Validate edges
        if 'edges' in kg_data:
            kg_data['edges'], edge_report = self._validate_edges(
                kg_data['edges'], 
                kg_data.get('nodes', [])
            )
            validation_report.update(edge_report)
        
        # Validate relationships and consistency
        kg_data = self._validate_relationships(kg_data, validation_report)
        
        # Final cleanup
        kg_data = self._final_cleanup(kg_data, validation_report)
        
        # Update statistics
        self._update_stats(validation_report)
        
        logger.info(f"✅ KG Validation complete: {validation_report['nodes_final']} nodes, {validation_report['edges_final']} edges")
        
        return kg_data, validation_report
    
    def _validate_basic_structure(self, kg_data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        """Validate basic knowledge graph structure."""
        
        if not isinstance(kg_data, dict):
            report['critical_issues'].append("Knowledge graph data is not a dictionary")
            return {'nodes': [], 'edges': []}
        
        # Ensure basic structure exists
        if 'nodes' not in kg_data:
            kg_data['nodes'] = []
            report['fixes_applied'].append("Added missing 'nodes' array")
        
        if 'edges' not in kg_data:
            kg_data['edges'] = []
            report['fixes_applied'].append("Added missing 'edges' array")
        
        # Validate that nodes and edges are lists
        if not isinstance(kg_data['nodes'], list):
            report['critical_issues'].append("'nodes' is not a list")
            kg_data['nodes'] = []
            report['fixes_applied'].append("Converted 'nodes' to empty list")
        
        if not isinstance(kg_data['edges'], list):
            report['critical_issues'].append("'edges' is not a list")
            kg_data['edges'] = []
            report['fixes_applied'].append("Converted 'edges' to empty list")
        
        # Record original counts
        report['nodes_original'] = len(kg_data['nodes'])
        report['edges_original'] = len(kg_data['edges'])
        
        return kg_data
    
    def _validate_nodes(self, nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate and clean node data."""
        
        report = {
            'node_issues': [],
            'node_fixes': []
        }
        
        valid_nodes = []
        node_ids_seen = set()
        
        for i, node in enumerate(nodes):
            try:
                cleaned_node = self._validate_single_node(node, i)
                if cleaned_node:
                    # Check for duplicate IDs
                    node_id = cleaned_node.get('id')
                    if node_id in node_ids_seen:
                        # Make ID unique
                        original_id = node_id
                        counter = 1
                        while f"{original_id}_{counter}" in node_ids_seen:
                            counter += 1
                        cleaned_node['id'] = f"{original_id}_{counter}"
                        report['node_fixes'].append(f"Made duplicate node ID unique: {original_id} -> {cleaned_node['id']}")
                    
                    node_ids_seen.add(cleaned_node['id'])
                    valid_nodes.append(cleaned_node)
                else:
                    report['node_issues'].append(f"Node {i} is invalid and was removed")
                    
            except Exception as e:
                report['node_issues'].append(f"Error validating node {i}: {e}")
                logger.warning(f"Error validating node {i}: {e}")
        
        report['nodes_final'] = len(valid_nodes)
        report['nodes_removed'] = len(nodes) - len(valid_nodes)
        
        if report['nodes_removed'] > 0:
            report['fixes_applied'] = report.get('fixes_applied', [])
            report['fixes_applied'].append(f"Removed {report['nodes_removed']} invalid nodes")
        
        return valid_nodes, report
    
    def _validate_single_node(self, node: Dict[str, Any], index: int) -> Optional[Dict[str, Any]]:
        """Validate individual node."""
        
        if not isinstance(node, dict):
            logger.debug(f"Node {index} is not a dictionary")
            return None
        
        # Ensure required fields
        if 'id' not in node or not node['id']:
            # Generate ID if missing
            node['id'] = f"node_{index}_{hash(str(node)) % 10000}"
            logger.debug(f"Generated ID for node {index}: {node['id']}")
        
        # Clean and validate ID
        node_id = str(node['id']).strip()
        if not node_id:
            node['id'] = f"node_{index}_empty"
        else:
            # Sanitize ID (remove problematic characters)
            node['id'] = self._sanitize_id(node_id)
        
        # Ensure type field
        if 'type' not in node:
            node['type'] = 'entity'
        
        # Ensure label field
        if 'label' not in node:
            node['label'] = node.get('name', node.get('title', f"Node {node['id']}"))
        
        # Clean text fields
        text_fields = ['label', 'name', 'title', 'description']
        for field in text_fields:
            if field in node and node[field]:
                node[field] = self._clean_text_field(str(node[field]))
        
        # Validate properties
        if 'properties' in node and not isinstance(node['properties'], dict):
            node['properties'] = {}
        
        return node
    
    def _validate_edges(self, edges: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate and clean edge data."""
        
        report = {
            'edge_issues': [],
            'edge_fixes': []
        }
        
        # Create set of valid node IDs for reference
        valid_node_ids = {node.get('id') for node in nodes if node.get('id')}
        
        valid_edges = []
        
        for i, edge in enumerate(edges):
            try:
                cleaned_edge = self._validate_single_edge(edge, i, valid_node_ids)
                if cleaned_edge:
                    valid_edges.append(cleaned_edge)
                else:
                    report['edge_issues'].append(f"Edge {i} is invalid and was removed")
                    
            except Exception as e:
                report['edge_issues'].append(f"Error validating edge {i}: {e}")
                logger.warning(f"Error validating edge {i}: {e}")
        
        report['edges_final'] = len(valid_edges)
        report['edges_removed'] = len(edges) - len(valid_edges)
        
        if report['edges_removed'] > 0:
            report['fixes_applied'] = report.get('fixes_applied', [])
            report['fixes_applied'].append(f"Removed {report['edges_removed']} invalid edges")
        
        return valid_edges, report
    
    def _validate_single_edge(self, edge: Dict[str, Any], index: int, valid_node_ids: set) -> Optional[Dict[str, Any]]:
        """Validate individual edge."""
        
        if not isinstance(edge, dict):
            logger.debug(f"Edge {index} is not a dictionary")
            return None
        
        # Check required fields
        required_fields = ['source', 'target']
        for field in required_fields:
            if field not in edge or not edge[field]:
                logger.debug(f"Edge {index} missing required field: {field}")
                return None
        
        # Validate source and target exist in nodes
        source_id = str(edge['source']).strip()
        target_id = str(edge['target']).strip()
        
        if source_id not in valid_node_ids:
            logger.debug(f"Edge {index} references invalid source node: {source_id}")
            return None
        
        if target_id not in valid_node_ids:
            logger.debug(f"Edge {index} references invalid target node: {target_id}")
            return None
        
        # Clean IDs
        edge['source'] = self._sanitize_id(source_id)
        edge['target'] = self._sanitize_id(target_id)
        
        # Ensure relation type
        if 'type' not in edge:
            edge['type'] = 'related_to'
        elif not edge['type']:
            edge['type'] = 'related_to'
        
        # Clean relation type
        edge['type'] = self._clean_text_field(str(edge['type']))
        
        # Add ID if missing
        if 'id' not in edge:
            edge['id'] = f"edge_{source_id}_{target_id}_{edge['type']}"
        
        # Clean properties
        if 'properties' in edge and not isinstance(edge['properties'], dict):
            edge['properties'] = {}
        
        return edge
    
    def _validate_relationships(self, kg_data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        """Validate relationship consistency and add metadata."""
        
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        
        # Create node lookup
        node_lookup = {node.get('id'): node for node in nodes if node.get('id')}
        
        # Count relationship types
        relationship_types = {}
        for edge in edges:
            rel_type = edge.get('type', 'unknown')
            relationship_types[rel_type] = relationship_types.get(rel_type, 0) + 1
        
        # Add metadata
        kg_data['metadata'] = kg_data.get('metadata', {})
        kg_data['metadata'].update({
            'node_count': len(nodes),
            'edge_count': len(edges),
            'relationship_types': relationship_types,
            'validation_timestamp': datetime.now().isoformat()
        })
        
        # Check for isolated nodes
        connected_nodes = set()
        for edge in edges:
            connected_nodes.add(edge.get('source'))
            connected_nodes.add(edge.get('target'))
        
        isolated_nodes = [node for node in nodes if node.get('id') not in connected_nodes]
        if isolated_nodes:
            report['warnings'].append(f"Found {len(isolated_nodes)} isolated nodes (no edges)")
        
        # Check for self-loops
        self_loops = [edge for edge in edges if edge.get('source') == edge.get('target')]
        if self_loops:
            report['warnings'].append(f"Found {len(self_loops)} self-loop edges")
        
        return kg_data
    
    def _final_cleanup(self, kg_data: Dict[str, Any], report: Dict[str, Any]) -> Dict[str, Any]:
        """Final cleanup and optimization."""
        
        # Remove empty or None values from nodes
        for node in kg_data.get('nodes', []):
            keys_to_remove = [k for k, v in node.items() if v is None or (isinstance(v, str) and not v.strip())]
            for key in keys_to_remove:
                if key not in ['id', 'type', 'label']:  # Keep essential fields even if empty
                    del node[key]
        
        # Remove empty or None values from edges
        for edge in kg_data.get('edges', []):
            keys_to_remove = [k for k, v in edge.items() if v is None or (isinstance(v, str) and not v.strip())]
            for key in keys_to_remove:
                if key not in ['source', 'target', 'type']:  # Keep essential fields even if empty
                    del edge[key]
        
        # Sort nodes and edges for consistency
        if 'nodes' in kg_data:
            kg_data['nodes'] = sorted(kg_data['nodes'], key=lambda x: x.get('id', ''))
        
        if 'edges' in kg_data:
            kg_data['edges'] = sorted(kg_data['edges'], key=lambda x: (x.get('source', ''), x.get('target', '')))
        
        return kg_data
    
    def _sanitize_id(self, id_value: str) -> str:
        """Sanitize ID values to ensure they're valid."""
        # Remove or replace problematic characters
        sanitized = str(id_value).strip()
        sanitized = sanitized.replace(' ', '_')
        sanitized = sanitized.replace('\n', '_')
        sanitized = sanitized.replace('\t', '_')
        
        # Ensure it's not empty
        if not sanitized:
            sanitized = f"id_{hash(id_value) % 10000}"
        
        return sanitized
    
    def _clean_text_field(self, text: str) -> str:
        """Clean text fields."""
        cleaned = str(text).strip()
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        # Remove null bytes
        cleaned = cleaned.replace('\x00', '')
        return cleaned
    
    def _update_stats(self, report: Dict[str, Any]) -> None:
        """Update global validation statistics."""
        self.validation_stats['files_processed'] += 1
        self.validation_stats['nodes_processed'] += report.get('nodes_original', 0)
        self.validation_stats['edges_processed'] += report.get('edges_original', 0)
        self.validation_stats['nodes_preserved'] += report.get('nodes_final', 0)
        self.validation_stats['edges_preserved'] += report.get('edges_final', 0)
        self.validation_stats['issues_found'] += len(report.get('issues_found', []))
        self.validation_stats['fixes_applied'] += len(report.get('fixes_applied', []))
    
    def save_cleaned_kg(self, kg_data: Dict[str, Any], original_path: Path, output_dir: Path) -> Path:
        """Save cleaned knowledge graph with validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_{original_path.stem}_{timestamp}.json"
        output_path = output_dir / clean_filename
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned KG
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(kg_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved cleaned knowledge graph: {output_path}")
        return output_path
    
    def save_validation_report(self, reports: List[Dict[str, Any]], output_dir: Path) -> Path:
        """Save comprehensive validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"kg_validation_report_{timestamp}.json"
        
        comprehensive_report = {
            'validation_summary': self.validation_stats,
            'file_reports': reports,
            'validation_timestamp': datetime.now().isoformat(),
            'validator_config': self.kg_config
        }
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Saved KG validation report: {report_path}")
        return report_path
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation operations."""
        summary = self.validation_stats.copy()
        
        if summary['nodes_processed'] > 0:
            summary['node_preservation_rate'] = summary['nodes_preserved'] / summary['nodes_processed']
        else:
            summary['node_preservation_rate'] = 0.0
            
        if summary['edges_processed'] > 0:
            summary['edge_preservation_rate'] = summary['edges_preserved'] / summary['edges_processed']
        else:
            summary['edge_preservation_rate'] = 0.0
        
        summary['overall_data_quality'] = (summary['node_preservation_rate'] + summary['edge_preservation_rate']) / 2
        
        return summary


class KGIntegrityChecker:
    """Additional integrity checks for knowledge graphs."""
    
    @staticmethod
    def check_graph_connectivity(kg_data: Dict[str, Any]) -> Dict[str, Any]:
        """Check graph connectivity and structure."""
        nodes = kg_data.get('nodes', [])
        edges = kg_data.get('edges', [])
        
        if not nodes:
            return {'error': 'No nodes in graph'}
        
        if not edges:
            return {'warning': 'No edges in graph - all nodes are isolated'}
        
        # Build adjacency list
        adjacency = {}
        for edge in edges:
            source = edge.get('source')
            target = edge.get('target')
            if source and target:
                if source not in adjacency:
                    adjacency[source] = []
                adjacency[source].append(target)
        
        # Find connected components
        visited = set()
        components = []
        
        def dfs(node):
            if node in visited:
                return []
            visited.add(node)
            component = [node]
            for neighbor in adjacency.get(node, []):
                component.extend(dfs(neighbor))
            return component
        
        node_ids = {node.get('id') for node in nodes if node.get('id')}
        for node_id in node_ids:
            if node_id not in visited:
                component = dfs(node_id)
                if component:
                    components.append(component)
        
        return {
            'total_nodes': len(nodes),
            'total_edges': len(edges),
            'connected_components': len(components),
            'largest_component_size': max(len(comp) for comp in components) if components else 0,
            'isolated_nodes': len(node_ids) - len(visited),
            'is_connected': len(components) <= 1 and len(visited) == len(node_ids)
        }
