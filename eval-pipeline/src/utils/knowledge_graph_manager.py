#!/usr/bin/env python3
"""
Knowledge Graph Manager for Pipeline

This module handles knowledge graph storage, loading, and reuse functionality.
"""

import os
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)

class KnowledgeGraphManager:
    """Manages knowledge graph storage and retrieval"""
    
    def __init__(self, base_output_dir: str):
        self.base_output_dir = Path(base_output_dir)
        self.kg_dir = self.base_output_dir / "knowledge_graphs"
        self.kg_dir.mkdir(parents=True, exist_ok=True)
    
    def save_knowledge_graph(self, kg, metadata: Dict[str, Any], run_id: str) -> str:
        """Save knowledge graph with metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save knowledge graph
        kg_filename = f"knowledge_graph_{timestamp}.pkl"
        kg_filepath = self.kg_dir / kg_filename
        
        with open(kg_filepath, 'wb') as f:
            pickle.dump(kg, f)
        
        # Save metadata
        metadata_filename = f"metadata_{timestamp}.json"
        metadata_filepath = self.kg_dir / metadata_filename
        
        metadata_with_info = {
            'created_at': timestamp,
            'run_id': run_id,
            'kg_file': kg_filename,
            'nodes_count': len(kg.nodes) if hasattr(kg, 'nodes') else 0,
            'relationships_count': len(kg.relationships) if hasattr(kg, 'relationships') else 0,
            **metadata
        }
        
        with open(metadata_filepath, 'w') as f:
            json.dump(metadata_with_info, f, indent=2)
        
        logger.info(f"💾 Saved knowledge graph: {kg_filepath}")
        logger.info(f"📋 Saved metadata: {metadata_filepath}")
        
        return str(kg_filepath)
    
    def load_knowledge_graph(self, kg_filepath: str):
        """Load knowledge graph from file"""
        try:
            with open(kg_filepath, 'rb') as f:
                kg = pickle.load(f)
            
            logger.info(f"📥 Loaded knowledge graph from: {kg_filepath}")
            return kg
        except Exception as e:
            logger.error(f"❌ Failed to load knowledge graph: {e}")
            return None
    
    def list_available_knowledge_graphs(self) -> List[Dict[str, Any]]:
        """List all available knowledge graphs with metadata"""
        kg_files = []
        
        for metadata_file in self.kg_dir.glob("metadata_*.json"):
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                kg_file = self.kg_dir / metadata.get('kg_file', '')
                if kg_file.exists():
                    metadata['kg_filepath'] = str(kg_file)
                    metadata['metadata_filepath'] = str(metadata_file)
                    metadata['file_size'] = kg_file.stat().st_size
                    kg_files.append(metadata)
            except Exception as e:
                logger.warning(f"Failed to read metadata from {metadata_file}: {e}")
        
        # Sort by creation time (newest first)
        kg_files.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        
        return kg_files
    
    def get_latest_knowledge_graph(self) -> Optional[str]:
        """Get the path to the latest knowledge graph"""
        available_kgs = self.list_available_knowledge_graphs()
        
        if available_kgs:
            latest_kg = available_kgs[0]
            return latest_kg.get('kg_filepath')
        
        return None
    
    def cleanup_old_knowledge_graphs(self, keep_count: int = 5):
        """Remove old knowledge graphs, keeping only the specified number"""
        available_kgs = self.list_available_knowledge_graphs()
        
        if len(available_kgs) > keep_count:
            to_remove = available_kgs[keep_count:]
            
            for kg_info in to_remove:
                try:
                    # Remove knowledge graph file
                    kg_filepath = Path(kg_info['kg_filepath'])
                    if kg_filepath.exists():
                        kg_filepath.unlink()
                    
                    # Remove metadata file
                    metadata_filepath = Path(kg_info['metadata_filepath'])
                    if metadata_filepath.exists():
                        metadata_filepath.unlink()
                    
                    logger.info(f"🗑️ Removed old knowledge graph: {kg_filepath.name}")
                except Exception as e:
                    logger.warning(f"Failed to remove old knowledge graph: {e}")


def find_and_use_latest_kg(config: Dict[str, Any], output_dir: str) -> Dict[str, Any]:
    """Find and configure to use the latest knowledge graph"""
    kg_manager = KnowledgeGraphManager(output_dir)
    latest_kg = kg_manager.get_latest_knowledge_graph()
    
    if latest_kg:
        logger.info(f"🔍 Found existing knowledge graph: {latest_kg}")
        
        # Update configuration to use existing KG
        if 'testset_generation' not in config:
            config['testset_generation'] = {}
        
        if 'ragas_config' not in config['testset_generation']:
            config['testset_generation']['ragas_config'] = {}
        
        if 'knowledge_graph_config' not in config['testset_generation']['ragas_config']:
            config['testset_generation']['ragas_config']['knowledge_graph_config'] = {}
        
        config['testset_generation']['ragas_config']['knowledge_graph_config']['existing_kg_file'] = latest_kg
        config['testset_generation']['ragas_config']['knowledge_graph_config']['enable_kg_loading'] = True
        
        logger.info("✅ Updated configuration to use existing knowledge graph")
    else:
        logger.info("ℹ️ No existing knowledge graphs found")
    
    return config