#!/usr/bin/env python3
"""
Pipeline File Saver - Comprehensive Solution for Missing File Save Functionality

This module provides a standardized file saving system for the evaluation pipeline:
1. Testset CSV files
2. Knowledge graph JSON files  
3. Personas and scenarios JSON files
4. Pipeline metadata
5. Standardized directory structure
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class PipelineFileSaver:
    """
    Comprehensive file saver for pipeline outputs with standardized directory structure
    """
    
    def __init__(self, base_output_dir: Path):
        self.base_output_dir = Path(base_output_dir)
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Standardized directory structure
        self.directories = {
            'testsets': self.base_output_dir / 'testsets',
            'knowledge_graphs': self.base_output_dir / 'testsets' / 'knowledge_graphs',
            'personas': self.base_output_dir / 'metadata' / 'personas', 
            'scenarios': self.base_output_dir / 'metadata' / 'scenarios',
            'metadata': self.base_output_dir / 'metadata',
            'evaluations': self.base_output_dir / 'evaluations',
            'reports': self.base_output_dir / 'reports',
            'cache': self.base_output_dir / 'cache',
            'logs': self.base_output_dir / 'logs'
        }
        
        # Create all directories
        self._create_directories()
        
    def _create_directories(self):
        """Create all required directories"""
        for dir_name, dir_path in self.directories.items():
            dir_path.mkdir(parents=True, exist_ok=True)
            logger.debug(f"📁 Created directory: {dir_path}")
    
    def save_testset_csv(self, test_samples: List[Dict], filename_prefix: str = "testset") -> str:
        """
        Save testset to CSV format with proper error handling
        
        Args:
            test_samples: List of test sample dictionaries
            filename_prefix: Prefix for the CSV filename
            
        Returns:
            Path to saved CSV file
        """
        if not test_samples:
            logger.warning("❌ No test samples to save")
            return ""
        
        try:
            # Convert to DataFrame
            df = pd.DataFrame(test_samples)
            
            # Generate filename
            csv_filename = f"{filename_prefix}_{self.timestamp}.csv"
            csv_path = self.directories['testsets'] / csv_filename
            
            # Save to CSV
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            logger.info(f"✅ Testset CSV saved: {csv_path} ({len(test_samples)} samples)")
            return str(csv_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save testset CSV: {e}")
            raise
    
    def save_knowledge_graph_json(self, kg_data: Dict[str, Any], filename_prefix: str = "knowledge_graph") -> str:
        """
        Save knowledge graph to JSON format
        
        Args:
            kg_data: Knowledge graph data dictionary
            filename_prefix: Prefix for the JSON filename
            
        Returns:
            Path to saved JSON file
        """
        if not kg_data:
            logger.warning("❌ No knowledge graph data to save")
            return ""
        
        try:
            # Generate filename
            json_filename = f"{filename_prefix}_{self.timestamp}.json"
            json_path = self.directories['knowledge_graphs'] / json_filename
            
            # Save to JSON with proper serialization
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(kg_data, f, indent=2, ensure_ascii=False, default=str)
            
            nodes_count = len(kg_data.get('nodes', []))
            relationships_count = len(kg_data.get('relationships', []))
            
            logger.info(f"✅ Knowledge graph JSON saved: {json_path} ({nodes_count} nodes, {relationships_count} relationships)")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save knowledge graph JSON: {e}")
            raise
    
    def save_personas_json(self, personas_data: List[Dict[str, Any]], filename_prefix: str = "personas") -> str:
        """
        Save personas to JSON format
        
        Args:
            personas_data: List of persona dictionaries
            filename_prefix: Prefix for the JSON filename
            
        Returns:
            Path to saved JSON file
        """
        if not personas_data:
            logger.warning("❌ No personas data to save")
            return ""
        
        try:
            # Generate filename
            json_filename = f"{filename_prefix}_{self.timestamp}.json"
            json_path = self.directories['personas'] / json_filename
            
            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(personas_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Personas JSON saved: {json_path} ({len(personas_data)} personas)")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save personas JSON: {e}")
            raise
    
    def save_scenarios_json(self, scenarios_data: List[Dict[str, Any]], filename_prefix: str = "scenarios") -> str:
        """
        Save scenarios to JSON format
        
        Args:
            scenarios_data: List of scenario dictionaries
            filename_prefix: Prefix for the JSON filename
            
        Returns:
            Path to saved JSON file
        """
        if not scenarios_data:
            logger.warning("❌ No scenarios data to save")
            return ""
        
        try:
            # Generate filename
            json_filename = f"{filename_prefix}_{self.timestamp}.json"
            json_path = self.directories['scenarios'] / json_filename
            
            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(scenarios_data, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Scenarios JSON saved: {json_path} ({len(scenarios_data)} scenarios)")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save scenarios JSON: {e}")
            raise
    
    def save_pipeline_metadata(self, metadata: Dict[str, Any], filename_prefix: str = "pipeline_metadata") -> str:
        """
        Save pipeline metadata to JSON format
        
        Args:
            metadata: Metadata dictionary
            filename_prefix: Prefix for the JSON filename
            
        Returns:
            Path to saved JSON file
        """
        try:
            # Add save timestamp to metadata
            metadata['save_timestamp'] = datetime.now().isoformat()
            metadata['directory_structure'] = {
                name: str(path) for name, path in self.directories.items()
            }
            
            # Generate filename
            json_filename = f"{filename_prefix}_{self.timestamp}.json"
            json_path = self.directories['metadata'] / json_filename
            
            # Save to JSON
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False, default=str)
            
            logger.info(f"✅ Pipeline metadata saved: {json_path}")
            return str(json_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to save pipeline metadata: {e}")
            raise
    
    def save_all_pipeline_outputs(self, 
                                  test_samples: List[Dict] = None,
                                  kg_data: Dict[str, Any] = None,
                                  personas_data: List[Dict] = None,
                                  scenarios_data: List[Dict] = None,
                                  metadata: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Save all pipeline outputs in one call
        
        Returns:
            Dictionary mapping output types to file paths
        """
        saved_files = {}
        
        try:
            # Save testset CSV
            if test_samples:
                saved_files['testset_csv'] = self.save_testset_csv(test_samples)
            
            # Save knowledge graph
            if kg_data:
                saved_files['knowledge_graph_json'] = self.save_knowledge_graph_json(kg_data)
            
            # Save personas
            if personas_data:
                saved_files['personas_json'] = self.save_personas_json(personas_data)
            
            # Save scenarios
            if scenarios_data:
                saved_files['scenarios_json'] = self.save_scenarios_json(scenarios_data)
            
            # Save metadata
            if metadata:
                saved_files['metadata_json'] = self.save_pipeline_metadata(metadata)
            
            logger.info(f"✅ Pipeline save complete: {len(saved_files)} files saved")
            return saved_files
            
        except Exception as e:
            logger.error(f"❌ Pipeline save failed: {e}")
            raise

    def get_directory_structure(self) -> Dict[str, str]:
        """Return the standardized directory structure"""
        return {name: str(path) for name, path in self.directories.items()}
