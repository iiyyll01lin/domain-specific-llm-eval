"""
Configuration Manager for RAG Evaluation Pipeline

Handles loading, validation, and management of pipeline configuration.
"""

import yaml
import os
import sys
from typing import Dict, Any, List, Optional
from pathlib import Path
import logging
from jsonschema import validate, ValidationError
import tempfile

logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages pipeline configuration loading and validation."""
    
    def __init__(self, config_path: str):
        """
        Initialize configuration manager.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config: Optional[Dict[str, Any]] = None
        
    def load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file with environment variable substitution.
        
        Returns:
            Loaded configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid YAML
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        try:
            with open(self.config_path, 'r') as f:
                config_content = f.read()
            
            # Substitute environment variables
            config_content = self._substitute_env_vars(config_content)
            
            # Parse YAML
            self.config = yaml.safe_load(config_content)
            
            # Apply defaults
            self.config = self._apply_defaults(self.config)
            
            logger.info(f"Configuration loaded from {self.config_path}")
            return self.config
            
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {self.config_path}: {e}")
    
    def validate_config(self) -> Dict[str, Any]:
        """
        Validate the loaded configuration.
        
        Returns:
            Dictionary with validation results:
            - success: bool
            - errors: List[str]
            - warnings: List[str]
        """
        if self.config is None:
            return {
                'success': False,
                'errors': ['Configuration not loaded'],
                'warnings': []
            }
        
        errors = []
        warnings = []
        
        # Validate required sections
        required_sections = [
            'pipeline',
            'data_sources', 
            'testset_generation',
            'rag_system',
            'evaluation',
            'reporting',
            'output'
        ]
        
        for section in required_sections:
            if section not in self.config:
                errors.append(f"Missing required section: {section}")
        
        # Validate data sources
        data_sources = self.config.get('data_sources', {})
        if 'documents' in data_sources:
            docs = data_sources['documents']
            if 'primary_docs' in docs and docs['primary_docs'] is not None:
                for doc_path in docs['primary_docs']:
                    if not self._check_file_exists(doc_path):
                        warnings.append(f"Primary document not found: {doc_path}")
            
            if 'additional_dirs' in docs and docs['additional_dirs'] is not None:
                for dir_path in docs['additional_dirs']:
                    if not self._check_dir_exists(dir_path):
                        warnings.append(f"Additional directory not found: {dir_path}")
        
        # Validate RAG system configuration
        rag_config = self.config.get('rag_system', {})
        if 'endpoint' not in rag_config:
            errors.append("RAG system endpoint not specified")
        
        # Validate evaluation settings
        eval_config = self.config.get('evaluation', {})
        
        # Check contextual keywords settings
        if eval_config.get('contextual_keywords', {}).get('enabled', False):
            if 'threshold' not in eval_config.get('contextual_keywords', {}):
                warnings.append("Contextual keywords threshold not specified, using default")
        
        # Check RAGAS settings
        if eval_config.get('ragas_metrics', {}).get('enabled', False):
            ragas_config = eval_config.get('ragas_metrics', {})
            if 'metrics' not in ragas_config:
                warnings.append("RAGAS metrics not specified, using defaults")
        
        # Validate output settings
        output_config = self.config.get('output', {})
        if 'base_dir' not in output_config:
            errors.append("Output base directory not specified")
        
        return {
            'success': len(errors) == 0,
            'errors': errors,
            'warnings': warnings
        }
    
    def get_section(self, section_name: str) -> Dict[str, Any]:
        """
        Get a specific configuration section.
        
        Args:
            section_name: Name of the configuration section
            
        Returns:
            Configuration section dictionary
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        
        return self.config.get(section_name, {})
    
    def update_config(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with new values.
        
        Args:
            updates: Dictionary of updates to apply
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        
        self._deep_update(self.config, updates)
    
    def save_config(self, output_path: Optional[str] = None) -> None:
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save config. If None, overwrites original.
        """
        if self.config is None:
            raise RuntimeError("Configuration not loaded")
        
        save_path = Path(output_path) if output_path else self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {save_path}")
    
    def _substitute_env_vars(self, content: str) -> str:
        """
        Substitute environment variables in configuration content.
        
        Args:
            content: Raw configuration content
            
        Returns:
            Content with environment variables substituted
        """
        import re
        
        # Pattern to match ${VAR_NAME} or ${VAR_NAME:default_value}
        pattern = r'\$\{([^}]+)\}'
        
        def replace_var(match):
            var_spec = match.group(1)
            
            # Check for default value
            if ':' in var_spec:
                var_name, default_value = var_spec.split(':', 1)
            else:
                var_name = var_spec
                default_value = ''
            
            # Get environment variable value
            value = os.environ.get(var_name, default_value)
            
            if not value and not default_value:
                logger.warning(f"Environment variable {var_name} not set")
            
            return value
        
        return re.sub(pattern, replace_var, content)
    
    def _apply_defaults(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply default values to configuration.
        
        Args:
            config: Configuration dictionary
            
        Returns:
            Configuration with defaults applied
        """
        defaults = {
            'pipeline': {
                'mode': 'hybrid',
                'version': '1.0.0'
            },
            'data_sources': {
                'documents': {
                    'file_types': ['pdf', 'docx', 'txt', 'md'],
                    'processing': {
                        'chunk_size': 512,
                        'chunk_overlap': 50,
                        'language': 'en',
                        'min_doc_length': 100
                    }
                }
            },
            'testset_generation': {
                'samples_per_document': 100,
                'max_total_samples': 1000,
                'strategies': {
                    'simple': 0.4,
                    'multi_context': 0.3,
                    'reasoning': 0.2,
                    'conditional': 0.1
                },
                'output': {
                    'format': 'excel',
                    'include_metadata': True,
                    'tag_source_files': True
                }
            },
            'rag_system': {
                'timeout': 30,
                'max_retries': 3
            },
            'evaluation': {
                'contextual_keywords': {
                    'enabled': True,
                    'weights': {
                        'mandatory': 0.8,
                        'optional': 0.2
                    },
                    'threshold': 0.6,
                    'similarity_model': 'all-MiniLM-L6-v2',
                    'spacy_model': 'en_core_web_sm'
                },
                'ragas_metrics': {
                    'enabled': True,
                    'metrics': [
                        'context_precision',
                        'context_recall',
                        'faithfulness',
                        'answer_relevancy'
                    ]
                },
                'human_feedback': {
                    'enabled': True,
                    'initial_threshold': 0.7,
                    'uncertainty_bounds': {
                        'min': 0.3,
                        'max': 0.9,
                        'buffer': 0.1
                    }
                }
            },
            'reporting': {
                'report_types': ['comprehensive'],
                'formats': ['html', 'excel'],
                'visualizations': {
                    'enabled': True
                }
            },
            'output': {
                'base_dir': './outputs',
                'use_timestamps': True
            },
            'logging': {
                'level': 'INFO',
                'destinations': ['console', 'file']
            }
        }
        
        # Deep merge defaults with provided config
        result = self._deep_merge(defaults, config)
        return result
    
    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            dict1: Base dictionary
            dict2: Dictionary to merge in (takes precedence)
            
        Returns:
            Merged dictionary
        """
        result = dict1.copy()
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _deep_update(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> None:
        """
        Deep update dictionary in place.
        
        Args:
            dict1: Dictionary to update
            dict2: Updates to apply
        """
        for key, value in dict2.items():
            if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
                self._deep_update(dict1[key], value)
            else:
                dict1[key] = value
    
    def _check_file_exists(self, path: str) -> bool:
        """Check if a file exists, handling relative paths."""
        file_path = Path(path)
        
        # If relative path, check relative to config file directory
        if not file_path.is_absolute():
            file_path = self.config_path.parent / file_path
        
        return file_path.exists() and file_path.is_file()
    
    def _check_dir_exists(self, path: str) -> bool:
        """Check if a directory exists, handling relative paths."""
        dir_path = Path(path)
        
        # If relative path, check relative to config file directory
        if not dir_path.is_absolute():
            dir_path = self.config_path.parent / dir_path
        
        return dir_path.exists() and dir_path.is_dir()
