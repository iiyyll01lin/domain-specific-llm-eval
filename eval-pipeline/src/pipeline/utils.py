"""
Utility functions for the RAG Evaluation Pipeline.
"""

import os
import sys
import shutil
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import importlib.util
import subprocess

logger = logging.getLogger(__name__)

def generate_run_id() -> str:
    """
    Generate a unique run ID for pipeline execution.
    
    Returns:
        Unique run identifier string
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = str(uuid.uuid4())[:8]
    return f"run_{timestamp}_{short_uuid}"

def create_output_directories(config: Dict[str, Any], run_id: str) -> Dict[str, Path]:
    """
    Create output directory structure for pipeline run.
    
    Args:
        config: Pipeline configuration
        run_id: Unique run identifier
        
    Returns:
        Dictionary mapping directory types to Path objects
    """
    output_config = config.get('output', {})
    base_dir = Path(output_config.get('base_dir', './outputs'))
    
    # Create timestamped subdirectory if enabled
    if output_config.get('use_timestamps', True):
        run_dir = base_dir / run_id
    else:
        run_dir = base_dir
    
    # Define directory structure
    directories = {
        'base': run_dir,
        'testsets': run_dir / 'testsets',
        'reports': run_dir / 'reports', 
        'metadata': run_dir / 'metadata',
        'logs': run_dir / 'logs',
        'cache': run_dir / 'cache',
        'temp': run_dir / 'temp'
    }
    
    # Create directories
    for dir_type, dir_path in directories.items():
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")
    
    logger.info(f"Output directories created under: {run_dir}")
    return directories

def validate_environment(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate the runtime environment for pipeline execution.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Dictionary with validation results:
        - success: bool
        - errors: List[str]
        - warnings: List[str]
    """
    errors = []
    warnings = []
    
    # Check Python version
    python_version = sys.version_info
    if python_version < (3, 8):
        errors.append(f"Python 3.8+ required, found {python_version.major}.{python_version.minor}")
    
    # Check required packages
    required_packages = [
        'numpy',
        'pandas', 
        'sentence_transformers',
        'spacy',
        'yaml',  # PyYAML package imports as 'yaml'
        'openpyxl'
    ]
    
    # Add conditional packages based on config
    eval_config = config.get('evaluation', {})
    
    if eval_config.get('contextual_keywords', {}).get('enabled', False):
        required_packages.extend(['spacy'])
    
    if eval_config.get('ragas_metrics', {}).get('enabled', False):
        required_packages.extend(['ragas', 'datasets', 'transformers'])
    
    # Check package availability
    missing_packages = []
    for package in required_packages:
        if not _check_package_available(package):
            missing_packages.append(package)
    
    if missing_packages:
        errors.append(f"Missing required packages: {', '.join(missing_packages)}")
    
    # Check spaCy model if needed
    if eval_config.get('contextual_keywords', {}).get('enabled', False):
        spacy_model = eval_config.get('contextual_keywords', {}).get('spacy_model', 'en_core_web_sm')
        if not _check_spacy_model(spacy_model):
            warnings.append(f"spaCy model '{spacy_model}' not found. Install with: python -m spacy download {spacy_model}")
    
    # Check write permissions for output directory
    output_dir = Path(config.get('output', {}).get('base_dir', './outputs'))
    if not _check_write_permission(output_dir):
        errors.append(f"No write permission for output directory: {output_dir}")
    
    # Check available disk space
    available_space_gb = _get_available_disk_space(output_dir)
    if available_space_gb < 1.0:  # Less than 1 GB
        warnings.append(f"Low disk space: {available_space_gb:.1f} GB available")
    
    # Check memory availability
    available_memory_gb = _get_available_memory()
    if available_memory_gb < 2.0:  # Less than 2 GB
        warnings.append(f"Low memory: {available_memory_gb:.1f} GB available")
    
    # Check RAG system connectivity (if configured)
    rag_config = config.get('rag_system', {})
    if 'endpoint' in rag_config:
        endpoint = rag_config['endpoint']
        if not _check_url_accessible(endpoint):
            warnings.append(f"RAG system endpoint not accessible: {endpoint}")
    
    return {
        'success': len(errors) == 0,
        'errors': errors,
        'warnings': warnings
    }

def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"

def format_file_size(bytes_size: int) -> str:
    """
    Format file size in bytes to human-readable string.
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} PB"

def safe_filename(filename: str) -> str:
    """
    Convert string to safe filename by removing/replacing invalid characters.
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename string
    """
    # Replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    safe_name = filename
    for char in invalid_chars:
        safe_name = safe_name.replace(char, '_')
    
    # Remove leading/trailing spaces and dots
    safe_name = safe_name.strip(' .')
    
    # Limit length
    if len(safe_name) > 200:
        safe_name = safe_name[:200]
    
    return safe_name

def copy_config_to_output(config_path: str, output_dir: Path) -> Path:
    """
    Copy configuration file to output directory for reproducibility.
    
    Args:
        config_path: Path to source configuration file
        output_dir: Output directory
        
    Returns:
        Path to copied configuration file
    """
    config_source = Path(config_path)
    config_dest = output_dir / f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    
    shutil.copy2(config_source, config_dest)
    logger.info(f"Configuration copied to: {config_dest}")
    
    return config_dest

def cleanup_temp_files(temp_dir: Path, keep_on_error: bool = True) -> None:
    """
    Clean up temporary files and directories.
    
    Args:
        temp_dir: Temporary directory to clean
        keep_on_error: Whether to keep files if there was an error
    """
    if temp_dir.exists() and temp_dir.is_dir():
        try:
            shutil.rmtree(temp_dir)
            logger.debug(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary directory {temp_dir}: {e}")

def get_system_info() -> Dict[str, Any]:
    """
    Get system information for debugging and reporting.
    
    Returns:
        Dictionary with system information
    """
    import platform
    
    return {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'architecture': platform.architecture()[0],
        'processor': platform.processor(),
        'available_memory_gb': _get_available_memory(),
        'available_disk_gb': _get_available_disk_space(Path.cwd())
    }

# Private helper functions

def _check_package_available(package_name: str) -> bool:
    """Check if a Python package is available for import."""
    try:
        importlib.import_module(package_name)
        return True
    except ImportError:
        return False

def _check_spacy_model(model_name: str) -> bool:
    """Check if a spaCy model is available."""
    try:
        import spacy
        spacy.load(model_name)
        return True
    except (ImportError, IOError):
        return False

def _check_write_permission(directory: Path) -> bool:
    """Check if we have write permission to a directory."""
    try:
        # Create directory if it doesn't exist
        directory.mkdir(parents=True, exist_ok=True)
        
        # Try to create a temporary file
        test_file = directory / f"test_write_{uuid.uuid4().hex[:8]}.tmp"
        test_file.write_text("test")
        test_file.unlink()
        
        return True
    except (PermissionError, OSError):
        return False

def _get_available_disk_space(path: Path) -> float:
    """Get available disk space in GB for given path."""
    try:
        statvfs = os.statvfs(path)
        available_bytes = statvfs.f_frsize * statvfs.f_available
        return available_bytes / (1024 ** 3)  # Convert to GB
    except (OSError, AttributeError):
        # Fallback for Windows
        try:
            import shutil
            _, _, free_bytes = shutil.disk_usage(path)
            return free_bytes / (1024 ** 3)  # Convert to GB
        except Exception:
            return float('inf')  # Unknown

def _get_available_memory() -> float:
    """Get available system memory in GB."""
    try:
        import psutil
        memory = psutil.virtual_memory()
        return memory.available / (1024 ** 3)  # Convert to GB
    except ImportError:
        # Fallback without psutil
        try:
            # Linux/Unix
            with open('/proc/meminfo', 'r') as f:
                meminfo = f.read()
            
            for line in meminfo.split('\n'):
                if line.startswith('MemAvailable:'):
                    available_kb = int(line.split()[1])
                    return available_kb / (1024 ** 2)  # Convert to GB
        except Exception:
            pass
        
        return float('inf')  # Unknown

def _check_url_accessible(url: str) -> bool:
    """Check if a URL is accessible."""
    try:
        import requests
        response = requests.head(url, timeout=5)
        return response.status_code < 400
    except Exception:
        return False
