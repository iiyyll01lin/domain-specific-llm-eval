"""
Logging configuration for the RAG Evaluation Pipeline.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

def setup_logging(config: Dict[str, Any], run_id: str) -> logging.Logger:
    """
    Set up logging configuration for the pipeline.
    
    Args:
        config: Pipeline configuration
        run_id: Unique run identifier
        
    Returns:
        Configured logger instance
    """
    logging_config = config.get('logging', {})
    
    # Get log level
    log_level = getattr(logging, logging_config.get('level', 'INFO').upper())
    
    # Create logger
    logger = logging.getLogger('rag_evaluation')
    logger.setLevel(log_level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if 'console' in logging_config.get('destinations', ['console']):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if 'file' in logging_config.get('destinations', ['console']):
        # Create logs directory
        output_dir = Path(config.get('output', {}).get('base_dir', './outputs'))
        
        if config.get('output', {}).get('use_timestamps', True):
            log_dir = output_dir / run_id / 'logs'
        else:
            log_dir = output_dir / 'logs'
        
        log_dir.mkdir(parents=True, exist_ok=True)
        
        # Log file path
        log_file = log_dir / f'pipeline_{run_id}.log'
        
        # File handler with rotation
        file_logging_config = logging_config.get('file_logging', {})
        max_size_mb = file_logging_config.get('max_size_mb', 100)
        backup_count = file_logging_config.get('backup_count', 5)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_size_mb * 1024 * 1024,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Performance logging if enabled
    if logging_config.get('performance', {}).get('enabled', False):
        perf_logger = _setup_performance_logging(config, run_id)
        logger.info(f"Performance logging enabled: {perf_logger.handlers[0].baseFilename}")
    
    logger.info(f"Logging initialized (level: {logging_config.get('level', 'INFO')})")
    return logger

def _setup_performance_logging(config: Dict[str, Any], run_id: str) -> logging.Logger:
    """
    Set up performance logging for timing and memory tracking.
    
    Args:
        config: Pipeline configuration
        run_id: Unique run identifier
        
    Returns:
        Performance logger instance
    """
    # Create performance logger
    perf_logger = logging.getLogger('rag_evaluation.performance')
    perf_logger.setLevel(logging.INFO)
    
    # Performance log directory
    output_dir = Path(config.get('output', {}).get('base_dir', './outputs'))
    
    if config.get('output', {}).get('use_timestamps', True):
        log_dir = output_dir / run_id / 'logs'
    else:
        log_dir = output_dir / 'logs'
    
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Performance log file
    perf_log_file = log_dir / f'performance_{run_id}.log'
    
    # Performance formatter
    perf_formatter = logging.Formatter(
        fmt='%(asctime)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Performance file handler
    perf_handler = logging.FileHandler(perf_log_file)
    perf_handler.setLevel(logging.INFO)
    perf_handler.setFormatter(perf_formatter)
    perf_logger.addHandler(perf_handler)
    
    return perf_logger

class PerformanceTimer:
    """Context manager for timing operations."""
    
    def __init__(self, operation_name: str, logger: Optional[logging.Logger] = None):
        """
        Initialize performance timer.
        
        Args:
            operation_name: Name of the operation being timed
            logger: Logger instance for recording timing
        """
        self.operation_name = operation_name
        self.logger = logger or logging.getLogger('rag_evaluation.performance')
        self.start_time = None
        
    def __enter__(self):
        """Start timing."""
        self.start_time = datetime.now()
        self.logger.info(f"STARTED | {self.operation_name}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log duration."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        if exc_type is None:
            self.logger.info(f"COMPLETED | {self.operation_name} | Duration: {duration:.2f}s")
        else:
            self.logger.error(f"FAILED | {self.operation_name} | Duration: {duration:.2f}s | Error: {exc_type.__name__}")

class MemoryTracker:
    """Track memory usage during pipeline execution."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize memory tracker.
        
        Args:
            logger: Logger instance for recording memory usage
        """
        self.logger = logger or logging.getLogger('rag_evaluation.performance')
        self._psutil_available = self._check_psutil()
        
    def log_memory_usage(self, operation: str) -> None:
        """
        Log current memory usage.
        
        Args:
            operation: Description of current operation
        """
        if not self._psutil_available:
            return
        
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024
            
            self.logger.info(f"MEMORY | {operation} | {memory_mb:.1f} MB")
            
        except Exception as e:
            self.logger.debug(f"Failed to log memory usage: {e}")
    
    def _check_psutil(self) -> bool:
        """Check if psutil is available."""
        try:
            import psutil
            return True
        except ImportError:
            return False

# Convenience functions for common logging patterns

def log_pipeline_start(logger: logging.Logger, config: Dict[str, Any], run_id: str) -> None:
    """Log pipeline start information."""
    logger.info("=" * 80)
    logger.info("🚀 RAG EVALUATION PIPELINE STARTED")
    logger.info("=" * 80)
    logger.info(f"Run ID: {run_id}")
    logger.info(f"Pipeline Mode: {config.get('pipeline', {}).get('mode', 'unknown')}")
    logger.info(f"Configuration: {config.get('pipeline', {}).get('name', 'unnamed')}")

def log_pipeline_end(logger: logging.Logger, success: bool, duration: float) -> None:
    """Log pipeline completion information."""
    logger.info("=" * 80)
    if success:
        logger.info("✅ RAG EVALUATION PIPELINE COMPLETED SUCCESSFULLY")
    else:
        logger.info("❌ RAG EVALUATION PIPELINE FAILED")
    logger.info(f"Total Duration: {duration:.2f} seconds")
    logger.info("=" * 80)

def log_stage_start(logger: logging.Logger, stage_name: str) -> None:
    """Log stage start."""
    logger.info(f"🔄 Starting stage: {stage_name}")

def log_stage_end(logger: logging.Logger, stage_name: str, success: bool, duration: float) -> None:
    """Log stage completion."""
    status = "✅ COMPLETED" if success else "❌ FAILED"
    logger.info(f"{status} | Stage: {stage_name} | Duration: {duration:.2f}s")

def log_metrics(logger: logging.Logger, metrics: Dict[str, Any]) -> None:
    """Log evaluation metrics."""
    logger.info("📊 EVALUATION METRICS")
    logger.info("-" * 40)
    for metric_name, metric_value in metrics.items():
        if isinstance(metric_value, float):
            logger.info(f"{metric_name}: {metric_value:.3f}")
        else:
            logger.info(f"{metric_name}: {metric_value}")
    logger.info("-" * 40)
