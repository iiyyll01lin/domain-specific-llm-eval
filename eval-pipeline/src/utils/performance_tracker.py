"""
Enhanced Performance Tracker for Domain-Specific RAG Evaluation Pipeline

This module provides comprehensive performance tracking including:
- Pipeline stage timings
- RAG response times
- LLM endpoint response times  
- Individual metric computation times
- Memory usage tracking
- Detailed timing breakdowns per question and metric
"""

import time
import logging
import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from contextlib import contextmanager

# Optional imports with fallbacks
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False
    
try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class TimingEntry:
    """Single timing measurement entry"""
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def finish(self) -> float:
        """Mark the timing entry as finished and calculate duration"""
        if self.end_time is None:
            self.end_time = time.time()
            self.duration = self.end_time - self.start_time
        return self.duration

@dataclass
class PerformanceMetrics:
    """Aggregated performance metrics"""
    total_duration: float
    min_duration: float
    max_duration: float
    mean_duration: float
    median_duration: float
    percentile_95: float
    percentile_99: float
    count: int
    memory_usage_mb: Dict[str, float] = field(default_factory=dict)

class PerformanceTracker:
    """
    Comprehensive performance tracker for the RAG evaluation pipeline
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize performance tracker with configuration"""
        self.config = config or {}
        self.enabled = self.config.get('enabled', True)
        
        if not self.enabled:
            return
            
        # Timing storage
        self.stage_timings: Dict[str, List[TimingEntry]] = defaultdict(list)
        self.rag_response_timings: List[TimingEntry] = []
        self.llm_response_timings: List[TimingEntry] = []
        self.metric_timings: Dict[str, List[TimingEntry]] = defaultdict(list)
        self.question_timings: List[TimingEntry] = []
        
        # Memory tracking
        self.memory_snapshots: List[Dict[str, float]] = []
        self.memory_tracking_enabled = self.config.get('track_memory_usage', True) and PSUTIL_AVAILABLE
        
        # Configuration
        self.track_stage_timings = self.config.get('track_stage_timings', True)
        self.track_rag_response_time = self.config.get('track_rag_response_time', True)
        self.track_llm_response_time = self.config.get('track_llm_response_time', True)
        self.track_metric_computation_time = self.config.get('track_metric_computation_time', True)
        self.detailed_timing_breakdown = self.config.get('detailed_timing_breakdown', True)
        self.timing_percentiles = self.config.get('timing_percentiles', [50, 95, 99])
        
        # Active timing contexts
        self._active_contexts: Dict[str, TimingEntry] = {}
        self._context_lock = threading.Lock()
        
        logger.info("✅ Performance tracker initialized")
        
    def start_stage_timing(self, stage_name: str, metadata: Dict[str, Any] = None) -> str:
        """Start timing a pipeline stage"""
        if not self.enabled or not self.track_stage_timings:
            return ""
            
        timing_id = f"stage_{stage_name}_{time.time()}"
        entry = TimingEntry(
            start_time=time.time(),
            metadata=metadata or {}
        )
        
        with self._context_lock:
            self._active_contexts[timing_id] = entry
            
        if self.memory_tracking_enabled:
            self._capture_memory_snapshot(f"stage_start_{stage_name}")
            
        logger.debug(f"⏱️ Started timing stage: {stage_name}")
        return timing_id
        
    def end_stage_timing(self, timing_id: str, stage_name: str) -> Optional[float]:
        """End timing a pipeline stage"""
        if not self.enabled or not timing_id:
            return None
            
        with self._context_lock:
            entry = self._active_contexts.pop(timing_id, None)
            
        if entry is None:
            logger.warning(f"⚠️ No active timing found for stage {stage_name}")
            return None
            
        duration = entry.finish()
        self.stage_timings[stage_name].append(entry)
        
        if self.memory_tracking_enabled:
            self._capture_memory_snapshot(f"stage_end_{stage_name}")
            
        logger.debug(f"✅ Stage {stage_name} completed in {duration:.3f}s")
        return duration
        
    @contextmanager
    def time_stage(self, stage_name: str, metadata: Dict[str, Any] = None):
        """Context manager for timing a pipeline stage"""
        timing_id = self.start_stage_timing(stage_name, metadata)
        try:
            yield
        finally:
            self.end_stage_timing(timing_id, stage_name)
            
    def time_rag_response(self, question: str, response_time: float, 
                         metadata: Dict[str, Any] = None):
        """Record RAG response timing"""
        if not self.enabled or not self.track_rag_response_time:
            return
            
        entry = TimingEntry(
            start_time=time.time() - response_time,
            end_time=time.time(),
            duration=response_time,
            metadata={
                'question': question[:100] + "..." if len(question) > 100 else question,
                **(metadata or {})
            }
        )
        
        self.rag_response_timings.append(entry)
        logger.debug(f"📊 RAG response time: {response_time:.3f}s")
        
    def time_llm_response(self, prompt: str, response_time: float, 
                         metadata: Dict[str, Any] = None):
        """Record LLM response timing"""
        if not self.enabled or not self.track_llm_response_time:
            return
            
        entry = TimingEntry(
            start_time=time.time() - response_time,
            end_time=time.time(),
            duration=response_time,
            metadata={
                'prompt_length': len(prompt),
                'prompt_preview': prompt[:50] + "..." if len(prompt) > 50 else prompt,
                **(metadata or {})
            }
        )
        
        self.llm_response_timings.append(entry)
        logger.debug(f"🤖 LLM response time: {response_time:.3f}s")
        
    @contextmanager
    def time_metric_computation(self, metric_name: str, metadata: Dict[str, Any] = None):
        """Context manager for timing metric computation"""
        if not self.enabled or not self.track_metric_computation_time:
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            entry = TimingEntry(
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                metadata=metadata or {}
            )
            
            self.metric_timings[metric_name].append(entry)
            logger.debug(f"📏 Metric {metric_name} computed in {duration:.3f}s")
            
    @contextmanager
    def time_question_evaluation(self, question_id: str, question: str):
        """Context manager for timing individual question evaluation"""
        if not self.enabled or not self.detailed_timing_breakdown:
            yield
            return
            
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            entry = TimingEntry(
                start_time=start_time,
                end_time=time.time(),
                duration=duration,
                metadata={
                    'question_id': question_id,
                    'question': question[:100] + "..." if len(question) > 100 else question
                }
            )
            
            self.question_timings.append(entry)
            logger.debug(f"❓ Question {question_id} evaluated in {duration:.3f}s")
            
    def _capture_memory_snapshot(self, label: str):
        """Capture current memory usage snapshot"""
        if not PSUTIL_AVAILABLE:
            return
            
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            snapshot = {
                'timestamp': time.time(),
                'label': label,
                'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size
                'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size
                'available_mb': psutil.virtual_memory().available / 1024 / 1024,
                'percent_used': psutil.virtual_memory().percent
            }
            
            self.memory_snapshots.append(snapshot)
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to capture memory snapshot: {e}")
            
    def _calculate_metrics(self, timings: List[TimingEntry]) -> Optional[PerformanceMetrics]:
        """Calculate performance metrics from timing entries"""
        if not timings:
            return None
            
        durations = [entry.duration for entry in timings if entry.duration is not None]
        if not durations:
            return None
        
        # Use numpy if available, otherwise use basic statistics
        if NUMPY_AVAILABLE:
            durations_array = np.array(durations)
            mean_duration = np.mean(durations_array)
            median_duration = np.median(durations_array)
            
            # Calculate percentiles
            percentiles = {}
            for p in self.timing_percentiles:
                percentiles[f'percentile_{p}'] = np.percentile(durations_array, p)
        else:
            # Fallback to basic statistics
            mean_duration = sum(durations) / len(durations)
            sorted_durations = sorted(durations)
            n = len(sorted_durations)
            median_duration = (sorted_durations[n//2] + sorted_durations[(n-1)//2]) / 2
            
            # Simple percentile calculation
            percentiles = {}
            for p in self.timing_percentiles:
                idx = int((p / 100.0) * (n - 1))
                percentiles[f'percentile_{p}'] = sorted_durations[min(idx, n-1)]
            
        return PerformanceMetrics(
            total_duration=sum(durations),
            min_duration=min(durations),
            max_duration=max(durations),
            mean_duration=mean_duration,
            median_duration=median_duration,
            percentile_95=percentiles.get('percentile_95', 0),
            percentile_99=percentiles.get('percentile_99', 0),
            count=len(durations)
        )
        
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary"""
        if not self.enabled:
            return {'performance_tracking': 'disabled'}
            
        summary = {
            'performance_tracking': {
                'enabled': True,
                'collection_time': datetime.now().isoformat(),
                'configuration': {
                    'track_stage_timings': self.track_stage_timings,
                    'track_rag_response_time': self.track_rag_response_time,
                    'track_llm_response_time': self.track_llm_response_time,
                    'track_metric_computation_time': self.track_metric_computation_time,
                    'memory_tracking_enabled': self.memory_tracking_enabled
                }
            }
        }
        
        # Stage timings
        if self.track_stage_timings and self.stage_timings:
            stage_metrics = {}
            for stage_name, timings in self.stage_timings.items():
                metrics = self._calculate_metrics(timings)
                if metrics:
                    stage_metrics[stage_name] = {
                        'total_duration_seconds': metrics.total_duration,
                        'average_duration_seconds': metrics.mean_duration,
                        'min_duration_seconds': metrics.min_duration,
                        'max_duration_seconds': metrics.max_duration,
                        'execution_count': metrics.count
                    }
            summary['stage_performance'] = stage_metrics
            
        # RAG response timings
        if self.track_rag_response_time and self.rag_response_timings:
            rag_metrics = self._calculate_metrics(self.rag_response_timings)
            if rag_metrics:
                summary['rag_response_performance'] = {
                    'total_requests': rag_metrics.count,
                    'total_response_time_seconds': rag_metrics.total_duration,
                    'average_response_time_seconds': rag_metrics.mean_duration,
                    'min_response_time_seconds': rag_metrics.min_duration,
                    'max_response_time_seconds': rag_metrics.max_duration,
                    'median_response_time_seconds': rag_metrics.median_duration,
                    'p95_response_time_seconds': rag_metrics.percentile_95,
                    'p99_response_time_seconds': rag_metrics.percentile_99
                }
                
        # LLM response timings
        if self.track_llm_response_time and self.llm_response_timings:
            llm_metrics = self._calculate_metrics(self.llm_response_timings)
            if llm_metrics:
                summary['llm_response_performance'] = {
                    'total_requests': llm_metrics.count,
                    'total_response_time_seconds': llm_metrics.total_duration,
                    'average_response_time_seconds': llm_metrics.mean_duration,
                    'min_response_time_seconds': llm_metrics.min_duration,
                    'max_response_time_seconds': llm_metrics.max_duration,
                    'median_response_time_seconds': llm_metrics.median_duration,
                    'p95_response_time_seconds': llm_metrics.percentile_95,
                    'p99_response_time_seconds': llm_metrics.percentile_99
                }
                
        # Metric computation timings
        if self.track_metric_computation_time and self.metric_timings:
            metric_performance = {}
            for metric_name, timings in self.metric_timings.items():
                metrics = self._calculate_metrics(timings)
                if metrics:
                    metric_performance[metric_name] = {
                        'total_computation_time_seconds': metrics.total_duration,
                        'average_computation_time_seconds': metrics.mean_duration,
                        'min_computation_time_seconds': metrics.min_duration,
                        'max_computation_time_seconds': metrics.max_duration,
                        'computation_count': metrics.count
                    }
            summary['metric_computation_performance'] = metric_performance
            
        # Question-level timings
        if self.detailed_timing_breakdown and self.question_timings:
            question_metrics = self._calculate_metrics(self.question_timings)
            if question_metrics:
                summary['question_evaluation_performance'] = {
                    'total_questions': question_metrics.count,
                    'total_evaluation_time_seconds': question_metrics.total_duration,
                    'average_evaluation_time_seconds': question_metrics.mean_duration,
                    'min_evaluation_time_seconds': question_metrics.min_duration,
                    'max_evaluation_time_seconds': question_metrics.max_duration,
                    'median_evaluation_time_seconds': question_metrics.median_duration
                }
                
        # Memory usage summary
        if self.memory_tracking_enabled and self.memory_snapshots:
            rss_values = [snapshot['rss_mb'] for snapshot in self.memory_snapshots]
            summary['memory_usage'] = {
                'peak_memory_usage_mb': max(rss_values),
                'min_memory_usage_mb': min(rss_values),
                'average_memory_usage_mb': sum(rss_values) / len(rss_values),
                'memory_snapshots_count': len(self.memory_snapshots),
                'final_memory_usage_mb': rss_values[-1] if rss_values else 0
            }
            
        return summary
        
    def get_detailed_timings(self) -> Dict[str, Any]:
        """Get detailed timing breakdown for debugging"""
        if not self.enabled:
            return {}
            
        detailed = {}
        
        # Individual question timings
        if self.detailed_timing_breakdown and self.question_timings:
            detailed['question_timings'] = [
                {
                    'question_id': entry.metadata.get('question_id'),
                    'question_preview': entry.metadata.get('question'),
                    'duration_seconds': entry.duration,
                    'timestamp': entry.start_time
                }
                for entry in self.question_timings
            ]
            
        # RAG response details
        if self.rag_response_timings:
            detailed['rag_response_timings'] = [
                {
                    'question_preview': entry.metadata.get('question'),
                    'response_time_seconds': entry.duration,
                    'timestamp': entry.start_time,
                    'metadata': entry.metadata
                }
                for entry in self.rag_response_timings
            ]
            
        # Memory snapshots
        if self.memory_snapshots:
            detailed['memory_snapshots'] = self.memory_snapshots
            
        return detailed
        
    def reset(self):
        """Reset all performance tracking data"""
        if not self.enabled:
            return
            
        self.stage_timings.clear()
        self.rag_response_timings.clear()
        self.llm_response_timings.clear()
        self.metric_timings.clear()
        self.question_timings.clear()
        self.memory_snapshots.clear()
        
        with self._context_lock:
            self._active_contexts.clear()
            
        logger.info("🔄 Performance tracker reset")
