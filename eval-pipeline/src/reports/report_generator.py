#!/usr/bin/env python3
"""
Report Generator - Comprehensive Evaluation Report Generation

This module generates detailed reports from hybrid evaluation results including:
1. Executive summary for users
2. Technical analysis for developers  
3. Visualizations and charts
4. Actionable recommendations
5. Metadata and statistics
"""

import json
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import the evaluation data formatter
try:
    from .evaluation_data_formatter import EvaluationDataFormatter
    FORMATTER_AVAILABLE = True
except ImportError:
    FORMATTER_AVAILABLE = False
    logging.warning("EvaluationDataFormatter not available")
from jinja2 import Template

class ReportGenerator:
    """
    Comprehensive report generator for hybrid RAG evaluation results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.report_config = config.get('reporting', {})
        
        # Initialize data formatter
        if FORMATTER_AVAILABLE:
            self.formatter = EvaluationDataFormatter()
        else:
            self.formatter = None
            self.logger.warning("EvaluationDataFormatter not available - some reports may fail")
        
        # Set up visualization style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        self.logger.info("ReportGenerator initialized")
    
    def _ensure_required_columns(self, df: pd.DataFrame) -> None:
        """Ensure DataFrame has all required columns with default values."""
        if df.empty:
            # For empty DataFrame, create a single row with defaults
            default_row = {
                'overall_pass': False,
                'overall_score': 0.0,
                'contextual_total_score': 0.0,
                'contextual_keyword_pass': False,
                'contextual_pass': False,
                'ragas_composite_score': 0.0,
                'ragas_pass': False,
                'semantic_similarity': 0.0,
                'semantic_pass': False,
                'question_index': 0,
                'question': 'No data available',
                'rag_answer': 'No data available',
                'expected_answer': 'No data available',
                'source_file': 'unknown'
            }
            # Don't add default row for empty DataFrame - let it stay empty
            for col, default_val in default_row.items():
                df[col] = []
            return
        
        # Enhanced column mappings to handle actual evaluation output
        column_mappings = {
            # Direct RAGAS metrics (these are the actual column names generated)
            'context_precision': 'ragas_context_precision',
            'context_recall': 'ragas_context_recall', 
            'faithfulness': 'ragas_faithfulness',
            'answer_relevancy': 'ragas_answer_relevancy',
            
            # Legacy mappings for backward compatibility
            'kw_metric': 'contextual_keyword_score',
            'keyword_score': 'contextual_total_score',
            'weighted_average_score': 'overall_score'
        }
        
        # Apply column mappings (copy to new columns, don't rename)
        for old_col, new_col in column_mappings.items():
            if old_col in df.columns and new_col not in df.columns:
                df[new_col] = df[old_col]
                self.logger.info(f"✅ Mapped column {old_col} → {new_col}")
        
        # Create composite RAGAS score from individual metrics
        ragas_metrics = ['context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        available_ragas = [col for col in ragas_metrics if col in df.columns]
        if available_ragas and 'ragas_composite_score' not in df.columns:
            df['ragas_composite_score'] = df[available_ragas].mean(axis=1)
            self.logger.info(f"✅ Created ragas_composite_score from {len(available_ragas)} RAGAS metrics")
        
        # Create overall_score as average of available scores
        score_columns = []
        if 'ragas_composite_score' in df.columns:
            score_columns.append('ragas_composite_score')
        if 'contextual_total_score' in df.columns:
            score_columns.append('contextual_total_score')
        if 'semantic_similarity' in df.columns:
            score_columns.append('semantic_similarity')
            
        if score_columns and 'overall_score' not in df.columns:
            df['overall_score'] = df[score_columns].mean(axis=1)
            self.logger.info(f"✅ Created overall_score from {len(score_columns)} metrics")
        
        # Create pass/fail columns based on thresholds
        if 'contextual_keyword_pass' not in df.columns and 'contextual_total_score' in df.columns:
            df['contextual_keyword_pass'] = df['contextual_total_score'] >= 0.5
        if 'ragas_pass' not in df.columns and 'ragas_composite_score' in df.columns:
            df['ragas_pass'] = df['ragas_composite_score'] >= 0.7
        if 'semantic_pass' not in df.columns and 'semantic_similarity' in df.columns:
            df['semantic_pass'] = df['semantic_similarity'] >= 0.7
        if 'overall_pass' not in df.columns and 'overall_score' in df.columns:
            df['overall_pass'] = df['overall_score'] >= 0.7
        
        required_columns = {
            'overall_pass': False,
            'overall_score': 0.0,
            'contextual_total_score': 0.0,
            'contextual_keyword_pass': False,
            'contextual_pass': False,
            'ragas_composite_score': 0.0,
            'ragas_pass': False,
            'semantic_similarity': 0.0,
            'semantic_pass': False,
            'question_index': 0,
            'question': '',
            'rag_answer': '',
            'expected_answer': '',
            'source_file': 'unknown'
        }
        
        # Add missing columns with defaults (but don't override existing data)
        for col, default_val in required_columns.items():
            if col not in df.columns:
                df[col] = default_val
                self.logger.debug(f"Added default column: {col}")
        
        # Ensure proper data types
        boolean_columns = ['overall_pass', 'contextual_keyword_pass', 'contextual_pass', 'ragas_pass', 'semantic_pass']
        for col in boolean_columns:
            if col in df.columns:
                df[col] = df[col].astype(bool)
        
        numeric_columns = ['overall_score', 'contextual_total_score', 'ragas_composite_score', 'semantic_similarity',
                          'context_precision', 'context_recall', 'faithfulness', 'answer_relevancy']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)
    
    def generate_reports(self, evaluation_files: List[Path], run_id: str) -> Dict[str, str]:
        """
        Generate reports from evaluation files (orchestrator interface)
        
        Args:
            evaluation_files: List of evaluation result JSON files
            run_id: Pipeline run identifier
            
        Returns:
            Dictionary mapping report types to file paths
        """
        self.logger.info(f"Generating reports for run {run_id} from {len(evaluation_files)} files")
        
        try:
            # Load evaluation data
            evaluation_data = []
            evaluation_summary = {
                'run_id': run_id,
                'timestamp': datetime.now().isoformat(),
                'total_files': len(evaluation_files),
                'status': 'success'
            }
            
            for eval_file in evaluation_files:
                try:
                    with open(eval_file, 'r') as f:
                        data = json.load(f)
                        evaluation_data.append(data)
                except Exception as e:
                    self.logger.warning(f"Failed to load {eval_file}: {e}")
            
            if not evaluation_data:
                self.logger.warning("No valid evaluation data found")
                return {'error': 'No valid evaluation data'}
            
            # Convert to DataFrame for comprehensive report
            try:
                # Extract evaluation results from the loaded data
                flattened_data = []
                for data in evaluation_data:
                    if isinstance(data, dict):
                        # Check for different possible structures
                        if 'rag_results' in data and isinstance(data['rag_results'], list):
                            # Structure: {"rag_results": [...]}
                            flattened_data.extend(data['rag_results'])
                            self.logger.info(f"Extracted {len(data['rag_results'])} results from rag_results")
                        elif 'results' in data and isinstance(data['results'], list):
                            # Structure: {"results": [...]}  
                            flattened_data.extend(data['results'])
                            self.logger.info(f"Extracted {len(data['results'])} results from results")
                        elif isinstance(data, list):
                            # Structure: [...]
                            flattened_data.extend(data)
                            self.logger.info(f"Extracted {len(data)} results from list")
                        else:
                            # Single result object
                            flattened_data.append(data)
                            self.logger.info("Extracted 1 result from single object")
                    elif isinstance(data, list):
                        # Direct list of results
                        flattened_data.extend(data)
                        self.logger.info(f"Extracted {len(data)} results from direct list")
                
                # Create DataFrame from flattened data
                if flattened_data:
                    results_df = pd.DataFrame(flattened_data)
                    self.logger.info(f"Created DataFrame with {len(results_df)} rows and {len(results_df.columns)} columns")
                    self.logger.info(f"Available columns: {list(results_df.columns)}")
                    
                    # Ensure required columns exist with default values
                    self._ensure_required_columns(results_df)
                else:
                    results_df = pd.DataFrame()
                    self.logger.warning("No evaluation data found - creating empty DataFrame")
                    self._ensure_required_columns(results_df)
                
            except Exception as e:
                self.logger.warning(f"Could not create DataFrame: {e}")
                results_df = pd.DataFrame()
                self._ensure_required_columns(results_df)
            
            # Update summary with data statistics
            evaluation_summary.update({
                'total_evaluations': len(flattened_data) if flattened_data else 0,
                'data_summary': self._create_data_summary(evaluation_data)
            })
            
            # Generate output directory
            output_dir = Path(f"outputs/reports/run_{run_id}")
            
            # Generate comprehensive report
            return self.generate_comprehensive_report(
                evaluation_results=results_df,
                evaluation_summary=evaluation_summary,
                output_dir=output_dir
            )
            
        except Exception as e:
            self.logger.error(f"Report generation failed: {e}")
            return {'error': str(e)}
    
    def _create_data_summary(self, evaluation_data: List[Dict]) -> Dict[str, Any]:
        """Create summary statistics from evaluation data"""
        try:
            total_items = len(evaluation_data)
            
            # Extract metrics if available
            metrics_found = set()
            for item in evaluation_data:
                if isinstance(item, dict):
                    metrics_found.update(item.keys())
            
            return {
                'total_items': total_items,
                'metrics_found': list(metrics_found),
                'has_data': total_items > 0
            }
        except Exception as e:
            self.logger.warning(f"Could not create data summary: {e}")
            return {'total_items': 0, 'has_data': False}
    
    def generate_comprehensive_report(self, 
                                    evaluation_results: pd.DataFrame,
                                    evaluation_summary: Dict[str, Any],
                                    output_dir: Path,
                                    performance_data: Dict[str, Any] = None,
                                    composition_data: Dict[str, Any] = None,
                                    final_parameters: Dict[str, Any] = None) -> Dict[str, str]:
        """
        Generate comprehensive evaluation report with multiple formats including enhanced performance data
        
        Args:
            evaluation_results: Detailed evaluation results DataFrame
            evaluation_summary: Summary statistics and metadata
            output_dir: Directory to save reports
            performance_data: Performance tracking data (timing, memory usage)
            composition_data: Composition elements data (personas, scenarios, etc.)
            final_parameters: Final parameters and fallback usage data
        
        Returns:
            Dictionary mapping report types to file paths
        """
        self.logger.info("Generating comprehensive evaluation report with enhanced data")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Apply column mapping and ensure required columns exist
        self.logger.info(f"Processing DataFrame with {len(evaluation_results)} rows and {len(evaluation_results.columns)} columns")
        self.logger.debug(f"Input columns: {list(evaluation_results.columns)}")
        self._ensure_required_columns(evaluation_results)
        self.logger.info(f"After column mapping: {len(evaluation_results.columns)} columns")
        
        # Generate different report components
        report_files = {}
        
        try:
            # 1. Executive Summary (HTML) - Enhanced with performance data
            exec_summary_file = output_dir / "executive_summary.html"
            self._generate_executive_summary(
                evaluation_results, evaluation_summary, exec_summary_file,
                performance_data=performance_data
            )
            report_files['executive_summary'] = str(exec_summary_file)
            
        except Exception as e:
            self.logger.error(f"Error generating executive summary: {e}")
            # Continue with other reports
        
        try:
            # 2. Technical Analysis Report (HTML) - Enhanced with all data
            tech_report_file = output_dir / "technical_analysis.html" 
            self._generate_technical_report(
                evaluation_results, evaluation_summary, tech_report_file,
                performance_data=performance_data,
                composition_data=composition_data,
                final_parameters=final_parameters
            )
            report_files['technical_analysis'] = str(tech_report_file)
            
        except Exception as e:
            self.logger.error(f"Error generating technical report: {e}")
            # Continue with other reports
        
        try:
            # 3. Performance Analysis Report (HTML) - NEW
            if performance_data:
                perf_report_file = output_dir / "performance_analysis.html"
                self._generate_performance_report(
                    performance_data, perf_report_file, evaluation_summary
                )
                report_files['performance_analysis'] = str(perf_report_file)
                
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            # Continue with other reports
        
        try:
            # 4. Composition Elements Report (HTML) - NEW
            if composition_data:
                comp_report_file = output_dir / "composition_analysis.html"
                self._generate_composition_report(
                    composition_data, comp_report_file, evaluation_summary
                )
                report_files['composition_analysis'] = str(comp_report_file)
                
        except Exception as e:
            self.logger.error(f"Error generating composition report: {e}")
            # Continue with other reports
        
        try:
            # 5. Final Parameters Report (JSON) - NEW
            if final_parameters:
                params_report_file = output_dir / "final_parameters.json"
                self._generate_parameters_report(
                    final_parameters, params_report_file
                )
                report_files['final_parameters'] = str(params_report_file)
                
        except Exception as e:
            self.logger.error(f"Error generating parameters report: {e}")
            # Continue with other reports
        
        try:
            # 3. Detailed Results (Excel)
            excel_file = output_dir / "detailed_results.xlsx"
            self._generate_excel_report(evaluation_results, excel_file)
            report_files['detailed_results'] = str(excel_file)
            
        except Exception as e:
            self.logger.error(f"Error generating Excel report: {e}")
            # Continue with other reports
        
        try:
            # 4. Metadata (JSON)
            metadata_file = output_dir / "evaluation_metadata.json"
            self._generate_metadata_report(evaluation_summary, metadata_file)
            report_files['metadata'] = str(metadata_file)
            
        except Exception as e:
            self.logger.error(f"Error generating metadata report: {e}")
            # Continue with other reports
        
        try:
            # 5. Visualizations
            viz_dir = output_dir / "visualizations"
            viz_files = self._generate_visualizations(evaluation_results, viz_dir)
            report_files.update(viz_files)
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            # Continue with other reports
        
        try:
            # 6. Summary Statistics (CSV)
            stats_file = output_dir / "summary_statistics.csv"
            self._generate_summary_csv(evaluation_summary, stats_file)
            report_files['summary_statistics'] = str(stats_file)
            
        except Exception as e:
            self.logger.error(f"Error generating summary CSV: {e}")
            # Continue anyway
            
            self.logger.info(f"Report generation completed. {len(report_files)} files created.")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            raise
        
        return report_files
    
    def _generate_executive_summary(self, 
                                  results_df: pd.DataFrame,
                                  summary: Dict[str, Any],
                                  output_file: Path,
                                  performance_data: Dict[str, Any] = None):
        """Generate executive summary HTML report with enhanced performance data"""
        self.logger.info("Generating executive summary")
        
        # Calculate key metrics with safety checks
        total_evals = len(results_df) if results_df is not None and not results_df.empty else 0
        
        if total_evals == 0:
            # Handle empty dataset
            overall_pass_rate = 0.0
            source_performance = {}
            common_failures = []
        else:
            # Check if required columns exist
            if 'overall_pass' in results_df.columns:
                overall_pass_rate = results_df['overall_pass'].mean() * 100
            else:
                self.logger.warning("'overall_pass' column missing, calculating from available metrics")
                overall_pass_rate = 0.0
            
            # Performance by source (with safety checks)
            source_performance = {}
            if 'source_file' in results_df.columns:
                try:
                    # Check which columns exist before trying to use them
                    agg_dict = {}
                    if 'overall_pass' in results_df.columns:
                        agg_dict['overall_pass'] = 'mean'
                    if 'overall_score' in results_df.columns:
                        agg_dict['overall_score'] = 'mean'
                    if 'contextual_keyword_pass' in results_df.columns:
                        agg_dict['contextual_keyword_pass'] = 'mean'
                    elif 'contextual_pass' in results_df.columns:
                        agg_dict['contextual_pass'] = 'mean'
                    if 'ragas_pass' in results_df.columns:
                        agg_dict['ragas_pass'] = 'mean'
                    
                    if agg_dict:
                        source_perf = results_df.groupby('source_file').agg(agg_dict).round(3)
                        source_performance = source_perf.to_dict('index')
                    else:
                        self.logger.warning("No valid columns for source performance calculation")
                        
                except Exception as e:
                    self.logger.warning(f"Could not calculate source performance: {e}")
                    source_performance = {}
            
            # Identify top issues (with safety checks)
            if 'overall_pass' in results_df.columns:
                failed_evals = results_df[results_df['overall_pass'] == False]
            else:
                failed_evals = pd.DataFrame()  # Empty DataFrame if column missing
                
            common_failures = []
            
            if len(failed_evals) > 0:
                # Count failures with safe column access
                contextual_failures = 0
                ragas_failures = 0
                semantic_failures = 0
                
                if 'contextual_keyword_pass' in failed_evals.columns:
                    contextual_failures = len(failed_evals[failed_evals['contextual_keyword_pass'] == False])
                elif 'contextual_pass' in failed_evals.columns:
                    contextual_failures = len(failed_evals[failed_evals['contextual_pass'] == False])
                    
                if 'ragas_pass' in failed_evals.columns:
                    ragas_failures = len(failed_evals[failed_evals['ragas_pass'] == False])
                    
                if 'semantic_pass' in failed_evals.columns:
                    semantic_failures = len(failed_evals[failed_evals['semantic_pass'] == False])
                
                if contextual_failures > 0 or ragas_failures > 0 or semantic_failures > 0:
                    common_failures = [
                        {'type': 'Contextual Keyword Failures', 'count': contextual_failures, 'percentage': (contextual_failures/len(failed_evals))*100 if len(failed_evals) > 0 else 0},
                        {'type': 'RAGAS Metric Failures', 'count': ragas_failures, 'percentage': (ragas_failures/len(failed_evals))*100 if len(failed_evals) > 0 else 0},
                        {'type': 'Semantic Similarity Failures', 'count': semantic_failures, 'percentage': (semantic_failures/len(failed_evals))*100 if len(failed_evals) > 0 else 0}
                    ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results_df, summary)
        
        # Extract metrics safely from summary
        def safe_get_nested(d, keys, default=0.0):
            """Safely get nested dictionary values"""
            try:
                for key in keys:
                    d = d[key]
                return d
            except (KeyError, TypeError):
                return default
        
        contextual_pass_rate = safe_get_nested(summary, ['overall_statistics', 'contextual_pass_rate'], 0.0) * 100
        avg_contextual_score = safe_get_nested(summary, ['score_statistics', 'contextual_scores', 'mean'], 0.0)
        ragas_pass_rate = safe_get_nested(summary, ['overall_statistics', 'ragas_pass_rate'], 0.0) * 100
        avg_ragas_score = safe_get_nested(summary, ['score_statistics', 'ragas_scores', 'mean'], 0.0)
        semantic_pass_rate = safe_get_nested(summary, ['overall_statistics', 'semantic_pass_rate'], 0.0) * 100
        avg_semantic_score = safe_get_nested(summary, ['score_statistics', 'semantic_scores', 'mean'], 0.0)
        human_feedback_needed = safe_get_nested(summary, ['human_feedback_statistics', 'feedback_needed_count'], 0)
        human_feedback_ratio = safe_get_nested(summary, ['human_feedback_statistics', 'feedback_needed_ratio'], 0.0) * 100
        evaluation_date = safe_get_nested(summary, ['evaluation_metadata', 'start_time'], 'Unknown')
        
        # HTML template
        html_template = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG System Evaluation - Executive Summary</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
        .header { background-color: #f4f4f4; padding: 20px; border-radius: 8px; }
        .metric { background-color: #e8f4fd; padding: 15px; margin: 10px 0; border-radius: 5px; }
        .metric h3 { margin-top: 0; color: #2c5282; }
        .recommendation { background-color: #f0fff4; padding: 15px; margin: 10px 0; border-left: 4px solid #48bb78; }
        .warning { background-color: #fff5f5; padding: 15px; margin: 10px 0; border-left: 4px solid #f56565; }
        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 12px; text-align: left; }
        .table th { background-color: #f2f2f2; }
        .pass { color: #48bb78; font-weight: bold; }
        .fail { color: #f56565; font-weight: bold; }
    </style>
</head>
<body>
    <div class="header">
        <h1>RAG System Evaluation Report</h1>
        <p><strong>Evaluation Date:</strong> {{ evaluation_date }}</p>
        <p><strong>Total Evaluations:</strong> {{ total_evaluations }}</p>
        <p><strong>Overall Pass Rate:</strong> <span class="{{ 'pass' if overall_pass_rate >= 70 else 'fail' }}">{{ "%.1f"|format(overall_pass_rate) }}%</span></p>
    </div>
    
    <h2>📊 Key Performance Metrics</h2>
    <div class="metric">
        <h3>Contextual Keyword Performance</h3>
        <p><strong>Pass Rate:</strong> {{ "%.1f"|format(contextual_pass_rate) }}%</p>
        <p><strong>Average Score:</strong> {{ "%.3f"|format(avg_contextual_score) }}</p>
        <p>Measures how well the RAG system includes domain-specific keywords in context.</p>
    </div>
    
    <div class="metric">
        <h3>RAGAS Metrics Performance</h3>
        <p><strong>Pass Rate:</strong> {{ "%.1f"|format(ragas_pass_rate) }}%</p>
        <p><strong>Average Score:</strong> {{ "%.3f"|format(avg_ragas_score) }}</p>
        <p>Evaluates faithfulness, relevancy, and correctness of generated responses.</p>
    </div>
    
    <div class="metric">
        <h3>Semantic Similarity Performance</h3>
        <p><strong>Pass Rate:</strong> {{ "%.1f"|format(semantic_pass_rate) }}%</p>
        <p><strong>Average Score:</strong> {{ "%.3f"|format(avg_semantic_score) }}</p>
        <p>Measures semantic alignment between generated and expected answers.</p>
    </div>
    
    {% if source_performance %}
    <h2>📁 Performance by Source Document</h2>
    <table class="table">
        <thead>
            <tr>
                <th>Source Document</th>
                <th>Overall Pass Rate</th>
                <th>Avg Score</th>
                <th>Contextual Pass Rate</th>
                <th>RAGAS Pass Rate</th>
            </tr>
        </thead>
        <tbody>
        {% for source, metrics in source_performance.items() %}
            <tr>
                <td>{{ source }}</td>
                <td class="{{ 'pass' if metrics.overall_pass >= 0.7 else 'fail' }}">{{ "%.1f"|format(metrics.overall_pass * 100) }}%</td>
                <td>{{ "%.3f"|format(metrics.overall_score) }}</td>
                <td>{{ "%.1f"|format(metrics.contextual_keyword_pass * 100) }}%</td>
                <td>{{ "%.1f"|format(metrics.ragas_pass * 100) }}%</td>
            </tr>
        {% endfor %}
        </tbody>
    </table>
    {% endif %}
    
    {% if common_failures %}
    <h2>⚠️ Common Failure Patterns</h2>
    {% for failure in common_failures %}
    <div class="warning">
        <h3>{{ failure.type }}</h3>
        <p><strong>Count:</strong> {{ failure.count }} ({{ "%.1f"|format(failure.percentage) }}% of failures)</p>
    </div>
    {% endfor %}
    {% endif %}
    
    <h2>💡 Recommendations</h2>
    {% for rec in recommendations %}
    <div class="recommendation">
        <h3>{{ rec.title }}</h3>
        <p>{{ rec.description }}</p>
        <p><strong>Priority:</strong> {{ rec.priority }}</p>
    </div>
    {% endfor %}
    
    <h2>🔍 Human Feedback Requirements</h2>
    <div class="metric">
        <p><strong>Samples requiring human feedback:</strong> {{ human_feedback_needed }}</p>
        <p><strong>Feedback ratio:</strong> {{ "%.1f"|format(human_feedback_ratio) }}%</p>
        <p>Lower ratios indicate higher system confidence and consistency.</p>
    </div>
    
    <hr>
    <p><small>Generated by Domain-Specific RAG Evaluation Pipeline on {{ generation_time }}</small></p>
</body>
</html>
        """
        
        # Render template
        template = Template(html_template)
        
        html_content = template.render(
            evaluation_date=evaluation_date,
            total_evaluations=total_evals,
            overall_pass_rate=overall_pass_rate,
            contextual_pass_rate=contextual_pass_rate,
            avg_contextual_score=avg_contextual_score,
            ragas_pass_rate=ragas_pass_rate,
            avg_ragas_score=avg_ragas_score,
            semantic_pass_rate=semantic_pass_rate,
            avg_semantic_score=avg_semantic_score,
            source_performance=source_performance,
            common_failures=common_failures,
            recommendations=recommendations,
            human_feedback_needed=human_feedback_needed,
            human_feedback_ratio=human_feedback_ratio,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Executive summary saved to {output_file}")
    
    def _generate_technical_report(self, 
                                 results_df: pd.DataFrame,
                                 summary: Dict[str, Any],
                                 output_file: Path,
                                 performance_data: Dict[str, Any] = None,
                                 composition_data: Dict[str, Any] = None,
                                 final_parameters: Dict[str, Any] = None):
        """Generate detailed technical analysis report with enhanced performance data"""
        self.logger.info("Generating technical analysis report")
        
        # Detailed statistical analysis
        stats_analysis = self._perform_statistical_analysis(results_df)
        
        # Correlation analysis
        correlation_analysis = self._analyze_metric_correlations(results_df)
        
        # HTML template for technical report
        tech_template = """
<!DOCTYPE html>
<html>
<head>
    <title>RAG System Evaluation - Technical Analysis</title>
    <style>
        body { font-family: 'Courier New', monospace; margin: 40px; line-height: 1.6; }
        .header { background-color: #f8f9fa; padding: 20px; border-radius: 8px; }
        .section { margin: 30px 0; padding: 20px; border: 1px solid #dee2e6; border-radius: 5px; }
        .metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
        .metric-box { background-color: #f8f9fa; padding: 15px; border-radius: 5px; }
        .table { width: 100%; border-collapse: collapse; margin: 20px 0; }
        .table th, .table td { border: 1px solid #ddd; padding: 8px; text-align: left; font-size: 12px; }
        .table th { background-color: #e9ecef; }
        pre { background-color: #f8f9fa; padding: 15px; border-radius: 5px; overflow-x: auto; }
    </style>
</head>
<body>
    <div class="header">
        <h1>Technical Analysis Report</h1>
        <p><strong>Generated:</strong> {{ generation_time }}</p>
        <p><strong>Total Samples:</strong> {{ total_samples }}</p>
    </div>
    
    <div class="section">
        <h2>📈 Statistical Analysis</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <h3>Score Distributions</h3>
                <pre>{{ score_distributions }}</pre>
            </div>
            <div class="metric-box">
                <h3>Performance Quartiles</h3>
                <pre>{{ performance_quartiles }}</pre>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🔗 Metric Correlations</h2>
        <pre>{{ correlation_matrix }}</pre>
        <p><strong>Key Insights:</strong></p>
        <ul>
        {% for insight in correlation_insights %}
            <li>{{ insight }}</li>
        {% endfor %}
        </ul>
    </div>
    
    <div class="section">
        <h2>🎯 Threshold Analysis</h2>
        <div class="metric-grid">
            <div class="metric-box">
                <h3>Current Thresholds</h3>
                <p><strong>Contextual:</strong> {{ contextual_threshold }}</p>
                <p><strong>RAGAS:</strong> {{ ragas_threshold }}</p>
                <p><strong>Semantic:</strong> {{ semantic_threshold }}</p>
            </div>
            <div class="metric-box">
                <h3>Threshold Recommendations</h3>
                {% for rec in threshold_recommendations %}
                <p><strong>{{ rec.metric }}:</strong> {{ rec.recommendation }}</p>
                {% endfor %}
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>📊 Detailed Results Preview</h2>
        <table class="table">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>Overall Score</th>
                    <th>Contextual</th>
                    <th>RAGAS</th>
                    <th>Semantic</th>
                    <th>Pass</th>
                    <th>Feedback Needed</th>
                </tr>
            </thead>
            <tbody>
            {% for _, row in sample_results.iterrows() %}
                <tr>
                    <td>{{ row.evaluation_id }}</td>
                    <td>{{ "%.3f"|format(row.overall_score) }}</td>
                    <td>{{ "%.3f"|format(row.contextual_total_score) }}</td>
                    <td>{{ "%.3f"|format(row.ragas_composite_score) }}</td>
                    <td>{{ "%.3f"|format(row.semantic_similarity) }}</td>
                    <td>{{ "✓" if row.overall_pass else "✗" }}</td>
                    <td>{{ "Yes" if row.needs_human_feedback else "No" }}</td>
                </tr>
            {% endfor %}
            </tbody>
        </table>
    </div>
</body>
</html>
        """
        
        # Render technical template
        template = Template(tech_template)
        
        # Ensure sample results has required columns for template
        sample_df = results_df.head(10).copy()
        required_cols = {
            'evaluation_id': 'eval_id',
            'overall_score': 0.0,
            'contextual_total_score': 0.0,
            'ragas_composite_score': 0.0,
            'semantic_similarity': 0.0,
            'overall_pass': False,
            'needs_human_feedback': False
        }
        
        for col, default_val in required_cols.items():
            if col not in sample_df.columns:
                if col == 'evaluation_id':
                    sample_df[col] = [f'eval_{i}' for i in range(len(sample_df))]
                else:
                    sample_df[col] = default_val
        
        html_content = template.render(
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total_samples=len(results_df),
            score_distributions=stats_analysis['distributions'],
            performance_quartiles=stats_analysis['quartiles'],
            correlation_matrix=correlation_analysis['matrix'],
            correlation_insights=correlation_analysis['insights'],
            contextual_threshold=0.6,  # From config
            ragas_threshold=0.7,  # From config
            semantic_threshold=0.6,  # From config
            threshold_recommendations=self._analyze_thresholds(results_df),
            sample_results=sample_df
        )
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Technical report saved to {output_file}")
    
    def _generate_excel_report(self, results_df: pd.DataFrame, output_file: Path):
        """Generate detailed Excel report with multiple sheets"""
        self.logger.info("Generating Excel report")
        
        # Debug: Log DataFrame info
        self.logger.debug(f"Excel generation - DataFrame shape: {results_df.shape}")
        self.logger.debug(f"Excel generation - DataFrame columns: {list(results_df.columns)}")
        
        # Debug: Check numeric columns
        numeric_columns = results_df.select_dtypes(include=[np.number]).columns
        self.logger.debug(f"Excel generation - Numeric columns: {list(numeric_columns)}")
        self.logger.debug(f"Excel generation - DataFrame dtypes: {dict(results_df.dtypes)}")
        
        try:
            with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                # Main results sheet
                results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
                
                # Summary statistics sheet - with safe handling
                try:
                    # Get only numeric columns for statistics
                    numeric_columns = results_df.select_dtypes(include=[np.number]).columns
                    if len(numeric_columns) > 0:
                        summary_stats = results_df[numeric_columns].describe()
                        summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
                        self.logger.info(f"Generated statistics for {len(numeric_columns)} numeric columns")
                    else:
                        # Create placeholder statistics sheet
                        placeholder_stats = pd.DataFrame({
                            'info': ['No numeric columns available for statistical analysis'],
                            'total_rows': [len(results_df)],
                            'total_columns': [len(results_df.columns)]
                        })
                        placeholder_stats.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                        self.logger.warning("No numeric columns found - created placeholder statistics")
                except Exception as stats_error:
                    self.logger.error(f"Error generating statistics: {stats_error}")
                    # Create error info sheet
                    error_info = pd.DataFrame({
                        'error': [f'Statistics generation failed: {stats_error}'],
                        'columns': [', '.join(results_df.columns.tolist())]
                    })
                    error_info.to_excel(writer, sheet_name='Summary_Statistics', index=False)
                
                # Pass/Fail analysis sheet (with safety checks)
                pass_fail_data = []
                
                if 'overall_pass' in results_df.columns:
                    pass_fail_data.append({
                        'Metric': 'Overall',
                        'Pass_Count': results_df['overall_pass'].sum(),
                        'Fail_Count': len(results_df) - results_df['overall_pass'].sum(),
                        'Pass_Rate': results_df['overall_pass'].mean()
                    })
                
                if 'contextual_keyword_pass' in results_df.columns:
                    pass_fail_data.append({
                        'Metric': 'Contextual Keywords',
                        'Pass_Count': results_df['contextual_keyword_pass'].sum(),
                        'Fail_Count': len(results_df) - results_df['contextual_keyword_pass'].sum(),
                        'Pass_Rate': results_df['contextual_keyword_pass'].mean()
                    })
                elif 'contextual_pass' in results_df.columns:
                    pass_fail_data.append({
                        'Metric': 'Contextual Keywords',
                        'Pass_Count': results_df['contextual_pass'].sum(),
                        'Fail_Count': len(results_df) - results_df['contextual_pass'].sum(),
                        'Pass_Rate': results_df['contextual_pass'].mean()
                    })
                
                if 'ragas_pass' in results_df.columns:
                    pass_fail_data.append({
                        'Metric': 'RAGAS',
                        'Pass_Count': results_df['ragas_pass'].sum(),
                        'Fail_Count': len(results_df) - results_df['ragas_pass'].sum(),
                        'Pass_Rate': results_df['ragas_pass'].mean()
                    })
                
                if 'semantic_pass' in results_df.columns:
                    pass_fail_data.append({
                        'Metric': 'Semantic',
                        'Pass_Count': results_df['semantic_pass'].sum(),
                        'Fail_Count': len(results_df) - results_df['semantic_pass'].sum(),
                        'Pass_Rate': results_df['semantic_pass'].mean()
                    })
                
                if pass_fail_data:
                    pass_fail_analysis = pd.DataFrame(pass_fail_data)
                    pass_fail_analysis.to_excel(writer, sheet_name='Pass_Fail_Analysis', index=False)
                
                # Source performance sheet (if available)
                if 'source_file' in results_df.columns:
                    agg_dict = {}
                    for col in ['overall_score', 'contextual_total_score', 'ragas_composite_score', 'semantic_similarity']:
                        if col in results_df.columns:
                            agg_dict[col] = ['mean', 'std', 'count']
                    
                    if 'overall_pass' in results_df.columns:
                        agg_dict['overall_pass'] = 'mean'
                    
                    if agg_dict:
                        source_perf = results_df.groupby('source_file').agg(agg_dict).round(4)
                        source_perf.to_excel(writer, sheet_name='Source_Performance')
            
            self.logger.info(f"Excel report saved to {output_file}")
            
        except Exception as e:
            self.logger.error(f"Error generating Excel report: {e}")
            # Create a simple CSV backup
            backup_file = output_file.with_suffix('.csv')
            results_df.to_csv(backup_file, index=False)
            self.logger.info(f"Excel failed, saved CSV backup to {backup_file}")
            raise
    
    def _generate_metadata_report(self, summary: Dict[str, Any], output_file: Path):
        """Generate JSON metadata report"""
        self.logger.info("Generating metadata report")
        
        metadata = {
            'report_generation_time': datetime.now().isoformat(),
            'evaluation_summary': summary,
            'pipeline_config': self.config,
            'report_version': '1.0'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Metadata report saved to {output_file}")
    
    def _generate_visualizations(self, results_df: pd.DataFrame, output_dir: Path) -> Dict[str, str]:
        """Generate visualization charts with safe column handling"""
        self.logger.info("Generating visualizations")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_files = {}
        
        # Enhanced score column mapping that includes actual RAGAS metric names
        available_score_columns = []
        score_column_mapping = {
            # Individual RAGAS metrics (actual column names from evaluation)
            'context_precision': 'RAGAS Context Precision',
            'context_recall': 'RAGAS Context Recall', 
            'faithfulness': 'RAGAS Faithfulness',
            'answer_relevancy': 'RAGAS Answer Relevancy',
            
            # Composite/calculated scores
            'contextual_total_score': 'Contextual Score',
            'ragas_composite_score': 'RAGAS Composite Score',
            'semantic_similarity': 'Semantic Score',
            'overall_score': 'Overall Score',
            
            # Legacy column names for backward compatibility
            'kw_metric': 'Keyword Metric',
            'keyword_score': 'Keyword Score',
            'weighted_average_score': 'Weighted Average'
        }
        
        for col, label in score_column_mapping.items():
            if col in results_df.columns:
                available_score_columns.append((col, label))
                self.logger.debug(f"Found score column: {col} → {label}")
        
        if not available_score_columns:
            self.logger.warning("No score columns available for visualization")
            self.logger.debug(f"Available DataFrame columns: {list(results_df.columns)}")
            return viz_files
        
        self.logger.info(f"Generating visualizations for {len(available_score_columns)} score columns")
        
        try:
            # 1. Score distribution plots
            num_plots = len(available_score_columns)
            cols = 2
            rows = (num_plots + 1) // 2
            
            fig, axes = plt.subplots(rows, cols, figsize=(15, 6 * rows))
            
            # Handle different subplot configurations
            if num_plots == 1:
                axes = [axes] if rows == 1 and cols == 1 else axes.flatten()
            elif rows == 1:
                axes = axes if hasattr(axes, '__len__') else [axes]
            else:
                axes = axes.flatten()
            
            # Ensure axes is always a list
            if not hasattr(axes, '__len__'):
                axes = [axes]
            
            fig.suptitle('Score Distributions', fontsize=16)
            
            colors = ['blue', 'green', 'red', 'purple']
            for i, (col, label) in enumerate(available_score_columns):
                if i < len(axes):
                    axes[i].hist(results_df[col].dropna(), bins=20, alpha=0.7, color=colors[i % len(colors)])
                    axes[i].set_title(f'{label} Distribution')
                    axes[i].set_xlabel('Score')
                    axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(available_score_columns), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            dist_file = output_dir / 'score_distributions.png'
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['score_distributions'] = str(dist_file)
            
            # 2. Pass rate comparison (if pass columns exist)
            pass_columns = []
            pass_column_mapping = {
                'contextual_keyword_pass': 'Contextual',
                'contextual_pass': 'Contextual',
                'ragas_pass': 'RAGAS',
                'semantic_pass': 'Semantic',
                'overall_pass': 'Overall'
            }
            
            for col, label in pass_column_mapping.items():
                if col in results_df.columns:
                    pass_columns.append((col, label))
                    break  # Only take the first contextual pass column
            
            # Remove duplicates while preserving order
            seen_labels = set()
            unique_pass_columns = []
            for col, label in pass_columns:
                if label not in seen_labels:
                    unique_pass_columns.append((col, label))
                    seen_labels.add(label)
            
            if unique_pass_columns:
                fig, ax = plt.subplots(1, 1, figsize=(10, 6))
                
                metrics = [label for _, label in unique_pass_columns]
                pass_rates = [results_df[col].mean() for col, _ in unique_pass_columns]
                
                bars = ax.bar(metrics, pass_rates, color=colors[:len(metrics)], alpha=0.7)
                ax.set_ylabel('Pass Rate')
                ax.set_title('Pass Rates by Evaluation Metric')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, rate in zip(bars, pass_rates):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                           f'{rate:.1%}', ha='center', va='bottom')
                
                plt.tight_layout()
                pass_rate_file = output_dir / 'pass_rates_comparison.png'
                plt.savefig(pass_rate_file, dpi=300, bbox_inches='tight')
                plt.close()
                viz_files['pass_rates'] = str(pass_rate_file)
            
            # 3. Correlation heatmap (if enough data and columns)
            if len(results_df) > 1 and len(available_score_columns) > 1:
                score_columns = [col for col, _ in available_score_columns]
                corr_data = results_df[score_columns].corr()
                
                fig, ax = plt.subplots(1, 1, figsize=(8, 6))
                sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=ax)
                ax.set_title('Score Correlation Matrix')
                
                plt.tight_layout()
                corr_file = output_dir / 'correlation_heatmap.png'
                plt.savefig(corr_file, dpi=300, bbox_inches='tight')
                plt.close()
                viz_files['correlation_heatmap'] = str(corr_file)
            
            self.logger.info(f"Generated {len(viz_files)} visualizations")
            
        except Exception as e:
            self.logger.error(f"Error generating visualizations: {e}")
            import traceback
            traceback.print_exc()
        
        return viz_files
    
    def _generate_summary_csv(self, summary: Dict[str, Any], output_file: Path):
        """Generate summary statistics CSV"""
        self.logger.info("Generating summary CSV")
        
        # Flatten summary statistics for CSV
        rows = []
        
        # Overall statistics (with safety checks)
        overall_stats = summary.get('overall_statistics', {})
        for key, value in overall_stats.items():
            rows.append({
                'Category': 'Overall Statistics',
                'Metric': key,
                'Value': value
            })
        
        # Score statistics (with safety checks)
        score_stats = summary.get('score_statistics', {})
        for score_type, stats in score_stats.items():
            if isinstance(stats, dict):
                for stat_name, stat_value in stats.items():
                    rows.append({
                        'Category': f'{score_type} Statistics',
                        'Metric': stat_name,
                        'Value': stat_value
                    })
        
        # Human feedback statistics (with safety checks)
        feedback_stats = summary.get('human_feedback_statistics', {})
        for key, value in feedback_stats.items():
            rows.append({
                'Category': 'Human Feedback Statistics',
                'Metric': key,
                'Value': value
            })
        
        # Evaluation metadata (with safety checks)
        eval_metadata = summary.get('evaluation_metadata', {})
        for key, value in eval_metadata.items():
            rows.append({
                'Category': 'Evaluation Metadata',
                'Metric': key,
                'Value': value
            })
        
        # If no data found, add basic info
        if not rows:
            rows = [
                {'Category': 'Summary', 'Metric': 'run_id', 'Value': summary.get('run_id', 'unknown')},
                {'Category': 'Summary', 'Metric': 'timestamp', 'Value': summary.get('timestamp', 'unknown')},
                {'Category': 'Summary', 'Metric': 'status', 'Value': summary.get('status', 'unknown')}
            ]
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Summary CSV saved to {output_file}")
    
    def _generate_recommendations(self, results_df: pd.DataFrame, 
                                summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        # Safe helper function
        def safe_get_nested(d, keys, default=0.0):
            try:
                for key in keys:
                    d = d[key]
                return d
            except (KeyError, TypeError):
                return default
        
        # Get statistics safely
        overall_pass_rate = safe_get_nested(summary, ['overall_statistics', 'overall_pass_rate'], 0.0)
        contextual_pass_rate = safe_get_nested(summary, ['overall_statistics', 'contextual_pass_rate'], 0.0)
        ragas_pass_rate = safe_get_nested(summary, ['overall_statistics', 'ragas_pass_rate'], 0.0)
        semantic_pass_rate = safe_get_nested(summary, ['overall_statistics', 'semantic_pass_rate'], 0.0)
        
        # Overall performance recommendations
        if overall_pass_rate < 0.5:
            recommendations.append({
                'title': 'Critical: Overall Performance Below 50%',
                'description': 'The RAG system requires immediate attention. Consider reviewing document chunking strategy, embedding models, and retrieval mechanisms.',
                'priority': 'HIGH'
            })
        elif overall_pass_rate < 0.7:
            recommendations.append({
                'title': 'Moderate: Performance Improvement Needed',
                'description': 'The RAG system shows moderate performance. Focus on the lowest-performing evaluation metrics for targeted improvements.',
                'priority': 'MEDIUM'
            })
        
        # Contextual keyword recommendations
        if contextual_pass_rate < 0.6:
            recommendations.append({
                'title': 'Improve Domain-Specific Knowledge',
                'description': 'Low contextual keyword scores suggest the RAG system struggles with domain-specific terminology. Consider fine-tuning embeddings or improving document preprocessing.',
                'priority': 'HIGH'
            })
        
        # RAGAS recommendations
        if ragas_pass_rate < 0.6:
            recommendations.append({
                'title': 'Enhance Retrieval Quality',
                'description': 'Low RAGAS scores indicate issues with faithfulness, relevancy, or context precision. Review retrieval strategy and chunk size optimization.',
                'priority': 'HIGH'
            })
        
        # Semantic recommendations
        if semantic_pass_rate < 0.6:
            recommendations.append({
                'title': 'Improve Answer Generation',
                'description': 'Low semantic similarity suggests the generation model may not be producing sufficiently relevant answers. Consider prompt engineering or model fine-tuning.',
                'priority': 'MEDIUM'
            })
        
        # Human feedback recommendations
        feedback_ratio = safe_get_nested(summary, ['human_feedback_statistics', 'feedback_needed_ratio'], 0.0)
        if feedback_ratio > 0.3:
            recommendations.append({
                'title': 'High Uncertainty Detected',
                'description': f'{feedback_ratio:.1%} of responses require human feedback, indicating high system uncertainty. Consider additional training data or threshold adjustments.',
                'priority': 'MEDIUM'
            })
        
        # If no specific issues found, add general recommendations
        if not recommendations:
            recommendations.append({
                'title': 'System Performance Looks Good',
                'description': 'The RAG system is performing well across all metrics. Consider monitoring performance over time and gradually increasing evaluation complexity.',
                'priority': 'LOW'
            })
        
        return recommendations
    
    def _perform_statistical_analysis(self, results_df: pd.DataFrame) -> Dict[str, str]:
        """Perform detailed statistical analysis with safe column handling"""
        # Find available score columns
        potential_score_columns = [
            'contextual_total_score', 'ragas_composite_score', 
            'semantic_similarity', 'overall_score'
        ]
        
        available_score_columns = [col for col in potential_score_columns if col in results_df.columns]
        
        if not available_score_columns:
            return {
                'distributions': 'No score columns available for analysis',
                'quartiles': 'No score columns available for analysis'
            }
        
        # Distribution analysis
        distributions = results_df[available_score_columns].describe().round(4)
        
        # Quartile analysis
        quartiles = results_df[available_score_columns].quantile([0.25, 0.5, 0.75]).round(4)
        
        return {
            'distributions': distributions.to_string(),
            'quartiles': quartiles.to_string()
        }
    
    def _analyze_metric_correlations(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different metrics with safe column handling"""
        # Find available score columns
        potential_score_columns = [
            'contextual_total_score', 'ragas_composite_score', 
            'semantic_similarity', 'overall_score'
        ]
        
        available_score_columns = [col for col in potential_score_columns if col in results_df.columns]
        
        if len(available_score_columns) < 2:
            return {
                'matrix': 'Not enough score columns for correlation analysis',
                'insights': ['Insufficient data for correlation analysis']
            }
        
        # Calculate correlation matrix
        corr_matrix = results_df[available_score_columns].corr().round(3)
        
        # Generate insights
        insights = []
        
        # Find strong correlations (>0.7)
        for i in range(len(available_score_columns)):
            for j in range(i+1, len(available_score_columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strength = "strong positive" if corr_val > 0 else "strong negative"
                    insights.append(f"{strength.title()} correlation ({corr_val:.3f}) between {available_score_columns[i]} and {available_score_columns[j]}")
        
        # Find weak correlations (<0.3)
        for i in range(len(available_score_columns)):
            for j in range(i+1, len(available_score_columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) < 0.3:
                    insights.append(f"Weak correlation ({corr_val:.3f}) between {available_score_columns[i]} and {available_score_columns[j]} - metrics evaluate different aspects")
        
        if not insights:
            insights = ["All correlations are moderate (0.3-0.7 range)"]
        
        return {
            'matrix': corr_matrix.to_string(),
            'insights': insights
        }
    
    def _analyze_thresholds(self, results_df: pd.DataFrame) -> List[Dict[str, str]]:
        """Analyze current thresholds and provide recommendations with safe column handling"""
        recommendations = []
        
        # Analyze contextual threshold (check for available columns)
        contextual_col = None
        if 'contextual_total_score' in results_df.columns:
            contextual_col = 'contextual_total_score'
        elif 'contextual_keyword_score' in results_df.columns:
            contextual_col = 'contextual_keyword_score'
            
        if contextual_col:
            contextual_scores = results_df[contextual_col]
            optimal_contextual = contextual_scores.quantile(0.7)  # 70th percentile
            recommendations.append({
                'metric': 'Contextual Keywords',
                'recommendation': f"Consider threshold of {optimal_contextual:.3f} for 70% pass rate"
            })
        
        # Analyze RAGAS threshold
        if 'ragas_composite_score' in results_df.columns:
            ragas_scores = results_df['ragas_composite_score']
            optimal_ragas = ragas_scores.quantile(0.7)
            recommendations.append({
                'metric': 'RAGAS Composite',
                'recommendation': f"Consider threshold of {optimal_ragas:.3f} for 70% pass rate"
            })
        
        # Analyze semantic threshold
        if 'semantic_similarity' in results_df.columns:
            semantic_scores = results_df['semantic_similarity']
            optimal_semantic = semantic_scores.quantile(0.7)
            recommendations.append({
                'metric': 'Semantic Similarity',
                'recommendation': f"Consider threshold of {optimal_semantic:.3f} for 70% pass rate"
            })
        
        if not recommendations:
            recommendations.append({
                'metric': 'General',
                'recommendation': 'No score columns available for threshold analysis'
            })
        
        return recommendations
    
    def _generate_performance_report(self, performance_data: Dict[str, Any], 
                                   output_file: Path, evaluation_summary: Dict[str, Any]):
        """Generate detailed performance analysis report"""
        self.logger.info("Generating performance analysis report")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Performance Analysis Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .metric-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #007bff; }}
                .timing-table {{ width: 100%; border-collapse: collapse; }}
                .timing-table th, .timing-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .timing-table th {{ background-color: #f2f2f2; }}
                .performance-highlight {{ color: #28a745; font-weight: bold; }}
                .performance-warning {{ color: #ffc107; font-weight: bold; }}
                .performance-danger {{ color: #dc3545; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 Performance Analysis Report</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>⏱️ Response Time Analysis</h2>
                {self._format_response_time_analysis(performance_data)}
            </div>
            
            <div class="section">
                <h2>🏗️ Pipeline Stage Performance</h2>
                {self._format_stage_performance(performance_data)}
            </div>
            
            <div class="section">
                <h2>📊 Metric Computation Times</h2>
                {self._format_metric_performance(performance_data)}
            </div>
            
            <div class="section">
                <h2>🧠 Memory Usage Analysis</h2>
                {self._format_memory_analysis(performance_data)}
            </div>
            
            <div class="section">
                <h2>📈 Performance Summary</h2>
                {self._format_performance_summary(performance_data)}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Performance report saved to {output_file}")
    
    def _generate_composition_report(self, composition_data: Dict[str, Any], 
                                   output_file: Path, evaluation_summary: Dict[str, Any]):
        """Generate composition elements analysis report"""
        self.logger.info("Generating composition analysis report")
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Composition Elements Analysis</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #8e44ad; color: white; padding: 20px; margin-bottom: 30px; }}
                .section {{ margin-bottom: 30px; }}
                .composition-card {{ background-color: #f8f9fa; padding: 15px; margin: 10px 0; border-left: 4px solid #8e44ad; }}
                .distribution-table {{ width: 100%; border-collapse: collapse; }}
                .distribution-table th, .distribution-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                .distribution-table th {{ background-color: #f2f2f2; }}
                .persona-highlight {{ color: #e74c3c; font-weight: bold; }}
                .style-highlight {{ color: #3498db; font-weight: bold; }}
                .node-highlight {{ color: #2ecc71; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎭 Composition Elements Analysis</h1>
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>👤 Persona Distribution</h2>
                {self._format_persona_analysis(composition_data)}
            </div>
            
            <div class="section">
                <h2>📝 Query Style & Length Analysis</h2>
                {self._format_query_style_analysis(composition_data)}
            </div>
            
            <div class="section">
                <h2>🔗 Knowledge Graph Usage</h2>
                {self._format_kg_usage_analysis(composition_data)}
            </div>
            
            <div class="section">
                <h2>🎯 Scenario Type Distribution</h2>
                {self._format_scenario_analysis(composition_data)}
            </div>
            
            <div class="section">
                <h2>📊 Composition Summary</h2>
                {self._format_composition_summary(composition_data)}
            </div>
        </body>
        </html>
        """
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Composition report saved to {output_file}")
    
    def _generate_parameters_report(self, final_parameters: Dict[str, Any], output_file: Path):
        """Generate final parameters and fallback usage report"""
        self.logger.info("Generating final parameters report")
        
        # Add timestamp
        final_parameters['report_metadata'] = {
            'generated_at': datetime.now().isoformat(),
            'report_type': 'final_parameters_and_fallbacks'
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_parameters, f, indent=2, default=str)
        
        self.logger.info(f"Parameters report saved to {output_file}")
    
    def _format_response_time_analysis(self, performance_data: Dict[str, Any]) -> str:
        """Format RAG and LLM response time analysis"""
        if 'rag_response_performance' not in performance_data:
            return "<p>No RAG response time data available</p>"
        
        rag_perf = performance_data['rag_response_performance']
        llm_perf = performance_data.get('llm_response_performance', {})
        
        return f"""
        <div class="metric-card">
            <h3>🔄 RAG System Response Times</h3>
            <table class="timing-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Requests</td><td>{rag_perf.get('total_requests', 0)}</td></tr>
                <tr><td>Average Response Time</td><td>{rag_perf.get('average_response_time_seconds', 0):.3f}s</td></tr>
                <tr><td>Min Response Time</td><td>{rag_perf.get('min_response_time_seconds', 0):.3f}s</td></tr>
                <tr><td>Max Response Time</td><td>{rag_perf.get('max_response_time_seconds', 0):.3f}s</td></tr>
                <tr><td>95th Percentile</td><td>{rag_perf.get('p95_response_time_seconds', 0):.3f}s</td></tr>
                <tr><td>99th Percentile</td><td>{rag_perf.get('p99_response_time_seconds', 0):.3f}s</td></tr>
            </table>
        </div>
        
        <div class="metric-card">
            <h3>🤖 LLM Response Times</h3>
            <table class="timing-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Total Requests</td><td>{llm_perf.get('total_requests', 0)}</td></tr>
                <tr><td>Average Response Time</td><td>{llm_perf.get('average_response_time_seconds', 0):.3f}s</td></tr>
                <tr><td>Min Response Time</td><td>{llm_perf.get('min_response_time_seconds', 0):.3f}s</td></tr>
                <tr><td>Max Response Time</td><td>{llm_perf.get('max_response_time_seconds', 0):.3f}s</td></tr>
                <tr><td>95th Percentile</td><td>{llm_perf.get('p95_response_time_seconds', 0):.3f}s</td></tr>
                <tr><td>99th Percentile</td><td>{llm_perf.get('p99_response_time_seconds', 0):.3f}s</td></tr>
            </table>
        </div>
        """
    
    def _format_stage_performance(self, performance_data: Dict[str, Any]) -> str:
        """Format pipeline stage performance data"""
        if 'stage_performance' not in performance_data:
            return "<p>No stage performance data available</p>"
        
        stage_perf = performance_data['stage_performance']
        
        table_rows = ""
        for stage_name, metrics in stage_perf.items():
            table_rows += f"""
            <tr>
                <td>{stage_name}</td>
                <td>{metrics.get('execution_count', 0)}</td>
                <td>{metrics.get('total_duration_seconds', 0):.3f}s</td>
                <td>{metrics.get('average_duration_seconds', 0):.3f}s</td>
                <td>{metrics.get('min_duration_seconds', 0):.3f}s</td>
                <td>{metrics.get('max_duration_seconds', 0):.3f}s</td>
            </tr>
            """
        
        return f"""
        <div class="metric-card">
            <table class="timing-table">
                <tr>
                    <th>Stage</th>
                    <th>Executions</th>
                    <th>Total Time</th>
                    <th>Average Time</th>
                    <th>Min Time</th>
                    <th>Max Time</th>
                </tr>
                {table_rows}
            </table>
        </div>
        """
    
    def _format_metric_performance(self, performance_data: Dict[str, Any]) -> str:
        """Format metric computation performance data"""
        if 'metric_computation_performance' not in performance_data:
            return "<p>No metric computation performance data available</p>"
        
        metric_perf = performance_data['metric_computation_performance']
        
        table_rows = ""
        for metric_name, metrics in metric_perf.items():
            table_rows += f"""
            <tr>
                <td>{metric_name}</td>
                <td>{metrics.get('computation_count', 0)}</td>
                <td>{metrics.get('total_computation_time_seconds', 0):.3f}s</td>
                <td>{metrics.get('average_computation_time_seconds', 0):.3f}s</td>
                <td>{metrics.get('min_computation_time_seconds', 0):.3f}s</td>
                <td>{metrics.get('max_computation_time_seconds', 0):.3f}s</td>
            </tr>
            """
        
        return f"""
        <div class="metric-card">
            <table class="timing-table">
                <tr>
                    <th>Metric</th>
                    <th>Computations</th>
                    <th>Total Time</th>
                    <th>Average Time</th>
                    <th>Min Time</th>
                    <th>Max Time</th>
                </tr>
                {table_rows}
            </table>
        </div>
        """
    
    def _format_memory_analysis(self, performance_data: Dict[str, Any]) -> str:
        """Format memory usage analysis"""
        if 'memory_usage' not in performance_data:
            return "<p>No memory usage data available</p>"
        
        memory_data = performance_data['memory_usage']
        
        return f"""
        <div class="metric-card">
            <table class="timing-table">
                <tr><th>Metric</th><th>Value</th></tr>
                <tr><td>Peak Memory Usage</td><td>{memory_data.get('peak_memory_usage_mb', 0):.2f} MB</td></tr>
                <tr><td>Min Memory Usage</td><td>{memory_data.get('min_memory_usage_mb', 0):.2f} MB</td></tr>
                <tr><td>Average Memory Usage</td><td>{memory_data.get('average_memory_usage_mb', 0):.2f} MB</td></tr>
                <tr><td>Final Memory Usage</td><td>{memory_data.get('final_memory_usage_mb', 0):.2f} MB</td></tr>
                <tr><td>Memory Snapshots</td><td>{memory_data.get('memory_snapshots_count', 0)}</td></tr>
            </table>
        </div>
        """
    
    def _format_performance_summary(self, performance_data: Dict[str, Any]) -> str:
        """Format performance summary"""
        return f"""
        <div class="metric-card">
            <h3>📈 Performance Insights</h3>
            <ul>
                <li><strong>Pipeline Tracking:</strong> {performance_data.get('performance_tracking', {}).get('enabled', False)}</li>
                <li><strong>Data Collection Time:</strong> {performance_data.get('performance_tracking', {}).get('collection_time', 'Unknown')}</li>
                <li><strong>Components Tracked:</strong> {len(performance_data.get('stage_performance', {}))}</li>
                <li><strong>Metrics Tracked:</strong> {len(performance_data.get('metric_computation_performance', {}))}</li>
            </ul>
        </div>
        """
    
    def _format_persona_analysis(self, composition_data: Dict[str, Any]) -> str:
        """Format persona distribution analysis"""
        stats = composition_data.get('composition_statistics', {})
        persona_dist = stats.get('distributions', {}).get('persona_distribution', {})
        
        if not persona_dist:
            return "<p>No persona distribution data available</p>"
        
        table_rows = ""
        for persona, count in persona_dist.items():
            percentage = (count / stats.get('overview', {}).get('total_scenarios_generated', 1)) * 100
            table_rows += f"""
            <tr>
                <td class="persona-highlight">{persona}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """
        
        return f"""
        <div class="composition-card">
            <table class="distribution-table">
                <tr><th>Persona</th><th>Usage Count</th><th>Percentage</th></tr>
                {table_rows}
            </table>
        </div>
        """
    
    def _format_query_style_analysis(self, composition_data: Dict[str, Any]) -> str:
        """Format query style and length analysis"""
        stats = composition_data.get('composition_statistics', {})
        style_dist = stats.get('distributions', {}).get('query_style_distribution', {})
        length_dist = stats.get('distributions', {}).get('query_length_distribution', {})
        
        style_rows = ""
        for style, count in style_dist.items():
            percentage = (count / stats.get('overview', {}).get('total_scenarios_generated', 1)) * 100
            style_rows += f"<tr><td class='style-highlight'>{style}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        length_rows = ""
        for length, count in length_dist.items():
            percentage = (count / stats.get('overview', {}).get('total_scenarios_generated', 1)) * 100
            length_rows += f"<tr><td class='style-highlight'>{length}</td><td>{count}</td><td>{percentage:.1f}%</td></tr>"
        
        return f"""
        <div class="composition-card">
            <h3>Query Styles</h3>
            <table class="distribution-table">
                <tr><th>Style</th><th>Count</th><th>Percentage</th></tr>
                {style_rows}
            </table>
        </div>
        
        <div class="composition-card">
            <h3>Query Lengths</h3>
            <table class="distribution-table">
                <tr><th>Length</th><th>Count</th><th>Percentage</th></tr>
                {length_rows}
            </table>
        </div>
        """
    
    def _format_kg_usage_analysis(self, composition_data: Dict[str, Any]) -> str:
        """Format knowledge graph usage analysis"""
        stats = composition_data.get('composition_statistics', {})
        usage_patterns = stats.get('usage_patterns', {})
        
        most_used_nodes = usage_patterns.get('most_used_nodes', [])[:5]
        most_used_rels = usage_patterns.get('most_used_relationships', [])[:5]
        
        node_rows = ""
        for node_id, count in most_used_nodes:
            node_rows += f"<tr><td class='node-highlight'>{node_id[:50]}...</td><td>{count}</td></tr>"
        
        rel_rows = ""
        for rel_type, count in most_used_rels:
            rel_rows += f"<tr><td class='node-highlight'>{rel_type}</td><td>{count}</td></tr>"
        
        return f"""
        <div class="composition-card">
            <h3>Most Used Nodes</h3>
            <table class="distribution-table">
                <tr><th>Node ID</th><th>Usage Count</th></tr>
                {node_rows}
            </table>
        </div>
        
        <div class="composition-card">
            <h3>Most Used Relationships</h3>
            <table class="distribution-table">
                <tr><th>Relationship Type</th><th>Usage Count</th></tr>
                {rel_rows}
            </table>
        </div>
        """
    
    def _format_scenario_analysis(self, composition_data: Dict[str, Any]) -> str:
        """Format scenario type analysis"""
        stats = composition_data.get('composition_statistics', {})
        scenario_dist = stats.get('distributions', {}).get('scenario_type_distribution', {})
        
        if not scenario_dist:
            return "<p>No scenario type data available</p>"
        
        table_rows = ""
        total_scenarios = stats.get('overview', {}).get('total_scenarios_generated', 1)
        for scenario_type, count in scenario_dist.items():
            percentage = (count / total_scenarios) * 100
            table_rows += f"""
            <tr>
                <td>{scenario_type}</td>
                <td>{count}</td>
                <td>{percentage:.1f}%</td>
            </tr>
            """
        
        return f"""
        <div class="composition-card">
            <table class="distribution-table">
                <tr><th>Scenario Type</th><th>Count</th><th>Percentage</th></tr>
                {table_rows}
            </table>
        </div>
        """
    
    def _format_composition_summary(self, composition_data: Dict[str, Any]) -> str:
        """Format composition summary"""
        overview = composition_data.get('composition_statistics', {}).get('overview', {})
        
        return f"""
        <div class="composition-card">
            <h3>📊 Composition Overview</h3>
            <ul>
                <li><strong>Total Scenarios Generated:</strong> {overview.get('total_scenarios_generated', 0)}</li>
                <li><strong>Unique Personas Used:</strong> {overview.get('unique_personas_used', 0)}</li>
                <li><strong>Unique Query Styles:</strong> {overview.get('unique_query_styles_used', 0)}</li>
                <li><strong>Unique Query Lengths:</strong> {overview.get('unique_query_lengths_used', 0)}</li>
                <li><strong>Nodes Involved:</strong> {overview.get('unique_nodes_involved', 0)}</li>
                <li><strong>Relationships Used:</strong> {overview.get('unique_relationships_used', 0)}</li>
            </ul>
        </div>
        """
