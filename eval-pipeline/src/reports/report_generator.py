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
from jinja2 import Template

class ReportGenerator:
    """
    Comprehensive report generator for hybrid RAG evaluation results
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.report_config = config.get('reporting', {})
        
        # Set up visualization style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        sns.set_palette("husl")
        
        self.logger.info("ReportGenerator initialized")
    
    def generate_comprehensive_report(self, 
                                    evaluation_results: pd.DataFrame,
                                    evaluation_summary: Dict[str, Any],
                                    output_dir: Path) -> Dict[str, str]:
        """
        Generate comprehensive evaluation report with multiple formats
        
        Args:
            evaluation_results: Detailed evaluation results DataFrame
            evaluation_summary: Summary statistics and metadata
            output_dir: Directory to save reports
        
        Returns:
            Dictionary mapping report types to file paths
        """
        self.logger.info("Generating comprehensive evaluation report")
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate different report components
        report_files = {}
        
        try:
            # 1. Executive Summary (HTML)
            exec_summary_file = output_dir / "executive_summary.html"
            self._generate_executive_summary(
                evaluation_results, evaluation_summary, exec_summary_file
            )
            report_files['executive_summary'] = str(exec_summary_file)
            
            # 2. Technical Analysis Report (HTML)
            tech_report_file = output_dir / "technical_analysis.html" 
            self._generate_technical_report(
                evaluation_results, evaluation_summary, tech_report_file
            )
            report_files['technical_analysis'] = str(tech_report_file)
            
            # 3. Detailed Results (Excel)
            excel_file = output_dir / "detailed_results.xlsx"
            self._generate_excel_report(evaluation_results, excel_file)
            report_files['detailed_results'] = str(excel_file)
            
            # 4. Metadata (JSON)
            metadata_file = output_dir / "evaluation_metadata.json"
            self._generate_metadata_report(evaluation_summary, metadata_file)
            report_files['metadata'] = str(metadata_file)
            
            # 5. Visualizations
            viz_dir = output_dir / "visualizations"
            viz_files = self._generate_visualizations(evaluation_results, viz_dir)
            report_files.update(viz_files)
            
            # 6. Summary Statistics (CSV)
            stats_file = output_dir / "summary_statistics.csv"
            self._generate_summary_csv(evaluation_summary, stats_file)
            report_files['summary_statistics'] = str(stats_file)
            
            self.logger.info(f"Report generation completed. {len(report_files)} files created.")
            
        except Exception as e:
            self.logger.error(f"Error generating reports: {e}")
            raise
        
        return report_files
    
    def _generate_executive_summary(self, 
                                  results_df: pd.DataFrame,
                                  summary: Dict[str, Any],
                                  output_file: Path):
        """Generate executive summary HTML report"""
        self.logger.info("Generating executive summary")
        
        # Calculate key metrics
        total_evals = len(results_df)
        overall_pass_rate = results_df['overall_pass'].mean() * 100
        
        # Performance by source
        source_performance = {}
        if 'source_file' in results_df.columns:
            source_perf = results_df.groupby('source_file').agg({
                'overall_pass': 'mean',
                'overall_score': 'mean',
                'contextual_keyword_pass': 'mean',
                'ragas_pass': 'mean'
            }).round(3)
            source_performance = source_perf.to_dict('index')
        
        # Identify top issues
        failed_evals = results_df[results_df['overall_pass'] == False]
        common_failures = []
        
        if len(failed_evals) > 0:
            contextual_failures = len(failed_evals[failed_evals['contextual_keyword_pass'] == False])
            ragas_failures = len(failed_evals[failed_evals['ragas_pass'] == False])
            semantic_failures = len(failed_evals[failed_evals['semantic_pass'] == False])
            
            common_failures = [
                {'type': 'Contextual Keyword Failures', 'count': contextual_failures, 'percentage': (contextual_failures/len(failed_evals))*100},
                {'type': 'RAGAS Metric Failures', 'count': ragas_failures, 'percentage': (ragas_failures/len(failed_evals))*100},
                {'type': 'Semantic Similarity Failures', 'count': semantic_failures, 'percentage': (semantic_failures/len(failed_evals))*100}
            ]
        
        # Generate recommendations
        recommendations = self._generate_recommendations(results_df, summary)
        
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
            evaluation_date=summary['evaluation_metadata']['start_time'],
            total_evaluations=total_evals,
            overall_pass_rate=overall_pass_rate,
            contextual_pass_rate=summary['overall_statistics']['contextual_pass_rate'] * 100,
            avg_contextual_score=summary['score_statistics']['contextual_scores']['mean'],
            ragas_pass_rate=summary['overall_statistics']['ragas_pass_rate'] * 100,
            avg_ragas_score=summary['score_statistics']['ragas_scores']['mean'],
            semantic_pass_rate=summary['overall_statistics']['semantic_pass_rate'] * 100,
            avg_semantic_score=summary['score_statistics']['semantic_scores']['mean'],
            source_performance=source_performance,
            common_failures=common_failures,
            recommendations=recommendations,
            human_feedback_needed=summary['human_feedback_statistics']['feedback_needed_count'],
            human_feedback_ratio=summary['human_feedback_statistics']['feedback_needed_ratio'] * 100,
            generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Executive summary saved to {output_file}")
    
    def _generate_technical_report(self, 
                                 results_df: pd.DataFrame,
                                 summary: Dict[str, Any],
                                 output_file: Path):
        """Generate detailed technical analysis report"""
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
            sample_results=results_df.head(10)
        )
        
        # Write to file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Technical report saved to {output_file}")
    
    def _generate_excel_report(self, results_df: pd.DataFrame, output_file: Path):
        """Generate detailed Excel report with multiple sheets"""
        self.logger.info("Generating Excel report")
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            # Main results sheet
            results_df.to_excel(writer, sheet_name='Detailed_Results', index=False)
            
            # Summary statistics sheet
            summary_stats = results_df.describe()
            summary_stats.to_excel(writer, sheet_name='Summary_Statistics')
            
            # Pass/Fail analysis sheet
            pass_fail_analysis = pd.DataFrame({
                'Metric': ['Overall', 'Contextual Keywords', 'RAGAS', 'Semantic'],
                'Pass_Count': [
                    results_df['overall_pass'].sum(),
                    results_df['contextual_keyword_pass'].sum(),
                    results_df['ragas_pass'].sum(),
                    results_df['semantic_pass'].sum()
                ],
                'Fail_Count': [
                    len(results_df) - results_df['overall_pass'].sum(),
                    len(results_df) - results_df['contextual_keyword_pass'].sum(),
                    len(results_df) - results_df['ragas_pass'].sum(),
                    len(results_df) - results_df['semantic_pass'].sum()
                ],
                'Pass_Rate': [
                    results_df['overall_pass'].mean(),
                    results_df['contextual_keyword_pass'].mean(),
                    results_df['ragas_pass'].mean(),
                    results_df['semantic_pass'].mean()
                ]
            })
            pass_fail_analysis.to_excel(writer, sheet_name='Pass_Fail_Analysis', index=False)
            
            # Source performance sheet (if available)
            if 'source_file' in results_df.columns:
                source_perf = results_df.groupby('source_file').agg({
                    'overall_score': ['mean', 'std', 'count'],
                    'overall_pass': 'mean',
                    'contextual_total_score': 'mean',
                    'ragas_composite_score': 'mean',
                    'semantic_similarity': 'mean'
                }).round(4)
                
                source_perf.to_excel(writer, sheet_name='Source_Performance')
        
        self.logger.info(f"Excel report saved to {output_file}")
    
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
        """Generate visualization charts"""
        self.logger.info("Generating visualizations")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        viz_files = {}
        
        try:
            # 1. Score distribution plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Score Distributions', fontsize=16)
            
            # Contextual scores
            axes[0, 0].hist(results_df['contextual_total_score'], bins=20, alpha=0.7, color='blue')
            axes[0, 0].set_title('Contextual Keyword Scores')
            axes[0, 0].set_xlabel('Score')
            axes[0, 0].set_ylabel('Frequency')
            
            # RAGAS scores
            axes[0, 1].hist(results_df['ragas_composite_score'], bins=20, alpha=0.7, color='green')
            axes[0, 1].set_title('RAGAS Composite Scores')
            axes[0, 1].set_xlabel('Score')
            axes[0, 1].set_ylabel('Frequency')
            
            # Semantic scores
            axes[1, 0].hist(results_df['semantic_similarity'], bins=20, alpha=0.7, color='red')
            axes[1, 0].set_title('Semantic Similarity Scores')
            axes[1, 0].set_xlabel('Score')
            axes[1, 0].set_ylabel('Frequency')
            
            # Overall scores
            axes[1, 1].hist(results_df['overall_score'], bins=20, alpha=0.7, color='purple')
            axes[1, 1].set_title('Overall Scores')
            axes[1, 1].set_xlabel('Score')
            axes[1, 1].set_ylabel('Frequency')
            
            plt.tight_layout()
            dist_file = output_dir / 'score_distributions.png'
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            plt.close()
            viz_files['score_distributions'] = str(dist_file)
            
            # 2. Pass rate comparison
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            
            metrics = ['Contextual', 'RAGAS', 'Semantic', 'Overall']
            pass_rates = [
                results_df['contextual_keyword_pass'].mean(),
                results_df['ragas_pass'].mean(),
                results_df['semantic_pass'].mean(),
                results_df['overall_pass'].mean()
            ]
            
            bars = ax.bar(metrics, pass_rates, color=['blue', 'green', 'red', 'purple'], alpha=0.7)
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
            
            # 3. Correlation heatmap
            if len(results_df) > 1:
                score_columns = [
                    'contextual_total_score', 'ragas_composite_score', 
                    'semantic_similarity', 'overall_score'
                ]
                
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
        
        return viz_files
    
    def _generate_summary_csv(self, summary: Dict[str, Any], output_file: Path):
        """Generate summary statistics CSV"""
        self.logger.info("Generating summary CSV")
        
        # Flatten summary statistics for CSV
        rows = []
        
        # Overall statistics
        for key, value in summary['overall_statistics'].items():
            rows.append({
                'Category': 'Overall Statistics',
                'Metric': key,
                'Value': value
            })
        
        # Score statistics
        for score_type, stats in summary['score_statistics'].items():
            for stat_name, stat_value in stats.items():
                rows.append({
                    'Category': f'{score_type} Statistics',
                    'Metric': stat_name,
                    'Value': stat_value
                })
        
        # Human feedback statistics
        for key, value in summary['human_feedback_statistics'].items():
            rows.append({
                'Category': 'Human Feedback Statistics',
                'Metric': key,
                'Value': value
            })
        
        # Convert to DataFrame and save
        summary_df = pd.DataFrame(rows)
        summary_df.to_csv(output_file, index=False)
        
        self.logger.info(f"Summary CSV saved to {output_file}")
    
    def _generate_recommendations(self, results_df: pd.DataFrame, 
                                summary: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate actionable recommendations based on results"""
        recommendations = []
        
        overall_pass_rate = summary['overall_statistics']['overall_pass_rate']
        contextual_pass_rate = summary['overall_statistics']['contextual_pass_rate']
        ragas_pass_rate = summary['overall_statistics']['ragas_pass_rate']
        semantic_pass_rate = summary['overall_statistics']['semantic_pass_rate']
        
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
        feedback_ratio = summary['human_feedback_statistics']['feedback_needed_ratio']
        if feedback_ratio > 0.3:
            recommendations.append({
                'title': 'High Uncertainty Detected',
                'description': f'{feedback_ratio:.1%} of responses require human feedback, indicating high system uncertainty. Consider additional training data or threshold adjustments.',
                'priority': 'MEDIUM'
            })
        
        return recommendations
    
    def _perform_statistical_analysis(self, results_df: pd.DataFrame) -> Dict[str, str]:
        """Perform detailed statistical analysis"""
        score_columns = [
            'contextual_total_score', 'ragas_composite_score', 
            'semantic_similarity', 'overall_score'
        ]
        
        # Distribution analysis
        distributions = results_df[score_columns].describe().round(4)
        
        # Quartile analysis
        quartiles = results_df[score_columns].quantile([0.25, 0.5, 0.75]).round(4)
        
        return {
            'distributions': distributions.to_string(),
            'quartiles': quartiles.to_string()
        }
    
    def _analyze_metric_correlations(self, results_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between different metrics"""
        score_columns = [
            'contextual_total_score', 'ragas_composite_score', 
            'semantic_similarity', 'overall_score'
        ]
        
        # Calculate correlation matrix
        corr_matrix = results_df[score_columns].corr().round(3)
        
        # Generate insights
        insights = []
        
        # Find strong correlations (>0.7)
        for i in range(len(score_columns)):
            for j in range(i+1, len(score_columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.7:
                    strength = "strong positive" if corr_val > 0 else "strong negative"
                    insights.append(f"{strength.title()} correlation ({corr_val:.3f}) between {score_columns[i]} and {score_columns[j]}")
        
        # Find weak correlations (<0.3)
        for i in range(len(score_columns)):
            for j in range(i+1, len(score_columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) < 0.3:
                    insights.append(f"Weak correlation ({corr_val:.3f}) between {score_columns[i]} and {score_columns[j]} - metrics evaluate different aspects")
        
        return {
            'matrix': corr_matrix.to_string(),
            'insights': insights
        }
    
    def _analyze_thresholds(self, results_df: pd.DataFrame) -> List[Dict[str, str]]:
        """Analyze current thresholds and provide recommendations"""
        recommendations = []
        
        # Analyze contextual threshold
        contextual_scores = results_df['contextual_total_score']
        optimal_contextual = contextual_scores.quantile(0.7)  # 70th percentile
        
        recommendations.append({
            'metric': 'Contextual Keywords',
            'recommendation': f"Consider threshold of {optimal_contextual:.3f} for 70% pass rate"
        })
        
        # Analyze RAGAS threshold
        ragas_scores = results_df['ragas_composite_score']
        optimal_ragas = ragas_scores.quantile(0.7)
        
        recommendations.append({
            'metric': 'RAGAS Composite',
            'recommendation': f"Consider threshold of {optimal_ragas:.3f} for 70% pass rate"
        })
        
        # Analyze semantic threshold
        semantic_scores = results_df['semantic_similarity']
        optimal_semantic = semantic_scores.quantile(0.7)
        
        recommendations.append({
            'metric': 'Semantic Similarity',
            'recommendation': f"Consider threshold of {optimal_semantic:.3f} for 70% pass rate"
        })
        
        return recommendations
