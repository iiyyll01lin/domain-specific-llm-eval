#!/usr/bin/env python3
"""
Test the fixed report generation with existing data
"""

import sys
import json
import pandas as pd
from pathlib import Path

import pytest

# Add the eval-pipeline src to path
sys.path.insert(0, str(Path(__file__).parent / 'eval-pipeline' / 'src'))

def test_fixed_reports():
    """Test report generation with the fixed data formatter."""
    print("🧪 Testing Fixed Report Generation")
    print("=" * 60)
    
    # Use existing evaluation data
    run_dir = "eval-pipeline/outputs/run_20250713_144623_8c5787af"
    
    # Load the comprehensive report to get structure
    report_file = f"{run_dir}/evaluations/comprehensive_report_20250713_144716.json"
    if not Path(report_file).exists():
        pytest.skip("Historical fixed report artifact is not present in this workspace")
    
    with open(report_file, 'r') as f:
        report_data = json.load(f)
    
    # Import the components we need
    from reports.evaluation_data_formatter import EvaluationDataFormatter
    from reports.report_generator import ReportGenerator
    
    # Format the evaluation data
    formatter = EvaluationDataFormatter()
    
    # Recreate the evaluation results structure
    evaluation_results = {
        'results': {
            'contextual_keyword': report_data['detailed_results']['contextual_keyword'],
            'ragas': report_data['detailed_results']['ragas']
        },
        'total_questions': report_data['detailed_results']['contextual_keyword']['total_questions'],
        'timestamp': report_data['evaluation_summary']['timestamp']
    }
    
    testset_file = report_data['evaluation_summary']['testset_file']
    
    # Format the data with the fixed formatter
    results_df = formatter.format_comprehensive_results(evaluation_results, testset_file)
    
    print(f"✅ Formatted DataFrame:")
    print(f"   Shape: {results_df.shape}")
    print(f"   Columns: {list(results_df.columns)}")
    
    # Show actual score values
    print(f"\n📊 Score values (first 3 rows):")
    score_cols = ['contextual_total_score', 'ragas_composite_score', 'overall_score']
    for col in score_cols:
        if col in results_df.columns:
            values = results_df[col].head(3).tolist()
            print(f"   {col}: {values}")
    
    # Create test summary
    evaluation_summary = {
        'overall_statistics': {
            'total_questions': len(results_df),
            'overall_pass_rate': results_df['overall_pass'].mean() if 'overall_pass' in results_df.columns else 0.0
        },
        'score_statistics': {
            'contextual_scores': {
                'mean': results_df['contextual_total_score'].mean(),
                'std': results_df['contextual_total_score'].std()
            },
            'ragas_scores': {
                'mean': results_df['ragas_composite_score'].mean(),
                'std': results_df['ragas_composite_score'].std()
            }
        }
    }
    
    # Test report generation
    test_config = {
        'reporting': {
            'visualizations': {'enabled': True}
        }
    }
    
    report_gen = ReportGenerator(test_config)
    
    # Create test output directory
    test_output_dir = Path("outputs/fixed_reports_test")
    test_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n🔧 Generating comprehensive reports...")
    
    # Generate comprehensive report
    report_files = report_gen.generate_comprehensive_report(
        evaluation_results=results_df,
        evaluation_summary=evaluation_summary,
        output_dir=test_output_dir
    )
    
    print(f"✅ Generated {len(report_files)} report files:")
    for report_type, file_path in report_files.items():
        file_path_obj = Path(file_path)
        exists = file_path_obj.exists()
        size = file_path_obj.stat().st_size if exists else 0
        print(f"   - {report_type}: {file_path} ({'✅' if exists else '❌'} {size} bytes)")
    
    # Check specifically for visualizations
    viz_dir = test_output_dir / "visualizations"
    if viz_dir.exists():
        viz_files = list(viz_dir.glob("*.png"))
        print(f"\n🎨 Visualization files:")
        if viz_files:
            for viz_file in viz_files:
                size = viz_file.stat().st_size
                print(f"   ✅ {viz_file.name} ({size} bytes)")
        else:
            print(f"   ❌ No PNG files found in visualizations directory")
    else:
        print(f"\n❌ No visualizations directory found")
    
    assert len(report_files) > 0
    assert viz_dir.exists()
    assert len(list(viz_dir.glob("*.png"))) > 0

if __name__ == "__main__":
    success = test_fixed_reports()
    
    if success:
        print(f"\n🎉 SUCCESS!")
        print(f"✅ Fixed data formatter working correctly")
        print(f"✅ Visualizations generated successfully")
        print(f"💡 The issue was in the data extraction - now fixed!")
    else:
        print(f"\n❌ STILL ISSUES")
        print(f"🔧 Need further debugging")
