#!/usr/bin/env python3
"""
Domain-Specific RAG Evaluation Pipeline - Main Entry Point

This script orchestrates the complete hybrid evaluation pipeline:
1. Document loading and processing
2. Testset generation with auto-keyword extraction  
3. RAG system evaluation using contextual keywords + RAGAS metrics
4. Dynamic human feedback integration
5. Comprehensive report generation

Usage:
    python run_pipeline.py --config config/pipeline_config.yaml
    python run_pipeline.py --config config/pipeline_config.yaml --mode dry-run
"""

# CRITICAL: Apply tiktoken patch IMMEDIATELY before any other imports
import sys
import os
from pathlib import Path

# Import startup hook first to patch tiktoken
try:
    import tiktoken_startup  # This applies the patch automatically
    print("✅ Tiktoken startup patch applied")
except Exception as e:
    print(f"⚠️ Could not apply tiktoken startup patch: {e}")
    
    # Fallback: Apply patch manually
    scripts_dir = Path(__file__).parent / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.append(str(scripts_dir))
    
    try:
        from tiktoken_fallback import patch_tiktoken_with_fallback
        patch_tiktoken_with_fallback()
        print("✅ Manual tiktoken patch applied")
    except Exception as e2:
        print(f"⚠️ Manual tiktoken patch also failed: {e2}")

import argparse
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, Optional

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from pipeline.orchestrator import PipelineOrchestrator
from pipeline.config_manager import ConfigManager
from pipeline.logger import setup_logging
from pipeline.utils import (
    validate_environment,
    create_output_directories,
    generate_run_id
)

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Domain-Specific RAG Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full evaluation with default config
  python run_pipeline.py
  
  # Run with custom config
  python run_pipeline.py --config config/my_config.yaml
  
  # Dry run to validate configuration
  python run_pipeline.py --mode dry-run
  
  # Generate testset only
  python run_pipeline.py --stage testset-generation
  
  # Run evaluation only (assumes testset exists)
  python run_pipeline.py --stage evaluation
  
  # Generate reports only
  python run_pipeline.py --stage reporting
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config/pipeline_config.yaml",
        help="Path to pipeline configuration file"
    )
    
    parser.add_argument(
        "--mode", "-m",
        choices=["full", "dry-run", "validate"],
        default="full",
        help="Pipeline execution mode"
    )
    
    parser.add_argument(
        "--stage", "-s",
        choices=["all", "testset-generation", "evaluation", "reporting"],
        default="all",
        help="Pipeline stage to execute"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Override output directory"
    )
    
    parser.add_argument(
        "--log-level", "-l",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite existing outputs"
    )
    
    parser.add_argument(
        "--resume",
        type=str,
        help="Resume from specific run ID"
    )
    
    return parser.parse_args()

def main():
    """Main pipeline entry point."""
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Load configuration
        config_manager = ConfigManager(args.config)
        config = config_manager.load_config()
        
        # Apply command line overrides
        if args.output_dir:
            config['output']['base_dir'] = args.output_dir
        if args.log_level:
            config['logging']['level'] = args.log_level
            
        # Generate run ID
        run_id = args.resume if args.resume else generate_run_id()
        
        # Setup logging
        logger = setup_logging(config, run_id)
        logger.info("=" * 80)
        logger.info("🚀 Starting Domain-Specific RAG Evaluation Pipeline")
        logger.info("=" * 80)
        logger.info(f"Run ID: {run_id}")
        logger.info(f"Config: {args.config}")
        logger.info(f"Mode: {args.mode}")
        logger.info(f"Stage: {args.stage}")
        
        # Validate environment
        logger.info("🔍 Validating environment...")
        validation_results = validate_environment(config)
        
        if not validation_results['success']:
            logger.error("❌ Environment validation failed:")
            for error in validation_results['errors']:
                logger.error(f"  - {error}")
            sys.exit(1)
        
        logger.info("✅ Environment validation passed")
        
        # Create output directories
        output_dirs = create_output_directories(config, run_id)
        logger.info(f"📁 Output directory: {output_dirs['base']}")
        
        # Dry run mode - validate configuration and exit
        if args.mode == "dry-run":
            logger.info("🧪 Dry run mode - validating configuration...")
            config_validation = config_manager.validate_config()
            
            if config_validation['success']:
                logger.info("✅ Configuration validation passed")
                logger.info("📋 Pipeline would execute the following steps:")
                
                if args.stage in ["all", "testset-generation"]:
                    logger.info("  1. 📄 Document loading and processing")
                    logger.info("  2. 🎯 Testset generation with auto-keyword extraction")
                
                if args.stage in ["all", "evaluation"]:
                    logger.info("  3. 🔄 RAG system evaluation")
                    logger.info("  4. ⚖️ Contextual keyword evaluation")
                    logger.info("  5. 📊 RAGAS metrics computation")
                    logger.info("  6. 🎯 Human feedback integration")
                
                if args.stage in ["all", "reporting"]:
                    logger.info("  7. 📈 Report generation")
                
                logger.info("🎉 Dry run completed successfully")
            else:
                logger.error("❌ Configuration validation failed:")
                for error in config_validation['errors']:
                    logger.error(f"  - {error}")
                sys.exit(1)
            
            return
        
        # Validate mode - check config only
        if args.mode == "validate":
            logger.info("🔍 Validation mode - checking configuration...")
            config_validation = config_manager.validate_config()
            
            if config_validation['success']:
                logger.info("✅ Configuration is valid")
            else:
                logger.error("❌ Configuration validation failed:")
                for error in config_validation['errors']:
                    logger.error(f"  - {error}")
                sys.exit(1)
            
            return
        
        # Initialize pipeline orchestrator
        logger.info("🏗️ Initializing pipeline orchestrator...")
        orchestrator = PipelineOrchestrator(
            config=config,
            run_id=run_id,
            output_dirs=output_dirs,
            force_overwrite=args.force
        )
        
        # Execute pipeline
        logger.info(f"🎬 Starting pipeline execution (stage: {args.stage})...")
        results = orchestrator.run(stage=args.stage)
        
        # Log results summary
        logger.info("=" * 80)
        logger.info("📊 PIPELINE EXECUTION SUMMARY")
        logger.info("=" * 80)
        
        if results['success']:
            logger.info("✅ Pipeline completed successfully!")
            
            # Log key metrics
            if 'testset_generation' in results:
                tg_results = results['testset_generation']
                logger.info(f"📄 Documents processed: {tg_results.get('documents_processed', 0)}")
                logger.info(f"🎯 Testsets generated: {tg_results.get('testsets_generated', 0)}")
                logger.info(f"❓ Total QA pairs: {tg_results.get('total_qa_pairs', 0)}")
            
            if 'evaluation' in results:
                eval_results = results['evaluation']
                logger.info(f"🔄 RAG queries executed: {eval_results.get('queries_executed', 0)}")
                logger.info(f"⚖️ Contextual keyword pass rate: {eval_results.get('keyword_pass_rate', 0):.1%}")
                logger.info(f"📊 Average RAGAS score: {eval_results.get('avg_ragas_score', 0):.3f}")
                logger.info(f"🎯 Human feedback requests: {eval_results.get('feedback_requests', 0)}")
            
            if 'reporting' in results:
                report_results = results['reporting']
                logger.info(f"📈 Reports generated: {len(report_results.get('reports_generated', []))}")
                logger.info(f"📁 Report directory: {report_results.get('report_directory', 'N/A')}")
        
        else:
            logger.error("❌ Pipeline execution failed!")
            if 'error' in results:
                logger.error(f"Error: {results['error']}")
        
        logger.info("=" * 80)
        logger.info(f"🏁 Pipeline finished. Run ID: {run_id}")
        logger.info("=" * 80)
        
        # Exit with appropriate code
        sys.exit(0 if results['success'] else 1)
        
    except KeyboardInterrupt:
        logger.warning("⚠️ Pipeline interrupted by user")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"💥 Unexpected error: {str(e)}")
        logger.error("Stack trace:")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    main()
