#!/usr/bin/env python3
"""
Robust Testset Generation Wrapper
=================================

This module provides a robust wrapper around RAGAS testset generation
with comprehensive error handling, fallbacks, and checkpointing.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Union

logger = logging.getLogger(__name__)

class RobustTestsetGenerationWrapper:
    """
    Robust wrapper for testset generation with multiple fallback strategies.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the robust wrapper."""
        self.config = config
        self.checkpoint_dir = Path("outputs/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Error tracking
        self.errors_encountered = []
        self.successful_batches = 0
        self.failed_batches = 0
        
        logger.info("🛡️ Robust Testset Generation Wrapper initialized")
    
    def generate_with_progressive_fallbacks(self, csv_files: List[str], output_dir: Path, target_size: int = 1000) -> Dict[str, Any]:
        """
        Generate testset using progressive fallback strategies.
        
        Args:
            csv_files: List of CSV file paths
            output_dir: Output directory
            target_size: Target number of samples
            
        Returns:
            Dictionary with generation results
        """
        logger.info(f"🎯 Starting progressive fallback generation for {target_size} samples...")
        
        results = {
            'success': False,
            'testset_file': None,
            'samples_generated': 0,
            'method_used': None,
            'errors': [],
            'fallbacks_used': []
        }
        
        # Progressive strategies from most ambitious to most conservative
        strategies = [
            ('full_ragas', lambda: self._try_full_ragas_generation(csv_files, output_dir, target_size)),
            ('reduced_ragas', lambda: self._try_reduced_ragas_generation(csv_files, output_dir, target_size // 2)),
            ('minimal_ragas', lambda: self._try_minimal_ragas_generation(csv_files, output_dir, min(10, target_size))),
            ('batch_ragas', lambda: self._try_batch_generation(csv_files, output_dir, target_size)),
            ('template_based', lambda: self._try_template_based_generation(csv_files, output_dir, target_size)),
            ('csv_conversion', lambda: self._try_csv_conversion_fallback(csv_files, output_dir, target_size))
        ]
        
        for strategy_name, strategy_func in strategies:
            try:
                logger.info(f"🔄 Trying strategy: {strategy_name}")
                strategy_result = strategy_func()
                
                if strategy_result and strategy_result.get('success'):
                    results.update(strategy_result)
                    results['method_used'] = strategy_name
                    logger.info(f"✅ Strategy {strategy_name} successful!")
                    break
                else:
                    error_msg = strategy_result.get('error', 'Unknown error') if strategy_result else 'Strategy returned None'
                    logger.warning(f"⚠️ Strategy {strategy_name} failed: {error_msg}")
                    results['fallbacks_used'].append({
                        'strategy': strategy_name,
                        'error': error_msg
                    })
                    
            except Exception as e:
                error_msg = f"Strategy {strategy_name} exception: {str(e)}"
                logger.error(error_msg)
                results['errors'].append(error_msg)
                results['fallbacks_used'].append({
                    'strategy': strategy_name,
                    'error': error_msg
                })
        
        if not results['success']:
            logger.error("❌ All testset generation strategies failed")
            results['error'] = "All generation strategies failed"
        
        return results
    
    def _try_full_ragas_generation(self, csv_files: List[str], output_dir: Path, target_size: int) -> Dict[str, Any]:
        """Try full RAGAS generation with all features."""
        try:
            # Apply all patches first
            self._apply_all_patches()
            
            # Import and initialize RAGAS generator
            from data.pure_ragas_testset_generator import PureRAGASTestsetGenerator
            
            generator = PureRAGASTestsetGenerator(self.config)
            result = generator.generate_comprehensive_testset(csv_files, output_dir)
            
            if result and result.get('success'):
                return {
                    'success': True,
                    'testset_file': result.get('testset_file'),
                    'samples_generated': result.get('samples_generated', 0),
                    'method': 'full_ragas'
                }
            else:
                return {
                    'success': False,
                    'error': result.get('error', 'Unknown RAGAS error') if result else 'RAGAS returned None'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Full RAGAS generation failed: {str(e)}"
            }
    
    def _try_batch_generation(self, csv_files: List[str], output_dir: Path, target_size: int) -> Dict[str, Any]:
        """Try batch generation with checkpointing."""
        try:
            logger.info(f"🔄 Trying batch generation: {target_size} samples in batches")
            
            batch_size = min(100, target_size // 10)  # Conservative batch size
            if batch_size < 1:
                batch_size = 1
                
            all_samples = []
            checkpoint_file = self.checkpoint_dir / f"batch_checkpoint_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            # Load existing checkpoint if available
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint_data = json.load(f)
                    all_samples = checkpoint_data.get('samples', [])
                    logger.info(f"📁 Loaded {len(all_samples)} samples from checkpoint")
            
            # Generate remaining samples in batches
            remaining = target_size - len(all_samples)
            batches_needed = (remaining + batch_size - 1) // batch_size
            
            for batch_idx in range(batches_needed):
                try:
                    logger.info(f"🔄 Processing batch {batch_idx + 1}/{batches_needed} (size: {batch_size})")
                    
                    # Try minimal RAGAS generation for this batch
                    batch_result = self._try_minimal_ragas_generation(csv_files, output_dir, batch_size)
                    
                    if batch_result and batch_result.get('success'):
                        # Load the generated samples
                        testset_file = batch_result.get('testset_file')
                        if testset_file and Path(testset_file).exists():
                            df = pd.read_csv(testset_file)
                            batch_samples = df.to_dict('records')
                            all_samples.extend(batch_samples)
                            self.successful_batches += 1
                            
                            # Save checkpoint
                            checkpoint_data = {
                                'timestamp': datetime.now().isoformat(),
                                'total_samples': len(all_samples),
                                'target_size': target_size,
                                'samples': all_samples
                            }
                            with open(checkpoint_file, 'w') as f:
                                json.dump(checkpoint_data, f, indent=2)
                            
                            logger.info(f"✅ Batch {batch_idx + 1} successful: {len(batch_samples)} samples")
                    else:
                        self.failed_batches += 1
                        logger.warning(f"⚠️ Batch {batch_idx + 1} failed")
                        
                except Exception as e:
                    self.failed_batches += 1
                    logger.error(f"❌ Batch {batch_idx + 1} exception: {e}")
                    
                # Stop early if we have enough samples
                if len(all_samples) >= target_size:
                    break
            
            # Save final testset
            if all_samples:
                final_df = pd.DataFrame(all_samples[:target_size])  # Trim to target size
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                final_file = output_dir / f"batch_generated_testset_{timestamp}.csv"
                final_df.to_csv(final_file, index=False)
                
                # Clean up checkpoint
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                
                return {
                    'success': True,
                    'testset_file': str(final_file),
                    'samples_generated': len(final_df),
                    'method': 'batch_generation',
                    'successful_batches': self.successful_batches,
                    'failed_batches': self.failed_batches
                }
            else:
                return {
                    'success': False,
                    'error': 'No samples generated in any batch'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Batch generation failed: {str(e)}"
            }
    
    def _try_csv_conversion_fallback(self, csv_files: List[str], output_dir: Path, target_size: int) -> Dict[str, Any]:
        """Fallback: Convert CSV directly to testset format."""
        try:
            logger.info(f"🔄 CSV conversion fallback for {target_size} samples")
            
            all_data = []
            for csv_file in csv_files:
                if Path(csv_file).exists():
                    df = pd.read_csv(csv_file)
                    all_data.append(df)
            
            if not all_data:
                return {'success': False, 'error': 'No valid CSV files found'}
            
            # Combine all CSV data
            combined_df = pd.concat(all_data, ignore_index=True)
            
            # Convert to testset format
            testset_data = []
            for idx, row in combined_df.iterrows():
                if idx >= target_size:
                    break
                    
                # Create question from content
                content = ""
                if 'content' in row and pd.notna(row['content']):
                    content = str(row['content'])
                elif 'display' in row and pd.notna(row['display']):
                    content = str(row['display'])
                else:
                    content = str(row.iloc[0])  # Use first non-null column
                
                # Create simple Q&A pair
                question = f"What is the information about item {idx + 1}?"
                answer = content[:500] + "..." if len(content) > 500 else content
                
                testset_data.append({
                    'user_input': question,
                    'reference_contexts': [content],
                    'reference': answer,
                    'auto_keywords': ""
                })
            
            # Save testset
            testset_df = pd.DataFrame(testset_data)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            testset_file = output_dir / f"csv_fallback_testset_{timestamp}.csv"
            testset_df.to_csv(testset_file, index=False)
            
            return {
                'success': True,
                'testset_file': str(testset_file),
                'samples_generated': len(testset_df),
                'method': 'csv_conversion_fallback'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"CSV conversion fallback failed: {str(e)}"
            }
    
    def _apply_all_patches(self):
        """Apply all available patches for robustness."""
        try:
            # Apply StringIO patches
            try:
                from utils.stringio_compatibility import patch_stringio_validation
                patch_stringio_validation()
                logger.info("✅ StringIO patches applied")
            except ImportError:
                logger.warning("⚠️ StringIO patches not available")
            
            # Apply TestsetGenerator patches
            try:
                from utils.ragas_testset_patches import patch_testset_generator
                patch_testset_generator()
                logger.info("✅ TestsetGenerator patches applied")
            except ImportError:
                logger.warning("⚠️ TestsetGenerator patches not available")
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to apply some patches: {e}")

# Example usage function
def create_robust_wrapper(config: Dict[str, Any]) -> RobustTestsetGenerationWrapper:
    """Create a robust testset generation wrapper."""
    return RobustTestsetGenerationWrapper(config)

if __name__ == "__main__":
    logger.info("🛡️ Robust Testset Generation Wrapper module loaded")
