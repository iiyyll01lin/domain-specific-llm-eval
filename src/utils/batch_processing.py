#!/usr/bin/env python3
"""
Batch Processing with Checkpointing
===================================

This module provides batch processing capabilities with checkpointing
for generating large testsets reliably.
"""

import os
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Batch processor with checkpointing for testset generation."""
    
    def __init__(self, config: Dict[str, Any], checkpoint_dir: str = "outputs/checkpoints"):
        """Initialize batch processor."""
        self.config = config
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def process_in_batches(self, csv_files: List[str], output_dir: Path, total_size: int = 1000, batch_size: int = 100) -> Dict[str, Any]:
        """Process testset generation in batches with checkpointing."""
        try:
            logger.info(f"🔄 Starting batch processing: {total_size} samples in batches of {batch_size}")
            
            session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_file = self.checkpoint_dir / f"batch_session_{session_id}.json"
            
            # Initialize or load checkpoint
            if checkpoint_file.exists():
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                logger.info(f"📁 Resuming from checkpoint: {len(checkpoint.get('completed_samples', []))} samples")
            else:
                checkpoint = {
                    'session_id': session_id,
                    'total_size': total_size,
                    'batch_size': batch_size,
                    'completed_samples': [],
                    'failed_batches': [],
                    'current_batch': 0
                }
            
            completed_samples = checkpoint.get('completed_samples', [])
            remaining = total_size - len(completed_samples)
            
            if remaining <= 0:
                logger.info("✅ All samples already completed!")
                # Save final testset
                final_file = self._save_final_testset(completed_samples, output_dir, session_id)
                return {
                    'success': True,
                    'testset_file': final_file,
                    'samples_generated': len(completed_samples),
                    'method': 'batch_processing_resumed'
                }
            
            # Process remaining batches
            num_batches = (remaining + batch_size - 1) // batch_size
            current_batch = checkpoint.get('current_batch', 0)
            
            for batch_idx in range(current_batch, num_batches):
                try:
                    logger.info(f"🔄 Processing batch {batch_idx + 1}/{num_batches}")
                    
                    # Calculate batch size for this iteration
                    actual_batch_size = min(batch_size, remaining - (batch_idx - current_batch) * batch_size)
                    
                    # Generate batch
                    batch_result = self._generate_batch(csv_files, actual_batch_size, batch_idx)
                    
                    if batch_result.get('success'):
                        batch_samples = batch_result.get('samples', [])
                        completed_samples.extend(batch_samples)
                        
                        # Update checkpoint
                        checkpoint['completed_samples'] = completed_samples
                        checkpoint['current_batch'] = batch_idx + 1
                        
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint, f, indent=2)
                        
                        logger.info(f"✅ Batch {batch_idx + 1} completed: {len(batch_samples)} samples")
                    else:
                        logger.warning(f"⚠️ Batch {batch_idx + 1} failed: {batch_result.get('error')}")
                        checkpoint['failed_batches'].append({
                            'batch_idx': batch_idx,
                            'error': batch_result.get('error'),
                            'timestamp': datetime.now().isoformat()
                        })
                        
                        # Save checkpoint even for failed batches
                        with open(checkpoint_file, 'w') as f:
                            json.dump(checkpoint, f, indent=2)
                
                except Exception as e:
                    logger.error(f"❌ Batch {batch_idx + 1} exception: {e}")
                    checkpoint['failed_batches'].append({
                        'batch_idx': batch_idx,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    })
            
            # Save final testset
            if completed_samples:
                final_file = self._save_final_testset(completed_samples, output_dir, session_id)
                
                # Clean up checkpoint
                if checkpoint_file.exists():
                    checkpoint_file.unlink()
                
                return {
                    'success': True,
                    'testset_file': final_file,
                    'samples_generated': len(completed_samples),
                    'method': 'batch_processing',
                    'session_id': session_id,
                    'failed_batches': len(checkpoint.get('failed_batches', []))
                }
            else:
                return {
                    'success': False,
                    'error': 'No samples generated in any batch'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Batch processing failed: {str(e)}"
            }
    
    def _generate_batch(self, csv_files: List[str], batch_size: int, batch_idx: int) -> Dict[str, Any]:
        """Generate a single batch of samples."""
        try:
            # Use fallback generation for batch
            from utils.fallback_testset_generation import FallbackTestsetGenerator
            
            fallback_generator = FallbackTestsetGenerator(self.config)
            temp_dir = self.checkpoint_dir / f"temp_batch_{batch_idx}"
            temp_dir.mkdir(exist_ok=True)
            
            result = fallback_generator.generate_from_csv_templates(
                csv_files, temp_dir, batch_size
            )
            
            if result.get('success'):
                # Load generated samples
                testset_file = result.get('testset_file')
                if testset_file and Path(testset_file).exists():
                    df = pd.read_csv(testset_file)
                    samples = df.to_dict('records')
                    
                    # Clean up temp file
                    Path(testset_file).unlink()
                    temp_dir.rmdir()
                    
                    return {
                        'success': True,
                        'samples': samples
                    }
            
            return {
                'success': False,
                'error': result.get('error', 'Unknown error')
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f"Batch generation failed: {str(e)}"
            }
    
    def _save_final_testset(self, samples: List[Dict[str, Any]], output_dir: Path, session_id: str) -> str:
        """Save final combined testset."""
        try:
            df = pd.DataFrame(samples)
            final_file = output_dir / f"batch_processed_testset_{session_id}.csv"
            df.to_csv(final_file, index=False)
            
            logger.info(f"💾 Final testset saved: {final_file} ({len(samples)} samples)")
            return str(final_file)
            
        except Exception as e:
            logger.error(f"Failed to save final testset: {e}")
            return ""

def create_batch_processor(config: Dict[str, Any]) -> BatchProcessor:
    """Create a batch processor instance."""
    return BatchProcessor(config)

if __name__ == "__main__":
    logger.info("🔄 Batch Processing with Checkpointing module loaded")
