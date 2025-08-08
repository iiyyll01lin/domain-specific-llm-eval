#!/usr/bin/env python3
"""
Keyword Metadata Manager

Manages saving and loading of keyword extraction metadata to separate files
for comprehensive tracking and analysis.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class KeywordMetadataManager:
    """
    Manager for keyword extraction metadata persistence and analysis
    """
    
    def __init__(self, output_dir: Path, run_id: str):
        self.output_dir = Path(output_dir)
        self.run_id = run_id
        self.metadata_dir = self.output_dir / "metadata"
        self.metadata_dir.mkdir(exist_ok=True)
        
    def save_keyword_metadata(self, keyword_metadata_list: List[Dict[str, Any]]) -> str:
        """
        Save comprehensive keyword extraction metadata to separate file
        
        Args:
            keyword_metadata_list: List of metadata for each testset sample
            
        Returns:
            Path to saved metadata file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metadata_file = self.metadata_dir / f"keyword_extraction_metadata_{timestamp}_{self.run_id}.json"
        
        try:
            # Create comprehensive metadata structure
            keyword_metadata = {
                "run_info": {
                    "run_id": self.run_id,
                    "timestamp": datetime.now().isoformat(),
                    "extraction_method": "enhanced_hybrid",
                    "total_samples": len(keyword_metadata_list)
                },
                "extraction_config": self._extract_config_from_samples(keyword_metadata_list),
                "samples": keyword_metadata_list,
                "summary_statistics": self._generate_keyword_extraction_stats(keyword_metadata_list)
            }
            
            # Save to JSON file with proper encoding
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(keyword_metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"💾 Saved keyword extraction metadata to: {metadata_file}")
            logger.info(f"📊 Metadata contains {len(keyword_metadata_list)} samples with full extraction details")
            
            return str(metadata_file)
            
        except Exception as e:
            logger.error(f"❌ Failed to save keyword metadata: {e}")
            return ""
    
    def _extract_config_from_samples(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Extract configuration info from sample metadata"""
        if not metadata_list:
            return {}
            
        # Get config from first sample (should be consistent across all)
        sample_config = metadata_list[0].get('extraction_config', {})
        
        return {
            "source_weights": sample_config.get('source_weights', {}),
            "models_available": sample_config.get('models_used', {}),
            "language_detection": True,
            "domain_validation": True,
            "hybrid_pipeline": True
        }
    
    def _generate_keyword_extraction_stats(self, metadata_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics for keyword extraction"""
        if not metadata_list:
            return {}
            
        stats = {
            "language_distribution": {"chinese": 0, "english": 0, "multilingual": 0, "mixed": 0},
            "extraction_methods": {"keybert": 0, "spacy": 0, "yake": 0},
            "source_utilization": {"user_query": 0, "reference_contexts": 0, "reference_answer": 0},
            "average_keywords_per_sample": 0,
            "total_unique_keywords": set(),
            "extraction_success_rate": 0,
            "language_percentages": {"chinese_avg": 0, "english_avg": 0},
            "method_effectiveness": {},
            "post_processing_stats": {
                "total_deduplication_stats": {
                    "exact_duplicates_removed": 0,
                    "substring_duplicates_removed": 0,
                    "linguistic_variations_removed": 0,
                    "semantic_duplicates_removed": 0,
                    "total_removed": 0
                },
                "semantic_deduplication_details": {
                    "samples_with_semantic_dedup": 0,
                    "total_semantic_pairs_compared": 0,
                    "semantic_methods_used": [],
                    "models_used": [],
                    "avg_similarity_threshold": 0.0,
                    "duplicate_pairs_found": []
                },
                "filter_effectiveness": {
                    "avg_score_filter_removal": 0,
                    "avg_deduplication_removal": 0,
                    "avg_final_selection_rate": 0
                },
                "processing_stages": []
            }
        }
        
        successful_extractions = 0
        total_chinese_pct = 0
        total_english_pct = 0
        method_keyword_counts = {"keybert": 0, "spacy": 0, "yake": 0}
        
        # Post-processing aggregation variables
        total_score_filter_removals = 0
        total_deduplication_removals = 0
        total_original_counts = 0
        
        # Semantic deduplication tracking
        semantic_methods_used = set()
        semantic_models_used = set()
        semantic_thresholds = []
        semantic_pairs_compared = 0
        semantic_duplicate_pairs = []
        samples_with_semantic_dedup = 0
        
        for sample_metadata in metadata_list:
            # Check if extraction was successful
            if not sample_metadata.get('extraction_failed', False) and sample_metadata.get('keywords'):
                successful_extractions += 1
                
                # Collect post-processing statistics
                post_process_meta = sample_metadata.get('post_process_metadata', {})
                if post_process_meta:
                    # Aggregate deduplication stats
                    dedup_stats = post_process_meta.get('deduplication_stats', {})
                    stats['post_processing_stats']['total_deduplication_stats']['exact_duplicates_removed'] += dedup_stats.get('exact_duplicates_removed', 0)
                    stats['post_processing_stats']['total_deduplication_stats']['substring_duplicates_removed'] += dedup_stats.get('substring_duplicates_removed', 0)
                    stats['post_processing_stats']['total_deduplication_stats']['linguistic_variations_removed'] += dedup_stats.get('linguistic_variations_removed', 0)
                    stats['post_processing_stats']['total_deduplication_stats']['semantic_duplicates_removed'] += dedup_stats.get('semantic_duplicates_removed', 0)
                    stats['post_processing_stats']['total_deduplication_stats']['total_removed'] += dedup_stats.get('total_removed', 0)
                    
                    # Collect semantic deduplication details
                    semantic_meta = dedup_stats.get('semantic_deduplication_metadata', {})
                    if semantic_meta:
                        samples_with_semantic_dedup += 1
                        
                        # Collect method and model information
                        method_used = semantic_meta.get('method_used', '')
                        if method_used:
                            semantic_methods_used.add(method_used)
                            
                        model_used = semantic_meta.get('model_used', '')
                        if model_used:
                            semantic_models_used.add(model_used)
                            
                        # Collect threshold and pairs compared
                        threshold = semantic_meta.get('similarity_threshold', 0)
                        if threshold > 0:
                            semantic_thresholds.append(threshold)
                            
                        pairs_compared = semantic_meta.get('pairs_compared', 0)
                        semantic_pairs_compared += pairs_compared
                        
                        # Collect duplicate pairs found
                        duplicate_pairs = semantic_meta.get('duplicate_pairs_found', [])
                        semantic_duplicate_pairs.extend(duplicate_pairs)
                    
                    # Track filter effectiveness
                    original_count = post_process_meta.get('original_count', 0)
                    after_score_filter = post_process_meta.get('after_score_filter', 0)
                    final_count = post_process_meta.get('final_count', 0)
                    
                    if original_count > 0:
                        total_original_counts += original_count
                        total_score_filter_removals += (original_count - after_score_filter)
                        total_deduplication_removals += dedup_stats.get('total_removed', 0)
            # Check if extraction was successful
            if not sample_metadata.get('extraction_failed', False) and sample_metadata.get('keywords'):
                successful_extractions += 1
                
                # Language distribution
                lang_dist = sample_metadata.get('language_distribution', {})
                if lang_dist.get('mixed_content', False):
                    stats['language_distribution']['mixed'] += 1
                elif lang_dist.get('chinese_percentage', 0) > 50:
                    stats['language_distribution']['chinese'] += 1
                elif lang_dist.get('english_percentage', 0) > 50:
                    stats['language_distribution']['english'] += 1
                else:
                    stats['language_distribution']['multilingual'] += 1
                
                # Accumulate language percentages for averaging
                total_chinese_pct += lang_dist.get('chinese_percentage', 0)
                total_english_pct += lang_dist.get('english_percentage', 0)
                
                # Extraction methods from metadata
                extraction_meta = sample_metadata.get('extraction_metadata', {})
                methods_used = extraction_meta.get('extraction_methods_used', [])
                for method in methods_used:
                    method_clean = method.lower().split('_')[0]  # keybert_enhanced -> keybert
                    if method_clean in stats['extraction_methods']:
                        stats['extraction_methods'][method_clean] += 1
                
                # Source utilization
                source_breakdown = sample_metadata.get('source_breakdown', {})
                for source in source_breakdown.keys():
                    if source in stats['source_utilization']:
                        stats['source_utilization'][source] += 1
                
                # Method effectiveness (count keywords per method)
                keyword_details = sample_metadata.get('keyword_details', [])
                for detail in keyword_details:
                    methods = detail.get('methods', [])
                    for method in methods:
                        method_clean = method.lower().split('_')[0]
                        if method_clean in method_keyword_counts:
                            method_keyword_counts[method_clean] += 1
                
                # Keywords
                keywords = sample_metadata.get('keywords', [])
                stats['total_unique_keywords'].update(keywords)
        
        # Calculate final statistics
        total_samples = len(metadata_list)
        if total_samples > 0:
            stats['extraction_success_rate'] = round((successful_extractions / total_samples) * 100, 2)
            stats['language_percentages']['chinese_avg'] = round(total_chinese_pct / total_samples, 2)
            stats['language_percentages']['english_avg'] = round(total_english_pct / total_samples, 2)
        
        # Convert set to count
        stats['total_unique_keywords'] = len(stats['total_unique_keywords'])
        
        # Calculate average keywords per sample
        total_keywords = sum(len(m.get('keywords', [])) for m in metadata_list)
        stats['average_keywords_per_sample'] = round(total_keywords / total_samples, 2) if total_samples > 0 else 0
        
        # Method effectiveness
        stats['method_effectiveness'] = {
            method: {
                "samples_used": stats['extraction_methods'][method],
                "keywords_generated": method_keyword_counts[method],
                "avg_keywords_per_use": round(
                    method_keyword_counts[method] / max(stats['extraction_methods'][method], 1), 2
                )
            }
            for method in ["keybert", "spacy", "yake"]
        }
        
        # Calculate post-processing effectiveness averages
        if successful_extractions > 0:
            stats['post_processing_stats']['filter_effectiveness']['avg_score_filter_removal'] = round(
                total_score_filter_removals / successful_extractions, 2
            )
            stats['post_processing_stats']['filter_effectiveness']['avg_deduplication_removal'] = round(
                total_deduplication_removals / successful_extractions, 2
            )
            if total_original_counts > 0:
                final_keywords_total = sum(len(m.get('keywords', [])) for m in metadata_list)
                stats['post_processing_stats']['filter_effectiveness']['avg_final_selection_rate'] = round(
                    (final_keywords_total / total_original_counts) * 100, 2
                )
        
        # Populate semantic deduplication summary
        stats['post_processing_stats']['semantic_deduplication_details'] = {
            'samples_with_semantic_dedup': samples_with_semantic_dedup,
            'total_semantic_pairs_compared': semantic_pairs_compared,
            'semantic_methods_used': list(semantic_methods_used),
            'models_used': list(semantic_models_used),
            'avg_similarity_threshold': round(sum(semantic_thresholds) / len(semantic_thresholds), 3) if semantic_thresholds else 0.0,
            'duplicate_pairs_found': semantic_duplicate_pairs
        }
        
        return stats
    
    def load_keyword_metadata(self, metadata_file_path: str) -> Optional[Dict[str, Any]]:
        """
        Load keyword extraction metadata from file
        
        Args:
            metadata_file_path: Path to metadata file
            
        Returns:
            Loaded metadata dict or None if failed
        """
        try:
            with open(metadata_file_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            logger.info(f"📖 Loaded keyword metadata from: {metadata_file_path}")
            return metadata
            
        except Exception as e:
            logger.error(f"❌ Failed to load keyword metadata: {e}")
            return None
    
    def get_metadata_file_path(self, timestamp: str = None) -> Path:
        """Get the expected metadata file path"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return self.metadata_dir / f"keyword_extraction_metadata_{timestamp}_{self.run_id}.json"
