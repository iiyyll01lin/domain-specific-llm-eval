#!/usr/bin/env python3
"""
NaN Handling Utilities for RAGAS Pipeline
=========================================

This module provides utilities for handling NaN values that can occur
during RAGAS testset generation and evaluation.
"""

import logging
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)

class NaNHandler:
    """Utilities for handling NaN values in RAGAS pipeline."""
    
    @staticmethod
    def clean_dataframe(df: pd.DataFrame, strategy: str = 'fallback') -> pd.DataFrame:
        """
        Clean DataFrame of NaN values using specified strategy.
        
        Args:
            df: DataFrame to clean
            strategy: Cleaning strategy ('drop', 'fillna', 'fallback')
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or len(df) == 0:
            return df
        
        logger.info(f"🧹 Cleaning DataFrame with {len(df)} rows using '{strategy}' strategy")
        
        original_count = len(df)
        
        if strategy == 'drop':
            # Drop rows with any NaN values
            df_clean = df.dropna()
            logger.info(f"   Dropped {original_count - len(df_clean)} rows with NaN values")
            
        elif strategy == 'fillna':
            # Fill NaN values with appropriate defaults
            df_clean = df.copy()
            
            for col in df_clean.columns:
                if df_clean[col].dtype == 'object':
                    df_clean[col] = df_clean[col].fillna('')
                elif df_clean[col].dtype in ['float64', 'int64']:
                    df_clean[col] = df_clean[col].fillna(0.0)
                else:
                    df_clean[col] = df_clean[col].fillna('')
            
            logger.info(f"   Filled NaN values in {len(df_clean.columns)} columns")
            
        elif strategy == 'fallback':
            # Use intelligent fallbacks based on column names
            df_clean = df.copy()
            
            for col in df_clean.columns:
                if col in ['user_input', 'question', 'query']:
                    df_clean[col] = df_clean[col].fillna(f'Sample question {df_clean.index}')
                elif col in ['reference', 'answer', 'response']:
                    df_clean[col] = df_clean[col].fillna('Sample answer')
                elif col in ['reference_contexts', 'contexts']:
                    df_clean[col] = df_clean[col].fillna(['Sample context'])
                elif col in ['auto_keywords', 'keywords']:
                    df_clean[col] = df_clean[col].fillna('')
                elif 'score' in col.lower() or 'metric' in col.lower():
                    df_clean[col] = df_clean[col].fillna(0.0)
                else:
                    if df_clean[col].dtype == 'object':
                        df_clean[col] = df_clean[col].fillna('')
                    else:
                        df_clean[col] = df_clean[col].fillna(0.0)
            
            logger.info(f"   Applied intelligent fallbacks for {len(df_clean.columns)} columns")
        
        else:
            logger.warning(f"Unknown strategy '{strategy}', returning original DataFrame")
            df_clean = df
        
        # Final validation
        remaining_nans = df_clean.isna().sum().sum()
        if remaining_nans > 0:
            logger.warning(f"⚠️ {remaining_nans} NaN values remain after cleaning")
        else:
            logger.info("✅ All NaN values cleaned successfully")
        
        return df_clean
    
    @staticmethod
    def validate_testset(df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate testset DataFrame for common issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Validation results dictionary
        """
        if df is None:
            return {
                'valid': False,
                'error': 'DataFrame is None',
                'issues': ['DataFrame is None']
            }
        
        if len(df) == 0:
            return {
                'valid': False,
                'error': 'DataFrame is empty',
                'issues': ['DataFrame is empty']
            }
        
        issues = []
        
        # Check for required columns
        required_columns = ['user_input', 'reference_contexts', 'reference']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(f"Missing required columns: {missing_columns}")
        
        # Check for NaN values
        nan_counts = df.isna().sum()
        nan_columns = nan_counts[nan_counts > 0].to_dict()
        if nan_columns:
            issues.append(f"NaN values found: {nan_columns}")
        
        # Check for empty strings in critical columns
        for col in ['user_input', 'reference']:
            if col in df.columns:
                empty_count = (df[col] == '').sum()
                if empty_count > 0:
                    issues.append(f"Empty strings in {col}: {empty_count} rows")
        
        # Check data types
        if 'reference_contexts' in df.columns:
            non_list_contexts = 0
            for idx, contexts in df['reference_contexts'].items():
                if not isinstance(contexts, (list, str)):
                    non_list_contexts += 1
            if non_list_contexts > 0:
                issues.append(f"Invalid reference_contexts format: {non_list_contexts} rows")
        
        valid = len(issues) == 0
        
        return {
            'valid': valid,
            'error': None if valid else f"{len(issues)} validation issues found",
            'issues': issues,
            'row_count': len(df),
            'column_count': len(df.columns),
            'columns': list(df.columns)
        }
    
    @staticmethod
    def safe_to_numeric(series: pd.Series, default: float = 0.0) -> pd.Series:
        """Safely convert series to numeric, handling errors."""
        try:
            return pd.to_numeric(series, errors='coerce').fillna(default)
        except Exception as e:
            logger.warning(f"Failed to convert to numeric: {e}")
            return pd.Series([default] * len(series), index=series.index)

def clean_testset_nan_values(testset_file: str, strategy: str = 'fallback') -> str:
    """
    Clean NaN values from a testset file.
    
    Args:
        testset_file: Path to testset CSV file
        strategy: Cleaning strategy
        
    Returns:
        Path to cleaned testset file
    """
    try:
        from pathlib import Path
        
        testset_path = Path(testset_file)
        if not testset_path.exists():
            raise FileNotFoundError(f"Testset file not found: {testset_file}")
        
        # Load testset
        df = pd.read_csv(testset_path)
        logger.info(f"📁 Loaded testset: {len(df)} rows, {len(df.columns)} columns")
        
        # Clean NaN values
        df_clean = NaNHandler.clean_dataframe(df, strategy)
        
        # Validate cleaned testset
        validation = NaNHandler.validate_testset(df_clean)
        if not validation['valid']:
            logger.warning(f"⚠️ Validation issues after cleaning: {validation['issues']}")
            # Apply additional fixes if needed
            df_clean = NaNHandler.clean_dataframe(df_clean, 'fillna')
        
        # Save cleaned testset
        clean_file = testset_path.with_stem(testset_path.stem + '_clean')
        df_clean.to_csv(clean_file, index=False)
        
        logger.info(f"✅ Cleaned testset saved: {clean_file}")
        return str(clean_file)
        
    except Exception as e:
        logger.error(f"❌ Failed to clean testset: {e}")
        return testset_file  # Return original file if cleaning fails

if __name__ == "__main__":
    logger.info("🧹 NaN Handling Utilities module loaded")
