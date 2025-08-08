"""
CSV Data Validator for RAG Evaluation Pipeline

This module provides comprehensive validation and cleaning of CSV data
before testset generation to prevent NaN-related failures and ensure
robust pipeline execution.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import logging
import json
import re
from datetime import datetime

logger = logging.getLogger(__name__)

class CSVDataValidator:
    """Validates and cleans CSV data before testset generation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Try multiple config paths for CSV processing configuration
        self.csv_config = (
            config.get('csv_processing', {}) or 
            config.get('data_sources', {}).get('csv', {}).get('format', {}) or
            config.get('validation', {}).get('csv_validation', {}) or
            {}
        )
        self.validation_stats = {
            'total_files_processed': 0,
            'total_rows_input': 0,
            'total_rows_output': 0,
            'total_issues_found': 0,
            'total_fixes_applied': 0
        }
        
    def validate_and_clean_csv(self, csv_file: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Validate and clean CSV data before processing.
        
        Args:
            csv_file: Path to the CSV file to validate
            
        Returns:
            Tuple of (cleaned_dataframe, validation_report)
        """
        logger.info(f"🔍 Validating CSV file: {csv_file}")
        
        try:
            # Load CSV with error handling
            df = self._safe_load_csv(csv_file)
            if df is None:
                raise ValueError(f"Failed to load CSV file: {csv_file}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load CSV {csv_file}: {e}")
            raise
            
        original_count = len(df)
        self.validation_stats['total_rows_input'] += original_count
        
        validation_report = {
            'file_name': csv_file.name,
            'original_rows': original_count,
            'issues_found': [],
            'fixes_applied': [],
            'final_rows': 0,
            'data_quality_score': 0.0,
            'validation_timestamp': datetime.now().isoformat(),
            'critical_issues': [],
            'warnings': []
        }
        
        # Step 1: Basic data integrity checks
        df = self._check_basic_integrity(df, validation_report)
        
        # Step 2: Handle NaN values strategically
        df = self._handle_nan_values(df, validation_report)
        
        # Step 3: Validate and clean text content
        df = self._validate_text_content(df, validation_report)
        
        # Step 4: Validate data types and schema
        df = self._validate_data_types(df, validation_report)
        
        # Step 5: Remove duplicates and invalid entries
        df = self._remove_duplicates_and_invalid(df, validation_report)
        
        # Step 6: Validate content quality
        df = self._validate_content_quality(df, validation_report)
        
        # Step 7: Validate JSON field mappings
        df = self._validate_json_mappings(df, validation_report)
        
        # Step 8: Final consistency checks
        df = self._final_consistency_checks(df, validation_report)
        
        # Calculate final metrics
        validation_report['final_rows'] = len(df)
        validation_report['data_quality_score'] = len(df) / original_count if original_count > 0 else 0.0
        validation_report['rows_preserved'] = len(df)
        validation_report['rows_removed'] = original_count - len(df)
        validation_report['preservation_rate'] = validation_report['data_quality_score']
        
        # Update global stats
        self.validation_stats['total_files_processed'] += 1
        self.validation_stats['total_rows_output'] += len(df)
        self.validation_stats['total_issues_found'] += len(validation_report['issues_found'])
        self.validation_stats['total_fixes_applied'] += len(validation_report['fixes_applied'])
        
        logger.info(f"✅ CSV validation complete: {original_count} → {len(df)} rows (quality: {validation_report['data_quality_score']:.2%})")
        
        # Log critical issues
        if validation_report['critical_issues']:
            logger.warning(f"⚠️ Critical issues found in {csv_file.name}:")
            for issue in validation_report['critical_issues']:
                logger.warning(f"   - {issue}")
        
        return df, validation_report
    
    def _safe_load_csv(self, csv_file: Path) -> Optional[pd.DataFrame]:
        """Safely load CSV with multiple encoding attempts."""
        encodings = ['utf-8', 'utf-8-sig', 'latin1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(csv_file, encoding=encoding)
                logger.debug(f"✅ Successfully loaded CSV with {encoding} encoding")
                return df
            except UnicodeDecodeError:
                logger.debug(f"⚠️ Failed to load with {encoding} encoding, trying next...")
                continue
            except Exception as e:
                logger.error(f"❌ Error loading CSV with {encoding}: {e}")
                continue
                
        logger.error(f"❌ Failed to load CSV with any encoding: {csv_file}")
        return None
    
    def _check_basic_integrity(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Check basic data integrity issues."""
        # Check for completely empty DataFrame
        if df.empty:
            report['critical_issues'].append("DataFrame is completely empty")
            return df
            
        # Check for columns
        if len(df.columns) == 0:
            report['critical_issues'].append("No columns found in DataFrame")
            return df
            
        # Check for unnamed columns
        unnamed_cols = [col for col in df.columns if str(col).startswith('Unnamed:')]
        if unnamed_cols:
            report['issues_found'].append(f"Found {len(unnamed_cols)} unnamed columns")
            # Drop unnamed columns if they're mostly empty
            for col in unnamed_cols:
                if df[col].notna().sum() / len(df) < 0.1:  # Less than 10% data
                    df = df.drop(columns=[col])
                    report['fixes_applied'].append(f"Dropped mostly empty unnamed column: {col}")
        
        return df
    
    def _handle_nan_values(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Handle NaN values strategically based on column importance."""
        
        # Count NaN values
        nan_counts = df.isnull().sum()
        total_nans = nan_counts.sum()
        
        if total_nans > 0:
            report['issues_found'].append(f"Found {total_nans} NaN values across {(nan_counts > 0).sum()} columns")
            logger.warning(f"⚠️ Found {total_nans} NaN values in CSV")
            
            # Get column importance levels
            critical_columns = self._get_critical_columns()
            important_columns = self._get_important_columns()
            optional_columns = self._get_optional_columns()
            
            # Strategy 1: Drop rows with NaN in critical columns
            critical_cols_present = [col for col in critical_columns if col in df.columns]
            if critical_cols_present:
                before_drop = len(df)
                df = df.dropna(subset=critical_cols_present)
                dropped = before_drop - len(df)
                if dropped > 0:
                    report['fixes_applied'].append(f"Dropped {dropped} rows with NaN in critical columns: {critical_cols_present}")
                    logger.info(f"✅ Dropped {dropped} rows with critical NaN values")
            
            # Strategy 2: Smart fill for important columns
            important_cols_present = [col for col in important_columns if col in df.columns]
            for col in important_cols_present:
                if df[col].isnull().any():
                    fill_value = self._get_smart_fill_value(df, col)
                    before_fill = df[col].isnull().sum()
                    df[col] = df[col].fillna(fill_value)
                    report['fixes_applied'].append(f"Filled {before_fill} NaN values in {col} with '{fill_value}'")
            
            # Strategy 3: Handle optional columns
            optional_cols_present = [col for col in optional_columns if col in df.columns]
            for col in optional_cols_present:
                if df[col].isnull().any():
                    fill_value = self._get_optional_fill_value(col)
                    before_fill = df[col].isnull().sum()
                    df[col] = df[col].fillna(fill_value)
                    report['fixes_applied'].append(f"Filled {before_fill} NaN values in optional column {col}")
            
            # Strategy 4: Handle remaining NaN values
            remaining_nans = df.isnull().sum().sum()
            if remaining_nans > 0:
                df = self._smart_fill_remaining_nan(df)
                report['fixes_applied'].append(f"Applied smart filling to {remaining_nans} remaining NaN values")
        
        return df
    
    def _validate_text_content(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Validate and clean text content."""
        text_columns = self._get_text_columns()
        
        for col in text_columns:
            if col in df.columns:
                # Check for empty or whitespace-only text
                empty_mask = df[col].astype(str).str.strip().eq('')
                empty_count = empty_mask.sum()
                
                if empty_count > 0:
                    report['issues_found'].append(f"Found {empty_count} empty text values in {col}")
                    # Fill empty text with meaningful defaults
                    df.loc[empty_mask, col] = f"[Missing {col.replace('_', ' ').title()}]"
                    report['fixes_applied'].append(f"Filled {empty_count} empty text values in {col}")
                
                # Check for very short content that might be invalid
                if col in ['display', 'content', 'text']:
                    short_mask = df[col].astype(str).str.len() < 3
                    short_count = short_mask.sum()
                    if short_count > 0:
                        report['warnings'].append(f"Found {short_count} very short text entries in {col}")
                
                # Clean text content
                df[col] = df[col].astype(str).str.strip()
                
                # Remove or fix problematic characters
                df[col] = df[col].str.replace(r'\x00', '', regex=True)  # Remove null bytes
                df[col] = df[col].str.replace(r'[\r\n]+', ' ', regex=True)  # Normalize line breaks
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Validate and fix data type issues."""
        json_mapping = self.csv_config.get('json_field_mapping', {})
        
        # Ensure text columns are proper strings
        text_columns = self._get_text_columns()
        for col in text_columns:
            if col in df.columns:
                original_dtype = df[col].dtype
                df[col] = df[col].astype(str)
                if str(original_dtype) != 'object':
                    report['fixes_applied'].append(f"Converted {col} from {original_dtype} to string")
        
        # Handle numeric columns if any
        numeric_columns = self._get_numeric_columns()
        for col in numeric_columns:
            if col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    # Fill NaN values created by coercion
                    if df[col].isnull().any():
                        df[col] = df[col].fillna(0)
                        report['fixes_applied'].append(f"Converted {col} to numeric and filled NaN with 0")
                except Exception as e:
                    report['warnings'].append(f"Failed to convert {col} to numeric: {e}")
        
        return df
    
    def _remove_duplicates_and_invalid(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Remove duplicates and completely invalid entries."""
        
        # Remove exact duplicates
        before_dedup = len(df)
        df = df.drop_duplicates()
        dedup_removed = before_dedup - len(df)
        if dedup_removed > 0:
            report['fixes_applied'].append(f"Removed {dedup_removed} exact duplicate rows")
        
        # Remove rows where all important fields are missing/invalid
        important_columns = self._get_critical_columns() + self._get_important_columns()
        important_cols_present = [col for col in important_columns if col in df.columns]
        
        if important_cols_present:
            # Check for rows where all important columns are effectively empty
            invalid_mask = True
            for col in important_cols_present:
                col_valid = ~(df[col].astype(str).str.strip().isin(['', 'nan', 'None', '[Missing', 'Unknown']))
                invalid_mask = invalid_mask & ~col_valid
            
            invalid_count = invalid_mask.sum()
            if invalid_count > 0:
                df = df[~invalid_mask]
                report['fixes_applied'].append(f"Removed {invalid_count} rows with no valid important data")
        
        return df
    
    def _validate_content_quality(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Validate content quality and remove poor quality entries."""
        json_mapping = self.csv_config.get('json_field_mapping', {})
        
        # Get primary content column
        text_col = None
        for potential_col in ['display', 'content', 'text', json_mapping.get('text', 'text')]:
            if potential_col in df.columns:
                text_col = potential_col
                break
        
        if text_col:
            # Remove entries with very short content (likely invalid)
            min_length = self.csv_config.get('min_content_length', 10)
            short_content = df[text_col].astype(str).str.len() < min_length
            short_count = short_content.sum()
            
            if short_count > 0:
                # Don't remove, but warn
                if short_count / len(df) > 0.5:  # More than 50% are short
                    report['warnings'].append(f"Over 50% of entries have short content in {text_col}")
                else:
                    df = df[~short_content]
                    report['fixes_applied'].append(f"Removed {short_count} entries with content < {min_length} chars in {text_col}")
            
            # Check for repetitive or low-quality content
            if len(df) > 1:
                # Check for too many identical entries
                value_counts = df[text_col].value_counts()
                frequent_values = value_counts[value_counts > len(df) * 0.1]  # More than 10% identical
                if len(frequent_values) > 0:
                    report['warnings'].append(f"Found {len(frequent_values)} values appearing in >10% of rows in {text_col}")
        
        return df
    
    def _validate_json_mappings(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Validate JSON field mappings are working correctly."""
        json_mapping = self.csv_config.get('json_field_mapping', {})
        
        if not json_mapping:
            return df
        
        # Check if mapped columns exist
        missing_columns = []
        for field_name, column_name in json_mapping.items():
            if column_name not in df.columns:
                missing_columns.append(f"{field_name} -> {column_name}")
        
        if missing_columns:
            report['warnings'].append(f"JSON mapping references missing columns: {', '.join(missing_columns)}")
        
        # Validate that mapped columns have reasonable data
        for field_name, column_name in json_mapping.items():
            if column_name in df.columns:
                empty_rate = df[column_name].astype(str).str.strip().eq('').sum() / len(df)
                if empty_rate > 0.8:  # More than 80% empty
                    report['warnings'].append(f"Mapped column {column_name} ({field_name}) is {empty_rate:.1%} empty")
        
        return df
    
    def _final_consistency_checks(self, df: pd.DataFrame, report: Dict[str, Any]) -> pd.DataFrame:
        """Perform final consistency checks."""
        
        # Check minimum viable dataset size
        min_rows = self.csv_config.get('min_viable_rows', 5)
        if len(df) < min_rows:
            report['critical_issues'].append(f"Dataset too small: {len(df)} rows < minimum {min_rows}")
        
        # Check for consistent data types within columns
        for col in df.columns:
            if df[col].dtype == 'object':
                # Check if column should be numeric but has mixed types
                numeric_convertible = pd.to_numeric(df[col], errors='coerce').notna().sum()
                if numeric_convertible > len(df) * 0.8:  # 80% could be numeric
                    report['warnings'].append(f"Column {col} might be intended as numeric but contains text")
        
        # Final data integrity check
        if df.isnull().any().any():
            remaining_nans = df.isnull().sum().sum()
            report['warnings'].append(f"Still has {remaining_nans} NaN values after cleaning")
        
        return df
    
    def _get_critical_columns(self) -> List[str]:
        """Get column names that cannot have NaN values."""
        json_mapping = self.csv_config.get('json_field_mapping', {})
        critical = [
            json_mapping.get('text', 'text'),
            json_mapping.get('title', 'title'),
            'display',
            'content'
        ]
        
        # Also check validation config for critical columns
        validation_config = self.config.get('validation', {}).get('csv_validation', {})
        user_critical = (
            self.csv_config.get('critical_columns', []) or
            validation_config.get('critical_columns', [])
        )
        
        # If no configuration found, use safe defaults to avoid empty critical columns
        if not user_critical and not json_mapping:
            critical = ['display', 'content']  # Safe defaults that likely exist
            
        return list(set(critical + user_critical))
    
    def _get_important_columns(self) -> List[str]:
        """Get column names that are important but can be filled."""
        json_mapping = self.csv_config.get('json_field_mapping', {})
        important = [
            'error_code',
            'category',
            'type',
            'description',
            json_mapping.get('metadata', 'metadata')
        ]
        
        # Also check validation config for important columns
        validation_config = self.config.get('validation', {}).get('csv_validation', {})
        user_important = (
            self.csv_config.get('important_columns', []) or
            validation_config.get('important_columns', [])
        )
        
        return list(set(important + user_important))
    
    def _get_optional_columns(self) -> List[str]:
        """Get column names that are optional."""
        return self.csv_config.get('optional_columns', ['id', 'timestamp', 'source', 'version'])
    
    def _get_text_columns(self) -> List[str]:
        """Get column names that contain text content."""
        json_mapping = self.csv_config.get('json_field_mapping', {})
        text_cols = [
            json_mapping.get('text', 'text'),
            json_mapping.get('title', 'title'),
            'display',
            'content',
            'description',
            'error_message'
        ]
        user_text = self.csv_config.get('text_columns', [])
        return list(set(text_cols + user_text))
    
    def _get_numeric_columns(self) -> List[str]:
        """Get column names that should be numeric."""
        return self.csv_config.get('numeric_columns', ['id', 'count', 'score', 'rating'])
    
    def _get_smart_fill_value(self, df: pd.DataFrame, column: str) -> str:
        """Get smart fill value for important columns based on context."""
        column_lower = column.lower()
        
        if 'error' in column_lower or 'code' in column_lower:
            return 'UNKNOWN_ERROR'
        elif 'category' in column_lower or 'type' in column_lower:
            return 'Uncategorized'
        elif 'description' in column_lower:
            return f'[Auto-generated description for {column}]'
        elif 'title' in column_lower:
            return f'[Auto-generated title]'
        else:
            return f'[Missing {column.replace("_", " ").title()}]'
    
    def _get_optional_fill_value(self, column: str) -> str:
        """Get fill value for optional columns."""
        column_lower = column.lower()
        
        if 'id' in column_lower:
            return f'auto_id_{hash(column) % 10000}'
        elif 'timestamp' in column_lower or 'date' in column_lower:
            return datetime.now().isoformat()
        elif 'source' in column_lower:
            return 'Unknown Source'
        elif 'version' in column_lower:
            return '1.0'
        else:
            return f'[Default {column}]'
    
    def _smart_fill_remaining_nan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply smart NaN filling strategies for remaining values."""
        for col in df.columns:
            if df[col].isnull().any():
                if df[col].dtype == 'object':  # Text columns
                    df[col] = df[col].fillna(f'[Unknown {col.replace("_", " ").title()}]')
                elif df[col].dtype in ['int64', 'float64']:  # Numeric columns
                    df[col] = df[col].fillna(0)
                else:
                    df[col] = df[col].fillna('Unknown')
        return df
    
    def save_cleaned_csv(self, df: pd.DataFrame, original_path: Path, output_dir: Path) -> Path:
        """Save cleaned CSV with validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        clean_filename = f"cleaned_{original_path.stem}_{timestamp}.csv"
        output_path = output_dir / clean_filename
        
        # Ensure output directory exists
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cleaned CSV
        df.to_csv(output_path, index=False, encoding='utf-8')
        logger.info(f"💾 Saved cleaned CSV: {output_path}")
        
        return output_path
    
    def save_validation_report(self, reports: List[Dict[str, Any]], output_dir: Path) -> Path:
        """Save comprehensive validation report."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = output_dir / f"csv_validation_report_{timestamp}.json"
        
        comprehensive_report = {
            'validation_summary': self.validation_stats,
            'file_reports': reports,
            'validation_timestamp': datetime.now().isoformat(),
            'validator_config': self.csv_config
        }
        
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(comprehensive_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"📄 Saved validation report: {report_path}")
        return report_path
    
    def get_validation_summary(self) -> Dict[str, Any]:
        """Get summary of all validation operations."""
        summary = self.validation_stats.copy()
        if summary['total_rows_input'] > 0:
            summary['overall_preservation_rate'] = summary['total_rows_output'] / summary['total_rows_input']
        else:
            summary['overall_preservation_rate'] = 0.0
        
        return summary
