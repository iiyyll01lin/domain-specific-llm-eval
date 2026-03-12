#!/usr/bin/env python3
"""
RAGAS Fixes for Column Mapping and Schema Issues
"""


def fix_ragas_dataset_schema(df):
    """
    Fix dataset schema to match RAGAS expectations
    """
    import logging

    import pandas as pd
    
    logger = logging.getLogger(__name__)
    
    # Create a copy to avoid modifying original
    fixed_df = df.copy()
    
    # Map columns to RAGAS expected names
    column_mapping = {
        'user_input': 'question',
        'response': 'answer', 
        'reference': 'ground_truth',
        'retrieved_contexts': 'contexts'
    }
    
    # Apply mapping
    for old_col, new_col in column_mapping.items():
        if old_col in fixed_df.columns and new_col not in fixed_df.columns:
            fixed_df[new_col] = fixed_df[old_col]
            logger.info(f"   Mapped {old_col} -> {new_col}")
    
    # Ensure required columns exist
    required_cols = ['question', 'answer', 'ground_truth']
    for col in required_cols:
        if col not in fixed_df.columns:
            if col == 'question' and 'user_input' in fixed_df.columns:
                fixed_df[col] = fixed_df['user_input']
            elif col == 'answer' and 'response' in fixed_df.columns:
                fixed_df[col] = fixed_df['response']
            elif col == 'ground_truth':
                if 'reference' in fixed_df.columns:
                    fixed_df[col] = fixed_df['reference']
                elif 'answer' in fixed_df.columns:
                    fixed_df[col] = fixed_df['answer']  # Use answer as ground truth fallback
            logger.info(f"   Created missing column: {col}")
    
    return fixed_df

def safe_ragas_evaluation(df, metrics, llm=None, embeddings=None):
    """
    Safe RAGAS evaluation with proper error handling
    """
    import logging

    import pandas as pd
    from ragas import evaluate
    from ragas.dataset_schema import SingleTurnSample
    
    logger = logging.getLogger(__name__)
    
    try:
        # Fix schema first
        fixed_df = fix_ragas_dataset_schema(df)
        
        # Create RAGAS dataset
        samples = []
        for _, row in fixed_df.iterrows():
            try:
                sample = SingleTurnSample(
                    user_input=str(row.get('question', row.get('user_input', ''))),
                    response=str(row.get('answer', row.get('response', ''))),
                    reference=str(row.get('ground_truth', row.get('reference', ''))),
                    retrieved_contexts=row.get('contexts', row.get('retrieved_contexts', []))
                )
                samples.append(sample)
            except Exception as e:
                logger.warning(f"Failed to create sample for row {row.name}: {e}")
                continue
        
        if not samples:
            logger.error("No valid samples created for RAGAS evaluation")
            return None
        
        logger.info(f"Created {len(samples)} valid samples for RAGAS evaluation")
        
        # Evaluate with proper error handling
        result = evaluate(
            dataset=samples,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings
        )
        
        return result
        
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return None
