import pandas as pd
import numpy as np
from loguru import logger
import gc

def optimize_memory(df: pd.DataFrame) -> pd.DataFrame:
    """Optimise memory usage of dataframe"""
    logger.info("Optimising memory usage...")
    
    initial_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
            else:
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
        else:
            if df[col].nunique() / len(df) < 0.5:
                df[col] = df[col].astype('category')
    
    final_memory = df.memory_usage(deep=True).sum() / 1024**2
    logger.info(f"Memory reduced from {initial_memory:.2f} MB to {final_memory:.2f} MB")
    logger.info(f"Reduction: {(1 - final_memory/initial_memory)*100:.1f}%")
    
    gc.collect()
    
    return df

def get_memory_usage(df: pd.DataFrame) -> dict:
    """Get detailed memory usage"""
    memory_usage = {}
    
    for col in df.columns:
        memory_usage[col] = {
            'dtype': str(df[col].dtype),
            'memory_mb': df[col].memory_usage(deep=True) / 1024**2,
            'unique_count': df[col].nunique()
        }
    
    total_memory = df.memory_usage(deep=True).sum() / 1024**2
    
    return {
        'total_mb': total_memory,
        'columns': memory_usage
    }