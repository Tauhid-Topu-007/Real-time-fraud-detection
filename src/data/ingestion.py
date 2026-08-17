import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from loguru import logger
from typing import Optional, Tuple
import gc
import time

class BigDataIngestion:
    """Optimised data ingestion for large datasets"""
    
    def __init__(self, config_path: str = "configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        raw_path = self.config['data']['raw_path']
        self.raw_path = Path(raw_path)
        
        if not self.raw_path.exists():
            alternatives = [
                Path("data/raw/fraud_detection.csv"),
                Path("data/raw/transactions.csv"),
                Path("data/raw/fraud.csv"),
                Path("data/raw/data.csv")
            ]
            for alt in alternatives:
                if alt.exists():
                    self.raw_path = alt
                    logger.info(f"Found dataset at: {self.raw_path}")
                    break
        
        self.processed_path = Path(self.config['data']['processed_path'])
        self.processed_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.dtype_dict = {
            'step': 'int32',
            'type': 'category',
            'amount': 'float32',
            'nameOrig': 'category',
            'oldbalanceOrg': 'float32',
            'newbalanceOrig': 'float32',
            'nameDest': 'category',
            'oldbalanceDest': 'float32',
            'newbalanceDest': 'float32',
            'isFraud': 'int8',
            'isFlaggedFraud': 'int8'
        }
        
        logger.info(f"BigDataIngestion initialized with data path: {self.raw_path}")
    
    def load_processed(self) -> Optional[pd.DataFrame]:
        """Load processed parquet file if exists"""
        if self.processed_path.exists():
            logger.info(f"Loading processed data from {self.processed_path}")
            df = pd.read_parquet(self.processed_path)
            logger.info(f"Loaded {len(df):,} rows")
            logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
            return df
        return None
    
    def load_raw_with_optimisation(self) -> pd.DataFrame:
        """Load raw data with memory optimisation"""
        logger.info(f"Loading raw data from {self.raw_path}")
        start_time = time.time()
        
        if not self.raw_path.exists():
            raise FileNotFoundError(f"Dataset not found: {self.raw_path}")
        
        file_size = self.raw_path.stat().st_size / (1024**2)
        logger.info(f"File size: {file_size:.2f} MB")
        
        try:
            df = pd.read_csv(
                self.raw_path,
                dtype=self.dtype_dict,
                low_memory=False
            )
        except Exception as e:
            logger.warning(f"Error with optimised dtypes: {e}")
            logger.info("Loading with default dtypes...")
            df = pd.read_csv(self.raw_path, low_memory=False)
        
        load_time = time.time() - start_time
        logger.info(f"Loaded {len(df):,} rows in {load_time:.2f} seconds")
        logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        
        return df
    
    def load_in_chunks(self, chunk_size: int = 100000) -> pd.DataFrame:
        """Load data in chunks to manage memory"""
        logger.info(f"Loading data in chunks of {chunk_size:,}")
        
        chunks = []
        total_rows = 0
        
        for chunk in pd.read_csv(
            self.raw_path, 
            chunksize=chunk_size,
            dtype=self.dtype_dict,
            low_memory=False
        ):
            chunks.append(chunk)
            total_rows += len(chunk)
            logger.info(f"Loaded chunk: {len(chunk):,} rows (Total: {total_rows:,})")
            
            if len(chunks) > 20:
                break
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"Total loaded: {len(df):,} rows")
        
        del chunks
        gc.collect()
        
        return df
    
    def sample_data(self, n: int = 200000, random_state: int = 42) -> pd.DataFrame:
        """Load a random sample of data"""
        logger.info(f"Loading {n:,} random samples")
        
        total_rows = sum(1 for _ in open(self.raw_path)) - 1
        logger.info(f"Total rows: {total_rows:,}")
        
        if n >= total_rows:
            return self.load_raw_with_optimisation()
        
        np.random.seed(random_state)
        skip_rows = np.random.choice(
            range(1, total_rows + 1),
            size=total_rows - n,
            replace=False
        )
        skip_rows = sorted(skip_rows)
        
        df = pd.read_csv(
            self.raw_path,
            skiprows=skip_rows,
            dtype=self.dtype_dict,
            low_memory=False
        )
        
        logger.info(f"Loaded {len(df):,} samples")
        return df
    
    def save_processed(self, df: pd.DataFrame) -> None:
        """Save as parquet for faster loading"""
        logger.info(f"Saving processed data to {self.processed_path}")
        df.to_parquet(self.processed_path, index=False, compression='snappy')
        logger.info(f"Saved {len(df):,} rows to parquet")
    
    def load_data(self, use_processed: bool = True, sample: Optional[int] = None) -> pd.DataFrame:
        """Main method to load data"""
        if use_processed:
            df = self.load_processed()
            if df is not None:
                if sample and len(df) > sample:
                    df = df.sample(n=sample, random_state=42)
                    logger.info(f"Sampled {len(df):,} rows")
                return df
        
        df = self.load_raw_with_optimisation()
        
        try:
            self.save_processed(df)
        except Exception as e:
            logger.warning(f"Could not save processed data: {e}")
        
        return df