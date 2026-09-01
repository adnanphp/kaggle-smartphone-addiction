import pandas as pd
import numpy as np
from typing import Tuple, List, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.config import config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer_num = SimpleImputer(strategy='median')
        self.imputer_cat = SimpleImputer(strategy='most_frequent')
        self.is_fitted = False
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        logger.info("Loading data...")
        train = pd.read_csv(config.TRAIN_FILE)
        test = pd.read_csv(config.TEST_FILE)
        sample = pd.read_csv(config.SAMPLE_SUBMISSION)
        
        # Clean column names
        train.columns = train.columns.str.strip()
        test.columns = test.columns.str.strip()
        sample.columns = sample.columns.str.strip()
        
        logger.info(f"Train shape: {train.shape}")
        logger.info(f"Test shape: {test.shape}")
        logger.info(f"Sample shape: {sample.shape}")
        
        return train, test, sample
    
    def explore_basic_stats(self, df: pd.DataFrame) -> dict:
        stats = {
            'shape': df.shape,
            'dtypes': df.dtypes.to_dict(),
            'missing': df.isnull().sum().to_dict(),
            'missing_pct': (df.isnull().sum() / len(df) * 100).to_dict(),
        }
        return stats
    
    def clean_data(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        df = df.copy()
        df.columns = df.columns.str.strip()
        
        # Remove duplicate columns
        df = df.loc[:, ~df.columns.duplicated()]
        
        # Drop leakage columns
        if is_train:
            for col in config.LEAKAGE_COLUMNS:
                if col in df.columns:
                    logger.info(f"Dropping leakage column: {col}")
                    df = df.drop(columns=[col])
        
        # Drop ID columns
        for col in config.DROP_COLUMNS:
            if col in df.columns:
                df = df.drop(columns=[col])
        
        # Identify column types
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if config.TARGET_COLUMN in num_cols:
            num_cols.remove(config.TARGET_COLUMN)
        if config.TARGET_COLUMN in cat_cols:
            cat_cols.remove(config.TARGET_COLUMN)
        
        logger.info(f"Numerical columns: {num_cols}")
        logger.info(f"Categorical columns: {cat_cols}")
        
        # Handle missing values
        if is_train:
            if len(num_cols) > 0:
                self.imputer_num.fit(df[num_cols])
            if len(cat_cols) > 0:
                self.imputer_cat.fit(df[cat_cols])
        
        if len(num_cols) > 0:
            df[num_cols] = self.imputer_num.transform(df[num_cols])
        if len(cat_cols) > 0:
            df[cat_cols] = self.imputer_cat.transform(df[cat_cols])
        
        # Encode categorical
        for col in cat_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                if is_train:
                    df[col] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    df[col] = df[col].astype(str)
                    known_labels = set(self.label_encoders[col].classes_)
                    df[col] = df[col].apply(lambda x: x if x in known_labels else 'unknown')
                    self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                    df[col] = self.label_encoders[col].transform(df[col])
            else:
                df[col] = df[col].astype(str)
                known_labels = set(self.label_encoders[col].classes_)
                df[col] = df[col].apply(lambda x: x if x in known_labels else 'unknown')
                self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                df[col] = self.label_encoders[col].transform(df[col])
        
        # Scale numerical
        if is_train:
            if len(num_cols) > 0:
                self.scaler.fit(df[num_cols])
        if len(num_cols) > 0:
            df[num_cols] = self.scaler.transform(df[num_cols])
        
        self.is_fitted = True
        return df
