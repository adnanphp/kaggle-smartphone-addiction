"""
Simple feature engineering - No overfitting!
"""
import pandas as pd
import numpy as np
from typing import List
from src.config import config
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self):
        self.feature_importance = None
        
    def create_features(self, df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
        """Return original features only - NO ENGINEERING!"""
        df = df.copy()
        
        # Extract target
        target = None
        if config.TARGET_COLUMN in df.columns:
            target = df[config.TARGET_COLUMN]
            df = df.drop(columns=[config.TARGET_COLUMN])
        
        # Keep only original features (no engineering)
        # Just return what we have
        logger.info(f"Using only original features: {df.shape[1]} columns")
        
        # Add target back
        if target is not None:
            df[config.TARGET_COLUMN] = target
        
        return df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series, n_features: int = 50) -> List[str]:
        from sklearn.feature_selection import SelectKBest, mutual_info_classif
        
        X_clean = X.fillna(0)
        selector = SelectKBest(mutual_info_classif, k=min(n_features, X_clean.shape[1]))
        selector.fit(X_clean, y)
        
        feature_scores = pd.DataFrame({
            'feature': X_clean.columns,
            'score': selector.scores_
        }).sort_values('score', ascending=False)
        
        self.feature_importance = feature_scores
        return feature_scores['feature'].head(n_features).tolist()
