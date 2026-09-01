import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
import catboost as cb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import joblib
from src.config import config
import logging

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.models = {}
        
    def train_logistic(self, X_train: pd.DataFrame, y_train: pd.Series) -> LogisticRegression:
        logger.info("Training Logistic Regression...")
        model = LogisticRegression(
            max_iter=1000,
            random_state=self.random_seed,
            C=0.1,  # Strong regularization
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        self.models['logistic'] = model
        logger.info("Logistic Regression trained successfully")
        return model
    
    def train_random_forest(self, X_train: pd.DataFrame, y_train: pd.Series) -> RandomForestClassifier:
        logger.info("Training Random Forest...")
        model = RandomForestClassifier(
            n_estimators=50,
            max_depth=3,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=self.random_seed,
            n_jobs=-1,
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        self.models['random_forest'] = model
        logger.info("Random Forest trained successfully")
        return model
    
    def train_xgboost(self, X_train: pd.DataFrame, y_train: pd.Series, params=None) -> xgb.XGBClassifier:
        logger.info("Training XGBoost (with regularization)...")
        if params is None:
            params = {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.05,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'min_child_weight': 5,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'random_state': self.random_seed,
                'tree_method': 'hist',
                'eval_metric': 'auc'
            }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        self.models['xgboost'] = model
        logger.info("XGBoost trained successfully")
        return model
    
    def train_lightgbm(self, X_train: pd.DataFrame, y_train: pd.Series, params=None) -> lgb.LGBMClassifier:
        logger.info("Training LightGBM (with regularization)...")
        if params is None:
            params = {
                'n_estimators': 50,
                'max_depth': 3,
                'learning_rate': 0.05,
                'num_leaves': 15,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.5,
                'reg_lambda': 0.5,
                'min_child_samples': 20,
                'random_state': self.random_seed
            }
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        self.models['lightgbm'] = model
        logger.info("LightGBM trained successfully")
        return model
    
    def train_catboost(self, X_train: pd.DataFrame, y_train: pd.Series, params=None) -> cb.CatBoostClassifier:
        logger.info("Training CatBoost (with regularization)...")
        if params is None:
            params = {
                'iterations': 50,
                'depth': 3,
                'learning_rate': 0.05,
                'l2_leaf_reg': 5,
                'border_count': 64,
                'random_seed': self.random_seed,
                'verbose': False
            }
        model = cb.CatBoostClassifier(**params)
        model.fit(X_train, y_train)
        self.models['catboost'] = model
        logger.info("CatBoost trained successfully")
        return model
    
    def compare_models(self, X_val: pd.DataFrame, y_val: pd.Series) -> pd.DataFrame:
        results = []
        for name, model in self.models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    y_pred = model.predict_proba(X_val)[:, 1]
                else:
                    y_pred = model.predict(X_val)
                
                y_pred_class = (y_pred > 0.5).astype(int)
                results.append({
                    'model': name,
                    'roc_auc': roc_auc_score(y_val, y_pred),
                    'accuracy': accuracy_score(y_val, y_pred_class),
                    'f1_score': f1_score(y_val, y_pred_class)
                })
            except Exception as e:
                logger.warning(f"Could not evaluate {name}: {e}")
        
        return pd.DataFrame(results)
    
    def save_models(self):
        for name, model in self.models.items():
            joblib.dump(model, config.MODELS / f'{name}.pkl')
        logger.info(f"{len(self.models)} models saved")
    
    def load_models(self):
        for name in ['logistic', 'random_forest', 'xgboost', 'lightgbm', 'catboost']:
            model_path = config.MODELS / f'{name}.pkl'
            if model_path.exists():
                self.models[name] = joblib.load(model_path)
        logger.info(f"{len(self.models)} models loaded")
