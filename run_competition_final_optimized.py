"""
FINAL OPTIMIZED PIPELINE - Break 0.54
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'sample_size': None,  # Full dataset
    'test_size': 0.2,
    'random_seed': 42,
    'target_column': 'academic_work_impact',
    'leakage_column': 'addicted_label'
}

print("="*70)
print("🚀 FINAL OPTIMIZED PIPELINE - Target: 0.54+")
print("="*70)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
sample = pd.read_csv('data/raw/sample_submission.csv')

# Drop leakage
if CONFIG['leakage_column'] in train.columns:
    train = train.drop(columns=[CONFIG['leakage_column']])

# Sample if needed
if CONFIG['sample_size'] and len(train) > CONFIG['sample_size']:
    train = train.sample(n=CONFIG['sample_size'], random_state=CONFIG['random_seed'])

# Prepare train
y = train[CONFIG['target_column']].map({'Yes': 1, 'No': 0})
X = train.drop(columns=[CONFIG['target_column'], 'id'])

# Drop NaN target
valid_mask = ~y.isna()
y = y[valid_mask]
X = X[valid_mask]

# Prepare test
X_test = test.drop(columns=['id'])
if CONFIG['target_column'] in X_test.columns:
    X_test = X_test.drop(columns=[CONFIG['target_column']])

# Add smart features
X['screen_sleep_ratio'] = X['daily_screen_time_hours'] / (X['sleep_hours'] + 0.01)
X_test['screen_sleep_ratio'] = X_test['daily_screen_time_hours'] / (X_test['sleep_hours'] + 0.01)

X['total_screen'] = X['daily_screen_time_hours'] + X['social_media_hours'] + X['gaming_hours']
X_test['total_screen'] = X_test['daily_screen_time_hours'] + X_test['social_media_hours'] + X_test['gaming_hours']

X['work_sleep_ratio'] = X['work_study_hours'] / (X['sleep_hours'] + 0.01)
X_test['work_sleep_ratio'] = X_test['work_study_hours'] / (X_test['sleep_hours'] + 0.01)

print(f"Train: {X.shape}, Test: {X_test.shape}")

# Split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=CONFIG['test_size'], 
    random_state=CONFIG['random_seed'], stratify=y
)

# Identify columns
num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()

print(f"Numerical: {len(num_cols)}, Categorical: {len(cat_cols)}")

# Impute
imputer_num = SimpleImputer(strategy='median')
imputer_cat = SimpleImputer(strategy='most_frequent')

if len(num_cols) > 0:
    X_train[num_cols] = imputer_num.fit_transform(X_train[num_cols])
    X_val[num_cols] = imputer_num.transform(X_val[num_cols])
    X_test[num_cols] = imputer_num.transform(X_test[num_cols])

if len(cat_cols) > 0:
    X_train[cat_cols] = imputer_cat.fit_transform(X_train[cat_cols])
    X_val[cat_cols] = imputer_cat.transform(X_val[cat_cols])
    X_test[cat_cols] = imputer_cat.transform(X_test[cat_cols])

# Encode categorical
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    
    known_labels = set(le.classes_)
    X_val[col] = X_val[col].astype(str).apply(
        lambda x: x if x in known_labels else 'unknown'
    )
    le.classes_ = np.append(le.classes_, 'unknown')
    X_val[col] = le.transform(X_val[col])
    
    X_test[col] = X_test[col].astype(str).apply(
        lambda x: x if x in known_labels else 'unknown'
    )
    X_test[col] = le.transform(X_test[col])

# Scale
scaler = StandardScaler()
if len(num_cols) > 0:
    X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
    X_val[num_cols] = scaler.transform(X_val[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])

# Ensure all numeric
for df in [X_train, X_val, X_test]:
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

print(f"Train: {X_train.shape}, Validation: {X_val.shape}, Test: {X_test.shape}")

# Train models
print("\nTraining models...")

models = {}

# LightGBM - 1000 estimators
lgb = LGBMClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.02,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=CONFIG['random_seed'],
    verbose=-1
)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict_proba(X_val)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_pred)
models['lightgbm'] = (lgb, lgb_pred, lgb_auc)
print(f"   LightGBM: {lgb_auc:.4f}")

# XGBoost - 1000 estimators
xgb = XGBClassifier(
    n_estimators=1000,
    max_depth=6,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=CONFIG['random_seed'],
    tree_method='hist',
    eval_metric='auc'
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_pred)
models['xgboost'] = (xgb, xgb_pred, xgb_auc)
print(f"   XGBoost: {xgb_auc:.4f}")

# CatBoost
cat = CatBoostClassifier(
    iterations=1000,
    depth=6,
    learning_rate=0.02,
    l2_leaf_reg=3,
    random_seed=CONFIG['random_seed'],
    verbose=False
)
cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
cat_pred = cat.predict_proba(X_val)[:, 1]
cat_auc = roc_auc_score(y_val, cat_pred)
models['catboost'] = (cat, cat_pred, cat_auc)
print(f"   CatBoost: {cat_auc:.4f}")

# Random Forest
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=CONFIG['random_seed'],
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_pred)
models['random_forest'] = (rf, rf_pred, rf_auc)
print(f"   Random Forest: {rf_auc:.4f}")

# Ensemble
scores = {name: auc for name, (_, _, auc) in models.items()}
total = sum(scores.values())
weights = {name: score/total for name, score in scores.items()}

print("\n   Model weights:")
for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"      {name}: {weight:.3f} (AUC: {scores[name]:.4f})")

# Weighted ensemble
ensemble_val = np.zeros(len(y_val))
for name, (model, pred, auc) in models.items():
    ensemble_val += weights[name] * pred

ensemble_auc = roc_auc_score(y_val, ensemble_val)
print(f"\n   🏆 Ensemble AUC: {ensemble_auc:.4f}")

# Make predictions on test
test_preds = []
for name, (model, _, _) in models.items():
    test_preds.append(model.predict_proba(X_test)[:, 1])

final_pred = np.zeros(len(X_test))
for name, pred in zip(weights.keys(), test_preds):
    final_pred += weights[name] * pred

# Create submission
submission = sample.copy()
submission['academic_work_impact'] = (final_pred > 0.5).astype(int)

import os
os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/final_optimized_submission.csv'
submission.to_csv(submission_path, index=False)

print(f"\n✅ Submission saved to {submission_path}")
print(f"📊 Best Validation AUC: {ensemble_auc:.4f}")

print("\n" + "="*70)
print("📊 FINAL RESULTS")
print("="*70)
print("\nModel Performance (Validation):")
print("-"*50)
for name in sorted(scores.keys(), key=lambda x: -scores[x]):
    print(f"   {name:15s}: AUC = {scores[name]:.4f}")
print("-"*50)
print(f"   {'ENSEMBLE':15s}: AUC = {ensemble_auc:.4f} 🏆")
print("="*70)

print(f"""
🚀 TARGET: 0.54

Current Ensemble AUC: {ensemble_auc:.4f}
Gap to 0.54: {0.54 - ensemble_auc:.4f}

✅ Next Steps if not at 0.54:
   1. Try LightGBM with learning_rate=0.01
   2. Try XGBoost with learning_rate=0.01  
   3. Add 'rank' transformation for key features
   4. Try stacking ensemble
""")
