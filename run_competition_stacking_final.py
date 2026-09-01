"""
FINAL - Stacking Ensemble to Break 0.54
No TabPFN - Just smarter ensembling
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore')

CONFIG = {
    'sample_size': None,
    'test_size': 0.2,
    'random_seed': 42,
    'target_column': 'academic_work_impact',
    'leakage_column': 'addicted_label'
}

print("="*70)
print("🚀 FINAL - Stacking Ensemble to Break 0.54")
print("="*70)

# Load data
train = pd.read_csv('data/raw/train.csv')
test = pd.read_csv('data/raw/test.csv')
sample = pd.read_csv('data/raw/sample_submission.csv')

# Drop leakage
if CONFIG['leakage_column'] in train.columns:
    train = train.drop(columns=[CONFIG['leakage_column']])

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

# ============================================
# FEATURE ENGINEERING - BEST ONES
# ============================================

# Rank transformations
rank_cols = ['daily_screen_time_hours', 'social_media_hours', 'gaming_hours', 
             'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 
             'weekend_screen_time', 'work_study_hours']

for col in rank_cols:
    if col in X.columns:
        X[f'{col}_rank'] = X[col].rank(pct=True)
        X_test[f'{col}_rank'] = X_test[col].rank(pct=True)

# Ratio features
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

# ============================================
# INDIVIDUAL MODELS (Record scores)
# ============================================

print("\n📊 Training individual models...")

models = {}
scores = {}

# LightGBM
print("   LightGBM...")
lgb = LGBMClassifier(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=CONFIG['random_seed'],
    verbose=-1
)
lgb.fit(X_train, y_train)
lgb_pred = lgb.predict_proba(X_val)[:, 1]
lgb_auc = roc_auc_score(y_val, lgb_pred)
scores['lightgbm'] = lgb_auc
print(f"   ✅ LightGBM: {lgb_auc:.4f}")

# XGBoost
print("   XGBoost...")
xgb = XGBClassifier(
    n_estimators=800,
    learning_rate=0.02,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    random_state=CONFIG['random_seed'],
    tree_method='hist',
    eval_metric='auc'
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb.predict_proba(X_val)[:, 1]
xgb_auc = roc_auc_score(y_val, xgb_pred)
scores['xgboost'] = xgb_auc
print(f"   ✅ XGBoost: {xgb_auc:.4f}")

# CatBoost
print("   CatBoost...")
cat = CatBoostClassifier(
    iterations=800,
    learning_rate=0.02,
    depth=6,
    l2_leaf_reg=3,
    random_seed=CONFIG['random_seed'],
    verbose=False
)
cat.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=50, verbose=False)
cat_pred = cat.predict_proba(X_val)[:, 1]
cat_auc = roc_auc_score(y_val, cat_pred)
scores['catboost'] = cat_auc
print(f"   ✅ CatBoost: {cat_auc:.4f}")

# Random Forest
print("   Random Forest...")
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=CONFIG['random_seed'],
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict_proba(X_val)[:, 1]
rf_auc = roc_auc_score(y_val, rf_pred)
scores['random_forest'] = rf_auc
print(f"   ✅ Random Forest: {rf_auc:.4f}")

# ============================================
# STACKING ENSEMBLE
# ============================================

print("\n🔮 Training Stacking Ensemble...")

# Use top 3 models for stacking
base_models = [
    ('lgb', LGBMClassifier(n_estimators=500, max_depth=6, random_state=42, verbose=-1)),
    ('xgb', XGBClassifier(n_estimators=500, max_depth=6, random_state=42, tree_method='hist')),
    ('cat', CatBoostClassifier(iterations=500, depth=6, random_state=42, verbose=False))
]

# Meta-model (Logistic Regression with regularization)
meta_model = LogisticRegression(C=0.1, max_iter=1000, random_state=42)

stack = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=3,  # 3-fold cross-validation
    stack_method='predict_proba',
    n_jobs=-1
)

print("   Training stack (may take 2-3 minutes)...")
stack.fit(X_train, y_train)
stack_pred = stack.predict_proba(X_val)[:, 1]
stack_auc = roc_auc_score(y_val, stack_pred)
scores['stacking'] = stack_auc
print(f"   ✅ Stacking AUC: {stack_auc:.4f}")

# ============================================
# WEIGHTED ENSEMBLE (with stacking included)
# ============================================

print("\n🔮 Creating weighted ensemble...")

# Combine all predictions
all_preds = {
    'lightgbm': lgb_pred,
    'xgboost': xgb_pred,
    'catboost': cat_pred,
    'random_forest': rf_pred,
    'stacking': stack_pred
}

# Calculate weights (exponential based on AUC)
weights = {}
for name, auc in scores.items():
    weights[name] = np.exp(auc * 10)  # Stronger weighting

total = sum(weights.values())
weights = {name: w/total for name, w in weights.items()}

print("\n   Model weights:")
for name, weight in sorted(weights.items(), key=lambda x: -x[1]):
    print(f"      {name}: {weight:.3f} (AUC: {scores[name]:.4f})")

# Weighted ensemble
ensemble_val = np.zeros(len(y_val))
for name, pred in all_preds.items():
    if name in weights:
        ensemble_val += weights[name] * pred

ensemble_auc = roc_auc_score(y_val, ensemble_val)
print(f"\n   🏆 Ensemble AUC: {ensemble_auc:.4f}")

# ============================================
# MAKE PREDICTIONS
# ============================================

print("\nMaking predictions...")

test_preds = {
    'lightgbm': lgb.predict_proba(X_test)[:, 1],
    'xgboost': xgb.predict_proba(X_test)[:, 1],
    'catboost': cat.predict_proba(X_test)[:, 1],
    'random_forest': rf.predict_proba(X_test)[:, 1],
    'stacking': stack.predict_proba(X_test)[:, 1]
}

final_pred = np.zeros(len(X_test))
for name, pred in test_preds.items():
    if name in weights:
        final_pred += weights[name] * pred

# ============================================
# SUBMISSION
# ============================================

print("\nCreating submission...")

submission = sample.copy()
submission['academic_work_impact'] = (final_pred > 0.5).astype(int)

import os
os.makedirs('submissions', exist_ok=True)
submission_path = 'submissions/stacking_final_submission.csv'
submission.to_csv(submission_path, index=False)

print(f"\n✅ Submission saved to {submission_path}")

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

if ensemble_auc >= 0.54:
    print("\n🎉🎉🎉 CONGRATULATIONS! YOU BROKE 0.54! 🎉🎉🎉")
else:
    gap = 0.54 - ensemble_auc
    print(f"\n📊 Gap to 0.54: {gap:.4f}")
    print("\n📈 What's left to try:")
    print("   - Increase Stacking CV to 5")
    print("   - Use different meta-model (RandomForest, XGBoost)")
    print("   - Add more features from the original data")
