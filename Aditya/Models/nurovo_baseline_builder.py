#!/usr/bin/env python3
"""
Nurovo Laser Intensity Prediction - Rapid Baseline Builder (v2)
Loads intensity from BIDS event files.

Predicts physical stimulus intensity (laser_power, Joules) from EEG features.

Changes from v1:
  - Subject-grouped train/test split and CV (GroupShuffleSplit / GroupKFold)
    so the model is evaluated on subjects it has never seen at all, not just
    unseen trials from subjects it partly trained on. This was the biggest
    risk in v1: ordinary train_test_split lets the same subject's trials
    land in both train and test.
  - Missing-value imputation is fit on the TRAIN split only, then applied to
    test (v1 computed the median using the full dataset, which leaks test
    statistics into training).
  - Random Forest is trained once and reused for feature importance instead
    of being fit a second time.
  - Added a per-dataset breakdown of test R² for the best model, to flag
    whether performance is concentrated in one experiment (a sign the model
    may be exploiting dataset-specific quirks rather than a general
    EEG-intensity relationship).
  - Bumped SVR's C and added early_stopping to the MLP as a first pass at
    the untuned-defaults issue (still not a real hyperparameter search).
"""

import pandas as pd
import numpy as np
import warnings
import os
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================
BASE_PATH = r'C:\Users\hi2ad\OneDrive\UT Austin\Nurovo\Code\Baseline Models'
FEATURES_CSV = os.path.join(BASE_PATH, 'features_combined.csv')

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print(f"\n📂 Loading features from: {FEATURES_CSV}")
if not os.path.exists(FEATURES_CSV):
    print("   ❌ File not found!")
    exit()

df = pd.read_csv(FEATURES_CSV)
print(f"   ✓ Loaded {len(df)} rows, {len(df.columns)} columns")
print(f"\n🎯 Non-null intensity values: {df['laser_power'].notna().sum()}")

# ============================================================================
# 2. PREPARE DATA
# ============================================================================
print("\n" + "=" * 80)
print("DATA PREPARATION")
print("=" * 80)

df_clean = df[df['laser_power'].notna()].copy()
print(f"✓ Rows with intensity: {len(df_clean)}")

metadata_cols = ['dataset', 'subject', 'epoch', 'vertex_channel', 'sfreq', 'rating']
feature_cols = [c for c in df_clean.columns if c not in metadata_cols and c != 'laser_power']
print(f"✓ Features: {len(feature_cols)}")

X = df_clean[feature_cols].copy()
y = df_clean['laser_power'].copy()
groups = df_clean['subject'].copy()          # subject id, used for grouped splitting
dataset_ids = df_clean['dataset'].copy()     # kept for the per-dataset breakdown

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.dropna(axis=1, how='all')
feature_cols = list(X.columns)

print(f"✓ Final: X={X.shape}, y={y.shape}")
print(f"   Target mean: {y.mean():.4f}, std: {y.std():.4f}")
print(f"   Subjects: {groups.nunique()}, Datasets: {dataset_ids.nunique()}")

# ============================================================================
# 3. SPLIT DATA (grouped by subject)
# ============================================================================
print("\n" + "=" * 80)
print("TRAIN-TEST SPLIT (grouped by subject)")
print("=" * 80)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=groups))

X_train, X_test = X.iloc[train_idx].copy(), X.iloc[test_idx].copy()
y_train, y_test = y.iloc[train_idx].copy(), y.iloc[test_idx].copy()
groups_train = groups.iloc[train_idx]
dataset_test = dataset_ids.iloc[test_idx]

print(f"✓ Train: {X_train.shape} ({groups_train.nunique()} subjects)")
print(f"✓ Test:  {X_test.shape} ({groups.iloc[test_idx].nunique()} subjects)")
overlap = set(groups_train) & set(groups.iloc[test_idx])
print(f"✓ Subject overlap between train/test: {len(overlap)} (should be 0)")

# Impute using TRAIN statistics only, then apply the same fill values to test
imputer = SimpleImputer(strategy='median')
X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=feature_cols, index=X_train.index)
X_test = pd.DataFrame(imputer.transform(X_test), columns=feature_cols, index=X_test.index)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 4. BUILD MODELS
# ============================================================================
print("\n" + "=" * 80)
print("MODEL TRAINING & EVALUATION")
print("=" * 80)

models = {
    'Linear Regression': LinearRegression(),
    'Ridge (α=1.0)': Ridge(alpha=1.0),
    'Lasso (α=0.1)': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(
        n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVM (RBF)': SVR(kernel='rbf', C=10.0),
    'KNN (k=5)': KNeighborsRegressor(n_neighbors=5),
    'Neural Net': MLPRegressor(
        hidden_layer_sizes=(128, 64), max_iter=500, early_stopping=True, random_state=42
    ),
}

SCALED_MODELS = (LinearRegression, Ridge, Lasso, SVR, KNeighborsRegressor, MLPRegressor)

results = []
fitted_models = {}
group_kfold = GroupKFold(n_splits=5)

for name, model in models.items():
    print(f"\n🔧 {name}...")

    uses_scaled = isinstance(model, SCALED_MODELS)
    X_train_use = X_train_scaled if uses_scaled else X_train
    X_test_use = X_test_scaled if uses_scaled else X_test

    model.fit(X_train_use, y_train)
    fitted_models[name] = model

    y_train_pred = model.predict(X_train_use)
    y_test_pred = model.predict(X_test_use)

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
    test_mae = mean_absolute_error(y_test, y_test_pred)

    # Subject-grouped CV: each fold holds out an entirely different set of subjects
    cv_scores = cross_val_score(
        model, X_train_use, y_train, cv=group_kfold,
        groups=groups_train, scoring='r2', n_jobs=-1
    )

    results.append({
        'Model': name,
        'Train R²': train_r2,
        'Test R²': test_r2,
        'CV R² (mean±std)': f"{cv_scores.mean():.4f}±{cv_scores.std():.4f}",
        'Test RMSE': test_rmse,
        'Test MAE': test_mae,
    })

    print(f"   Train R²: {train_r2:.4f} | Test R²: {test_r2:.4f}")
    print(f"   RMSE: {test_rmse:.4f} | MAE: {test_mae:.4f}")

# ============================================================================
# 5. RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)
results_df = pd.DataFrame(results).sort_values('Test R²', ascending=False)
print("\n" + results_df.to_string(index=False))

# Per-dataset breakdown for the best model
best_name = results_df.iloc[0]['Model']
best_model = fitted_models[best_name]
uses_scaled = isinstance(best_model, SCALED_MODELS)
X_test_best = X_test_scaled if uses_scaled else X_test
y_test_pred_best = best_model.predict(X_test_best)

print("\n" + "=" * 80)
print(f"PER-DATASET TEST R² ({best_name})")
print("=" * 80)
per_dataset = pd.DataFrame({
    'dataset': dataset_test.values,
    'y_true': y_test.values,
    'y_pred': y_test_pred_best
})
for ds, grp in per_dataset.groupby('dataset'):
    if len(grp) >= 20:
        r2 = r2_score(grp['y_true'], grp['y_pred'])
        print(f"   {ds}: n={len(grp)}, R²={r2:.4f}")

# ============================================================================
# 6. FEATURE IMPORTANCE (reuse the already-fitted Random Forest)
# ============================================================================
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE (Random Forest)")
print("=" * 80)
rf_model = fitted_models['Random Forest']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)
print("\n" + feature_importance.head(15).to_string(index=False))

# ============================================================================
# 7. SAVE RESULTS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)
results_df.to_csv('baseline_results.csv', index=False)
feature_importance.to_csv('feature_importance.csv', index=False)
print("✓ baseline_results.csv")
print("✓ feature_importance.csv")

# ============================================================================
# 8. VISUALIZATION
# ============================================================================
print("\nGenerating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax1 = axes[0, 0]
results_sorted = results_df.sort_values('Test R²')
ax1.barh(results_sorted['Model'], results_sorted['Test R²'], color='steelblue')
ax1.set_xlabel('Test R²')
ax1.set_title('Model Performance (subject-grouped split)')
ax1.grid(axis='x', alpha=0.3)

ax2 = axes[0, 1]
top_features = feature_importance.head(10)
ax2.barh(range(len(top_features)), top_features['importance'], color='forestgreen')
ax2.set_yticks(range(len(top_features)))
ax2.set_yticklabels(top_features['feature'], fontsize=9)
ax2.set_xlabel('Importance')
ax2.set_title('Top 10 Features')
ax2.grid(axis='x', alpha=0.3)

ax3 = axes[1, 0]
x_pos = np.arange(len(results_df))
width = 0.35
ax3.bar(x_pos - width/2, results_df['Train R²'], width, label='Train', alpha=0.8)
ax3.bar(x_pos + width/2, results_df['Test R²'], width, label='Test', alpha=0.8)
ax3.set_ylabel('R²')
ax3.set_title('Train vs Test')
ax3.set_xticks(x_pos)
ax3.set_xticklabels([m[:10] for m in results_df['Model']], rotation=45, ha='right', fontsize=8)
ax3.legend()
ax3.grid(axis='y', alpha=0.3)

ax4 = axes[1, 1]
results_rmse = results_df.sort_values('Test RMSE')
ax4.barh(results_rmse['Model'], results_rmse['Test RMSE'], color='coral')
ax4.set_xlabel('Test RMSE')
ax4.set_title('Test RMSE')
ax4.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('baseline_comparison.png', dpi=150, bbox_inches='tight')
print("✓ baseline_comparison.png")

print("\n" + "=" * 80)
print("✅ COMPLETE")
print("=" * 80) 