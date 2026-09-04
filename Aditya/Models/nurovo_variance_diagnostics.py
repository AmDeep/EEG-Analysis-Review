#!/usr/bin/env python3
"""
Nurovo Diagnostics: between- vs within-dataset variance decomposition
Checks whether the baseline model's pooled test R² is mostly explained by
between-dataset differences in laser_power (a confound) rather than genuine
EEG-based decoding of intensity within a single protocol.
"""

import pandas as pd
import numpy as np
import os
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

BASE_PATH = r'C:\Users\hi2ad\OneDrive\UT Austin\Nurovo\Code\Baseline Models'
FEATURES_CSV = os.path.join(BASE_PATH, 'features_combined.csv')

df = pd.read_csv(FEATURES_CSV)
df_clean = df[df['laser_power'].notna()].copy()

metadata_cols = ['dataset', 'subject', 'epoch', 'vertex_channel', 'sfreq', 'rating']
feature_cols = [c for c in df_clean.columns if c not in metadata_cols and c != 'laser_power']

X = df_clean[feature_cols].copy()
y = df_clean['laser_power'].copy()
groups = df_clean['subject'].copy()
dataset_ids = df_clean['dataset'].copy()

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')
X = X.dropna(axis=1, how='all')
feature_cols = list(X.columns)

# ---------------------------------------------------------------------------
# 1. VARIANCE DECOMPOSITION
#    How much of laser_power's total variance is between-dataset (a protocol/
#    hardware confound) vs within-dataset (the real trial-to-trial signal a
#    decoder should be reading from EEG)?
# ---------------------------------------------------------------------------
print("=" * 80)
print("PER-DATASET laser_power STATS")
print("=" * 80)
stats = df_clean.groupby('dataset')['laser_power'].agg(['count', 'mean', 'std'])
print(stats.to_string())

grand_mean = y.mean()
ss_total = ((y - grand_mean) ** 2).sum()
ss_between = sum(
    len(grp) * (grp['laser_power'].mean() - grand_mean) ** 2
    for _, grp in df_clean.groupby('dataset')
)
ss_within = ss_total - ss_between

print(f"\nTotal variance (SS):   {ss_total:.2f}")
print(f"Between-dataset SS:    {ss_between:.2f}  ({100 * ss_between / ss_total:.1f}% of total)")
print(f"Within-dataset SS:     {ss_within:.2f}  ({100 * ss_within / ss_total:.1f}% of total)")
print("\n-> A large between-dataset share means a chunk of pooled R2 could come")
print("   from the model implicitly identifying which protocol/hardware a trial")
print("   belongs to, rather than decoding real intensity variation from EEG.")

# ---------------------------------------------------------------------------
# 2. RESIDUALIZED TARGET
#    Subtract each dataset's own mean laser_power, forcing the model to
#    explain ONLY within-dataset variation. Same subject-grouped split as
#    the main baseline script, so results are comparable.
# ---------------------------------------------------------------------------
df_clean['laser_power_centered'] = (
    df_clean.groupby('dataset')['laser_power'].transform(lambda v: v - v.mean())
)
y_centered = df_clean['laser_power_centered']

imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=feature_cols, index=X.index)

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X_imputed, y_centered, groups=groups))

X_train, X_test = X_imputed.iloc[train_idx], X_imputed.iloc[test_idx]
y_train_c, y_test_c = y_centered.iloc[train_idx], y_centered.iloc[test_idx]
dataset_test = dataset_ids.iloc[test_idx]

rf = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train_c)
y_pred_c = rf.predict(X_test)

print("\n" + "=" * 80)
print("RANDOM FOREST ON DATASET-CENTERED laser_power (within-dataset signal only)")
print("=" * 80)
print(f"Overall Test R2: {r2_score(y_test_c, y_pred_c):.4f}")
print("(Compare this to the pooled 0.276 from the main script. If this number")
print(" is much lower or near zero, most of that 0.276 was the between-dataset")
print(" confound, not real EEG-based intensity decoding.)")

per_dataset = pd.DataFrame({
    'dataset': dataset_test.values,
    'y_true': y_test_c.values,
    'y_pred': y_pred_c,
})
print("\nPer-dataset R2 on centered target:")
for ds, grp in per_dataset.groupby('dataset'):
    if len(grp) >= 20:
        print(f"   {ds}: n={len(grp)}, R2={r2_score(grp['y_true'], grp['y_pred']):.4f}")

# ---------------------------------------------------------------------------
# 3. PER-SUBJECT CENTERED TARGET
#    Some designs (e.g. binary Low/High) calibrate laser_power to each
#    subject's own pain threshold, so a chunk of "within-dataset" variation
#    may really be between-subject calibration differences rather than
#    trial-to-trial signal. Centering by subject instead isolates genuine
#    within-subject decoding. Reuses the same split (same gss, same inputs).
# ---------------------------------------------------------------------------
df_clean['laser_power_subj_centered'] = (
    df_clean.groupby('subject')['laser_power'].transform(lambda v: v - v.mean())
)
y_subj_centered = df_clean['laser_power_subj_centered']

y_train_s, y_test_s = y_subj_centered.iloc[train_idx], y_subj_centered.iloc[test_idx]

rf_subj = RandomForestRegressor(
    n_estimators=100, max_depth=10, min_samples_leaf=5, random_state=42, n_jobs=-1
)
rf_subj.fit(X_train, y_train_s)
y_pred_s = rf_subj.predict(X_test)

print("\n" + "=" * 80)
print("RANDOM FOREST ON SUBJECT-CENTERED laser_power (within-subject signal only)")
print("=" * 80)
print(f"Overall Test R2: {r2_score(y_test_s, y_pred_s):.4f}")

per_dataset_s = pd.DataFrame({
    'dataset': dataset_test.values,
    'y_true': y_test_s.values,
    'y_pred': y_pred_s,
})
print("\nPer-dataset R2 on subject-centered target:")
for ds, grp in per_dataset_s.groupby('dataset'):
    if len(grp) >= 20:
        print(f"   {ds}: n={len(grp)}, R2={r2_score(grp['y_true'], grp['y_pred']):.4f}")

# ---------------------------------------------------------------------------
# 4. MACRO-AVERAGED R2
#    Pooled R2 is a single sum-of-squares calc across all test trials, so a
#    large-variance dataset (e.g. ds005293) can dominate it even when most
#    other datasets do poorly. Macro-averaging treats every dataset equally.
# ---------------------------------------------------------------------------
def macro_avg_r2(df_pred, min_n=20):
    r2s = [
        r2_score(grp['y_true'], grp['y_pred'])
        for _, grp in df_pred.groupby('dataset') if len(grp) >= min_n
    ]
    return float(np.mean(r2s)), r2s

macro_c, _ = macro_avg_r2(per_dataset)
macro_s, _ = macro_avg_r2(per_dataset_s)

print("\n" + "=" * 80)
print("MACRO-AVERAGED (UNWEIGHTED) R2 ACROSS DATASETS")
print("=" * 80)
print(f"  Dataset-centered:  {macro_c:.4f}")
print(f"  Subject-centered:  {macro_s:.4f}")
print("(This treats every dataset equally instead of letting the largest")
print(" dataset dominate the summary number - a fairer view given how much")
print(" decodability varies by protocol/hardware.)")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)