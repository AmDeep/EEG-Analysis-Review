# """
# NUROVO TUESDAY DELIVERABLE PIPELINE
# Scoped to ds005293 + ds005473 (the two datasets with confirmed real signal
# from today's regression + clustering + intensity-range triangulation).

# Runs, in order:
#   1. Load fixed features, subset to the 2 target datasets
#   2. Random Forest vs Gradient Boosting, global_subject-grouped 5-fold CV
#      (fixes the earlier subject-collision leakage bug)
#   3. Per-dataset AND combined metrics (R², MAE) -- no misleading pooled
#      macro-average across all 9
#   4. Subject-centered target ablation (Oura/Whoop-style personal baseline)
#   5. Trial-aggregation test (Fitbit/Apple-style persistence-over-trials)
#   6. Both `laser_power` (fully labeled) and `rating` (96%+ labeled,
#      rating-NaN dropped) run as separate targets so you can compare which
#      one the team wants to report Tuesday

# Requirements: pip install scikit-learn pandas numpy
# """

# import os
# import numpy as np
# import pandas as pd
# from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
# from sklearn.model_selection import GroupKFold
# from sklearn.metrics import r2_score, mean_absolute_error

# BASE_PATH = r'C:\Users\hi2ad\OneDrive\UT Austin\Nurovo\Code\Baseline Models\KNNs'
# FEATURES_CSV = os.path.join(BASE_PATH, 'features_all_epochs_FIXED.csv')

# TARGET_DATASETS = ['ds005293', 'ds005473']
# N_FOLDS = 5
# RANDOM_STATE = 42
# AGGREGATION_WINDOW = 5  # trials to average per aggregated prediction

# pd.set_option('display.width', 140)


# # ============================================================================
# # 1. LOAD + SUBSET
# # ============================================================================

# def load_subset():
#     df = pd.read_csv(FEATURES_CSV)
#     df = df[df['dataset'].isin(TARGET_DATASETS)].copy()
#     print(f"Loaded {len(df)} trials from {TARGET_DATASETS}")
#     print(df['dataset'].value_counts())
#     print(f"global_subject nunique: {df['global_subject'].nunique()}\n")
#     return df


# def get_feature_cols(df):
#     metadata = ['dataset', 'subject', 'global_subject', 'epoch', 'sfreq',
#                 'laser_power', 'rating', 'source_file']
#     return [c for c in df.columns if c not in metadata]


# # ============================================================================
# # 2-3. MODEL COMPARISON, GROUPED CV, PER-DATASET + COMBINED METRICS
# # ============================================================================

# def run_grouped_cv(df, feature_cols, target_col, subject_centered=False):
#     """GroupKFold on global_subject. Returns a DataFrame of per-fold,
#     per-model, per-dataset results, plus out-of-fold predictions for the
#     trial-aggregation step."""
#     work = df.dropna(subset=[target_col]).copy()
#     if subject_centered:
#         # Subtract each subject's own mean target (Oura/Whoop-style personal
#         # baseline) -- model predicts DEVIATION from personal baseline, not
#         # the raw value. Predictions get added back for reporting.
#         subj_means = work.groupby('global_subject')[target_col].transform('mean')
#         work['_target_centered'] = work[target_col] - subj_means
#         work['_subj_mean'] = subj_means
#         y_col = '_target_centered'
#     else:
#         y_col = target_col

#     X = work[feature_cols].fillna(work[feature_cols].median()).values
#     y = work[y_col].values
#     groups = work['global_subject'].values

#     models = {
#         'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=12,
#                                                min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
#         'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=4,
#                                                        learning_rate=0.05, random_state=RANDOM_STATE),
#     }

#     gkf = GroupKFold(n_splits=N_FOLDS)
#     oof_preds = {name: np.full(len(work), np.nan) for name in models}
#     fold_results = []

#     for fold_idx, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
#         for name, model in models.items():
#             m = type(model)(**model.get_params())
#             m.fit(X[train_idx], y[train_idx])
#             preds = m.predict(X[test_idx])
#             oof_preds[name][test_idx] = preds

#             # add personal baseline back for real-scale metrics
#             if subject_centered:
#                 actual = work[target_col].values[test_idx]
#                 preds_real = preds + work['_subj_mean'].values[test_idx]
#             else:
#                 actual = y[test_idx]
#                 preds_real = preds

#             fold_results.append({
#                 'fold': fold_idx, 'model': name,
#                 'r2': r2_score(actual, preds_real),
#                 'mae': mean_absolute_error(actual, preds_real),
#                 'n_test': len(test_idx),
#             })

#     results_df = pd.DataFrame(fold_results)
#     work_out = work.copy()
#     for name in models:
#         real_scale = oof_preds[name] + (work['_subj_mean'].values if subject_centered else 0)
#         work_out[f'pred_{name}'] = real_scale

#     return results_df, work_out


# def summarize_cv(results_df, label):
#     print(f"\n--- {label} ---")
#     summary = results_df.groupby('model')[['r2', 'mae']].agg(['mean', 'std'])
#     print(summary.round(4))


# def per_dataset_metrics(work_out, target_col, model_names):
#     print(f"\n--- Per-dataset metrics (target={target_col}) ---")
#     for ds in TARGET_DATASETS:
#         sub = work_out[work_out['dataset'] == ds]
#         for name in model_names:
#             valid = sub[f'pred_{name}'].notna()
#             if valid.sum() > 1:
#                 r2 = r2_score(sub.loc[valid, target_col], sub.loc[valid, f'pred_{name}'])
#                 mae = mean_absolute_error(sub.loc[valid, target_col], sub.loc[valid, f'pred_{name}'])
#                 print(f"  {ds:12s} {name:18s} R2={r2:.4f}  MAE={mae:.4f}  n={valid.sum()}")


# # ============================================================================
# # 5. TRIAL-AGGREGATION TEST
# # ============================================================================

# def test_trial_aggregation(work_out, target_col, model_names, window=AGGREGATION_WINDOW):
#     """Fitbit/Apple-style persistence: does averaging predictions over a
#     rolling window of consecutive trials (within subject) beat single-trial
#     predictions?"""
#     print(f"\n--- Trial aggregation test (window={window} trials, target={target_col}) ---")
#     work_sorted = work_out.sort_values(['global_subject', 'epoch']).copy()

#     for name in model_names:
#         pred_col = f'pred_{name}'
#         valid = work_sorted[pred_col].notna()
#         sub = work_sorted[valid].copy()

#         # rolling mean of predictions within each subject
#         sub['pred_agg'] = sub.groupby('global_subject')[pred_col].transform(
#             lambda s: s.rolling(window, min_periods=1, center=True).mean()
#         )

#         r2_single = r2_score(sub[target_col], sub[pred_col])
#         mae_single = mean_absolute_error(sub[target_col], sub[pred_col])
#         r2_agg = r2_score(sub[target_col], sub['pred_agg'])
#         mae_agg = mean_absolute_error(sub[target_col], sub['pred_agg'])

#         print(f"  {name:18s} single-trial: R2={r2_single:.4f} MAE={mae_single:.4f}  |  "
#               f"aggregated: R2={r2_agg:.4f} MAE={mae_agg:.4f}  "
#               f"({'IMPROVED' if r2_agg > r2_single else 'no improvement'})")


# # ============================================================================
# # MAIN
# # ============================================================================

# def main():
#     df = load_subset()
#     feature_cols = get_feature_cols(df)
#     print(f"Using {len(feature_cols)} feature columns\n")

#     for target_col in ['laser_power', 'rating']:
#         print("\n" + "=" * 100)
#         print(f"TARGET: {target_col}")
#         print("=" * 100)
#         n_available = df[target_col].notna().sum()
#         print(f"{n_available} / {len(df)} trials have this target")

#         # --- raw target ---
#         results_raw, work_raw = run_grouped_cv(df, feature_cols, target_col, subject_centered=False)
#         summarize_cv(results_raw, f"{target_col} - RAW target - grouped CV (mean +/- std across {N_FOLDS} folds)")
#         per_dataset_metrics(work_raw, target_col, ['RandomForest', 'GradientBoosting'])
#         test_trial_aggregation(work_raw, target_col, ['RandomForest', 'GradientBoosting'])

#         # --- subject-centered target ---
#         results_centered, work_centered = run_grouped_cv(df, feature_cols, target_col, subject_centered=True)
#         summarize_cv(results_centered, f"{target_col} - SUBJECT-CENTERED target - grouped CV")
#         per_dataset_metrics(work_centered, target_col, ['RandomForest', 'GradientBoosting'])

#     print("\n" + "=" * 100)
#     print("DONE. Compare RAW vs SUBJECT-CENTERED and single-trial vs aggregated")
#     print("numbers above to decide what to lock in for Tuesday.")
#     print("=" * 100)


# if __name__ == "__main__":
#     main()







"""
Corrected laser_power prediction pipeline. Fixes a leakage bug from the
previous run: subject-centering computed each subject's personal baseline
using ALL of that subject's trials, including the ones being predicted
(since GroupKFold puts each subject entirely in one fold). This version
instead uses a genuine calibration/evaluation split per subject -- the
personal baseline is computed ONLY from a subject's first N trials
(calibration), and applied to their remaining trials (evaluation), which
were never used to compute it. This mirrors Garmin/Whoop's onboarding-
period pattern and is honestly deployable.

Also adds `dataset` as an explicit feature (legitimate -- known upfront,
not leakage) instead of letting the model implicitly discover it and
inflate the pooled score, and checks whether laser_power is actually a
small number of discrete levels rather than continuous.
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score, mean_absolute_error

BASE_PATH = r'C:\Users\hi2ad\OneDrive\UT Austin\Nurovo\Code\Baseline Models\KNNs'
FEATURES_CSV = os.path.join(BASE_PATH, 'features_all_epochs_FIXED.csv')

TARGET_DATASETS = ['ds005293', 'ds005473']
N_FOLDS = 5
RANDOM_STATE = 42
CALIBRATION_TRIALS = 10  # trials reserved per subject purely to compute their baseline offset

pd.set_option('display.width', 140)


def load_subset():
    df = pd.read_csv(FEATURES_CSV)
    df = df[df['dataset'].isin(TARGET_DATASETS)].copy()
    df = df.dropna(subset=['laser_power']).copy()
    print(f"Loaded {len(df)} trials from {TARGET_DATASETS}")
    print(f"global_subject nunique: {df['global_subject'].nunique()}\n")
    return df


def check_discreteness(df):
    print("=" * 100)
    print("LASER_POWER DISCRETENESS CHECK")
    print("=" * 100)
    for ds in TARGET_DATASETS:
        vals = sorted(df[df['dataset'] == ds]['laser_power'].unique())
        print(f"{ds}: {len(vals)} unique values -> {vals}")
    print()


def get_feature_cols(df):
    metadata = ['dataset', 'subject', 'global_subject', 'epoch', 'sfreq',
                'laser_power', 'rating', 'source_file', 'trial_rank', 'is_calibration']
    return [c for c in df.columns if c not in metadata]


def run_config(df, feature_cols, use_dataset_feature, use_calibration):
    """Single grouped-CV run. If use_calibration, splits each subject's
    trials into calibration (first CALIBRATION_TRIALS by epoch, used ONLY
    to compute that subject's offset) and evaluation (the rest, used for
    metrics). Calibration trials of TRAINING-fold subjects are still used
    for model training (legitimate -- not being evaluated on)."""
    work = df.sort_values(['global_subject', 'epoch']).copy()
    work['dataset_orig'] = work['dataset']  # keep before any dummy-encoding

    feat_cols = feature_cols.copy()
    if use_dataset_feature:
        work = pd.get_dummies(work, columns=['dataset'], prefix='ds')
        feat_cols = feature_cols + [c for c in work.columns if c.startswith('ds_')]

    if use_calibration:
        work['trial_rank'] = work.groupby('global_subject').cumcount()
        work['is_calibration'] = work['trial_rank'] < CALIBRATION_TRIALS
        trial_counts = work.groupby('global_subject').size()
        valid_subjects = trial_counts[trial_counts > CALIBRATION_TRIALS].index
        n_dropped = work['global_subject'].nunique() - len(valid_subjects)
        if n_dropped:
            print(f"    [note] dropping {n_dropped} subject(s) with <= {CALIBRATION_TRIALS} trials "
                  f"(not enough for calibration split)")
        work = work[work['global_subject'].isin(valid_subjects)].copy()
    else:
        work['is_calibration'] = False

    groups = work['global_subject'].values
    gkf = GroupKFold(n_splits=N_FOLDS)

    models = {
        'RandomForest': RandomForestRegressor(n_estimators=300, max_depth=12,
                                               min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=300, max_depth=4,
                                                       learning_rate=0.05, random_state=RANDOM_STATE),
    }

    all_results = {name: {'true': [], 'pred': [], 'dataset': []} for name in models}

    for train_idx, test_idx in gkf.split(work, work['laser_power'], groups):
        train_df = work.iloc[train_idx]
        test_df = work.iloc[test_idx]

        X_train_raw = train_df[feat_cols]
        median_fill = X_train_raw.median()  # fit imputation on TRAIN only
        X_train = X_train_raw.fillna(median_fill).values
        y_train = train_df['laser_power'].values

        test_calib = test_df[test_df['is_calibration']]
        test_eval = test_df[~test_df['is_calibration']] if use_calibration else test_df

        X_eval = test_eval[feat_cols].fillna(median_fill).values

        for name, model_template in models.items():
            model = type(model_template)(**model_template.get_params())
            model.fit(X_train, y_train)

            pred_eval = model.predict(X_eval)

            if use_calibration and len(test_calib) > 0:
                X_calib = test_calib[feat_cols].fillna(median_fill).values
                pred_calib = model.predict(X_calib)
                calib_df = test_calib.copy()
                calib_df['pred'] = pred_calib
                offsets = calib_df.groupby('global_subject').apply(
                    lambda g: (g['laser_power'] - g['pred']).mean())
                eval_offsets = test_eval['global_subject'].map(offsets).values
                pred_eval = pred_eval + eval_offsets

            all_results[name]['true'].extend(test_eval['laser_power'].tolist())
            all_results[name]['pred'].extend(pred_eval.tolist())
            all_results[name]['dataset'].extend(test_eval['dataset_orig'].tolist())

    return all_results


def report_fixed(results, label):
    print(f"\n--- {label} ---")
    for name, r in results.items():
        true = np.array(r['true'])
        pred = np.array(r['pred'])
        ds_arr = np.array(r['dataset'])
 
        r2 = r2_score(true, pred)
        mae = mean_absolute_error(true, pred)
        within_half_overall = np.mean(np.abs(pred - true) <= 0.5)
        print(f"  {name:18s} OVERALL  R2={r2:.4f}  MAE={mae:.4f}  "
              f"within+/-0.5J={within_half_overall*100:.1f}%  n={len(true)}")
 
        for ds in TARGET_DATASETS:
            mask = ds_arr == ds
            if mask.sum() > 1:
                r2_ds = r2_score(true[mask], pred[mask])
                mae_ds = mean_absolute_error(true[mask], pred[mask])
                within_half_ds = np.mean(np.abs(pred[mask] - true[mask]) <= 0.5)
                print(f"    {ds:12s} R2={r2_ds:.4f}  MAE={mae_ds:.4f}  "
                      f"within+/-0.5J={within_half_ds*100:.1f}%  n={mask.sum()}")


def main():
    df = load_subset()
    check_discreteness(df)
    feature_cols = get_feature_cols(df)
    print(f"Using {len(feature_cols)} feature columns\n")

    print("=" * 100)
    print("CONFIG A: raw model, no dataset feature (baseline -- same as before, redone with")
    print("          train-only imputation to remove even the mild feature-side leakage)")
    print("=" * 100)
    results_a = run_config(df, feature_cols, use_dataset_feature=False, use_calibration=False)
    report_fixed(results_a, "Config A: no dataset feature, no calibration")

    print("\n" + "=" * 100)
    print("CONFIG B: + dataset as explicit feature (legitimate, known upfront)")
    print("=" * 100)
    results_b = run_config(df, feature_cols, use_dataset_feature=True, use_calibration=False)
    report_fixed(results_b, "Config B: + dataset feature, no calibration")

    print("\n" + "=" * 100)
    print(f"CONFIG C: + calibration-trial personal baseline (first {CALIBRATION_TRIALS} trials/subject,")
    print("          leak-free -- offset computed only from calibration trials, applied to the rest)")
    print("=" * 100)
    results_c = run_config(df, feature_cols, use_dataset_feature=True, use_calibration=True)
    report_fixed(results_c, "Config C: + dataset feature + calibration offset")

    print("\n" + "=" * 100)
    print("Compare A vs B vs C above. C is the only config with a personal-baseline")
    print("correction that is actually honest and deployable.")
    print("=" * 100)


if __name__ == "__main__":
    main()