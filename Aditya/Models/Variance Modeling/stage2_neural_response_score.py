"""Stage 2: combine the Stage 1-significant, subject-consistent features into
a single candidate "Nurovo Neural Response Score":

    R = w1*Z_1 + w2*Z_2 + ... + intercept

Weights are NOT chosen by hand. They're learned by ridge regression predicting
laser_power from the selected Z-scored features, evaluated with subject-
grouped cross-validation (a subject's trials never appear in both train and
test) so the reported performance isn't inflated by within-subject leakage --
same discipline as ../CNN/train_eegnet.py's fold splitting.

Steps:
  1. Select features from stage1_results.csv: normalization == "z",
     p_linear < --p-thresh, subject_consistency >= --consistency-thresh.
  2. Greedy-drop redundant features (|correlation| > --corr-thresh), keeping
     the stronger (lower p_linear) of each pair -- e.g. rms/variance/std/
     hjorth_activity are near-duplicates of the same broadband-amplitude
     signal and would otherwise all pull the score in the same direction.
  3. GroupKFold (group = global_subject) ridge regression, standardized
     per-fold on the training subjects only. Reports out-of-fold correlation
     between R and laser_power as the generalization estimate.
  4. Refit on all data with the same recipe for the final deployable weights
     (reported separately from the CV performance estimate, not in place of it).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from plots import group_scatter, subject_spaghetti, SURFACE
import matplotlib.pyplot as plt


def select_features(results, features_df, p_thresh, consistency_thresh, corr_thresh):
    z_results = results[results["normalization"] == "z"].copy()
    sig = z_results[(z_results["p_linear"] < p_thresh) &
                     (z_results["subject_consistency"] >= consistency_thresh)]
    sig = sig.sort_values("p_linear")  # strongest evidence first

    candidate_cols = [f"{f}_z" for f in sig["feature"]]
    corr = features_df[candidate_cols].corr().abs()

    kept, dropped = [], {}
    for c in candidate_cols:
        collision = next((k for k in kept if corr.loc[c, k] > corr_thresh), None)
        if collision is not None:
            dropped[c] = collision
        else:
            kept.append(c)

    print(f"{len(z_results)} Z-normalized features tested -> {len(sig)} pass "
          f"p<{p_thresh:g} and consistency>={consistency_thresh:g} -> {len(kept)} kept after "
          f"dropping {len(dropped)} redundant (|r|>{corr_thresh:g}): {dropped}")
    return kept


def cross_validated_weights(X, y, groups, n_splits):
    gkf = GroupKFold(n_splits=n_splits)
    fold_rows, weight_rows = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler().fit(X.iloc[tr])
        Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])
        model = Ridge(alpha=1.0).fit(Xtr, y.iloc[tr])
        pred = model.predict(Xte)
        r, p = pearsonr(pred, y.iloc[te])
        fold_rows.append({"fold": fold, "n_test_trials": len(te),
                           "n_test_subjects": groups.iloc[te].nunique(), "r": r, "p": p})
        weight_rows.append(dict(zip(X.columns, model.coef_)))
    return pd.DataFrame(fold_rows), pd.DataFrame(weight_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_normalized.csv")
    ap.add_argument("--results", default="stage1_results.csv")
    ap.add_argument("--output-prefix", default="stage2")
    ap.add_argument("--p-thresh", type=float, default=1e-3)
    ap.add_argument("--consistency-thresh", type=float, default=0.8)
    ap.add_argument("--corr-thresh", type=float, default=0.95)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--plots-dir", default="plots")
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    results = pd.read_csv(a.results)

    kept = select_features(results, df, a.p_thresh, a.consistency_thresh, a.corr_thresh)
    if len(kept) < 2:
        raise SystemExit(f"Only {len(kept)} feature(s) survived selection -- loosen thresholds.")

    sub = df[["global_subject", "laser_power", *kept]].dropna()
    X, y, groups = sub[kept], sub["laser_power"], sub["global_subject"]

    fold_results, fold_weights = cross_validated_weights(X, y, groups, a.n_splits)
    print(f"\nPer-fold held-out correlation (R vs laser_power):")
    print(fold_results.to_string(index=False))
    print(f"\nCross-validated r = {fold_results['r'].mean():.3f} +/- {fold_results['r'].std():.3f} "
          f"across {a.n_splits} subject-grouped folds")

    print("\nPer-fold standardized weights (stability check):")
    print(fold_weights.describe().loc[["mean", "std"]].to_string())

    # Final deployable weights: refit on all data with the same recipe.
    final_scaler = StandardScaler().fit(X)
    final_model = Ridge(alpha=1.0).fit(final_scaler.transform(X), y)
    weights = pd.DataFrame({
        "feature": kept,
        "weight_standardized": final_model.coef_,
        "scaler_mean": final_scaler.mean_,
        "scaler_scale": final_scaler.scale_,
    })
    weights.attrs["intercept"] = final_model.intercept_
    weights_path = f"{a.output_prefix}_weights.csv"
    weights.to_csv(weights_path, index=False)
    print(f"\nsaved {weights_path} (intercept={final_model.intercept_:.4g}, "
          f"trained on all {X.shape[0]} trials / {groups.nunique()} subjects)")

    cv_path = f"{a.output_prefix}_cv_results.csv"
    fold_results.to_csv(cv_path, index=False)
    print(f"saved {cv_path}")

    # Score every trial (in-sample, for inspection/plotting -- NOT the generalization estimate above).
    df = df.copy()
    df["neural_response_score"] = np.nan
    df.loc[sub.index, "neural_response_score"] = final_model.predict(final_scaler.transform(X))
    scored_path = f"{a.output_prefix}_scored_trials.csv"
    df.to_csv(scored_path, index=False)
    print(f"saved {scored_path}")

    plot_sub = sub.copy()
    plot_sub["y"] = df.loc[sub.index, "neural_response_score"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    group_scatter(axes[0], plot_sub, "Neural Response Score", "R")
    subject_spaghetti(axes[1], plot_sub, "Neural Response Score", "R")
    fig.suptitle(f"Nurovo Neural Response Score vs. laser power "
                 f"[CV r={fold_results['r'].mean():.3f}]", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out_path = Path(a.plots_dir) / f"{a.output_prefix}_neural_response_score.png"
    out_path.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
