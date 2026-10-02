"""Stage 7: a clinically deployable score that needs NO stimulus and NO known
intensity -- just a resting/pre-stimulus baseline EEG recording.

Motivation: Stages 1-6 all depend on knowing laser_power, which won't exist
in a clinical deployment (patient gets a headset, not a calibrated research
laser). But 5 of our 9 datasets include an independently-measured, stimulus-
free ground truth: `Pain_Threshold` in participants.tsv -- the laser energy
that produces a pain rating of 7/10 for that person, from psychophysical
testing, not derived from any single EEG trial. That's the external
validation anchor Stage 4/6 didn't have (their "sensitivity" was itself
derived from the EEG).

Per instruction, every subject in all 9 datasets is treated as the HEALTHY
reference population for now (chronic pain comparison groups come later, once
that data exists). This script:

1. Pulls Pain_Threshold (or the differently-named equivalent column) from
   participants.tsv for the 255 subjects across 5 datasets that have it.
2. Ridge-regresses it on subject-averaged BASELINE features (K-fold CV, no
   stimulus/intensity involved at all -- purely resting-state characteristics)
   to test whether baseline EEG predicts a real, externally-measured
   sensitivity trait.
3. Builds the "healthy norm": population mean/SD of both the raw baseline
   features and the predicted-sensitivity score, computed over ALL 678
   subjects (not just the 255 with a measured threshold) via a model refit on
   the labeled subset and applied to everyone. This norm is what a new
   patient's baseline gets compared against at deployment -- no stimulus
   needed, no intensity needed, just their own baseline recording.
"""
import argparse
import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from plots import SURFACE, INK_PRIMARY, INK_MUTED, GRIDLINE, BASELINE_AXIS, BLUE

THRESHOLD_COLS = {
    "ds005284": "Pain_Threshold_7",
    "ds005286": "Pain_Threthold",
    "ds005289": "Energy",
    "ds005291": "Pain_Threthold",
    "ds005293": "Pain_Threshod",
}


def load_pain_thresholds(data_root):
    rows = []
    for ds, col in THRESHOLD_COLS.items():
        path = os.path.join(data_root, ds, "participants.tsv")
        if not os.path.exists(path):
            print(f"  [!] {path} not found, skipping {ds}")
            continue
        p = pd.read_csv(path, sep="\t")
        for _, r in p.iterrows():
            if pd.notna(r.get(col)):
                m = re.search(r"(\d+)", str(r["participant_id"]))
                sub_num = int(m.group(1))
                rows.append({"dataset": ds, "global_subject": f"{ds}_sub-{sub_num:03d}",
                             "pain_threshold": float(r[col])})
    out = pd.DataFrame(rows)
    print(f"loaded pain_threshold for {len(out)} subjects across {out['dataset'].nunique()} datasets")
    return out


def fit_cv_ridge(X, y, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    oof = np.full(len(y), np.nan)
    for tr, te in kf.split(X):
        scaler = StandardScaler().fit(X[tr])
        model = Ridge(alpha=1.0).fit(scaler.transform(X[tr]), y[tr])
        oof[te] = model.predict(scaler.transform(X[te]))
    r, p = stats.pearsonr(oof, y)
    return oof, r, p


def plot_validation(y_true, y_pred, r, out_path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true, y_pred, s=14, color=BLUE, alpha=0.5, linewidths=0)
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color=BASELINE_AXIS, lw=1, ls="--")
    ax.set_xlabel("Actual pain threshold (measured, J)")
    ax.set_ylabel("Predicted from baseline EEG (out-of-fold)")
    ax.set_title(f"Baseline EEG vs. independently-measured pain threshold\n(cross-validated r={r:.3f})",
                 fontsize=10)
    ax.grid(True, lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


def plot_healthy_distribution(scores, out_path):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(scores, bins=40, color=BLUE, alpha=0.75, edgecolor=SURFACE)
    ax.axvline(scores.mean(), color=INK_PRIMARY, lw=1.5, label=f"mean={scores.mean():.3g}")
    for k in (1, 2, 3):
        ax.axvline(scores.mean() + k * scores.std(), color=INK_MUTED, lw=1, ls="--")
        ax.axvline(scores.mean() - k * scores.std(), color=INK_MUTED, lw=1, ls="--")
    ax.set_xlabel("Predicted sensitivity (pain-threshold-equivalent, J)")
    ax.set_ylabel("Subjects")
    ax.set_title("Healthy reference distribution (all 9 datasets, n={})\n"
                 "dashed lines at +-1/2/3 SD".format(len(scores)), fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default="../data", help="root containing dsXXXXXX/participants.tsv")
    ap.add_argument("--features", default="features_normalized_all.csv")
    ap.add_argument("--output-prefix", default="stage7")
    ap.add_argument("--plots-dir", default="plots")
    a = ap.parse_args()

    thresholds = load_pain_thresholds(a.data_root)
    features = pd.read_csv(a.features)
    baseline_cols = [c for c in features.columns if c.endswith("_baseline")]
    per_subj_baseline = features.groupby("global_subject")[baseline_cols].mean().reset_index()
    print(f"{len(per_subj_baseline)} total subjects with baseline features (the full 'healthy' population)")

    labeled = thresholds.merge(per_subj_baseline, on="global_subject")
    print(f"{len(labeled)} subjects have both baseline features and a measured pain threshold")

    X_labeled = labeled[baseline_cols].fillna(labeled[baseline_cols].mean()).to_numpy()
    y_labeled = labeled["pain_threshold"].to_numpy()

    oof, r, p = fit_cv_ridge(X_labeled, y_labeled)
    print(f"\n=== Baseline EEG -> measured pain threshold (5-fold CV, {len(y_labeled)} subjects) ===")
    print(f"cross-validated r={r:.4f}, p={p:.3g}")

    plots_dir = a.plots_dir
    os.makedirs(plots_dir, exist_ok=True)
    plot_validation(y_labeled, oof, r, f"{plots_dir}/{a.output_prefix}_validation.png")

    # Refit on ALL labeled subjects (not CV-folded) to score the full population,
    # including subjects who never had a measured threshold.
    scaler = StandardScaler().fit(X_labeled)
    final_model = Ridge(alpha=1.0).fit(scaler.transform(X_labeled), y_labeled)

    X_all = per_subj_baseline[baseline_cols].fillna(per_subj_baseline[baseline_cols].mean()).to_numpy()
    per_subj_baseline = per_subj_baseline.copy()
    per_subj_baseline["predicted_sensitivity"] = final_model.predict(scaler.transform(X_all))

    healthy_mean = per_subj_baseline["predicted_sensitivity"].mean()
    healthy_std = per_subj_baseline["predicted_sensitivity"].std()
    per_subj_baseline["z_vs_healthy_norm"] = (per_subj_baseline["predicted_sensitivity"] - healthy_mean) / healthy_std
    print(f"\nHealthy reference: predicted_sensitivity mean={healthy_mean:.4g}, std={healthy_std:.4g}, "
          f"n={len(per_subj_baseline)}")

    plot_healthy_distribution(per_subj_baseline["predicted_sensitivity"].to_numpy(),
                               f"{plots_dir}/{a.output_prefix}_healthy_distribution.png")

    norm_reference = pd.DataFrame({
        "baseline_feature": baseline_cols,
        "healthy_mean": per_subj_baseline[baseline_cols].mean().to_numpy(),
        "healthy_std": per_subj_baseline[baseline_cols].std().to_numpy(),
    })
    norm_reference.loc[len(norm_reference)] = ["predicted_sensitivity", healthy_mean, healthy_std]
    norm_reference.to_csv(f"{a.output_prefix}_healthy_norm_reference.csv", index=False)
    per_subj_baseline.to_csv(f"{a.output_prefix}_subject_scores.csv", index=False)

    scaler_params = pd.DataFrame({"baseline_feature": baseline_cols,
                                   "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                                   "ridge_coef": final_model.coef_})
    scaler_params.attrs["intercept"] = final_model.intercept_
    scaler_params.to_csv(f"{a.output_prefix}_model_weights.csv", index=False)
    print(f"\nsaved {a.output_prefix}_healthy_norm_reference.csv, {a.output_prefix}_subject_scores.csv, "
          f"{a.output_prefix}_model_weights.csv (intercept={final_model.intercept_:.4g})")


if __name__ == "__main__":
    main()
