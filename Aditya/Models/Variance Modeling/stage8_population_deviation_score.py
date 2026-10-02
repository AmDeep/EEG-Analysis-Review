"""Stage 8: score a trial by how far its RESPONSE is from the POPULATION-
AVERAGE baseline -- not each subject's own baseline (that's Stage 1-2/6).

The idea: build one reference "what does a calm/resting healthy person's EEG
look like" from everyone in the dataset (already computed in Stage 7 --
`stage7_healthy_norm_reference.csv` has the population mean/std of every
baseline feature, subject-averaged). Then for any trial, however intense the
stimulus, measure

    Z_population = (response - population_baseline_mean) / population_baseline_std

i.e. how many population-baseline-standard-deviations this particular
response sits from "normal calm." Combine the features that Stage 2 already
found predictive into one composite score with the same recipe (ridge +
subject-grouped CV against laser_power, for validation only -- laser_power is
NOT an input to the score itself). Bucket the composite into 0-1/1-2/2-3/3+ SD
bands and build the calibration table: what actual laser power (pain
intensity) does each band correspond to, on average, in the labeled data?
That table IS the clinical lookup -- a new patient's SD band tells you which
row to read off, with no stimulus calibration or intensity input needed.

This is a different, simpler reference scheme than Stage 6: no per-subject
random effects, no per-subject historical baseline required at inference time
at all -- only the fixed population reference (computed once, here) and the
patient's own current EEG reading.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from stage3_consistency_analysis import BANDS, band_of, SEQUENTIAL_BLUE_CMAP
from stage2_neural_response_score import cross_validated_weights
from plots import SURFACE, INK_PRIMARY, INK_MUTED, GRIDLINE, BASELINE_AXIS, BLUE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_normalized_all.csv")
    ap.add_argument("--healthy-norm", default="stage7_healthy_norm_reference.csv")
    ap.add_argument("--feature-list-from", default="stage2_all_weights.csv",
                     help="reuse this stage's already-selected feature set")
    ap.add_argument("--output-prefix", default="stage8")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--n-splits", type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    norm = pd.read_csv(a.healthy_norm)
    norm = norm[norm["baseline_feature"] != "predicted_sensitivity"].set_index("baseline_feature")

    feature_bases = [f.replace("_z", "") for f in pd.read_csv(a.feature_list_from)["feature"]]
    print(f"Reusing {len(feature_bases)} features from {a.feature_list_from}: {feature_bases}")

    pop_cols = []
    for f in feature_bases:
        mean, std = norm.loc[f"{f}_baseline", "healthy_mean"], norm.loc[f"{f}_baseline", "healthy_std"]
        df[f"{f}_z_pop"] = (df[f"{f}_response"] - mean) / std
        pop_cols.append(f"{f}_z_pop")

    sub = df[["global_subject", "laser_power", *pop_cols]].dropna()
    X, y, groups = sub[pop_cols], sub["laser_power"], sub["global_subject"]
    print(f"{len(sub)} trials, {groups.nunique()} subjects")

    fold_results, fold_weights = cross_validated_weights(X, y, groups, a.n_splits)
    print("\n=== Population-baseline-referenced composite: held-out correlation with laser_power ===")
    print(fold_results.to_string(index=False))
    print(f"Cross-validated r = {fold_results['r'].mean():.3f} +/- {fold_results['r'].std():.3f} "
          f"(Stage 2's subject-referenced version, same 9 datasets: r=0.327)")

    scaler = StandardScaler().fit(X)
    model = Ridge(alpha=1.0).fit(scaler.transform(X), y)
    df = df.copy()
    df["population_deviation_score"] = np.nan
    df.loc[sub.index, "population_deviation_score"] = model.predict(scaler.transform(X))

    scored = df.dropna(subset=["population_deviation_score"]).copy()
    score_mean, score_std = scored["population_deviation_score"].mean(), scored["population_deviation_score"].std()
    scored["score_sd_units"] = (scored["population_deviation_score"] - score_mean) / score_std
    scored["band"] = scored["score_sd_units"].apply(band_of)

    band_order = [b[2] for b in BANDS]
    calibration = scored.groupby("band")["laser_power"].agg(["mean", "std", "min", "max", "count"]).reindex(band_order)
    calibration.to_csv(f"{a.output_prefix}_band_calibration.csv")
    print(f"\n=== SD band -> actual laser power calibration (n={len(scored)} trials) ===")
    print(calibration.to_string())

    rho = scored[["score_sd_units", "laser_power"]].corr().iloc[0, 1]
    print(f"\nPearson r(score_sd_units, laser_power) = {rho:.3f} (sanity check: higher band should mean higher power)")

    plots_dir = Path(a.plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    fig, ax = plt.subplots(figsize=(7, 5))
    groups_for_box = [scored.loc[scored["band"] == b, "laser_power"].to_numpy() for b in band_order]
    colors = [SEQUENTIAL_BLUE_CMAP(t) for t in np.linspace(0.15, 1.0, len(band_order))]
    bp = ax.boxplot(groups_for_box, tick_labels=band_order, patch_artist=True,
                     medianprops=dict(color=INK_PRIMARY, lw=1.5),
                     whiskerprops=dict(color=BASELINE_AXIS), capprops=dict(color=BASELINE_AXIS),
                     flierprops=dict(markeredgecolor=INK_MUTED, markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(BASELINE_AXIS)
    ax.set_xlabel("SD band (deviation from population-average baseline)")
    ax.set_ylabel("Actual laser power (J)")
    ax.set_title("Calibration: does the population-referenced SD band track real intensity?", fontsize=10)
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = plots_dir / f"{a.output_prefix}_band_calibration.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")

    scored.to_csv(f"{a.output_prefix}_scored_trials.csv", index=False)
    weights_out = pd.DataFrame({"feature": pop_cols, "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_,
                                 "ridge_coef": model.coef_})
    weights_out.attrs["intercept"] = model.intercept_
    weights_out.to_csv(f"{a.output_prefix}_weights.csv", index=False)
    print(f"\nsaved {a.output_prefix}_scored_trials.csv, {a.output_prefix}_weights.csv, "
          f"{a.output_prefix}_band_calibration.csv (intercept={model.intercept_:.4g}, "
          f"score_mean={score_mean:.4g}, score_std={score_std:.4g})")


if __name__ == "__main__":
    main()
