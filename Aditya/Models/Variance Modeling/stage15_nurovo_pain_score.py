"""Stage 15: "NurovoPain" -- push the population-referenced deviation score
(Stage 8) as far as it honestly goes toward a single-reading, no-calibration
pain-severity index, by folding in the one validated lever Stage 8 left out
(Stage 6's baseline-predicted sensitivity) and testing whether a nonlinear
model captures more of the signal than ridge.

Deployment story stays identical to Stage 8: at inference time you only need
(a) a resting baseline reading and (b) the current response reading -- no
known stimulus intensity, no per-patient historical calibration. This stage
does NOT change that; it only adds a second thing computable from (a) alone
(sensitivity_hat) as an extra predictor, and compares model families.

sensitivity_hat leakage discipline: Stage 6 already produced a leave-one-
subject-out estimate of `fitted_slope` (only identifiable for the 124-subject
within-subject cohort, since a slope needs a subject's own intensity
variation) -- reuse that AS-IS for those 124 subjects. For the other 554
subjects (between-subjects-only designs, never used to fit that model), refit
the same ridge-on-baseline-features model on all 124 and apply it
out-of-sample. Every subject's sensitivity_hat is therefore either LOSO or
fully out-of-sample -- never in-sample.

Both a linear (Ridge) and nonlinear (GradientBoosting) mapping from
[9 population-referenced Z-features + sensitivity_hat] -> laser_power are
compared under the same subject-grouped CV as every other stage here, and the
weaker one is not silently dropped -- if the nonlinear model wins, that also
means the effective conversion from score to a laser_power (pain-intensity)
estimate needs to use that model, not a linear formula.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from stage3_consistency_analysis import BANDS, band_of, SEQUENTIAL_BLUE_CMAP
from plots import SURFACE, INK_PRIMARY, INK_MUTED, GRIDLINE, BASELINE_AXIS


def fit_sensitivity_for_all_subjects(features_all, blups, sensitivity_loso):
    """Return a DataFrame [global_subject, sensitivity_hat] covering every
    subject in features_all, leakage-free (see module docstring)."""
    baseline_cols = [c for c in features_all.columns if c.endswith("_baseline")]
    per_subj = features_all.groupby("global_subject")[baseline_cols].mean().reset_index()

    trained_cohort = set(blups["global_subject"])
    train_rows = per_subj[per_subj["global_subject"].isin(trained_cohort)].merge(
        blups[["global_subject", "fitted_slope"]], on="global_subject")
    X_train = train_rows[baseline_cols].fillna(train_rows[baseline_cols].mean()).to_numpy()
    y_train = train_rows["fitted_slope"].to_numpy()

    scaler = StandardScaler().fit(X_train)
    model = Ridge(alpha=1.0).fit(scaler.transform(X_train), y_train)

    holdout_rows = per_subj[~per_subj["global_subject"].isin(trained_cohort)].copy()
    X_holdout = holdout_rows[baseline_cols].fillna(train_rows[baseline_cols].mean()).to_numpy()
    holdout_rows["sensitivity_hat"] = model.predict(scaler.transform(X_holdout))

    loso_rows = sensitivity_loso[["global_subject", "sensitivity_hat"]].copy()
    out = pd.concat([loso_rows, holdout_rows[["global_subject", "sensitivity_hat"]]], ignore_index=True)
    print(f"sensitivity_hat: {len(loso_rows)} subjects from Stage 6's LOSO fit (in-cohort), "
          f"{len(holdout_rows)} subjects from out-of-sample application of that same model "
          f"({len(out)} total, covering {per_subj['global_subject'].nunique()} in features file)")
    return out


def cross_validated(X, y, groups, n_splits, model_name):
    gkf = GroupKFold(n_splits=n_splits)
    rows = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups)):
        scaler = StandardScaler().fit(X.iloc[tr])
        Xtr, Xte = scaler.transform(X.iloc[tr]), scaler.transform(X.iloc[te])
        if model_name == "ridge":
            model = Ridge(alpha=1.0).fit(Xtr, y.iloc[tr])
        else:
            model = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05,
                subsample=0.8, random_state=fold).fit(Xtr, y.iloc[tr])
        pred = model.predict(Xte)
        r, p = pearsonr(pred, y.iloc[te])
        rows.append({"fold": fold, "n_test_trials": len(te),
                     "n_test_subjects": groups.iloc[te].nunique(), "r": r, "p": p})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_normalized_all.csv")
    ap.add_argument("--healthy-norm", default="stage7_healthy_norm_reference.csv")
    ap.add_argument("--feature-list-from", default="stage2_all_weights.csv")
    ap.add_argument("--blups", default="stage4_subject_blups.csv")
    ap.add_argument("--sensitivity-hat", default="stage6_sensitivity_hat.csv")
    ap.add_argument("--output-prefix", default="stage15")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--n-splits", type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    norm = pd.read_csv(a.healthy_norm)
    norm = norm[norm["baseline_feature"] != "predicted_sensitivity"].set_index("baseline_feature")
    blups = pd.read_csv(a.blups)
    sensitivity_loso = pd.read_csv(a.sensitivity_hat)

    feature_bases = [f.replace("_z", "") for f in pd.read_csv(a.feature_list_from)["feature"]]
    print(f"Reusing {len(feature_bases)} Stage 2 features: {feature_bases}")

    pop_cols = []
    for f in feature_bases:
        mean, std = norm.loc[f"{f}_baseline", "healthy_mean"], norm.loc[f"{f}_baseline", "healthy_std"]
        df[f"{f}_z_pop"] = (df[f"{f}_response"] - mean) / std
        pop_cols.append(f"{f}_z_pop")

    sensitivity = fit_sensitivity_for_all_subjects(df, blups, sensitivity_loso)
    df = df.merge(sensitivity, on="global_subject", how="left")

    all_cols = pop_cols + ["sensitivity_hat"]
    sub = df[["global_subject", "laser_power", *all_cols]].dropna()
    X, y, groups = sub[all_cols], sub["laser_power"], sub["global_subject"]
    print(f"{len(sub)} trials, {groups.nunique()} subjects, {len(all_cols)} predictors "
          f"(vs. Stage 8's {len(pop_cols)})")

    print("\n=== Model comparison: held-out correlation with laser_power (subject-grouped CV) ===")
    results = {}
    for model_name in ("ridge", "gbr"):
        fold_results = cross_validated(X, y, groups, a.n_splits, model_name)
        r_mean, r_std = fold_results["r"].mean(), fold_results["r"].std()
        print(f"{model_name:>6}: r = {r_mean:.3f} +/- {r_std:.3f}")
        results[model_name] = (fold_results, r_mean, r_std)
    print(f"\n(Stage 8 baseline, same features minus sensitivity_hat, ridge only: r=0.320)")

    print("\nModel selection is NOT just highest CV r: the deployable output is a "
          "monotonic SD-band -> intensity table, and gradient boosting's raw score "
          "can win on aggregate r while still non-monotonically inverting the rare, "
          "clinically important extreme band (small-n tail extrapolation). Checked below.")
    best_name = "ridge"
    for candidate in ("gbr", "ridge"):
        fold_results, r_mean, r_std = results[candidate]
        scaler_c = StandardScaler().fit(X)
        Xs_c = scaler_c.transform(X)
        if candidate == "ridge":
            probe = Ridge(alpha=1.0).fit(Xs_c, y)
        else:
            probe = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(Xs_c, y)
        probe_pred = probe.predict(Xs_c)
        probe_z = (probe_pred - probe_pred.mean()) / probe_pred.std()
        probe_band = pd.Series(probe_z, index=sub.index).apply(band_of)
        band_means = y.groupby(probe_band).mean().reindex([b[2] for b in BANDS])
        monotonic = band_means.is_monotonic_increasing
        print(f"{candidate:>6}: CV r={r_mean:.3f}, in-sample band means monotonic={monotonic} "
              f"({band_means.round(2).to_dict()})")
        if candidate == "gbr" and monotonic and r_mean > results["ridge"][1]:
            best_name = "gbr"
    fold_results, r_mean, r_std = results[best_name]
    print(f"\nSelected model: {best_name} (r={r_mean:.3f} +/- {r_std:.3f}) "
          f"-- {'nonlinear model kept its calibration monotonic, so it wins outright' if best_name == 'gbr' else 'ridge kept for a trustworthy calibration table even though its raw CV r may be lower'}")
    fold_results.to_csv(f"{a.output_prefix}_cv_results.csv", index=False)

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    if best_name == "ridge":
        final_model = Ridge(alpha=1.0).fit(Xs, y)
        weights_out = pd.DataFrame({"feature": all_cols, "scaler_mean": scaler.mean_,
                                     "scaler_scale": scaler.scale_, "ridge_coef": final_model.coef_})
        weights_out.attrs["intercept"] = final_model.intercept_
        weights_out.to_csv(f"{a.output_prefix}_weights.csv", index=False)
    else:
        final_model = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0).fit(Xs, y)
        importances = pd.DataFrame({"feature": all_cols, "importance": final_model.feature_importances_})
        importances.to_csv(f"{a.output_prefix}_feature_importances.csv", index=False)
        pd.DataFrame({"scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
                      index=all_cols).to_csv(f"{a.output_prefix}_scaler.csv")

    df["nurovo_pain_raw"] = np.nan
    df.loc[sub.index, "nurovo_pain_raw"] = final_model.predict(Xs)

    scored = df.dropna(subset=["nurovo_pain_raw"]).copy()
    score_mean, score_std = scored["nurovo_pain_raw"].mean(), scored["nurovo_pain_raw"].std()
    scored["nurovo_pain_z"] = (scored["nurovo_pain_raw"] - score_mean) / score_std
    scored["band"] = scored["nurovo_pain_z"].apply(band_of)

    band_order = [b[2] for b in BANDS]
    calibration = scored.groupby("band")["laser_power"].agg(["mean", "std", "min", "max", "count"]).reindex(band_order)
    calibration.to_csv(f"{a.output_prefix}_band_calibration.csv")
    print(f"\n=== SD band -> actual laser power calibration (n={len(scored)} trials, model={best_name}) ===")
    print(calibration.to_string())

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
    ax.set_xlabel("NurovoPain SD band")
    ax.set_ylabel("Actual laser power (J)")
    ax.set_title(f"Stage 15 NurovoPain calibration [{best_name}, CV r={r_mean:.3f}]", fontsize=10)
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = plots_dir / f"{a.output_prefix}_band_calibration.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")

    scored.to_csv(f"{a.output_prefix}_scored_trials.csv", index=False)
    print(f"\nsaved {a.output_prefix}_scored_trials.csv, {a.output_prefix}_cv_results.csv, "
          f"{a.output_prefix}_band_calibration.csv")


if __name__ == "__main__":
    main()
