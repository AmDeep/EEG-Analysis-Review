"""Stage 6: use what Stage 4 found (baseline state predicts sensitivity) as an
actual covariate, then build a standardized deviation score that's fair across
the whole intensity range rather than assuming one global SD.

Three steps:

1. PREDICT SENSITIVITY FROM BASELINE STATE (honestly). Stage 4 found several
   individual baseline features correlate with a subject's fitted slope.
   Ridge-regress the fitted slope on all ~20 baseline features with
   leave-one-subject-out CV, so `sensitivity_hat` for each subject is an
   out-of-fold prediction, not a circular in-sample fit.

2. MODERATED MIXED MODEL: score ~ laser_power * sensitivity_hat, with a
   subject random intercept + slope, same as Stage 4's model but now with a
   covariate that can explain away *some* of that random slope variance.
   Comparing the random-slope variance here against Stage 4's baseline model
   quantifies how much of the "who responds more" question the baseline
   covariate answers.

3. HETEROSCEDASTICITY-CORRECTED STANDARDIZATION: instead of one global SD
   (which Stage 3/4 showed is wrong -- variance grows ~4-5x with intensity),
   fit a smooth variance function log(residual^2) ~ laser_power, and divide
   each trial's residual by the *predicted* SD at its own intensity. This is
   the properly standardized version of the "N SDs from baseline" scale: a
   deviation at high intensity isn't automatically flagged as "more extreme"
   just because everyone is noisier there.

Validated by re-running Stage 3's exact same checks (Levene/BP, SD-band
composition per level) on the corrected score -- if this worked, the bands
should now look similar across intensity levels instead of concentrating in
the high bands only at high power.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

from stage3_consistency_analysis import BANDS, band_of, SEQUENTIAL_BLUE_CMAP
from plots import SURFACE, INK_PRIMARY, INK_MUTED, GRIDLINE, BASELINE_AXIS


def predict_sensitivity_loo(blups, features_df):
    baseline_cols = [c for c in features_df.columns if c.endswith("_baseline")]
    per_subj = features_df.groupby("global_subject")[baseline_cols].mean().reset_index()
    merged = blups.merge(per_subj, on="global_subject").dropna(subset=["fitted_slope"])
    X = merged[baseline_cols].fillna(merged[baseline_cols].mean()).to_numpy()
    y = merged["fitted_slope"].to_numpy()

    loo = LeaveOneOut()
    pred = np.full(len(y), np.nan)
    for tr, te in loo.split(X):
        scaler = StandardScaler().fit(X[tr])
        model = Ridge(alpha=1.0).fit(scaler.transform(X[tr]), y[tr])
        pred[te] = model.predict(scaler.transform(X[te]))

    r, p = stats.pearsonr(pred, y)
    print(f"=== Sensitivity prediction from baseline state (leave-one-subject-out CV) ===")
    print(f"predicted vs. actual fitted_slope: r={r:.3f}, p={p:.3g} ({len(y)} subjects)")

    out = merged[["global_subject"]].copy()
    out["sensitivity_hat"] = pred
    return out, r, p


def moderated_model(df):
    m_mod = smf.mixedlm("score ~ laser_power * sensitivity_hat", df, groups=df["global_subject"],
                         re_formula="~laser_power").fit(reml=False)
    print("\n=== Moderated random-slopes model (score ~ laser_power * sensitivity_hat) ===")
    print(m_mod.summary().tables[1])

    slope_var = m_mod.cov_re.loc["laser_power", "laser_power"]
    print(f"\nRandom slope variance WITH sensitivity_hat covariate: {slope_var:.4g} "
          f"(Stage 4, without covariate: 0.02202 -- "
          f"{100 * (1 - slope_var / 0.02202):.1f}% reduction)")
    return m_mod


def fit_variance_function(df, resid_col="mod_resid"):
    """log(e^2) ~ laser_power via OLS, then predict E[e^2] with Duan's smearing
    estimator rather than naively exponentiating the log-scale fit. Jensen's
    inequality means E[log(e^2)] < log(E[e^2]), so a naive exp(fitted) of the
    log-OLS prediction systematically UNDERESTIMATES the variance at every
    power level (verified empirically: predicted sigma came out 1.4-2.2x too
    small before this fix). The smearing estimator -- multiplying by the mean
    of exp(residuals) from this same fit -- corrects that bias nonparametrically,
    without assuming a distribution for e^2."""
    X = sm.add_constant(df["laser_power"])
    log_resid2 = np.log(df[resid_col] ** 2 + 1e-8)
    vf = sm.OLS(log_resid2, X).fit()
    smear = float(np.mean(np.exp(vf.resid)))
    print(f"\nVariance function: log(resid^2) ~ laser_power, slope={vf.params['laser_power']:.4g}, "
          f"p={vf.pvalues['laser_power']:.3g}, smearing factor={smear:.3g}")
    return vf, smear


def predict_sigma(vf, smear, power):
    X = sm.add_constant(power, has_constant="add")
    return np.sqrt(np.exp(vf.predict(X)) * smear)


def check_heteroscedasticity(df, col, label):
    groups = [df.loc[df["laser_power"] == lv, col].to_numpy() for lv in sorted(df["laser_power"].unique())]
    groups = [g for g in groups if len(g) >= 2]
    stat, p = stats.levene(*groups)
    print(f"Levene's test on {label} across intensity levels: stat={stat:.3g}, p={p:.3g} "
          f"({'still heteroscedastic' if p < 0.05 else 'variance now roughly constant across intensity'})")
    return stat, p


def plot_band_by_level(df, col, out_path, title):
    levels = sorted(df["laser_power"].unique())
    band_order = [b[2] for b in BANDS]
    crosstab = pd.crosstab(df["laser_power"], df[col], normalize="index").reindex(columns=band_order, fill_value=0)

    fig, ax = plt.subplots(figsize=(9, 5))
    bottom = np.zeros(len(levels))
    colors = [SEQUENTIAL_BLUE_CMAP(t) for t in np.linspace(0.15, 1.0, len(band_order))]
    for band, color in zip(band_order, colors):
        vals = crosstab.loc[levels, band].to_numpy() * 100
        ax.bar([str(lv) for lv in levels], vals, bottom=bottom, color=color, label=band, edgecolor=SURFACE)
        bottom += vals

    ax.set_xlabel("Laser power")
    ax.set_ylabel("% of observations")
    ax.set_title(title, fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="stage2_scored_trials.csv")
    ap.add_argument("--blups", required=True, help="stage4_subject_blups.csv")
    ap.add_argument("--output-prefix", default="stage6")
    ap.add_argument("--plots-dir", default="plots")
    a = ap.parse_args()

    df = pd.read_csv(a.scored).rename(columns={"neural_response_score": "score"})
    df = df.dropna(subset=["score", "laser_power"])
    blups = pd.read_csv(a.blups)

    sensitivity, r, p = predict_sensitivity_loo(blups, df)
    sensitivity.to_csv(f"{a.output_prefix}_sensitivity_hat.csv", index=False)

    df = df.merge(sensitivity, on="global_subject", how="inner").copy()
    m_mod = moderated_model(df)
    df["mod_resid"] = df["score"] - m_mod.fittedvalues  # includes each subject's own random effects -- diagnostic only

    print("\n=== Heteroscedasticity check BEFORE variance-function correction ===")
    check_heteroscedasticity(df, "mod_resid", "moderated-model residual")

    # DEPLOYABLE score: predict using only population-level info (intensity +
    # this subject's baseline-predicted sensitivity), NOT their own fitted
    # random intercept/slope -- a real new trial wouldn't have those yet. Built
    # directly from df's own columns (not the model's internal design matrix)
    # so there's no ambiguity about row order matching back up.
    fe = m_mod.fe_params
    df["population_expected"] = (
        fe["Intercept"] + fe["laser_power"] * df["laser_power"]
        + fe["sensitivity_hat"] * df["sensitivity_hat"]
        + fe["laser_power:sensitivity_hat"] * df["laser_power"] * df["sensitivity_hat"]
    )
    df["deviation"] = df["score"] - df["population_expected"]

    vf, smear = fit_variance_function(df, resid_col="deviation")
    sigma_hat = predict_sigma(vf, smear, df["laser_power"])
    df["score_z_corrected"] = df["deviation"] / sigma_hat
    df["band"] = df["score_z_corrected"].apply(band_of)

    print("\n=== Heteroscedasticity check AFTER variance-function correction ===")
    check_heteroscedasticity(df, "score_z_corrected", "corrected standardized score")

    plots_dir = Path(a.plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    plot_band_by_level(df, "band", plots_dir / f"{a.output_prefix}_band_by_level.png",
                        "SD-band composition per intensity level, AFTER baseline-state + "
                        "heteroscedasticity correction\n(flatter across levels = fairer standardized scale)")

    band_order = [b[2] for b in BANDS]
    print("\n=== Corrected SD band composition per level (row %) ===")
    crosstab = pd.crosstab(df["laser_power"], df["band"], normalize="index").reindex(columns=band_order, fill_value=0)
    print((crosstab * 100).round(1).to_string())

    df.to_csv(f"{a.output_prefix}_scored_trials.csv", index=False)
    print(f"\nsaved {a.output_prefix}_sensitivity_hat.csv, {a.output_prefix}_scored_trials.csv")


if __name__ == "__main__":
    main()
