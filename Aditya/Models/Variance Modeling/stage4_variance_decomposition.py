"""Stage 4: decompose the variance the Stage 1-3 models don't explain.

Stage 3 found the spread of the composite score around the population trend
grows ~4-5x from lowest to highest laser intensity. Before treating that as
real, this script tests it formally and asks what's driving it:

1. HETEROSCEDASTICITY TEST: fit a random-intercept mixed model
   S_ij = b0 + b1*I_ij + u_i + e_ij, then test whether Var(e | I) actually
   grows with I -- Breusch-Pagan-style regression of squared residuals on
   intensity, plus Levene's test across the exact intensity levels (no
   binning needed; all 9 datasets share one 1.0-4.5-by-0.25 grid).

2. RANDOM-SLOPES MODEL: S_ij = b0 + b1*I_ij + u_i + v_i*I_ij + e_ij. If v_i
   (per-subject slope deviation) has real variance -- tested by a likelihood-
   ratio test against the random-intercept-only model -- some people are
   genuinely more "sensitive" to intensity than others, not just noisier.
   The fitted per-subject slopes (BLUPs) are the closest thing we have to an
   individual sensitivity trait. Only meaningful for subjects with enough
   distinct intensity levels of their own (see README: that's the 124
   ds005293/ds005473 subjects, not the 554 single-level subjects from the
   other 7 datasets -- run this on --scored/--features from THAT cohort).

3. PER-SUBJECT CURVATURE: for those same subjects, is their trajectory closer
   to a plateau (saturating), accelerating, or linear?

4. INDIVIDUAL DIFFERENCES: do subjects' own (non-normalized) baseline EEG
   characteristics correlate with their fitted sensitivity slope? If so,
   that's the evidence needed before investing in a model that reads raw
   baseline EEG to predict sensitivity -- this checks whether the hand-picked
   baseline features already carry that signal, or whether it's plausible
   raw waveform structure carries more than they do.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

from plots import SURFACE, INK_PRIMARY, INK_MUTED, GRIDLINE, BASELINE_AXIS
from matplotlib.colors import LinearSegmentedColormap

MIN_LEVELS_FOR_SLOPE = 3
MIN_LEVELS_FOR_CURVATURE = 4

# Diverging blue<->red, for signed slope magnitude (Nurovo palette diverging pair).
DIVERGING = LinearSegmentedColormap.from_list("nurovo_diverging", ["#e34948", "#f0efec", "#2a78d6"])


def heteroscedasticity_test(df):
    m0 = smf.mixedlm("score ~ laser_power", df, groups=df["global_subject"]).fit(reml=False)
    resid = m0.resid
    resid2 = resid ** 2

    X = sm.add_constant(df["laser_power"])
    bp = sm.OLS(resid2, X).fit()
    bp_slope, bp_p = bp.params["laser_power"], bp.pvalues["laser_power"]

    groups = [resid[df["laser_power"] == lv].to_numpy() for lv in sorted(df["laser_power"].unique())]
    groups = [g for g in groups if len(g) >= 2]
    levene_stat, levene_p = stats.levene(*groups)

    print("=== Heteroscedasticity test (residuals from random-intercept model) ===")
    print(f"Breusch-Pagan-style: resid^2 ~ laser_power slope={bp_slope:.4g}, p={bp_p:.3g} "
          f"({'variance grows with intensity' if bp_p < 0.05 and bp_slope > 0 else 'not significant / not increasing'})")
    print(f"Levene's test across {len(groups)} intensity levels: stat={levene_stat:.3g}, p={levene_p:.3g}")
    return m0, dict(bp_slope=bp_slope, bp_p=bp_p, levene_stat=levene_stat, levene_p=levene_p)


def random_slopes_model(df):
    m_intercept = smf.mixedlm("score ~ laser_power", df, groups=df["global_subject"]).fit(reml=False)
    m_slope = smf.mixedlm("score ~ laser_power", df, groups=df["global_subject"],
                           re_formula="~laser_power").fit(reml=False)

    lr_stat = max(0.0, 2 * (m_slope.llf - m_intercept.llf))
    p_lrt = float(stats.chi2.sf(lr_stat, df=2))  # +variance, +covariance; boundary case -> conservative/approximate

    print("\n=== Random-slopes vs. random-intercept-only (LRT) ===")
    print(f"LR stat={lr_stat:.3g}, df=2 (approximate, boundary case), p={p_lrt:.3g}")
    print(f"Random-intercept-only: group Var={m_intercept.cov_re.iloc[0, 0]:.4g}, residual Var={m_intercept.scale:.4g}")
    print(f"Random-slopes model: {m_slope.cov_re}")
    print(f"Random-slopes model residual Var={m_slope.scale:.4g}")

    slope_var = m_slope.cov_re.loc["laser_power", "laser_power"]
    intercept_var = m_slope.cov_re.loc["Group", "Group"]
    total_between = intercept_var + slope_var
    print(f"Slope variance={slope_var:.4g} vs intercept variance={intercept_var:.4g} "
          f"({100 * slope_var / total_between:.1f}% of between-subject variance is slope, not just intercept)")

    re = m_slope.random_effects
    blups = pd.DataFrame({
        "global_subject": list(re.keys()),
        "intercept_blup": [v["Group"] for v in re.values()],
        "slope_blup": [v["laser_power"] for v in re.values()],
    })
    blups["fitted_slope"] = m_slope.fe_params["laser_power"] + blups["slope_blup"]
    return m_slope, p_lrt, blups


def per_subject_curvature(df):
    rows = []
    for subj, g in df.groupby("global_subject"):
        g = g.dropna(subset=["laser_power", "score"])
        if g["laser_power"].nunique() < MIN_LEVELS_FOR_CURVATURE:
            continue
        lin = np.polyfit(g["laser_power"], g["score"], 1)
        quad = np.polyfit(g["laser_power"], g["score"], 2)
        rows.append({"global_subject": subj, "linear_slope": lin[0], "curvature": quad[0],
                      "n_levels": g["laser_power"].nunique()})
    out = pd.DataFrame(rows)
    print(f"\n=== Per-subject curvature ({len(out)} subjects with >= {MIN_LEVELS_FOR_CURVATURE} levels) ===")
    print(f"curvature (quadratic coef): mean={out['curvature'].mean():.4g}, "
          f"{100 * (out['curvature'] < 0).mean():.1f}% negative (saturating/plateau), "
          f"{100 * (out['curvature'] > 0).mean():.1f}% positive (accelerating)")
    r, p = stats.pearsonr(out["linear_slope"], out["curvature"])
    if p >= 0.05:
        interp = "no clear slope/curvature relationship"
    elif r < 0:
        interp = "steeper responders tend to plateau more (curvature offsets slope)"
    else:
        interp = "steeper responders ALSO curve upward more -- they accelerate, not plateau"
    print(f"corr(linear_slope, curvature) = {r:.3f}, p={p:.3g} ({interp})")
    return out


def individual_differences(blups, features_df):
    baseline_cols = [c for c in features_df.columns if c.endswith("_baseline")]
    per_subj_baseline = features_df.groupby("global_subject")[baseline_cols].mean().reset_index()
    merged = blups.merge(per_subj_baseline, on="global_subject")

    rows = []
    for col in baseline_cols:
        x = merged[col]
        if x.std() == 0 or x.isna().all():
            continue
        r, p = stats.pearsonr(merged["fitted_slope"], x.fillna(x.mean()))
        rows.append({"baseline_feature": col, "r": r, "p": p})
    out = pd.DataFrame(rows).sort_values("p")
    out["p_fdr"] = multipletests(out["p"], method="fdr_bh")[1]
    n_sig = int((out["p_fdr"] < 0.05).sum())
    print(f"\n=== Individual differences: subject's own baseline features vs. fitted sensitivity slope ===")
    print(f"{n_sig} / {len(out)} baseline features survive FDR<0.05 correction")
    print(out.head(10).to_string(index=False))
    return out, merged


def plot_colored_trajectories(df, blups, out_path):
    per_level = df.groupby(["global_subject", "laser_power"])["score"].mean().reset_index()
    per_level = per_level.merge(blups[["global_subject", "slope_blup"]], on="global_subject")

    # slope_blup is already the deviation from the population-average slope (the
    # random effect itself), so centering the diverging colormap at 0 here is
    # exactly "below/above average" -- coloring by fitted_slope (population +
    # deviation) instead would make almost everyone the same color, since the
    # fixed effect dominates the much smaller deviations.
    vmax = per_level["slope_blup"].abs().quantile(0.95) or 1.0
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for subj, g in per_level.groupby("global_subject"):
        g = g.sort_values("laser_power")
        deviation = g["slope_blup"].iloc[0]
        color = DIVERGING(0.5 + 0.5 * np.clip(deviation / vmax, -1, 1))
        ax.plot(g["laser_power"], g["score"], color=color, alpha=0.6, lw=1.0)

    ax.set_xlabel("Laser power")
    ax.set_ylabel("Neural response score")
    ax.set_title("Per-subject trajectories, colored by sensitivity slope deviation\n"
                 "(red = below-average slope, blue = above-average slope)", fontsize=10)
    ax.grid(True, lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nsaved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="stage2_*_scored_trials.csv")
    ap.add_argument("--output-prefix", default="stage4")
    ap.add_argument("--plots-dir", default="plots")
    a = ap.parse_args()

    df = pd.read_csv(a.scored)
    df = df.rename(columns={"neural_response_score": "score"}).dropna(subset=["score", "laser_power"])

    m0, hetero = heteroscedasticity_test(df)
    m_slope, p_lrt, blups = random_slopes_model(df)
    curvature = per_subject_curvature(df)

    blups = blups.merge(curvature[["global_subject", "curvature", "n_levels"]], on="global_subject", how="left")
    blups.to_csv(f"{a.output_prefix}_subject_blups.csv", index=False)

    diffs, merged = individual_differences(blups, df)
    diffs.to_csv(f"{a.output_prefix}_individual_differences.csv", index=False)

    # Trial-level residuals from the random-slopes model: score minus what
    # population intensity trend + this subject's own intercept/slope predict.
    # This is the part left over after Stage 1-4's tabular-feature analysis --
    # the target for testing whether raw baseline EEG explains anything more.
    df["mixedlm_fitted"] = m_slope.fittedvalues
    df["residual"] = df["score"] - df["mixedlm_fitted"]
    residual_cols = ["dataset", "global_subject", "epoch", "laser_power", "score",
                      "mixedlm_fitted", "residual"]
    df[residual_cols].to_csv(f"{a.output_prefix}_trial_residuals.csv", index=False)
    print(f"saved {a.output_prefix}_trial_residuals.csv "
          f"(residual std={df['residual'].std():.4g}, for the raw-EEG follow-up)")

    plots_dir = Path(a.plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    plot_colored_trajectories(df, blups, plots_dir / f"{a.output_prefix}_trajectories_by_slope.png")

    print(f"\nsaved {a.output_prefix}_subject_blups.csv, {a.output_prefix}_individual_differences.csv")


if __name__ == "__main__":
    main()
