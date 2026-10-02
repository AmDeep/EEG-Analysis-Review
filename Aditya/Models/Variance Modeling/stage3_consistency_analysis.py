"""Stage 3: do different people at the SAME laser intensity show similar
baseline-normalized responses, does that hold across intensities, and can the
composite score be turned into a standardized "N SDs away" scale?

This is a different question from Stage 1/2's regression (does response track
intensity within/across subjects) -- it's about CLUSTERING: at a fixed
intensity, is the across-subject spread small relative to how far apart
different intensities are? Two parts:

1. CONSISTENCY: one-way ANOVA of the Stage 2 composite score (and the
   strongest Stage 1 features) across laser_power levels, treated as a
   categorical/ordinal factor (all 9 datasets sample the same 1.0-4.5, step
   0.25 grid, so levels are exact, not binned). Reported as eta^2 (fraction of
   total variance explained by intensity level -- the ANOVA analogue of R^2,
   without assuming a linear/monotonic shape) plus the per-level mean/std
   table, which shows the across-subject spread at each intensity directly.
   Aggregated to one row per (subject, laser_power) first, so a subject with
   several trials at the same level doesn't get counted multiple times.

2. SD SCALE: re-standardize the composite score against ITS OWN population
   distribution (R_z = (R - mean(R)) / std(R)) and bucket every subject-level
   observation into |R_z| bands: 0-1, 1-2, 2-3, 3+. If the scale means
   anything, the fraction of observations in the higher bands should rise
   monotonically with laser_power.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from matplotlib.colors import LinearSegmentedColormap

from plots import SURFACE, INK_PRIMARY, INK_MUTED, GRIDLINE, BASELINE_AXIS

# Sequential blue ramp (light -> dark) from the Nurovo palette's 100->700 steps,
# sampled to however many distinct laser_power levels are present.
SEQUENTIAL_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "nurovo_sequential_blue",
    ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"],
)

BANDS = [(0, 1, "0-1 SD"), (1, 2, "1-2 SD"), (2, 3, "2-3 SD"), (3, np.inf, "3+ SD")]


def eta_squared_anova(groups):
    """groups: list of 1D arrays, one per level. Returns (F, p, eta2)."""
    f_stat, p = stats.f_oneway(*groups)
    grand_mean = np.concatenate(groups).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in groups)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in groups)
    eta2 = ss_between / ss_total if ss_total > 0 else np.nan
    return f_stat, p, eta2


def band_of(z):
    az = abs(z)
    for lo, hi, label in BANDS:
        if lo <= az < hi:
            return label
    return BANDS[-1][2]


def plot_per_level_spread(subject_level, col, out_path):
    levels = sorted(subject_level["laser_power"].unique())
    groups = [subject_level.loc[subject_level["laser_power"] == lv, col].dropna().to_numpy() for lv in levels]
    colors = [SEQUENTIAL_BLUE_CMAP(t) for t in np.linspace(0.15, 1.0, len(levels))]

    fig, ax = plt.subplots(figsize=(9, 5))
    bp = ax.boxplot(groups, positions=levels, widths=0.15, patch_artist=True,
                     medianprops=dict(color=INK_PRIMARY, lw=1.5),
                     whiskerprops=dict(color=BASELINE_AXIS), capprops=dict(color=BASELINE_AXIS),
                     flierprops=dict(markeredgecolor=INK_MUTED, markersize=3, alpha=0.5))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_edgecolor(BASELINE_AXIS)

    ax.set_xlabel("Laser power")
    ax.set_ylabel(col)
    ax.set_title(f"Spread of {col} across subjects, per laser intensity level\n"
                 f"(tighter boxes = more consistent people at that intensity)", fontsize=10)
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


def analyze_column(subject_level, col, level_col="laser_power"):
    groups = [g[col].to_numpy() for _, g in subject_level.groupby(level_col) if len(g) >= 2]
    if len(groups) < 2:
        return None
    f_stat, p, eta2 = eta_squared_anova(groups)

    per_level = subject_level.groupby(level_col)[col].agg(["mean", "std", "count"]).reset_index()
    per_level.columns = [level_col, "mean", "std", "n_subjects"]
    return {"feature": col, "F": f_stat, "p": p, "eta2": eta2}, per_level


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", required=True, help="stage2_*_scored_trials.csv (has neural_response_score)")
    ap.add_argument("--results", required=True, help="stage1_results*.csv")
    ap.add_argument("--output-prefix", default="stage3")
    ap.add_argument("--top-n-features", type=int, default=3)
    ap.add_argument("--plots-dir", default="plots")
    a = ap.parse_args()

    df = pd.read_csv(a.scored)
    results = pd.read_csv(a.results)

    top_features = (results[results["normalization"] == "z"]
                    .sort_values("p_linear").head(a.top_n_features)["feature"])
    score_cols = ["neural_response_score"] + [f"{f}_z" for f in top_features]
    print(f"Analyzing: {score_cols}")

    subject_level = (df.groupby(["global_subject", "laser_power"])[score_cols]
                      .mean().reset_index())
    print(f"{len(subject_level)} (subject, laser_power) observations from "
          f"{subject_level['global_subject'].nunique()} subjects")

    plots_dir = Path(a.plots_dir)
    plots_dir.mkdir(exist_ok=True, parents=True)
    plot_per_level_spread(subject_level, "neural_response_score",
                           plots_dir / f"{a.output_prefix}_score_spread_by_level.png")

    anova_rows, per_level_tables = [], {}
    for col in score_cols:
        sub = subject_level.dropna(subset=[col])
        out = analyze_column(sub, col)
        if out is None:
            continue
        anova_row, per_level = out
        anova_rows.append(anova_row)
        per_level_tables[col] = per_level

    anova_df = pd.DataFrame(anova_rows)
    anova_df.to_csv(f"{a.output_prefix}_anova.csv", index=False)
    print("\n=== Consistency (one-way ANOVA across laser_power levels) ===")
    print(anova_df.to_string(index=False))

    score_per_level = per_level_tables["neural_response_score"]
    score_per_level.to_csv(f"{a.output_prefix}_score_per_level.csv", index=False)
    print("\n=== neural_response_score: mean +/- std per laser_power level ===")
    print(score_per_level.to_string(index=False))

    # --- SD scale ---
    sub = subject_level.dropna(subset=["neural_response_score"]).copy()
    mu, sigma = sub["neural_response_score"].mean(), sub["neural_response_score"].std()
    sub["score_z"] = (sub["neural_response_score"] - mu) / sigma
    sub["band"] = sub["score_z"].apply(band_of)

    band_order = [b[2] for b in BANDS]
    crosstab = pd.crosstab(sub["laser_power"], sub["band"], normalize="index")[
        [b for b in band_order if b in sub["band"].unique()]
    ]
    crosstab.to_csv(f"{a.output_prefix}_sd_band_by_level.csv")
    print(f"\n=== SD band composition per laser_power level (row %, mu={mu:.3g}, sigma={sigma:.3g}) ===")
    print((crosstab * 100).round(1).to_string())

    band_index = {b: i for i, b in enumerate(band_order)}
    sub["band_index"] = sub["band"].map(band_index)
    rho, p = stats.spearmanr(sub["laser_power"], sub["band_index"])
    print(f"\nSpearman correlation (laser_power vs. SD band): rho={rho:.3f}, p={p:.3g}")

    sub.to_csv(f"{a.output_prefix}_sd_scale.csv", index=False)
    print(f"\nsaved {a.output_prefix}_anova.csv, {a.output_prefix}_score_per_level.csv, "
          f"{a.output_prefix}_sd_band_by_level.csv, {a.output_prefix}_sd_scale.csv")


if __name__ == "__main__":
    main()
