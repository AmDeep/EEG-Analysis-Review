"""PCA variance graph on the features Stage 2 selected for the composite
score (`stage2_all_weights.csv`) -- how much of their combined variance is
genuinely independent information vs. redundant/overlapping?

Bar = variance explained by each individual principal component. Line =
cumulative variance explained. Both are percentages of the same total, so
they share one axis (never a dual-axis chart) -- unlike a typical bar+line
combo where the two series are on different scales.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from plots import SURFACE, INK_MUTED, GRIDLINE, BASELINE_AXIS, BLUE, ORANGE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_normalized_all.csv")
    ap.add_argument("--feature-list-from", default="stage2_all_weights.csv")
    ap.add_argument("--output-prefix", default="stage10")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--all-baseline-features", action="store_true",
                     help="Use every *_baseline column instead of --feature-list-from -- the set "
                          "actually available with no stimulus/intensity involved at all (Stage 7's "
                          "feature set), rather than Stage 2's feature list (which was itself SELECTED "
                          "by correlation with laser_power).")
    ap.add_argument("--target-col", default=None,
                     help="If set (e.g. laser_power), instead of showing how much of the FEATURES' "
                          "own variance each PC explains, show how much of this target's variance "
                          "each PC explains (R^2 of a simple regression of target on that PC alone). "
                          "PCs are orthogonal by construction, so these per-PC R^2 values sum exactly "
                          "to the multiple-regression R^2 using all PCs together -- the bar+cumulative "
                          "framing stays mathematically valid.")
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    if a.all_baseline_features:
        cols = [c for c in df.columns if c.endswith("_baseline")]
        print(f"Running PCA on all {len(cols)} baseline-only features (no intensity/stimulus involved "
              f"in computing OR selecting these): {cols}")
    else:
        cols = pd.read_csv(a.feature_list_from)["feature"].tolist()
        print(f"Running PCA on {len(cols)} chosen features: {cols}")

    dropna_cols = cols + [a.target_col] if a.target_col else cols
    sub = df[dropna_cols].dropna()
    print(f"{len(sub)} trials with complete data on all {len(cols)} features"
          + (f" + {a.target_col}" if a.target_col else ""))

    X = StandardScaler().fit_transform(sub[cols].to_numpy())
    pca = PCA()
    pca.fit(X)

    if a.target_col:
        y = sub[a.target_col].to_numpy()
        scores = pca.transform(X)
        var_pct = np.array([pearsonr(scores[:, k], y)[0] ** 2 for k in range(scores.shape[1])]) * 100
        cum_pct = np.cumsum(var_pct)
        # Sanity check: since PCA components are orthogonal, the sum of individual
        # R^2's should equal the multiple-regression R^2 using all components together
        # (and that, in turn, should equal regressing the target on the raw features
        # directly, since PCA is just an orthogonal rotation of them).
        import statsmodels.api as sm
        full_r2 = sm.OLS(y, sm.add_constant(scores)).fit().rsquared * 100
        print(f"[sanity check] sum of per-PC R^2 = {cum_pct[-1]:.2f}%, "
              f"full multiple-regression R^2 = {full_r2:.2f}% (should match)")
    else:
        var_pct = pca.explained_variance_ratio_ * 100
        cum_pct = np.cumsum(var_pct)

    reaches_90 = cum_pct[-1] >= 90
    n_for_90 = int(np.searchsorted(cum_pct, 90) + 1) if reaches_90 else None

    label = f"variance in {a.target_col}" if a.target_col else "variance in the features themselves"
    print(f"\n=== {label.capitalize()} explained per component ===")
    for i, (v, c) in enumerate(zip(var_pct, cum_pct), 1):
        print(f"PC{i}: {v:5.1f}%  (cumulative {c:5.1f}%)")
    if reaches_90:
        print(f"\n{n_for_90} of {len(cols)} components needed to reach 90% cumulative {label}")
    else:
        print(f"\nAll {len(cols)} components together only reach {cum_pct[-1]:.1f}% cumulative {label} "
              f"(never hits 90%)")

    # Loadings: which original features dominate the first 2 components
    loadings = pd.DataFrame(pca.components_[:3].T, index=cols, columns=["PC1", "PC2", "PC3"])
    loadings.to_csv(f"{a.output_prefix}_pca_loadings.csv")
    print("\n=== Top loadings (|weight| >= 0.3) ===")
    for pc in ["PC1", "PC2", "PC3"]:
        top = loadings[pc].reindex(loadings[pc].abs().sort_values(ascending=False).index)
        top = top[top.abs() >= 0.3]
        print(f"{pc}: " + ", ".join(f"{name}={w:+.2f}" for name, w in top.items()))

    pd.DataFrame({
        "component": [f"PC{i}" for i in range(1, len(cols) + 1)],
        "variance_pct": var_pct,
        "cumulative_pct": cum_pct,
    }).to_csv(f"{a.output_prefix}_pca_variance.csv", index=False)

    # --- plot ---
    n = len(cols)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(1, n + 1)
    bar_label = f"Variance in {a.target_col} explained (this PC)" if a.target_col else "Variance explained (this PC)"
    line_label = f"Cumulative variance in {a.target_col} explained" if a.target_col else "Cumulative variance explained"
    ax.bar(x, var_pct, color=BLUE, width=0.6, label=bar_label)
    ax.plot(x, cum_pct, color=ORANGE, marker="o", markersize=5, lw=2, label=line_label)

    y_label = f"% of {a.target_col}'s variance" if a.target_col else "% of total variance"
    ax.set_xlabel("Principal component")
    ax.set_ylabel(y_label)
    ax.set_xticks(x)

    title_subject = "all-baseline (no intensity involved)" if a.all_baseline_features else "chosen for the composite score"
    if a.target_col:
        ax.set_ylim(0, max(cum_pct[-1] * 1.15, 10))
        ax.set_title(f"PCA on the {n} features, {title_subject}\n"
                     f"mapped against {a.target_col}: all {n} components together explain "
                     f"{cum_pct[-1]:.1f}% of its variance", fontsize=10)
    else:
        ax.axhline(90, color=BASELINE_AXIS, lw=1, ls="--")
        ax.text(n, 91, "90%", ha="right", color=INK_MUTED, fontsize=8)
        ax.set_ylim(0, 105)
        ax.set_title(f"PCA on the {n} features, {title_subject}\n"
                     f"({n_for_90} components needed for 90% of the variance -- {100*(1-n_for_90/n):.0f}% redundancy)",
                     fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out_path = f"{a.plots_dir}/{a.output_prefix}_pca_variance.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nsaved {out_path}, {a.output_prefix}_pca_variance.csv, {a.output_prefix}_pca_loadings.csv")


if __name__ == "__main__":
    main()
