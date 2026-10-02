"""Stage 13: repeat Stage 10b/11/12's question (do baseline-EEG dimensionality-
reduction axes relate to laser intensity?) with t-SNE and UMAP instead of PCA.

IMPORTANT ASYMMETRY, stated up front rather than glossed over: PCA and UMAP
can both be fit on a training fold and then used to embed new, unseen points
(`.transform()`), which is what let Stage 12 do a proper subject-grouped
cross-validated regression. scikit-learn's t-SNE CANNOT embed new points at
all -- it only jointly optimizes positions for the exact set of points it's
given, with no held-out extension. That means:

  - t-SNE here is DESCRIPTIVE ONLY: one embedding fit on a subsample (t-SNE
    doesn't scale to 29,515 points in reasonable time), visualized and
    checked with an IN-SAMPLE regression. This is not a leak-free predictive
    test and isn't presented as one.
  - UMAP gets the full Stage 12 treatment: fit on training folds only,
    `.transform()` on held-out folds, linear + Random Forest regressors,
    across several embedding dimensionalities, subject-grouped CV.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

import umap

from plots import SURFACE, INK_PRIMARY, GRIDLINE, BLUE, ORANGE

SEQUENTIAL_BLUE_CMAP = LinearSegmentedColormap.from_list(
    "nurovo_sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"],
)


def load_baseline_data(features_path):
    df = pd.read_csv(features_path)
    baseline_cols = [c for c in df.columns if c.endswith("_baseline")]
    sub = df[["global_subject", "laser_power", *baseline_cols]].dropna()
    return sub, baseline_cols


def run_tsne(sub, baseline_cols, plots_dir, output_prefix, n_subsample, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sub), size=min(n_subsample, len(sub)), replace=False)
    sample = sub.iloc[idx]
    print(f"\n=== t-SNE (descriptive only, n={len(sample)} subsampled trials) ===")

    X = StandardScaler().fit_transform(sample[baseline_cols].to_numpy())
    y = sample["laser_power"].to_numpy()

    embedding = TSNE(n_components=2, init="pca", random_state=seed, perplexity=30).fit_transform(X)

    r1, p1 = pearsonr(embedding[:, 0], y)
    r2, p2 = pearsonr(embedding[:, 1], y)
    full_r2 = sm.OLS(y, sm.add_constant(embedding)).fit().rsquared
    print(f"dim1 vs laser_power: r={r1:.3f} (p={p1:.3g})")
    print(f"dim2 vs laser_power: r={r2:.3f} (p={p2:.3g})")
    print(f"IN-SAMPLE R^2 using both dims together: {full_r2:.4f} "
          f"(NOT cross-validated -- see module docstring)")

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=SEQUENTIAL_BLUE_CMAP, s=8, alpha=0.7, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Laser power (J)", color=INK_PRIMARY)
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.set_title(f"t-SNE on baseline EEG features, colored by laser intensity\n"
                 f"(descriptive only, n={len(sample)}; in-sample R^2={full_r2:.3f})", fontsize=10)
    ax.grid(True, lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = f"{plots_dir}/{output_prefix}_tsne_scatter.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")
    return {"dim1_r": r1, "dim2_r": r2, "in_sample_r2": full_r2, "n_subsample": len(sample)}


def cross_validated_r2_umap(X, y, groups, n_components, model_fns, n_splits, seed=42):
    """model_fns: dict of {name: zero-arg constructor}. Fits UMAP once per fold
    (it doesn't depend on which downstream model consumes the embedding) and
    reuses that embedding for every model, instead of refitting UMAP per model."""
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = {name: np.full(len(y), np.nan) for name in model_fns}
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        reducer = umap.UMAP(n_components=n_components, random_state=seed, n_jobs=1).fit(scaler.transform(X[tr]))
        Xtr = reducer.transform(scaler.transform(X[tr]))
        Xte = reducer.transform(scaler.transform(X[te]))

        for name, model_fn in model_fns.items():
            model = model_fn()
            model.fit(Xtr, y[tr])
            oof_pred[name][te] = model.predict(Xte)

    out = {}
    for name, pred in oof_pred.items():
        r, p = pearsonr(pred, y)
        out[name] = (r, r ** 2, p)
    return out


def run_umap_cv(sub, baseline_cols, plots_dir, output_prefix, component_counts, n_splits, n_subsample, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sub), size=min(n_subsample, len(sub)), replace=False)
    sample = sub.iloc[idx]
    print(f"\n=== UMAP cross-validated regression (n={len(sample)} subsampled trials, "
          f"subject-grouped {n_splits}-fold CV) ===")

    X = sample[baseline_cols].to_numpy()
    y = sample["laser_power"].to_numpy()
    groups = sample["global_subject"].to_numpy()

    model_fns = {
        "linear": lambda: LinearRegression(),
        "random_forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1),
    }

    results = []
    for k in component_counts:
        per_model = cross_validated_r2_umap(X, y, groups, k, model_fns, n_splits, seed)
        for model_name, (r, r2, p) in per_model.items():
            results.append({"model": model_name, "n_components": k, "r": r, "r2": r2, "p": p})
            print(f"{model_name:14s} n_components={k:2d}  r={r:.4f}  r2={r2:.4f}  p={p:.3g}")

    out = pd.DataFrame(results)
    out.to_csv(f"{output_prefix}_umap_cv_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"linear": BLUE, "random_forest": ORANGE}
    for model_name, color in colors.items():
        subset = out[out["model"] == model_name].sort_values("n_components")
        ax.plot(subset["n_components"], subset["r2"] * 100, color=color, marker="o",
                markersize=5, lw=2, label=model_name.replace("_", " ").title())
    ax.set_xlabel("Number of UMAP components used")
    ax.set_ylabel("Cross-validated R^2 predicting laser_power (%)")
    ax.set_title(f"Can baseline-EEG UMAP components predict laser intensity?\n"
                 f"(subject-grouped CV, n={len(sample)} subsampled trials)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = f"{plots_dir}/{output_prefix}_umap_cv_results.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}, {output_prefix}_umap_cv_results.csv")
    return out


def run_umap_visualization(sub, baseline_cols, plots_dir, output_prefix, n_subsample, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(sub), size=min(n_subsample, len(sub)), replace=False)
    sample = sub.iloc[idx]

    X = StandardScaler().fit_transform(sample[baseline_cols].to_numpy())
    y = sample["laser_power"].to_numpy()
    embedding = umap.UMAP(n_components=2, random_state=seed).fit_transform(X)

    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(embedding[:, 0], embedding[:, 1], c=y, cmap=SEQUENTIAL_BLUE_CMAP, s=8, alpha=0.7, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("Laser power (J)", color=INK_PRIMARY)
    ax.set_xlabel("UMAP dim 1")
    ax.set_ylabel("UMAP dim 2")
    ax.set_title(f"UMAP on baseline EEG features, colored by laser intensity (n={len(sample)})", fontsize=10)
    ax.grid(True, lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = f"{plots_dir}/{output_prefix}_umap_scatter.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"saved {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_normalized_all.csv")
    ap.add_argument("--output-prefix", default="stage13")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--tsne-n-subsample", type=int, default=8000)
    ap.add_argument("--umap-cv-n-subsample", type=int, default=12000)
    ap.add_argument("--umap-viz-n-subsample", type=int, default=15000)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--component-counts", type=int, nargs="+", default=[2, 5, 10, 20])
    a = ap.parse_args()

    sub, baseline_cols = load_baseline_data(a.features)
    print(f"{len(sub)} total trials, {len(baseline_cols)} baseline features")

    tsne_summary = run_tsne(sub, baseline_cols, a.plots_dir, a.output_prefix, a.tsne_n_subsample)
    run_umap_visualization(sub, baseline_cols, a.plots_dir, a.output_prefix, a.umap_viz_n_subsample)
    umap_results = run_umap_cv(sub, baseline_cols, a.plots_dir, a.output_prefix,
                                a.component_counts, a.n_splits, a.umap_cv_n_subsample)

    pd.DataFrame([tsne_summary]).to_csv(f"{a.output_prefix}_tsne_summary.csv", index=False)
    best = umap_results.loc[umap_results["r2"].idxmax()]
    print(f"\nBest UMAP result: {best['model']} with {int(best['n_components'])} components -> "
          f"r={best['r']:.3f}, r2={best['r2']:.3f}")
    print(f"For comparison -- Stage 12 (PCA) best: Random Forest, 20 components, r2=0.037")


if __name__ == "__main__":
    main()
