"""Stage 14: three more nonlinear dimensionality-reduction methods, same
question as Stages 10b-13 (do reduced baseline-EEG axes relate to laser
intensity?), still with no intensity involved in computing or selecting the
underlying features.

Kernel PCA, Isomap, and Locally Linear Embedding were picked specifically
because -- unlike t-SNE, Spectral Embedding, or MDS -- scikit-learn's
implementations all support `.transform()` on new, unseen data after fitting.
That means all three get the full Stage 12/13-UMAP treatment: fit on a
training fold only, transform the held-out fold, subject-grouped CV, linear +
Random Forest regressors, across several embedding dimensionalities. No
descriptive-only shortcuts here, unlike t-SNE in Stage 13.

Isomap and LLE build a k-nearest-neighbor graph per fold, which is
expensive at scale, so they run on a smaller subsample than Kernel PCA.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import KernelPCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.manifold import Isomap, LocallyLinearEmbedding
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from plots import SURFACE, GRIDLINE, BLUE, ORANGE

AQUA = "#1baf7a"  # third categorical slot of the Nurovo palette

REDUCERS = {
    "kernel_pca": {
        "n_subsample": 12000,
        "component_counts": [2, 5, 10, 20],
        "make": lambda k, seed: KernelPCA(n_components=k, kernel="rbf", random_state=seed, n_jobs=-1),
    },
    "isomap": {
        "n_subsample": 5000,
        "component_counts": [2, 5, 10, 20],
        "make": lambda k, seed: Isomap(n_components=k, n_neighbors=10, n_jobs=-1),
    },
    "lle": {
        "n_subsample": 5000,
        "component_counts": [2, 5, 10, 20],
        "make": lambda k, seed: LocallyLinearEmbedding(n_components=k, n_neighbors=10, random_state=seed),
    },
}

MODEL_FNS = {
    "linear": lambda seed: LinearRegression(),
    "random_forest": lambda seed: RandomForestRegressor(n_estimators=200, max_depth=6, random_state=seed, n_jobs=-1),
}


def load_baseline_data(features_path):
    df = pd.read_csv(features_path)
    baseline_cols = [c for c in df.columns if c.endswith("_baseline")]
    sub = df[["global_subject", "laser_power", *baseline_cols]].dropna()
    return sub, baseline_cols


def cross_validated_r2(X, y, groups, reducer_fn, n_components, n_splits, seed=42):
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = {name: np.full(len(y), np.nan) for name in MODEL_FNS}
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        reducer = reducer_fn(n_components, seed).fit(scaler.transform(X[tr]))
        Xtr = reducer.transform(scaler.transform(X[tr]))
        Xte = reducer.transform(scaler.transform(X[te]))

        for name, model_fn in MODEL_FNS.items():
            model = model_fn(seed)
            model.fit(Xtr, y[tr])
            oof_pred[name][te] = model.predict(Xte)

    out = {}
    for name, pred in oof_pred.items():
        r, p = pearsonr(pred, y)
        out[name] = (r, r ** 2, p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_normalized_all.csv")
    ap.add_argument("--output-prefix", default="stage14")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    sub, baseline_cols = load_baseline_data(a.features)
    print(f"{len(sub)} total trials, {len(baseline_cols)} baseline features")

    rng = np.random.default_rng(a.seed)
    all_results = []
    for method_name, cfg in REDUCERS.items():
        idx = rng.choice(len(sub), size=min(cfg["n_subsample"], len(sub)), replace=False)
        sample = sub.iloc[idx]
        X = sample[baseline_cols].to_numpy()
        y = sample["laser_power"].to_numpy()
        groups = sample["global_subject"].to_numpy()

        print(f"\n=== {method_name} (n={len(sample)} subsampled trials, "
              f"subject-grouped {a.n_splits}-fold CV) ===")
        for k in cfg["component_counts"]:
            per_model = cross_validated_r2(X, y, groups, cfg["make"], k, a.n_splits, a.seed)
            for model_name, (r, r2, p) in per_model.items():
                all_results.append({"method": method_name, "model": model_name,
                                     "n_components": k, "r": r, "r2": r2, "p": p,
                                     "n_subsample": len(sample)})
                print(f"{model_name:14s} n_components={k:2d}  r={r:.4f}  r2={r2:.4f}  p={p:.3g}")

    out = pd.DataFrame(all_results)
    out.to_csv(f"{a.output_prefix}_cv_results.csv", index=False)

    # Summary plot: best-of-{linear, random_forest} R^2 per method per component count.
    best_per_method = out.loc[out.groupby(["method", "n_components"])["r2"].idxmax()]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"kernel_pca": BLUE, "isomap": ORANGE, "lle": AQUA}
    for method_name, color in colors.items():
        subset = best_per_method[best_per_method["method"] == method_name].sort_values("n_components")
        ax.plot(subset["n_components"], subset["r2"] * 100, color=color, marker="o", markersize=5, lw=2,
                label=method_name.replace("_", " ").title())

    ax.set_xlabel("Number of components used")
    ax.set_ylabel("Cross-validated R^2 predicting laser_power (%)\n(best of linear / Random Forest)")
    ax.set_title("Three more nonlinear dimensionality reductions vs. laser intensity\n"
                 "(subject-grouped CV, baseline EEG only)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out_path = f"{a.plots_dir}/{a.output_prefix}_cv_results.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nsaved {out_path}, {a.output_prefix}_cv_results.csv")

    best = out.loc[out["r2"].idxmax()]
    print(f"\nBest overall: {best['method']} / {best['model']}, {int(best['n_components'])} components -> "
          f"r={best['r']:.3f}, r2={best['r2']:.3f}")
    print("For comparison: PCA (Stage 12) r2=0.037, UMAP (Stage 13) r2=0.004, t-SNE (Stage 13, in-sample) r2=0.003")


if __name__ == "__main__":
    main()
