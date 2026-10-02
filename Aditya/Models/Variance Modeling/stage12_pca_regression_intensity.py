"""Stage 12: take the PCA-reduced baseline features from Stage 10b/11 and
build actual regression models predicting laser_power from them, properly
cross-validated (Stage 11's ~3% number was an in-sample sanity check on the
whole pooled dataset, not held-out, and didn't group by subject).

For several numbers of retained components (1, 2, 3, 5, 9, 15, 20) and two
model types (plain linear regression, and a Random Forest to catch any
nonlinear relationship a linear model would miss), fits with GroupKFold on
global_subject (a subject's trials never split across train/test) and reports
held-out r/R^2. PCA itself is refit on the training fold only each time --
same discipline as every other cross-validated result in this project.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from plots import SURFACE, INK_MUTED, GRIDLINE, BASELINE_AXIS, BLUE, ORANGE

COMPONENT_COUNTS = [1, 2, 3, 5, 9, 15, 20]
MODELS = {
    "linear": lambda: LinearRegression(),
    "random_forest": lambda: RandomForestRegressor(n_estimators=200, max_depth=6, random_state=42, n_jobs=-1),
}


def cross_validated_r2(X, y, groups, n_components, model_fn, n_splits=5):
    gkf = GroupKFold(n_splits=n_splits)
    oof_pred = np.full(len(y), np.nan)
    for tr, te in gkf.split(X, y, groups):
        scaler = StandardScaler().fit(X[tr])
        pca = PCA(n_components=n_components).fit(scaler.transform(X[tr]))
        Xtr = pca.transform(scaler.transform(X[tr]))
        Xte = pca.transform(scaler.transform(X[te]))

        model = model_fn()
        model.fit(Xtr, y[tr])
        oof_pred[te] = model.predict(Xte)

    r, p = pearsonr(oof_pred, y)
    return r, r ** 2, p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="features_normalized_all.csv")
    ap.add_argument("--output-prefix", default="stage12")
    ap.add_argument("--plots-dir", default="plots")
    ap.add_argument("--n-splits", type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(a.features)
    baseline_cols = [c for c in df.columns if c.endswith("_baseline")]
    sub = df[["global_subject", "laser_power", *baseline_cols]].dropna()
    X = sub[baseline_cols].to_numpy()
    y = sub["laser_power"].to_numpy()
    groups = sub["global_subject"].to_numpy()
    print(f"{len(sub)} trials, {len(np.unique(groups))} subjects, {len(baseline_cols)} baseline features")

    results = []
    for model_name, model_fn in MODELS.items():
        for k in COMPONENT_COUNTS:
            r, r2, p = cross_validated_r2(X, y, groups, k, model_fn, a.n_splits)
            results.append({"model": model_name, "n_components": k, "r": r, "r2": r2, "p": p})
            print(f"{model_name:14s} n_components={k:2d}  r={r:.4f}  r2={r2:.4f}  p={p:.3g}")

    out = pd.DataFrame(results)
    out.to_csv(f"{a.output_prefix}_cv_results.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"linear": BLUE, "random_forest": ORANGE}
    for model_name, color in colors.items():
        subset = out[out["model"] == model_name].sort_values("n_components")
        ax.plot(subset["n_components"], subset["r2"] * 100, color=color, marker="o",
                markersize=5, lw=2, label=model_name.replace("_", " ").title())

    ax.set_xlabel("Number of PCA components used")
    ax.set_ylabel("Cross-validated R^2 predicting laser_power (%)")
    ax.set_title("Can baseline-EEG PCA components predict laser intensity?\n"
                 "(subject-grouped CV -- PCA refit on training folds only)", fontsize=10)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, axis="y", lw=0.5, color=GRIDLINE)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    out_path = f"{a.plots_dir}/{a.output_prefix}_cv_results.png"
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"\nsaved {a.output_prefix}_cv_results.csv, {out_path}")

    best = out.loc[out["r2"].idxmax()]
    print(f"\nBest: {best['model']} with {int(best['n_components'])} components -> "
          f"r={best['r']:.3f}, r2={best['r2']:.3f}")


if __name__ == "__main__":
    main()
