"""Discover discrete pain states with t-SNE + K-means, then learn them with KNN.

Pipeline
--------
1. Keep epochs from ds005289, ds005280, and ds005473 and the 24 EEG features.
2. Median-impute, standardize, and reduce to two dimensions with t-SNE.
3. Sweep K-means over the embedding and pick a state count with the elbow method.
4. Refit K-means at that count; states are relabeled by mean pain rating (0 = lowest).
5. Fit a KNN classifier on the states so new epochs can be assigned, and score it
   with subject-held-out folds.

Note on terminology: K-means does the clustering and supplies the elbow curve. KNN
is supervised, so it cannot invent states -- it learns the K-means partition and is
scored on how well that partition transfers to unseen subjects.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, silhouette_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from eeg_model_utils import EEG_FEATURES

DATASETS = ("ds005289", "ds005280", "ds005473")
DEFAULT_CSV = Path(__file__).resolve().parents[1] / "Aditya" / "Feature Extraction" / "features_combined.csv"
RANDOM_SEED = 42
PERPLEXITY = 30.0
MAX_K = 10
N_NEIGHBORS = 15
N_SPLITS = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help=f"Input CSV (default: {DEFAULT_CSV})")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "tsne_knn_state_results",
                        help="Directory for figures, tables, and summary.json")
    parser.add_argument("--perplexity", type=float, default=PERPLEXITY, help="t-SNE perplexity")
    parser.add_argument("--max-k", type=int, default=MAX_K, help="Largest cluster count to sweep")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for t-SNE and K-means")
    return parser.parse_args()


def load_data(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return metadata and the numeric EEG feature block for the three datasets."""
    data = pd.read_csv(csv_path)
    missing = set(EEG_FEATURES + ["dataset", "subject"]).difference(data.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    data = data.loc[data["dataset"].isin(DATASETS)].reset_index(drop=True)
    found = set(data["dataset"].unique())
    if found != set(DATASETS):
        raise ValueError(f"Expected all of {DATASETS}; found {sorted(found)}")

    X = data[EEG_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    # A feature that is empty for every epoch carries no information about any
    # epoch, so drop it rather than imputing one constant into every distance.
    empty = X.columns[X.isna().all()].tolist()
    X = X.drop(columns=empty)
    # An epoch with no usable feature would be placed entirely by the imputer.
    usable = X.notna().any(axis=1)
    print(f"Loaded {len(data)} epochs; dropped {len(empty)} empty features {empty} "
          f"and {int((~usable).sum())} all-missing epochs.")
    return data.loc[usable].reset_index(drop=True), X.loc[usable].reset_index(drop=True)


def embed(X: pd.DataFrame, perplexity: float, seed: int) -> np.ndarray:
    """Standardize the features and project them to two t-SNE dimensions."""
    scaled = make_pipeline(SimpleImputer(strategy="median"), StandardScaler()).fit_transform(X)
    tsne = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                max_iter=1000, random_state=seed)
    return tsne.fit_transform(scaled)


def sweep_k(Z: np.ndarray, max_k: int, seed: int) -> pd.DataFrame:
    """Record inertia and silhouette for each candidate number of states."""
    rows = []
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
        rows.append({"k": k, "inertia": float(model.inertia_),
                     "silhouette": float(silhouette_score(Z, model.labels_))})
        print(f"  k={k:2d}  inertia={rows[-1]['inertia']:12.1f}  silhouette={rows[-1]['silhouette']:.3f}")
    return pd.DataFrame(rows)


def elbow(curve: pd.DataFrame) -> int:
    """Knee of the inertia curve: the k furthest below the chord joining its ends.

    Both axes are normalized to [0, 1] first, so the choice does not depend on the
    units of inertia or on how wide the swept range happens to be.
    """
    k = curve["k"].to_numpy(dtype=float)
    inertia = curve["inertia"].to_numpy(dtype=float)
    x = (k - k[0]) / (k[-1] - k[0])
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    return int(k[np.argmax(1.0 - x - y)])


def order_states_by_pain(labels: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    """Relabel clusters so state 0 has the lowest mean rating and the last the highest.

    K-means IDs are arbitrary; ordering them makes states comparable across runs and
    readable as a severity axis. Ratings are used only for this cosmetic ordering and
    for interpretation, never as a clustering input.
    """
    key = "rating" if data["rating"].notna().any() else "laser_power"
    means = pd.Series(data[key].to_numpy(), index=labels).groupby(level=0).mean()
    ranks = {old: new for new, old in enumerate(means.sort_values().index)}
    return np.array([ranks[value] for value in labels])


def plot_elbow(curve: pd.DataFrame, chosen: int, path: Path) -> None:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.plot(curve["k"], curve["inertia"], "o-")
    left.axvline(chosen, color="crimson", linestyle="--", label=f"elbow k={chosen}")
    left.set(xlabel="number of states (k)", ylabel="within-cluster sum of squares",
             title="Elbow method")
    left.legend()
    right.plot(curve["k"], curve["silhouette"], "o-", color="seagreen")
    right.axvline(chosen, color="crimson", linestyle="--")
    right.set(xlabel="number of states (k)", ylabel="mean silhouette",
              title="Cluster separation")
    for axis in (left, right):
        axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_embedding(Z: np.ndarray, states: np.ndarray, data: pd.DataFrame, path: Path) -> None:
    codes, names = pd.factorize(data["dataset"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for axis, values, title, cmap in (
        (axes[0], states, f"K-means states (k={states.max() + 1})", "tab10"),
        (axes[1], codes, "dataset: " + ", ".join(f"{i}={name}" for i, name in enumerate(names)), "Set2"),
        (axes[2], data["rating"].to_numpy(), "pain rating", "viridis"),
    ):
        points = axis.scatter(Z[:, 0], Z[:, 1], c=values, cmap=cmap, s=4, alpha=0.6)
        axis.set(xlabel="t-SNE 1", ylabel="t-SNE 2", title=title)
        fig.colorbar(points, ax=axis, shrink=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def knn_transfer(Z: np.ndarray, states: np.ndarray, groups: pd.Series, seed: int) -> tuple[float, np.ndarray]:
    """Cross-validated agreement between KNN and the K-means states.

    Every epoch from a given subject stays in one fold, so the score reflects whether
    the state boundaries hold for subjects the classifier has not seen.
    """
    model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS, weights="distance", n_jobs=-1)
    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
    predicted = cross_val_predict(model, Z, states, groups=groups, cv=splitter)
    return float(accuracy_score(states, predicted)), predicted


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data, X = load_data(args.csv)

    print(f"Running t-SNE on {X.shape[0]} epochs x {X.shape[1]} features (perplexity={args.perplexity})...")
    Z = embed(X, args.perplexity, args.seed)

    print("Sweeping K-means over the embedding:")
    curve = sweep_k(Z, args.max_k, args.seed)
    chosen = elbow(curve)
    best_silhouette = int(curve.loc[curve["silhouette"].idxmax(), "k"])
    print(f"Elbow suggests k={chosen} states (best silhouette at k={best_silhouette}).")

    kmeans = KMeans(n_clusters=chosen, n_init=20, random_state=args.seed).fit(Z)
    states = order_states_by_pain(kmeans.labels_, data)

    # A subject ID is only unique within its own dataset.
    groups = data["dataset"].astype(str) + "__" + data["subject"].astype(str)
    agreement, predicted = knn_transfer(Z, states, groups, args.seed)
    print(f"KNN ({N_NEIGHBORS} neighbors) reproduces the states with "
          f"{agreement:.3f} subject-held-out agreement.")

    assignments = data[["dataset", "subject", "epoch", "rating", "laser_power"]].copy()
    assignments["tsne_1"], assignments["tsne_2"] = Z[:, 0], Z[:, 1]
    assignments["state"] = states
    assignments["knn_predicted_state"] = predicted
    assignments.to_csv(args.output_dir / "state_assignments.csv", index=False)

    profile = assignments.groupby("state").agg(
        n_epochs=("state", "size"),
        n_subjects=("subject", "nunique"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        mean_laser_power=("laser_power", "mean"),
    )
    profile["pct_epochs"] = 100 * profile["n_epochs"] / len(assignments)
    profile.to_csv(args.output_dir / "state_profiles.csv")
    pd.crosstab(assignments["state"], assignments["dataset"]).to_csv(
        args.output_dir / "state_dataset_counts.csv")
    curve.to_csv(args.output_dir / "elbow_curve.csv", index=False)

    plot_elbow(curve, chosen, args.output_dir / "elbow.png")
    plot_embedding(Z, states, data, args.output_dir / "tsne_states.png")

    summary = {
        "datasets": list(DATASETS),
        "n_epochs": int(len(assignments)),
        "n_features": int(X.shape[1]),
        "n_dataset_specific_subjects": int(groups.nunique()),
        "tsne": {"perplexity": args.perplexity, "seed": args.seed, "n_components": 2},
        "k_sweep": curve.to_dict(orient="records"),
        "elbow_k": chosen,
        "best_silhouette_k": best_silhouette,
        "silhouette_at_elbow": float(curve.loc[curve["k"] == chosen, "silhouette"].iloc[0]),
        "knn": {"n_neighbors": N_NEIGHBORS, "weights": "distance",
                "cv": f"{N_SPLITS}-fold StratifiedGroupKFold by dataset-specific subject",
                "held_out_agreement_with_kmeans": agreement},
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nState profiles:")
    print(profile.round(2).to_string())
    print(f"\nWrote figures and tables to {args.output_dir}")


if __name__ == "__main__":
    main()
