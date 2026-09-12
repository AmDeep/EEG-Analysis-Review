"""Exploratory t-SNE/K-means states and subject-held-out PCA/KNN transfer."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score, accuracy_score, silhouette_score
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from eeg_model_utils import EEG_FEATURES

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ("ds005289", "ds005280", "ds005473")


def load_features(path):
    data = pd.read_csv(path)
    required = set(EEG_FEATURES + ["dataset", "subject", "epoch"])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    data = data.loc[data.dataset.isin(DATASETS)].copy()
    if set(data.dataset.unique()) != set(DATASETS):
        raise ValueError(f"Input must contain all three requested datasets: {DATASETS}")
    if data[["dataset", "subject"]].isna().any().any():
        raise ValueError("Missing subject identifiers; cannot safely split subjects.")
    data.insert(0, "source_row", data.index)  # zero-based data row, excluding CSV header
    data = data.reset_index(drop=True)
    X = data[EEG_FEATURES].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    usable = X.notna().any(axis=1)
    dropped = int((~usable).sum())
    data, X = data.loc[usable].reset_index(drop=True), X.loc[usable].reset_index(drop=True)
    if len(X) < 6:
        raise ValueError("At least six usable epochs are required.")
    return data, X, dropped


def preprocessing():
    # Entirely missing training columns become constants rather than using test data.
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                         StandardScaler(), PCA(n_components=0.95, svd_solver="full"))


def elbow(curve):
    """Largest gap below the normalized endpoint chord; no forced elbow on a line."""
    ks = curve.k.to_numpy()
    inertia = curve.inertia.to_numpy()
    if len(ks) < 3 or inertia[0] - inertia[-1] <= np.finfo(float).eps:
        return None, 0.0
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (inertia - inertia[-1]) / (inertia[0] - inertia[-1])
    gaps = 1 - x - y
    index = int(np.argmax(gaps))
    if index in (0, len(ks) - 1) or gaps[index] < 0.05:
        return None, float(gaps[index])
    return int(ks[index]), float(gaps[index])


def sweep(Z, max_k, seed, silhouette=True):
    upper = min(max_k, len(Z) - 1, len(np.unique(Z, axis=0)))
    rows, models = [], {}
    for k in range(1, upper + 1):
        model = KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)
        score = np.nan
        if silhouette and 1 < len(np.unique(model.labels_)) < len(Z):
            try:
                score = silhouette_score(Z, model.labels_, sample_size=min(2000, len(Z)), random_state=seed)
            except ValueError:
                pass  # A tiny cluster can disappear from the silhouette sample.
        rows.append({"k": k, "inertia": model.inertia_, "silhouette": score})
        models[k] = model
    curve = pd.DataFrame(rows)
    k, strength = elbow(curve)
    return curve, k, strength, models


def cluster_outputs(name, Z, data, args):
    curve, k, strength, models = sweep(Z, args.max_k, args.seed)
    curve.to_csv(args.output_dir / f"{name}_elbow.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    axes[0].plot(curve.k, curve.inertia, "o-")
    axes[0].set(xlabel="Number of candidate states (K-means k)", ylabel="Within-cluster sum of squares", title=f"{name}: elbow")
    axes[1].plot(curve.k, curve.silhouette, "o-")
    axes[1].set(xlabel="Number of candidate states", ylabel="Silhouette (sample up to 2,000)", title="Separation in this representation")
    if k is not None:
        for ax in axes:
            ax.axvline(k, color="tab:red", linestyle="--", label=f"Elbow candidate: {k}")
            ax.legend()
    fig.savefig(args.output_dir / f"{name}_elbow.png", dpi=160)
    plt.close(fig)
    result = {"candidate_k": k, "normalized_elbow_gap": strength,
              "pca_dimensions" if name == "pca" else "dimensions": int(Z.shape[1])}
    if k is None:
        result["note"] = "No clear elbow under the configured heuristic; no states assigned."
        return result, None
    labels = models[k].labels_
    result["silhouette"] = float(curve.loc[curve.k == k, "silhouette"].iloc[0])
    result["dataset_adjusted_mutual_information"] = float(adjusted_mutual_info_score(data.dataset, labels))
    subject_groups = data.dataset.astype(str) + "__" + data.subject.astype(str)
    result["subject_adjusted_mutual_information"] = float(adjusted_mutual_info_score(subject_groups, labels))
    result["cluster_sizes"] = {str(c): int(n) for c, n in zip(*np.unique(labels, return_counts=True))}
    metadata = [c for c in ["source_row", "dataset", "subject", "epoch", "rating", "laser_power"] if c in data]
    assignments = data[metadata].copy()
    assignments["cluster"] = labels
    for dim in range(Z.shape[1]):
        assignments[f"component_{dim + 1}"] = Z[:, dim]
    assignments.to_csv(args.output_dir / f"{name}_assignments.csv", index=False)
    pd.crosstab(assignments.cluster, assignments.dataset).to_csv(args.output_dir / f"{name}_dataset_counts.csv")
    for target in ("rating", "laser_power"):
        if target in assignments:
            assignments[target] = pd.to_numeric(assignments[target], errors="coerce")
            assignments.groupby(["dataset", "cluster"])[target].agg(["count", "median", "mean", "std", "min", "max"]).to_csv(
                args.output_dir / f"{name}_{target}_profiles.csv")
    if Z.shape[1] >= 2:
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
        axes[0].scatter(Z[:, 0], Z[:, 1], c=labels, cmap="tab10", s=3, alpha=0.5, rasterized=True)
        axes[0].set_title(f"{name}: {k} candidate clusters (unordered)")
        for dataset in DATASETS:
            mask = data.dataset == dataset
            axes[1].scatter(Z[mask, 0], Z[mask, 1], label=dataset, s=3, alpha=0.5, rasterized=True)
        axes[1].set_title("Same coordinates colored by source dataset")
        axes[1].legend(markerscale=3)
        for ax in axes:
            ax.set(xlabel="Component 1", ylabel="Component 2")
        fig.savefig(args.output_dir / f"{name}_embedding.png", dpi=160)
        plt.close(fig)
    return result, labels


def evaluate_transfer(data, X, args):
    """KNN imitates training-only PCA/K-means states; agreement is NOT pain accuracy."""
    subjects = data.dataset.astype(str) + "__" + data.subject.astype(str)
    if subjects.nunique() < 2:
        raise ValueError("At least two subjects are needed for held-out evaluation.")
    schemes = {
        "subject_held_out": GroupKFold(n_splits=min(5, subjects.nunique())).split(X, groups=subjects),
        "dataset_held_out": LeaveOneGroupOut().split(X, groups=data.dataset),
    }
    rows, predictions = [], []
    for scheme, splits in schemes.items():
        for fold, (train, test) in enumerate(splits, 1):
            assert set(subjects.iloc[train]).isdisjoint(subjects.iloc[test])
            prep = preprocessing()
            Ztrain = prep.fit_transform(X.iloc[train])
            Ztest = prep.transform(X.iloc[test])
            curve, k, strength, models = sweep(Ztrain, args.max_k, args.seed, silhouette=False)
            curve.to_csv(args.output_dir / f"{scheme}_fold{fold}_elbow.csv", index=False)
            row = {"split": scheme, "fold": fold, "train_epochs": len(train), "test_epochs": len(test),
                   "train_subjects": subjects.iloc[train].nunique(), "test_subjects": subjects.iloc[test].nunique(),
                   "test_datasets": ",".join(sorted(data.dataset.iloc[test].unique())),
                   "candidate_k": k, "normalized_elbow_gap": strength}
            if k is not None:
                clusterer = models[k]
                knn = KNeighborsClassifier(n_neighbors=min(args.neighbors, len(train)), weights="distance")
                knn.fit(Ztrain, clusterer.labels_)
                reference, predicted = clusterer.predict(Ztest), knn.predict(Ztest)
                majority = int(np.bincount(clusterer.labels_).argmax())
                row.update(knn_cluster_agreement=accuracy_score(reference, predicted),
                           majority_cluster_agreement=float(np.mean(reference == majority)),
                           adjusted_rand_agreement=adjusted_rand_score(reference, predicted))
                frame = data.iloc[test][["source_row", "dataset", "subject", "epoch"]].copy()
                frame["split"], frame["fold"] = scheme, fold
                frame["kmeans_reference_cluster"], frame["knn_cluster"] = reference, predicted
                predictions.append(frame)
            rows.append(row)
            print(f"{scheme}, fold {fold}: candidate k={k}", flush=True)
    pd.DataFrame(rows).to_csv(args.output_dir / "knn_transfer_metrics.csv", index=False)
    if predictions:
        pd.concat(predictions).to_csv(args.output_dir / "knn_transfer_predictions.csv", index=False)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=ROOT / "Aditya" / "Feature Extraction" / "features_combined.csv")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "state_discovery_results")
    parser.add_argument("--max-k", type=int, default=10)
    parser.add_argument("--perplexities", type=float, nargs="+", default=[30, 50])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 7])
    parser.add_argument("--seed", type=int, default=42, help="K-means and silhouette seed")
    parser.add_argument("--neighbors", type=int, default=15)
    parser.add_argument("--tsne-iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.max_k < 3 or args.neighbors < 1 or args.tsne_iterations < 250:
        parser.error("max-k must be >=3, neighbors >=1, tsne-iterations >=250")
    data, X, dropped = load_features(args.csv)
    if any(p <= 0 or p >= len(X) for p in args.perplexities):
        parser.error("Each perplexity must be positive and smaller than the filtered sample count.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.groupby("dataset").agg(epochs=("subject", "size"), subjects=("subject", "nunique")).to_csv(args.output_dir / "dataset_summary.csv")
    X.isna().mean().rename("missing_fraction").to_csv(args.output_dir / "feature_missingness.csv")
    prep = preprocessing()
    Z = prep.fit_transform(X)
    if not np.isfinite(Z).all() or np.var(Z, axis=0).sum() <= 0:
        raise ValueError("No usable feature variation after preprocessing.")
    summary = {"source": str(args.csv.resolve()), "source_sha256": hashlib.sha256(args.csv.read_bytes()).hexdigest(),
               "datasets": list(DATASETS), "epochs": len(X), "dropped_all_missing_epochs": dropped,
               "features": EEG_FEATURES, "sklearn_version": sklearn.__version__,
               "config": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
               "interpretation": "Exploratory clusters are unordered, unvalidated candidate states. KNN agreement measures imitation of K-means, not pain prediction accuracy.",
               "representations": {}}
    summary["representations"]["pca"], _ = cluster_outputs("pca", Z, data, args)
    reference_labels = None
    for perplexity in dict.fromkeys(args.perplexities):
        for seed in dict.fromkeys(args.seeds):
            name = f"tsne_p{perplexity:g}_seed{seed}"
            print(f"Fitting {name} on {len(Z)} epochs...", flush=True)
            embedding = TSNE(n_components=2, perplexity=perplexity, init="pca", learning_rate="auto",
                             max_iter=args.tsne_iterations, random_state=seed).fit_transform(Z)
            result, labels = cluster_outputs(name, embedding, data, args)
            if reference_labels is not None and labels is not None:
                result["adjusted_rand_vs_first_tsne"] = float(adjusted_rand_score(reference_labels, labels))
            elif labels is not None:
                reference_labels = labels
            summary["representations"][name] = result
            print(f"{name}: {result}", flush=True)
    summary["knn_transfer"] = evaluate_transfer(data, X, args)
    # pandas/numpy metrics are explicitly converted to JSON-compatible scalars.
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=lambda x: x.item()), encoding="utf-8")
    print(f"Results saved to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    # Prevent BLAS/OpenMP oversubscription during repeated small K-means fits.
    with threadpool_limits(limits=2):
        main()
