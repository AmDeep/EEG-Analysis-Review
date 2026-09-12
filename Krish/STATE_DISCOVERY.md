# EEG state discovery experiment

Run from the repository root:

```powershell
python -m pip install -r Krish/requirements_state_discovery.txt
python Krish/07_state_discovery.py
```

If using the project virtual environment, substitute `.\.venv\Scripts\python.exe` for `python`.

The source is `Aditya/Feature Extraction/features_combined.csv`. Only rows with dataset IDs `ds005289`, `ds005280`, and `ds005473` are included, even when `--csv` points to another file. The experiment reuses the 24 EEG feature names in `eeg_model_utils.py`. Dataset, subject, epoch, acquisition metadata, rating, and laser power are never input features. Source CSVs and existing classification scripts are unchanged.

## What the experiment does

1. Convert nonnumeric and infinite feature values to missing values; discard rows with no usable EEG features. Median-impute, standardize, and retain 95% of variance with PCA. Entirely missing columns become constants.
2. Fit two-dimensional t-SNE with perplexities 30 and 50 and seeds 42 and 7. Each embedding uses all filtered usable epochs. Compare K-means clustering in each t-SNE embedding with K-means in the retained PCA space.
3. Sweep K-means cluster counts from 1 to 10 with 20 initializations. Plot inertia (within-cluster sum of squares) and silhouette scores. Select an elbow using the largest vertical gap below the chord joining the normalized curve endpoints. If the gap is below 0.05, report no clear elbow; this threshold is a heuristic, not a statistical test. Increase `--max-k` to inspect sensitivity to the search range.
4. Save epoch assignments, cluster sizes, dataset counts, and rating/laser-power profiles within each dataset. Adjusted mutual information with dataset and subject identity helps diagnose nuisance structure. Adjusted Rand agreement between t-SNE runs measures sensitivity to seed/perplexity, even when the selected counts differ.
5. Evaluate KNN transfer in five subject-held-out folds and three leave-one-dataset-out folds. Within every training fold, refit imputation, scaling, PCA, elbow selection, and K-means. Train a distance-weighted 15-neighbor classifier on those training cluster IDs. For held-out data, compare its predictions with the training K-means model's nearest-centroid assignments and a training-majority-cluster baseline. No held-out rows contribute to preprocessing or selecting the number of clusters. Cluster IDs are local to each fold and must not be pooled as a single label space.

KNN is a supervised classifier: it needs labels. Here it learns **pseudo-labels** from K-means. Its held-out agreement is a test of whether it reproduces the discovered partition, **not pain classification accuracy**. No pain severity ordering is assigned to cluster IDs. K-means itself can assign new PCA-transformed epochs, so the KNN comparison tests whether a neighbor-based mapping is useful.

The t-SNE branch is an exploratory analysis fitted to the full filtered dataset, not a held-out performance estimate. Standard scikit-learn t-SNE has no `transform` for unseen observations; separately embedding train and test data would produce incompatible coordinates. PCA provides a training-fitted transform for the KNN generalization experiment.

## Outputs

Results are written to `Krish/state_discovery_results/` by default:

- `summary.json`: source fingerprint, settings, candidate counts, separation and stability diagnostics.
- `*_elbow.png` / `.csv`: elbow and silhouette plots and values for each representation.
- `*_embedding.png`: identical coordinates colored by cluster and by dataset.
- `*_assignments.csv`: original zero-based source row, identifiers, available outcomes, cluster, and coordinates.
- `*_rating_profiles.csv`, `*_laser_power_profiles.csv`, `*_dataset_counts.csv`: post-hoc interpretation tables.
- `knn_transfer_metrics.csv` / `knn_transfer_predictions.csv`: held-out pseudo-label agreement and fold-specific predictions.
- `dataset_summary.csv` and `feature_missingness.csv`: input audit.

Example sensitivity run:

```powershell
python Krish/07_state_discovery.py --max-k 15 --perplexities 10 30 50 --seeds 42 7 --output-dir Krish/state_discovery_sensitivity
```

## Interpretation and next experiments

The elbow suggests a candidate number of geometric clusters; it does not establish an optimal biological number of pain states. t-SNE distorts global distances and densities, so its clusters can be artifacts of the representation. Compare PCA findings, cluster sizes, stability, and source-dataset composition before interpreting a state. Silhouette values in t-SNE and PCA describe different geometries and should not be treated as directly comparable measures of biological validity.

The studies contribute unequal numbers of epochs and subjects. Pooled fitting weights epochs equally, so large studies and participants with more epochs have more influence. This first run does not correct artifacts, harmonize study protocols, or standardize pain rating scales. Inspect within-study rating profiles and confirm the original scale definitions before naming low/moderate/high pain states. Laser power is a stimulus setting, not a direct pain label.

Next candidates are PCA + Gaussian mixtures (soft state probabilities, choose count with BIC), and density-based clustering if clusters are irregular or many epochs are outliers. For a validated pain predictor, define a consistent label from study protocols/ratings, train KNN on those labels, and evaluate on unseen subjects and datasets with training-only tuning. A temporal state model would additionally require verified timing and trial continuity; this script performs epoch clustering, not temporal segmentation.

Sources: [scikit-learn KNN](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html), [t-SNE API](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html), [silhouette analysis](https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html), and [How to Use t-SNE Effectively](https://distill.pub/2016/misread-tsne/).
