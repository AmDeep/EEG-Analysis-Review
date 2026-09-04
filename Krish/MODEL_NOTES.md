# EEG laser-power classification baselines

## What every script does

The target is `laser_power`, treated as 15 discrete classes from 1.00 to 4.50. This matches the request for classification accuracy. The scripts use 24 EEG-derived time-domain, spectral, connectivity, entropy, and complexity features. They deliberately exclude `dataset`, `subject`, `epoch`, `vertex_channel`, `sfreq`, `gamma_band_hz`, and `rating`. Those fields are identifiers, acquisition metadata, constant fields, or a behavioral outcome rather than an EEG feature. Including them could make a model identify a study or subject instead of learning an EEG–power relationship.

The split is five-fold `StratifiedGroupKFold` with seed 42. The group is `dataset + subject`, because labels such as `sub-001` recur in different studies. Every epoch for one dataset-specific subject stays entirely in either train or test for a fold. This is the key safeguard: random epoch splits would let the model see a participant's EEG in training and again in testing, leading to overly optimistic accuracy.

The data have 29,515 epochs, 678 dataset-specific subjects, 15 intensity classes, and an imbalanced class distribution. The `plv_FCz-CPz` feature is missing for many epochs. Each model median-imputes missing feature values using only its training fold. Scaled models also standardize using their training fold only.

Read `balanced_accuracy` and `macro_f1` alongside ordinary `accuracy`. Accuracy is dominated by the common 3.00 and 3.50 classes. Balanced accuracy gives each intensity equal weight; macro-F1 also penalizes a model that never predicts rare levels.

## 1. Logistic regression

**Intuition:** each EEG feature adds a weighted push toward or away from each power level. The final boundary is a flat plane in feature space.

**Settings:** multinomial logistic regression, `C=1.0`, `max_iter=3000`, and `class_weight="balanced"`; median imputation and z-scoring precede it. `C=1.0` is a moderate amount of L2 regularization, appropriate because EEG features such as band powers can be correlated. The larger iteration limit makes convergence reliable. Class weighting reduces the pull of common intensities.

**Strengths:** fast, stable, comparatively interpretable, and a rigorous linear reference. Its coefficient patterns can be inspected later to understand which features track a class.

**Weaknesses:** it cannot naturally express threshold effects or feature interactions, such as a pattern that occurs only when both an ERP amplitude and a gamma feature cross a range.

## 2. Random forest

**Intuition:** many decision trees each vote for a power level. A tree asks simple questions such as “is beta power above this threshold?”; the forest averages many such trees.

**Settings:** 300 trees, `max_features="sqrt"`, `min_samples_leaf=3`, and `class_weight="balanced_subsample"`. Trying about the square root of 24 features at each split makes the trees less alike. Three epochs per leaf prevents extremely specific splits. Balancing within each bootstrap sample addresses class imbalance.

**Strengths:** captures nonlinear effects and interactions, needs no scaling, and gives feature-importance estimates.

**Weaknesses:** importance values can favor correlated or high-variance features; predictions are less transparent than logistic regression. It can still learn study-specific patterns if the study designs differ, even with subjects held out.

## 3. Extra trees

**Intuition:** like a random forest, but it also randomizes where it tries thresholds. That extra randomness can reduce variance when measurements are noisy.

**Settings:** 300 trees, `max_features="sqrt"`, `min_samples_leaf=3`, and `class_weight="balanced"`. They match the forest settings to make the comparison fair; random thresholds provide the main difference.

**Strengths:** often faster than a random forest, robust to nonlinear EEG relationships, and a useful alternative ensemble when individual epoch features are noisy.

**Weaknesses:** may miss a precise informative threshold because it samples thresholds randomly. Like the forest, it does not show a simple single rule for a prediction.

## 4. Histogram gradient boosting

**Intuition:** trees are built one after another. Each new small tree concentrates on the mistakes made by the earlier trees, gradually refining the boundary between nearby intensity levels.

**Settings:** `learning_rate=0.08`, `max_iter=300`, `max_leaf_nodes=15`, and `l2_regularization=1.0`; balanced training-sample weights are passed in each fold. A small learning rate with many iterations learns gradually. Fifteen leaves keeps each tree shallow enough for a baseline rather than memorizing subjects, and L2 regularization further discourages overly sharp decisions.

**Strengths:** often one of the best models for engineered/tabular EEG features because it learns smooth nonlinear interactions efficiently.

**Weaknesses:** more sensitive to hyperparameters than the forest approaches, less directly interpretable, and can exploit dataset differences if datasets are systematically associated with certain powers.

## 5. Linear support-vector machine

**Intuition:** it finds the widest straight separating boundary between power classes after the features have been placed on the same scale. Unlike logistic regression, it focuses most on epochs near a class boundary.

**Settings:** linear SVM, `C=0.1`, `class_weight="balanced"`, `dual=False`, and `max_iter=10000`, plus median imputation and z-scoring. `C=0.1` prioritizes a wider, more regularized margin to reduce overfitting. `dual=False` is efficient here because there are far more epochs than features.

**Strengths:** computationally efficient for this many epochs, robust in a modest-dimensional feature space, and complementary to logistic regression because it uses a different loss function.

**Weaknesses:** it cannot represent nonlinear or interaction effects without feature engineering. It also has no native feature-importance measure.

## 6. k-nearest neighbors

**Intuition:** a new epoch receives the power level favored by its most similar training epochs. It makes almost no assumptions about the shape of the boundary.

**Settings:** 15 neighbors, inverse-distance voting, Euclidean distance (`p=2`), after median imputation and z-scoring. Fifteen neighbors smooths out noisy single epochs; closer neighbors carry more influence than distant ones.

**Strengths:** simple nonlinear comparison with few modeling assumptions. It is a useful diagnostic of whether nearby EEG feature vectors tend to share an intensity.

**Weaknesses:** high-dimensional feature spaces make distances less meaningful, and the large majority classes can dominate local neighborhoods. It offers little interpretability and can be slow at prediction time.

## Important interpretation limit

The model scripts test whether EEG features predict power for held-out subjects within the pooled studies. They do not establish a causal EEG response to power. Dataset is excluded as a feature, but the datasets have very different power distributions; a later, stricter check would be leave-one-dataset-out validation. That test asks whether an association learned in some studies transfers to an entirely unseen study, and its performance will usually be lower but more transportable.

## Running a script

Install dependencies first if needed:

```powershell
python -m pip install scikit-learn pandas numpy
```

Run any model from this folder, optionally passing another CSV and output directory:

```powershell
python .\04_hist_gradient_boosting.py
python .\04_hist_gradient_boosting.py --csv "C:\path\to\features.csv" --output-dir .\results
```

Each run writes a one-row metrics CSV/JSON, epoch-level out-of-fold predictions, and a confusion matrix. All scripts use the same cross-validation folds, so their metrics are directly comparable.
