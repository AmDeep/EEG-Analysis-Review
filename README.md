# EEG-Analysis-Review

This repository contains a reproducible, leakage‑aware pipeline and exploratory analysis for pain vs no‑pain classification using OpenNeuro dataset ds005284 (26 Biosemi subjects). The code extracts a broad set of time‑domain, spectral, multiscale, nonlinear, and connectivity features from selected somatosensory electrodes (C3, CZ, C4), evaluates feature candidates, and validates classification performance under a subject-held-out (LOSO) protocol.

## Quick summary
- Best single-channel biomarkers: nonlinear complexity measures at CZ (Higuchi FD, DFA, sample/perm entropy, Petrosian FD).
- Best single band: Low‑gamma (30–45 Hz), but multiband multifeature extraction provides substantially better performance (multiband AUC >> single-band AUC).
- Classifiers: LDA (shrinkage) provides state-of-the-art performance with a parsimonious model; ensembles are comparable but tend to overfit in some folds.
- Validation: Leave‑One‑Subject‑Out with training-only scaling, variance filtering, and ANOVA-F selection is used to avoid leakage.

---

## Repo structure (top-level)
```
Aditya/                 EDA and feature-extraction scripts
  Feature Extraction/   CSV feature matrices and lep_feature_extraction.py
  Initial EDA/          EDA scripts and PNG figures
Amar/                   Notebooks and reproducible LOSO ablation script
  preprocessed/         CSV / NPZ outputs and LOSO-ablation outputs
  01-05.ipynb           Package check, preprocessing, features, classification, biomarker exploration
  06_biomarker_loso_ablation.py  Leakage-safe ablation script
Hrishi/                 Feature importance notebook
README.md               (this file)
Amar/requirements.txt   Python package pins used for reproducible runs
```

## How to reproduce the main results (short path)
1. Clone the repository and create a Python 3.10+ environment.
2. Install dependencies listed in Amar/requirements.txt.
3. Place the ds005284 BDF files (or configure OpenNeuro downloader) and ensure channels.tsv mapping is present.
4. Run the preprocessing & feature extraction notebooks or scripts (Amar/02 & Amar/03), then the classification notebook (Amar/04) or the leakage-safe ablation script:

```bash
# from repository root
python Amar/06_biomarker_loso_ablation.py
```

Outputs will be written to Amar/preprocessed/ and notebooks/data/preprocessed/ (see `Amar/README.md` for exact filenames created by the notebooks).

---

## Installation (recommended)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r Amar/requirements.txt
# optional: pip install -r Amar/requirements.txt may require manual installs for some git-only packages
```

See Amar/requirements.txt for pinned package versions used in executed runs.

---

## Data preprocessing (exact recommended steps)
Follow these steps to ensure reproducibility and leakage safety. All training-step statistics must be fit on training folds only.

1. Channel mapping
   - Map device channel names to standard 10–20 labels using `channels.tsv` (Biosemi native BDF names must be renamed before ROI selection).
2. Resample
   - Target sampling rate: 256 Hz (the repository resamples from 1024 → 256).
3. Filtering
   - Notch: 50 Hz and 100 Hz (adjust if mains frequency is 60 Hz in your recordings).
   - Bandpass: 0.5 – 80 Hz (adjust upper bound depending on acquisition).
   - Reference: average reference (the repo uses average reference after filtering).
4. Epoching
   - Pain epochs: 0 → +2 s relative to stimulus onset (condition ID used in ds005284 = 54 in this repo).
   - No-pain baseline epochs: −5 → −3 s relative to same onset (pre-stimulus window used here).
   - Adjust windows to the new dataset's timing if necessary.
5. Artifact rejection
   - Amplitude threshold: reject epochs with |amplitude| > 200 µV (dataset-specific — raise/lower if acquisition differs).
   - Optional: additional channel/epoch SNR or flat-channel criteria. We recommend computing a per-subject epoch SNR metric and keeping a QC log.
6. ROI/channel selection
   - Default ROI: C3, CZ, C4 (S1 focus used in ds005284). For other datasets, set ROI tuple as needed.
7. Output
   - Save epoch arrays, event metadata, and a per-subject summary CSV so runs are auditable.

---

## Feature engineering — families and canonical items
The repository extracts a baseline set of 135 features and optionally an expanded set of ~165 features. For a new dataset we recommend extracting the full baseline family list below (grouped by family). Keep exact feature names consistent to ease cross-dataset comparisons.

1) Band power & derived spectral features (per ROI channel)
   - Absolute band power: delta (1–4 Hz), theta (4–8 Hz), alpha (8–13 Hz), beta (13–30 Hz), low-gamma (30–45 Hz)
   - Relative band power (band / total power)
   - Log-transformed band powers
   - Band-power ratios (e.g., alpha/beta, gamma/alpha)
   - Spectral centroid, median frequency, 50% / 95% spectral edges
   - Alpha peak frequency
   - Aperiodic (1/f) slope estimate

2) Nonlinear & complexity measures (per ROI channel) — (antropy)
   - Higuchi fractal dimension (Higuchi FD)
   - Petrosian fractal dimension
   - Detrended fluctuation analysis (DFA)
   - Sample entropy
   - Permutation entropy
   - Spectral entropy
   - Lempel-Ziv complexity
   - Hjorth parameters: activity / mobility / complexity

3) Time-domain waveform & robust statistics (per ROI channel)
   - RMS (root mean square)
   - Standard deviation
   - Peak-to-peak amplitude
   - Line length
   - Zero-crossing rate
   - Crest factor
   - Skewness, kurtosis

4) Multiscale / wavelet (PyWavelets)
   - db4 decomposition band energy ratios (approximation/detail ratios)
   - Wavelet entropy

5) Spectral envelope / MFCC (librosa)
   - MFCC mean coefficients (e.g., first 5 MFCC means per channel)

6) Autoregressive dynamics (spectrum / Burg)
   - Burg AR coefficients (low order coefficients) and innovation variance

7) Connectivity / inter-channel features (pairs: C3-CZ, C3-C4, CZ-C4)
   - Coherence per band
   - Phase Locking Value (PLV) per band
   - Correlation / covariance in time or envelope

Notes on naming: use channel prefix (e.g., CZ_higuchi, CZ_alpha_power, C3_MFCC3_mean, CZ_C4_lowgamma_plv) so feature matrices are self-descriptive.

---

## Constructing the full feature matrix for a new dataset
1. Extract all features above per epoch and per ROI channel and form a feature matrix X with one row per epoch and a matching label vector y.
2. Store a `feature_names` list in the same order as X columns (for auditability). The repository's `features_xy.npz` provides this pattern (X, y, subjects, feature_names).
3. Save raw features to CSV and compressed NPZ: `biomarker_exploration_features.csv` and `.npz`.

---

## Validation process (recommended, leakage-safe)
Always follow training-only statistics and selection. The Amar code enforces these rules; replicate them for new datasets.

1. Cross-validation design
   - Primary strategy: Leave‑One‑Subject‑Out (LOSO / LeaveOneGroupOut) for subject-generalisation. If subjects are not independent, use an appropriate grouping (session, participant).
2. Preprocessing & selection order (apply on training folds only)
   - Handle non-finite values (replace, or drop features that are entirely non-finite in training)
   - Variance threshold: remove zero-variance features (computed on training data)
   - Fit StandardScaler on training data and transform both training and test using those parameters
   - Compute ANOVA-F (univariate) ranking on training data and select top-K (e.g., K=60 as used here) — apply the same selection to test features
3. Classifiers & aggregations
   - Fit classifier on training features (after scaling & selection) and predict probabilities for held-out subject
   - Aggregate per-fold AUCs by averaging across LOSO folds to report mean AUC ± SD (the repository uses mean of per-subject AUCs)
4. Ablations
   - For each feature family, run a baseline vs baseline+family comparison using the same fold splits and compute paired tests across folds (Wilcoxon signed-rank recommended, as used in the repo)
5. Per-subject diagnostics
   - Save per_subject_accuracy.csv and per-fold confusion matrices. Flag subjects with chance-level performance for inspection.
6. Reproducibility
   - Record exact preprocessing & extractor configuration in a JSON hash (the repository writes `biomarker_loso_ablation_config.json`). Save fold audit rows to `biomarker_loso_ablation_folds.csv`.

---

## Deployment recommendations
- For low-latency or interpretability: LDA with shrinkage on a compact feature set (e.g., top 3–10 features from CZ: Higuchi, DFA, SampEn) provides strong performance.
- For research/maximum performance: expanded multiband + wavelet + MFCC families can be added; use LOSO to verify gains are consistent across subjects.
- Implement per-subject QC gates before deployment to prevent biases from outlier subjects.

---

## Outputs created by this repository (where to find them)
- Amar/preprocessed/
  - all_subjects_features.csv (baseline 135 features)
  - features_xy.npz (X, y, subjects, feature_names)
  - classification_results.csv
  - per_subject_accuracy.csv
  - subject_summary.csv
  - biomarker_exploration_features.csv / .npz
  - biomarker_exploration_screening.csv
  - biomarker_loso_ablation_results.csv
  - biomarker_loso_ablation_folds.csv
- Figures: Amar/results_panel.png, Amar/feature_heatmap.png, Amar/erp_analysis.png, Amar/biomarker_exploration_panel.png

---

## Practical checklist for applying to a new dataset (copy into dataset-specific README)
1. Confirm dataset channel naming; create channels.tsv to map to 10–20 labels.
2. Run preprocessing with target sampling rate 256 Hz, notch 50/100, bandpass 0.5–80 Hz.
3. Epoch definitions: align to stimulus timing; set pain & baseline windows.
4. Run artifact rejection and save per-epoch QC metrics.
5. Extract baseline 135 features (all families listed above) and store X, y, subjects, feature_names in NPZ.
6. Run LOSO with training-only scaling & ANOVA-F top-60 selection.
7. Run family-wise ablations using Amar/06_biomarker_loso_ablation.py pattern and save audit CSVs.
8. Produce report figures (ROC, feature heatmap, per-subject accuracy) and a summary table.

---

## Slide deck (high level) and handoff
A simple slide deck is included in `docs/pipeline_slides.md` (markdown slides) summarising pipeline, preprocessing, features, validation, and recommended actions for a new dataset.

---

## Acknowledgements & provenance
- Dataset: OpenNeuro ds005284 (Zhao et al., CC0)
- Core code and executed notebooks: Amar/*, Aditya/*, Hrishi/*

---

If you want, I can:
- Commit this README (I will update the repository root README.md), and add the slide deck file under `docs/` in the same commit.
- Optionally add a `CONTRIBUTING.md` or update Amar/requirements.txt with pinned versions for any extra optional packages.
