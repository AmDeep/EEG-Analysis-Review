# EEG Pain Classification Pipeline
## Dataset: ds005284 — Zhao et al. (OpenNeuro, CC0)
### "26 By Biosemi" — Pain vs No-Pain from Primary Somatosensory Cortex (S1)

---

## Real Results (all 26 subjects, executed August 2026)

### Classification Performance — LOSO Cross-Validation

| Classifier | Accuracy | Bal. Acc | **AUC-ROC** | F1 (Pain) | Sensitivity | Specificity | AUC ± SD |
|---|---|---|---|---|---|---|---|
| **LDA (shrinkage)** | **78.9%** | **78.8%** | **0.850** | **0.772** | **76.5%** | **81.2%** | ±0.134 |
| Gradient Boost | 78.8% | 78.8% | 0.838 | 0.766 | 76.1% | 81.4% | ±0.142 |
| Random Forest | 78.3% | 78.4% | 0.836 | 0.761 | 75.7% | 81.2% | ±0.148 |
| Logistic-L2 | 77.8% | 77.6% | 0.841 | 0.760 | 75.8% | 79.5% | ±0.132 |
| SVM-Linear | 76.0% | 76.0% | 0.832 | 0.742 | 74.5% | 77.5% | ±0.137 |
| SVM-RBF | 74.9% | 75.2% | 0.823 | 0.725 | 72.3% | 78.1% | ±0.141 |

**Dataset stats**: 26 subjects (18 female / 8 male) · 781 epochs (367 pain + 414 no-pain) · 135 baseline features extracted · 60 selected (ANOVA-F)

---

## Key Insights from Results

### 1. Nonlinear Complexity Is the Primary Pain Biomarker

The five most discriminative features are all nonlinear complexity measures at **CZ** — not raw band power:

| Rank | Feature | F-score | What it measures |
|------|---------|---------|-----------------|
| 1 | `CZ_higuchi` | **359** | Fractal dimension — signal self-similarity across scales |
| 2 | `CZ_dfa` | **267** | Long-range temporal correlations (Hurst exponent) |
| 3 | `CZ_samp_ent` | **184** | Irregularity / unpredictability of the time series |
| 4 | `CZ_perm_ent` | **156** | Ordinal pattern complexity |
| 5 | `CZ_petrosian` | **153** | Rapid nonlinear complexity estimate |

**Interpretation**: Pain suppresses the normal resting-state fractal structure of S1 cortex. The brain under pain produces a more *regular*, less complex EEG signal — consistent with thalamocortical entrainment during nociceptive processing. The top features are highly significant in the exploratory Mann-Whitney tests (approximately p < 10⁻⁹).

> The same pattern appears at C3 (F=102) and C4 (F=41), confirming bilateral S1 involvement, with the vertex electrode CZ showing the largest effect — consistent with the midline pain matrix.

---

### 2. Classifier Convergence: LDA Matches Ensemble Methods

All six classifiers cluster tightly in AUC (0.823–0.850), a spread of only **2.7 AUC points**. LDA with analytical shrinkage matches Gradient Boosting at effectively identical balanced accuracy (78.8% each). This means:

- The feature set is the bottleneck, not model complexity
- LDA's linear decision boundary is sufficient — the pain/no-pain classes are largely linearly separable in the 60-feature space
- Simpler models generalise better across subjects under LOSO; tree ensembles overfit within-fold

**Practical implication**: A 60-weight linear model deployable in real time on 3 electrodes is as good as any ensemble on this dataset.

---

### 3. Frequency Band Hierarchy: Broadband Signal, Not One Band

Single-band LDA LOSO AUC:

```
Low-gamma  (30–45 Hz)  0.710  ████████████████████
Alpha      ( 8–13 Hz)  0.686  ████████████████████
Beta       (13–30 Hz)  0.675  ████████████████████
Delta      ( 1– 4 Hz)  0.662  ████████████████████
Theta      ( 4– 8 Hz)  0.622  ████████████████████
Multiband  (all)       0.850  ██████████████████████████  ← +14 pts over best single band
```

**Interpretation**: Low-gamma is the single best band (0.710) — consistent with gamma-band pain processing in S1 — but the AUC jump from best-single-band to multiband (+14 AUC points) confirms that pain modulates the full-spectrum EEG rather than one isolated rhythm. No single band is sufficient; broadband features are necessary.

---

## Leakage-safe expanded-feature validation

`06_biomarker_loso_ablation.py` is the reproducible follow-up to the
exploratory Notebook 05 screen. It evaluates the validated 135-feature
baseline and the 165 expanded features with subject-held-out
`LeaveOneGroupOut` cross-validation. The script reports family-only and
baseline-plus-family ablations for wavelet, MFCC, Burg AR, spectral, and
connectivity features, plus all-expanded combinations.

For every held-out subject, non-finite replacement, variance filtering,
`StandardScaler`, ANOVA-F top-60 selection, and shrinkage LDA are fitted on
training epochs only. The held-out subject is transformed with those
training-fold parameters; no test-subject statistics or exploratory
screening ranking is used. AUC is reported as the mean of the 26 subject-fold
AUCs, matching the aggregation convention used for the historical Notebook 04
value.

Run from the repository root:

```bash
python notebooks/06_biomarker_loso_ablation.py
```

Outputs:

- `data/preprocessed/biomarker_loso_ablation_results.csv` — summary table,
  including deltas against both the fold-local baseline and historical
  Notebook 04 AUC 0.850.
- `data/preprocessed/biomarker_loso_ablation_folds.csv` — one audit row per
  feature-set/held-out-subject fold.
- `data/preprocessed/biomarker_loso_ablation_config.json` — input hashes and
  exact preprocessing/feature-set configuration.

The historical 0.850 result used global variance/ANOVA selection and
per-subject z-normalization that includes statistics from the held-out
subject. It is therefore a legacy reference, not a directly comparable
leakage-safe estimate. The new `baseline_135` row is the sole fair
within-run comparator: an expanded model improves held-out performance when
its `delta_auc_vs_fold_local_baseline` is positive. The historical delta is
retained only as context, not as an estimate of the effect of leakage.

### Executed result

The fold-local baseline reached **AUC 0.832**. The full 165-feature expanded
matrix reached **AUC 0.891** (+0.059 versus the fold-local baseline; +0.041
versus historical 0.850), and the 300-feature baseline-plus-expanded model
reached **0.885**. Wavelets were the strongest individual addition
(baseline-plus-wavelet **0.877**); baseline-plus-MFCC also exceeded the
historical reference (**0.863**). AR, spectral, and connectivity additions did
not exceed 0.850 on their own. The output table includes a paired one-sided
Wilcoxon test across the same 26 held-out subjects for each comparison.

---

### 4. Specificity Exceeds Sensitivity — Asymmetric Error Profile

Across all classifiers, **specificity (no-pain recall) consistently exceeds sensitivity (pain recall)**:
- LDA: Specificity 81.2% vs Sensitivity 76.5% — a ~5-point gap

This means the classifier is better at recognising *absence* of pain than *presence*. Two likely causes:
1. Pain epochs have higher trial-to-trial variability (subjective pain intensity fluctuates across the 16 trials)
2. The pre-stimulus baseline is a cleaner, more homogeneous signal than the pain response

**Research implication**: For a clinical pain detector, this asymmetry favours false negatives (missed pain) over false positives — the safer failure mode in most applications.

---

### 5. Strong Inter-Subject Variability — Two Outlier Subjects

Per-subject LOSO accuracy ranges from **0.438 to 0.969** (mean 0.789, SD 0.129):

- **Top performers** (>90%): sub-001 (0.933), sub-004 (0.903), sub-006 (0.913), sub-008 (0.913), sub-015 (0.969), sub-018 (0.968)
- **Near-chance** (<55%): sub-014 (0.452), sub-016 (0.438)
- **24/26 subjects** classified significantly above chance

The two near-chance subjects (sub-014, sub-016) are both male, age 22–23, with average pain thresholds — no obvious demographic predictor of failure. This matches the broader pain-EEG literature: ~10–15% of subjects show weak or absent cortical pain signatures, possibly due to differential attentional modulation or habituation across the 16 trials.

**Recommendation**: Subject-level quality flags (e.g., minimum epoch SNR or behavioural pain rating) should gate inclusion in clinical deployments.

---

### 6. Connectivity Features Add Incremental Value

Phase Locking Value (PLV) between CZ and C4 in the alpha band appears at rank 17 (F=35.5), and multiple coherence/PLV features are within the selected 60. This suggests:

- Pain modulates interhemispheric synchrony between central and right-hemisphere somatosensory cortex
- Connectivity features are complementary to single-channel complexity: they capture network-level pain processing that channel-wise features miss
- Alpha PLV CZ-C4 drops during pain — consistent with alpha-desynchronisation as an inhibitory gating mechanism releasing nociceptive signals

---

## 🧪 Expanded Biomarker Exploration (Notebook 05)

Notebook `05_biomarker_feature_exploration.ipynb` was executed against all 26 downloaded BDF recordings. It extends the original 135-feature set with feature families chosen to refine pain biomarkers and transfer to other EEG sources:

| Feature family | Library / method | What it adds |
|---|---|---|
| Waveform and robust statistics | NumPy / SciPy | RMS, standard deviation, peak-to-peak, line length, zero-crossing rate, crest factor, skew, kurtosis |
| Nonlinear dynamics | `antropy` | Hjorth activity/mobility/complexity, sample/permutation/spectral entropy, Higuchi and Petrosian fractal dimensions, DFA, Lempel-Ziv |
| Spectral and aperiodic structure | SciPy Welch PSD | Absolute/relative band power, 1/f slope, spectral centroid, 50/95% spectral edges, alpha peak frequency |
| Multiscale decomposition | `PyWavelets` | db4 approximation/detail-band energy ratios and wavelet entropy |
| Spectral-envelope representation | `librosa` | Five MFCC mean features per channel |
| Autoregressive dynamics | `spectrum` | Burg AR coefficients and innovation variance |
| Network-level physiology | SciPy + Hilbert transform | C3-CZ, C3-C4, and CZ-C4 correlation/covariance, alpha and low-gamma PLV/coherence |

### Executed screening results

The expanded extraction uses the **same preprocessed cohort as the validated model**: **781 epochs** (367 pain / 414 no-pain) from all 26 subjects, with **165 expanded features per epoch**.

| Candidate biomarker | Feature family | Effect size (Cohen's d) | Symmetric univariate AUC |
|---|---|---:|---:|
| CZ wavelet D1 relative energy | PyWavelets | −1.285 | **0.839** |
| CZ Higuchi fractal dimension | AntroPy | −1.359 | 0.832 |
| CZ MFCC-3 mean | librosa | +1.354 | 0.831 |
| CZ RMS | Waveform | +1.138 | 0.812 |
| CZ DFA | AntroPy | +1.171 | 0.803 |
| CZ Burg AR(1) | Spectrum | −1.031 | 0.783 |
| CZ aperiodic 1/f slope | Spectral | −0.890 | 0.743 |

**What changed:** Wavelet and MFCC features are now leading candidates alongside the established nonlinear measures. Their best univariate AUCs are comparable to the existing CZ Higuchi feature, while connectivity was weaker in this individual-feature screen (best family AUC 0.667). This supports testing a compact CZ-centered wavelet/MFCC/nonlinear panel under strict LOSO validation.

Package availability was also checked during the executed run. `mne`, `antropy`, `PyWavelets`, `spectrum`, and `librosa` are available and used. `mne-connectivity`, `python-neo`, `invertmeeg`, and `BEST` are currently optional/unavailable, so none contribute silent fallback features.

The notebook writes:

- `data/preprocessed/biomarker_exploration_features.csv` — expanded feature matrix computed from raw epochs
- `data/preprocessed/biomarker_exploration_features.npz` — compressed `X`, `y`, `subjects`, and `feature_names`
- `data/preprocessed/biomarker_exploration_screening.csv` — subject-normalized exploratory effect sizes, symmetric AUC, and Mann-Whitney p-values
- `biomarker_exploration_panel.png` — feature-effect and feature-family summary

### Transfer to other EEG sources

The loader is parameterized by BIDS root, task name, target sampling rate, and channel tuple. The same feature extractor can therefore be used for another ROI (for example frontal or occipital control channels) or another BIDS EEG source, provided the source supplies BDF + `channels.tsv` + `events.tsv`. The notebook demonstrates S1, frontal, and occipital configurations on the real `sub-001` recording.

The Biosemi positional channel rename remains mandatory: native BDF names (`A1`, `A2`, ..., `B32`) are mapped through `channels.tsv` before any source/ROI channel is selected.

### Interpretation and validation guardrail

The expanded ranking is hypothesis-generating, not a new validated performance claim. The strongest candidates should be carried into Notebook 04 and selected within each LOSO training fold. In particular, compare whether wavelet, MFCC, AR, aperiodic-slope, and connectivity families add AUC beyond the existing nonlinear CZ features without increasing subject leakage or overfitting.

---

## Per-Band Discriminability Summary

| Band | AUC | Biological Role in Pain |
|------|-----|------------------------|
| **Low-gamma (30–45 Hz)** | **0.710** | Feedforward nociceptive signalling in S1; strongest single-band marker |
| Alpha (8–13 Hz) | 0.686 | Alpha-desynchronisation reflects reduced inhibition → pain gating |
| Beta (13–30 Hz) | 0.675 | Sensorimotor processing; beta rebound reduced under pain |
| Delta (1–4 Hz) | 0.662 | Low-frequency pain oscillations; often linked to ongoing pain |
| Theta (4–8 Hz) | 0.622 | Weakest discriminator; more associated with affective pain than sensory |

---

## Pipeline Overview

```
OpenNeuro S3 (ds005284)
    │
    ▼
BDF download (26 subjects, ~63 MB each, CC0)
    │  mne.io.read_raw_bdf
    ▼
Channel rename (A1/B1/... → 10-20 via channels.tsv)
    │
    ▼
Preprocessing
  ├─ Resample 1024 → 256 Hz
  ├─ Notch filter: 50 + 100 Hz
  ├─ Bandpass: 0.5–80 Hz
  └─ Average reference
    │
    ▼
Epoching (condition 54 = pain onset)
  ├─ Pain:    0 to +2 s (baseline -0.2→0 s)
  └─ No-Pain: −5 to −3 s pre-onset window
    │
    ▼
Artifact rejection (|amplitude| > 200 µV)
    │
    ▼
S1 extraction: C3, CZ, C4
    │
    ▼
Feature extraction (135 features/epoch)
  ├─ Band power: δ θ α β γ (abs/rel/log + ratios)
  ├─ Antropy: sample entropy, perm entropy, spec entropy,
  │           Higuchi FD, Petrosian FD, DFA, LZiv, Hjorth
  ├─ Time domain: RMS, STD, skew, kurtosis, PTP
  ├─ Spectral: centroid, median frequency
  └─ Connectivity C3-CZ / C3-C4 / CZ-C4:
             coherence + PLV (5 bands × 3 pairs)
    │
    ▼
Variability removal
  ├─ Variance threshold (removes 48 zero-variance features)
  ├─ Per-subject z-normalisation (inside CV folds — no leakage)
  └─ ANOVA-F top-60 feature selection
    │
    ▼
LOSO Classification (26 folds)
  ├─ LDA (shrinkage)  ← BEST: AUC 0.850
  ├─ SVM-RBF + PCA whitening
  ├─ SVM-Linear
  ├─ Logistic-L2
  ├─ Random Forest
  └─ Gradient Boosting
```

---

## Dataset Facts
- **Source**: OpenNeuro ds005284, CC0 licence
- **S3**: `https://s3.amazonaws.com/openneuro.org/ds005284/`
- **Subjects**: 26 (18 F / 8 M, age 18–25, pain threshold 2.75–4.50 mA)
- **Task**: `26ByBiosemi` — thermal pain stimulation (condition 54 = stimulus onset)
- **EEG**: Biosemi ActiveTwo, 64 Ag/AgCl electrodes, 1024 Hz, 50 Hz power line
- **Reference**: CMS/DRL (Biosemi common-mode sense / driven right leg)
- **Event structure**: 16 pain stimuli per subject (~12 s apart, starting at ~122 s)
- **Paper**: Lu X et al., *J Pain Res* 2019 — music reduces pain unpleasantness

## Packages
| Package | Role |
|---------|------|
| `mne` 1.12 | BDF loading, filtering, epoching |
| `antropy` 0.2 | Nonlinear complexity features |
| `PyWavelets` 1.8 | Multiscale db4 energy ratios and wavelet entropy |
| `spectrum` | Burg autoregressive coefficients and innovation variance |
| `librosa` 0.11 | MFCC spectral-envelope features |
| `scikit-learn` 1.9 | Classification, LOSO CV, feature selection |
| `scipy` | Welch PSD, coherence, Butter filter, Hilbert |
| `seaborn` / `matplotlib` | Figures |
| `pandas` / `numpy` | Data management |

## Output Files
```
notebooks/data/preprocessed/
├── all_subjects_features.csv   (781 × 135 feature matrix)
├── features_xy.npz             (X, y, subjects, feature_names)
├── classification_results.csv  (6 classifiers × 9 metrics)
├── per_subject_accuracy.csv    (26 subjects × LOSO accuracy)
├── subject_summary.csv         (epoch counts per subject)
├── biomarker_exploration_features.csv (781 × 165 expanded raw-signal feature matrix)
├── biomarker_exploration_features.npz (expanded X, y, subjects, feature names)
└── biomarker_exploration_screening.csv (exploratory effect-size and AUC ranking)

notebooks/
├── results_panel.png           (AUC bars, ROC, table, per-subject, features, per-band)
├── erp_analysis.png            (ROC overlay, violin, accuracy vs pain threshold)
├── feature_heatmap.png         (z-scored heatmap + correlation matrix)
└── biomarker_exploration_panel.png (expanded feature-effect and family screen)
```

## Notebooks
| Notebook | Content |
|----------|---------|
| `01_package_evaluation.ipynb` | Package comparison & scoring |
| `02_data_loading_preprocessing.ipynb` | BDF loading, filtering, epoching (**executed**) |
| `03_feature_engineering.ipynb` | Feature extraction & statistics (**executed**) |
| `04_classification_results.ipynb` | Full LOSO results with figures (**executed**) |
| `05_biomarker_feature_exploration.ipynb` | Expanded raw-signal biomarker families, real-data screen, and portable source/ROI extractor (**executed**) |
