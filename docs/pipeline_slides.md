# EEG Pain Biomarker Pipeline — Overview

Slide 1 — Title
- EEG Pain Biomarker Pipeline
- Reproducible preprocessing, feature engineering, and LOSO validation for ds005284 (26 subjects)
- Presenter: EEG Analysis Team

---

Slide 2 — Goals
- Identify robust EEG biomarkers that discriminate pain vs no‑pain
- Produce leakage-safe held‑out validation (LOSO)
- Provide a deployable, interpretable model and an audit trail for reproducibility

---

Slide 3 — Dataset & Key facts
- OpenNeuro ds005284 (Zhao et al.), Biosemi 64-channel, 1024 Hz (resampled to 256 Hz)
- 26 subjects, 781 epochs (367 pain / 414 no‑pain)
- ROI: C3, CZ, C4 focused on somatosensory responses

---

Slide 4 — Preprocessing pipeline (exact steps)
1. Channel mapping: map raw device labels → 10–20 (channels.tsv)
2. Resample: 1024 → 256 Hz
3. Notch: 50 Hz & 100 Hz (adjust to 60 Hz mains if needed)
4. Bandpass: 0.5 – 80 Hz
5. Reference: average reference
6. Epoching:
   - Pain: 0 → +2 s relative to stimulus onset
   - No‑pain baseline: −5 → −3 s
7. Artifact rejection: amplitude threshold (|µV| > 200)
8. Save per-epoch QC and per-subject summaries (audit CSVs)

---

Slide 5 — Feature families (extract all for new datasets)
- Spectral: abs/rel band power (δ,θ,α,β,low‑γ), band ratios, spectral centroid, median freq, 1/f slope
- Nonlinear: Higuchi FD, Petrosian FD, DFA, sample & permutation entropy, Lempel‑Ziv, Hjorth
- Time-domain: RMS, STD, peak‑to‑peak, line length, zero-crossing rate, skew, kurtosis
- Wavelet: db4 multiscale energy ratios, wavelet entropy
- MFCC: first 5 MFCC means per channel
- AR dynamics: Burg AR coefficients + innovation variance
- Connectivity: coherence & PLV for C3-CZ, C3-C4, CZ-C4 across bands

---

Slide 6 — Feature naming & storage
- Use descriptive names: e.g., CZ_higuchi, C3_alpha_power, CZ_C4_lowgamma_plv
- Save:
  - features CSV (rows = epochs, cols = features)
  - features NPZ (X, y, subjects, feature_names)
  - per-epoch QC CSV and subject_summary.csv

---

Slide 7 — Validation (leakage-safe procedure)
1. Use Leave‑One‑Subject‑Out (LOSO) or appropriate grouped CV
2. For each training fold:
   - Replace non-finite, apply variance threshold (train-only)
   - Fit StandardScaler (train-only) → transform test
   - Compute ANOVA-F ranking on train → select top‑K (e.g., K=60)
   - Fit classifier (e.g., LDA with shrinkage)
3. Predict on held‑out subject; collect per-fold AUCs, sensitivities, specificities
4. Report mean AUC ± SD and per-subject metrics; use paired tests for ablations (Wilcoxon)

---

Slide 8 — Recommended classifiers & deployment
- Research / exploration: RF / Gradient Boosting for ablation; check overfitting
- Deployment / interpretability: LDA (shrinkage) on parsimonious set (e.g., top nonlinear CZ features)
- Implement per-subject QC gating before decision-making

---

Slide 9 — Ablation strategy
- Baseline (135 features) vs Baseline + family (wavelet / MFCC / AR / connectivity)
- Use the same LOSO splits and compute paired Wilcoxon tests across folds
- Record fold-level outputs and config JSON for auditability

---

Slide 10 — Typical outputs & audit files
- all_subjects_features.csv, features_xy.npz, classification_results.csv
- per_subject_accuracy.csv, subject_summary.csv
- biomarker_exploration_screening.csv, biomarker_loso_ablation_results.csv
- Figures: ROC curves, feature heatmaps, per-subject violin plots

---

Slide 11 — Action items for a new dataset
- Prepare channels.tsv; verify event structure and stimulus timing
- Run preprocessing; save QC; extract baseline 135 features
- Run LOSO with training-only scaling & selection; run family ablations
- If stable biomarkers appear, retrain compact model for deployment; validate on external dataset if available

---

Slide 12 — Contact / Next steps
- Commit run outputs and config JSON for every experiment
- Add subject-level QC thresholds and automated gating
- Move toward real-time inference test with LDA on streaming ROI signals

---

