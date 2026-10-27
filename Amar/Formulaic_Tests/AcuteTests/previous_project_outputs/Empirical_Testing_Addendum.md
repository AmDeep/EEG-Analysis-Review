# EEG-only pain-scale empirical testing addendum

Date: 2026-10-26

## Status and evidence boundary

This addendum distinguishes (A) published study facts, (B) tests run on software only, and (C) tests that require raw/trial-level EEG data. An attempted direct fetch from the public data/code host failed in this runtime because DNS/network access is unavailable (`Temporary failure in name resolution`). Therefore this run did not access subject-level signals, trial-level ratings, or the authors' MAT/EEGLAB files. No patient/participant records were fabricated and no subject-level performance is reported.

## Public acute experimental pain datasets verified from source pages

| Dataset | Published/source-reported design | What it can test | Access / caveat |
|---|---|---|---|
| ds005293 / NEMAR on005293, “95 By BP” | 95 participants; 60 EEG channels; 80 trials per participant; 4 calibrated pain target levels (2, 4, 6, 8 NRS) crossed with left/right hand; 10 trials per condition; source reports average laser energies E1 2.0±0.2 J, E2 2.7±0.3 J, E3 3.4±0.3 J, E4 4.1±0.4 J | Within-person ordinal trend, held-out-subject generalization, laterality, calibrated target-level robustness | 300 GB source dataset; full download and preprocessing not performed in this runtime |
| ds005291 / NEMAR on005291, “65 By ANT” | 65 participants; 32-channel EEG; 30 laser stimuli per participant after calibration toward mean rating 7; ratings 0–10 | Trial-level continuous/ordinal regression, calibration, within-person variability | Published archive listed as 38.1 GB; raw trial-level fit not run |
| ds005285 / NEMAR on005285, “29 By ANT” | 29 participants; 32-channel EEG; 160 trials per participant, half low and half high intensity; ratings 0–10 | Binary/ordinal contrast, within-person time drift, cross-paradigm robustness | Published archive listed as 23.5 GB; raw trial-level fit not run |
| ds005284 / NEMAR on005284, “26 By Biosemi” | 26 participants; 64-channel EEG; calibrated laser pain paradigm; study-specific fixed-intensity design | Sensor montage sensitivity and independent paradigm replication | Raw trial-level fit not run |
| Zhao et al. 2025 multi-experiment LEP collection | 678 healthy participants across nine experiments; single-trial pain ratings and high-density EEG; stimulus intensities 2.5–4.5 J; authors report average LEP responses and pain ratings trend upward with stimulus intensity/rating | Large multi-experiment replication, ERP features, across-experiment holdout | Paper reports a relationship; this is not independent validation of our proposed score formula |

## Formula families to test (all inputs are EEG-derived; ratings are labels for training/evaluation only)

Let `x_t` be the feature vector extracted from a prespecified EEG epoch, `z_j` a robust within-recording standardized feature, `y_t` the observed NRS label used only as target, and `clip10(a)=min(10,max(0,a))`.

1. **F1: robust alpha-only sigmoid**: `S_t = 10 / (1 + exp(-(b0 + b1*z_alpha)))`. Tests whether a single EEG feature provides any out-of-subject signal. Coefficients must be fitted in training folds only.
2. **F2: regularized spectral linear**: `S_t=clip10(b0 + sum_j b_j*z_j)`, with alpha/beta/theta/gamma/entropy/peak-alpha features. Ridge/elastic-net penalties selected only inside nested training folds.
3. **F3: ordinal cumulative-link**: `P(Y>=k|x)=sigmoid(a_k + bᵀx)`, monotone thresholds; expected NRS score `S=sum_{k=1}^{10}P(Y>=k|x)`. Appropriate for discrete NRS labels; threshold ordering must be enforced.
4. **F4: monotone GAM**: `S=clip10(b0 + Σ f_j(z_j))`; smooth terms can expose nonlinearity without unrestricted tree complexity. Smoothing is selected within training folds.
5. **F5: interaction model**: `S=clip10(b0 + bᵀx + Σ_{j<k} c_jk*z_j*z_k)`, limited to prespecified EEG interactions such as alpha×theta and ERP amplitude×latency; strong regularization required.
6. **F6: event-related potential (ERP) model**: features from fixed, preregistered windows/channels around N2/P2 (amplitude, latency, area); ridge/ordinal fit. Windows/channels cannot be selected using held-out subjects.
7. **F7: spatial EEG model**: regularized channel/time features or covariance/tangent-space representation; evaluate only with nested CV and leakage-safe scaling. More data-hungry and higher overfit risk.
8. **F8: non-linear tree baseline**: random forest / gradient boosting on fixed EEG features. Useful comparator, not inherently physiologically interpretable; tune only inside training folds.
9. **F9: within-person calibrated score**: `S_it=clip10(g_i(x_it))`, where `g_i` is calibrated from that subject's calibration trials. Must report separately from zero-shot subject-held-out performance; never let test trials leak into calibration.
10. **F10: temporal state score**: `S_t=λS_(t-1)+(1-λ)raw_t`. This is post-processing, not evidence that EEG predicts pain. Compare against raw scores and report lag/oversmoothing; tune λ on training participants only.
11. **F11: robust quantile/rank mapping**: map a training-set EEG score monotonically to NRS quantiles; fit mapping on training participants only. Useful when absolute calibration varies, but may discard absolute scale.
12. **F12: nuisance/artifact-aware model**: EEG predictors plus explicit quality flags and EMG/EOG contamination indicators as *quality-control variables*, with separate EEG-only predictor ablation. Do not allow artifact features to masquerade as neural pain signal; report performance with contaminated epochs removed and sensitivity analysis.
13. **F13: subject-normalized mixed-effects model**: fixed EEG effects plus participant random intercept/slope for within-person association. Valid for estimating population/within-person relationships, but a random effect for a held-out subject is unavailable without calibration; do not claim zero-shot prediction from it.
14. **F14: latent EEG composite**: prespecified PCA/PLS component(s) fit inside each training fold, then regression/ordinal model. Component extraction must never see held-out rows; compare to direct feature models.

## Statistical tests and reporting plan

- **Primary predictive comparison:** participant-grouped outer CV, all trials from a participant held out together. Report MAE, RMSE, bias, out-of-sample R², Spearman rho, and calibration slope/intercept. Compare each candidate with a training-fold mean/median baseline.
- **Uncertainty:** cluster bootstrap by participant (not by trial) for metric confidence intervals and paired differences. If comparing many models/features, adjust confirmatory p-values (e.g. Holm); mark broad feature search exploratory.
- **Within-person trend:** mixed-effects regression or participant-level slope of EEG feature/score versus NRS; report confidence interval and heterogeneity, not pooled trial-level p-values as if independent.
- **Ordinal validity:** ordinal log loss/Brier score, ranked probability score, confusion by NRS band, and calibration of cumulative probabilities; compare with ordinal prevalence baseline.
- **Permutation:** permute ratings within participant only when trial exchangeability is justified; for temporal tasks use block/circular-shift permutations to preserve autocorrelation. Re-run the entire training pipeline for each permutation. Report empirical p-value `(1 + count(null >= observed))/(1 + B)`.
- **Temporal generalization:** chronological train-early/test-late within participants, with a separate zero-shot subject-held-out analysis. Report performance degradation by trial order and temporal autocorrelation; do not randomly split adjacent trials.
- **Cross-dataset generalization:** train on one dataset and test on another only after matching target definitions and feature extraction; harmonize channels/windows without using target labels from test data. Report domain shift and no recalibration separately from recalibration.
- **Robustness:** EEG feature ablations; N2/P2 vs spectral-only; left/right hand; montage; line-noise/artifact rejection thresholds; gamma with EMG-sensitive epochs excluded; alternate baseline intervals; NRS target scaling; influential-participant analysis.
- **Negative controls:** shuffled labels, pre-stimulus EEG-only predictors, non-pain/non-nociceptive event windows where available, and stimulus intensity as a comparator rather than a predictor in an EEG-only model. A valid EEG-only formula must use EEG-derived predictors at inference.
- **Subgroup/fairness:** only if source metadata and sample sizes support it; report uncertainty and avoid interpreting underpowered nulls as equivalence.

## Results actually available at this point

- **Published facts:** dataset sizes/designs above and the source paper's qualitative technical-validation finding that average LEP responses increase with stimulus intensity and pain rating.
- **Software tests:** score clipping to [0,10], sigmoid output bounds, EWMA recurrence, zero-MAD baseline behavior, and rejection of invalid smoothing parameter passed in the existing code's unit tests.
- **Not run:** raw EEG extraction; coefficients; participant-level CV; confidence intervals; p-values; permutation tests; temporal tests; cross-dataset tests; subgroup analyses; formula ranking. These require downloaded EEG/ratings or the preprocessed trial-level files.

## Reproducibility checklist before reporting a winning formula

1. Pin each dataset DOI/version/license and record exact file hashes.
2. Read BIDS event definitions; verify rating field, stimulus onset, sampling rate, channel units/reference, and trial exclusions.
3. Prespecify preprocessing and epochs (including baseline), bad-channel/epoch rejection, ocular/muscle artifact handling, and feature windows.
4. Save one row per trial with subject/session/trial order, rating label, quality flags, and EEG-only feature columns; keep raw-to-feature provenance.
5. Freeze subject-level folds before feature selection. Fit imputation, scaling, dimensionality reduction, feature selection, calibration and tuning within training folds only.
6. Run model bank and primary/secondary tests. Preserve predictions for every held-out trial, not just aggregate metrics.
7. Save software versions, random seeds, dataset hashes, exclusion log, fold assignments, per-subject metrics, bootstrap draws/seed, permutation distribution, and all warnings.
8. Treat the result as exploratory until independently replicated; these datasets are healthy volunteers undergoing experimental acute pain, not necessarily clinical patients with spontaneous acute pain.
