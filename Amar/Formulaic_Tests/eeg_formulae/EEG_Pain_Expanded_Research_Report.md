# Expanded EEG-only pain prediction research package
**Version:** 0.2 | **Date:** 2026-10-16  
**Evidence status:** literature synthesis + pre-specified analysis protocol. **No new raw EEG model was fitted in this session.**

## Executive conclusion
EEG contains pain-related information, particularly in stimulus-evoked responses and some multivariate connectivity patterns, but it is not yet a reliable stand-alone, universally calibrated pain meter. A large chronic-pain EEG mega-analysis (614 people, eight datasets) found little evidence that individual resting theta/alpha/beta power or peak alpha frequency reliably tracked pain intensity across discovery and replication samples. A multivariate connectivity model had only modest out-of-sample association (reported predicted-vs-observed pain correlation r=0.06 in independent replication data; combined-data cross-validated r=0.14). A separate external-validation study of experimental pressure pain reported RF classification accuracy 68.32% on a same-parameter external sample and 60.42% when stimulation parameters changed; its 2026 continuous-pain external-validation paper reported that EEG regression did not outperform a mean-rating baseline and binary AUC was 0.62–0.68. These findings argue for strict baselines and external validation.

**Crucial limitation:** although public datasets and papers were located, raw EEG files could not be downloaded in this execution environment (network/DNS unavailable in the data-analysis runtime). Therefore, this package does not claim a new empirical fit, coefficient, p-value, AUC, MAE, or confidence interval for the proposed formulas. The plotted results below are explicitly literature-reported benchmarks, not results generated from raw recordings here.

## 1. Target definition: do not mix prediction tasks
1. **Within-person acute intensity:** predict a participant's trial-wise NRS after a stimulus. Use participant-grouped cross-validation for generalization to new people; within-person calibration is a separate task.
2. **Chronic pain intensity:** predict a person's reported average pain over a defined period. This is a different target from evoked laser/pneumatic trial ratings.
3. **Pain/no-pain or low/high classification:** define thresholds before analysis; report class prevalence and balanced accuracy/AUROC/AUPRC.
4. **Stimulus intensity decoding is not pain decoding.** Laser energy or pressure can correlate with reported pain; include a stimulus-only baseline and test whether EEG adds information beyond stimulus intensity.

## 2. EEG-only feature families and candidate formulas
All formulas are candidates. Direction/sign and weights must be learned only in training folds; no hand-picked signs or universal weights should be treated as validated.

Let \(x_{i,t}\) be the feature vector for person \(i\), trial \(t\), and \(y_{i,t}\) the observed NRS. Feature standardization must be fit inside each training fold. For band powers, use log power and clearly specify absolute/relative power, reference, montage, epoch, artifact rejection, and whether evoked activity is included.

### F1 — single-feature baseline models
Fit separate models for each feature: log alpha power, log theta power, log beta power, peak alpha frequency (PAF), N2-P2 amplitude, N2 latency, P2 amplitude, spectral entropy, aperiodic exponent, and each pre-registered connectivity summary. These are essential baselines because they reveal whether a composite truly adds value.

### F2 — sensory-evoked potential model
\[
\hat y = \beta_0+\beta_1 A_{N2P2}+\beta_2 A_{N2}+\beta_3 A_{P2}+\beta_4 L_{N2}+\beta_5 L_{P2}
\]
Measure amplitudes/latencies in fixed, pre-specified windows and electrodes or use a training-only spatial filter. Do not choose windows on the held-out test data. Report single-trial reliability and sensitivity to artifact rejection.

### F3 — event-related spectral response model
\[
\hat y = \beta_0+\sum_{b\in\{\theta,\alpha,\beta,\gamma\}}\beta_b\Delta\log P_b
\]
where \(\Delta\log P_b=\log P_{post,b}-\log P_{baseline,b}\), with baseline and post-stimulus windows pre-specified. Gamma features require strong EMG and line-noise checks. Separate evoked and induced power where feasible.

### F4 — within-person alpha suppression
\[
S_\alpha=-\log\left(\frac{P_{\alpha,post}+\epsilon}{P_{\alpha,base}+\epsilon}\right)
\]
This is a within-trial reactivity feature, not an absolute pain scale. Test only in designs with comparable pre/post epochs. Use a small pre-specified \(\epsilon\) tied to the signal units; do not tune it on the test set.

### F5 — multiband robust linear model
\[
\hat y=\beta_0+\beta_A z(\log P_\alpha)+\beta_T z(\log P_\theta)+\beta_B z(\log P_\beta)+\beta_G z(\log P_\gamma)+\beta_F z(PAF)
\]
Use ridge/elastic net with nested group CV. The signs are learned, not assumed. Compare to participant-mean and stimulus-only baselines.

### F6 — nonlinear EEG-only model
Random forest / gradient boosting on the same pre-specified EEG features, with all hyperparameter selection inside inner grouped CV. This is a benchmark, not inherently more valid than a linear model. Report feature importance only as exploratory unless stability is tested.

### F7 — connectivity summary model
Use a small, pre-registered set of theta/alpha/beta connectivity measures (e.g., debiased wPLI or imaginary coherence; source-space measures only if the source pipeline is defensible). Avoid raw coherence without addressing volume conduction. Fit regularized models; high-dimensional connectivity requires nested CV and preferably external validation.

### F8 — amplitude-envelope dynamics
Features: envelope mean, variance, autocorrelation time, burst rate, and trial-to-trial variability in pre-specified bands/regions. Normalize only using training data or the participant's baseline if that is part of the intended deployment protocol. Test incremental value beyond mean band power.

### F9 — spectral entropy / permutation entropy
Compute entropy using a fixed estimator, epoch length, sampling rate, and normalization. Entropy is not pain-specific; compare against simple power features and evaluate test-retest reliability.

### F10 — aperiodic 1/f exponent and offset
Fit a documented aperiodic spectral model (with oscillatory peaks separated if possible). Include exponent and offset as separate features. Published evidence includes a 264-person study (149 chronic pain, 115 controls) that found no reliable relationship between prefrontal aperiodic exponent and chronic pain intensity; therefore this is a candidate to falsify, not a presumed biomarker.

### F11 — phase-amplitude coupling (PAC)
Define phase band, amplitude band, region pair, estimator, surrogate procedure, and epoch a priori. Use surrogate-normalized PAC and test whether it adds value beyond power and ERP features. PAC is vulnerable to non-sinusoidal waveforms, artifacts, and multiple testing.

### F12 — temporal/habituation model
\[
\hat y_{i,t}=\beta_0+\beta_1 f_{i,t}+\beta_2 \mathrm{trialIndex}_t+\beta_3(f_{i,t}\times\mathrm{trialIndex}_t)
\]
where \(f\) is an EEG feature (e.g., N2-P2 amplitude or band response). Use mixed-effects models for repeated trials and test habituation slopes without leaking trials across folds. Trial index is design metadata, not EEG; for a strict EEG-only model, remove it from predictors and retain only an EEG-derived slope calculated under a deployment-compatible procedure.

### F13 — multi-view regularized model
Combine ERP, oscillatory, aperiodic, entropy, and connectivity blocks using elastic net or a training-only dimensionality reduction (PLS). Compare every block and the full model using identical outer folds and a pre-specified ablation analysis.

### F14 — ordinal pain-class model
For categories such as low/moderate/high, pre-register clinical/research thresholds based on the rating scale and intended use. Prefer ordinal regression when category ordering matters. Do not select thresholds after seeing model performance. Report macro-F1, balanced accuracy, per-class recall, confusion matrix, AUROC (binary only), calibration, and confidence intervals.

### F15 — personal-baseline change score
\[
I_{i,t}=\sum_k w_k\frac{f_{i,t,k}-\mu_{i,baseline,k}}{\sigma_{i,baseline,k}}
\]
This is a relative change index, not a universal 0–10 scale. It requires an explicit baseline session and a pre-defined rule for near-zero baseline variance. Calibration to NRS requires supervised data and independent validation.

## 3. Validation protocol required before claiming validity
- **Split unit:** participant, never randomly split trials across train/test when claiming generalization to new people.
- **Nested CV:** outer GroupKFold for unbiased evaluation; inner GroupKFold for feature selection/hyperparameter tuning. Fit imputation, scaling, feature selection, dimensionality reduction, and calibration inside the training folds.
- **Baselines:** (a) training-fold mean predictor; (b) EEG single-feature models; (c) stimulus-only model where stimulus intensity is available; (d) EEG + stimulus model to test incremental EEG value.
- **Regression metrics:** MAE, RMSE, \(R^2\), Spearman correlation and calibration slope/intercept. Negative test \(R^2\) is possible and informative.
- **Classification metrics:** balanced accuracy, macro-F1, per-class recall, AUROC/AUPRC, Brier score/calibration, confusion matrix; compare with majority/chance baseline appropriate to class balance.
- **Uncertainty:** participant-cluster bootstrap confidence intervals; permutation test that shuffles labels at participant level or within participant according to the estimand. Do not treat trials as independent observations.
- **Multiplicity:** pre-register a small primary feature set or control FDR across exploratory univariate tests. Report effect sizes and CIs, not only p-values.
- **Generalization:** leave-one-dataset/study-out validation or a locked external dataset. Report acute experimental and chronic clinical results separately.
- **Subgroups:** report performance by dataset/site and, where sample size supports it, age/sex/gender and pain diagnosis. Do not use these as predictors unless justified; do not claim subgroup validity from tiny groups.
- **Leakage checks:** verify that participant identity, repeated epochs, adjacent trials, stimulus condition labels, and preprocessing decisions cannot leak across the split.
- **Acceptance criteria:** declare in advance what MAE improvement over mean baseline, balanced accuracy/AUROC, calibration, and external replication would be useful. Statistical significance alone is insufficient.

## 4. Real datasets and evidence sources
- **Large acute laser-evoked EEG resource:** 678 healthy participants, high-density EEG, single-trial subjective pain ratings across nine OpenNeuro experiments. Data descriptor: Zhao et al., *Scientific Data* (2025), https://www.nature.com/articles/s41597-025-05900-1. Experiments include ds005284, ds005285, ds005289, ds005286, ds005291, ds005293, ds005292, ds005280, ds005473. Check the specific dataset's events and rating availability before assuming trial-wise labels are present.
- **ds005293 / NEMAR on005293:** four calibrated pain levels (NRS 2, 4, 6, 8), 80 trials/participant; https://nemar.org/dataset/on005293 and https://github.com/OpenNeuroDatasets/ds005293.
- **ds005285 / NEMAR on005285:** 160 laser trials/participant, low/high stimulation, ratings on 0–10 scale; https://nemar.org/dataset/on005285.
- **OSF uqt9z:** independent experiments for external validation of EEG low/high experimental-pain classification; https://osf.io/uqt9z/. Mari et al. (2023), https://doi.org/10.1038/s41598-022-27298-1.
- **Chronic pain connectivity mega-analysis:** 614 participants from eight datasets; published report and results: https://pmc.ncbi.nlm.nih.gov/articles/PMC12517086/ .
- **Aperiodic component study:** 149 chronic-pain participants and 115 controls; no reliable pain-intensity relationship with prefrontal aperiodic exponent; https://elifesciences.org/articles/101727.
- **External continuous prediction study (2026):** 91 participants across validation samples; regression models did not outperform mean baseline, binary classification AUC 0.62–0.68; https://www.sciencedirect.com/science/article/pii/S1526590026002348.

## 5. Published-result visualizations
The attached figures summarize values reported by cited papers. They are not generated from raw EEG in this session and must not be interpreted as performance of F1–F15.

## 6. What has and has not been tested here
**Completed:** literature review; candidate formula specification; pre-specified validation design; scripts/workbook/report; visualizations of published benchmark results.  
**Not completed:** downloading raw recordings, extracting a new EEG feature matrix, fitting F1–F15 to observed participant-level data, running new permutation tests or confidence intervals, or externally validating newly fitted coefficients. The data-analysis runtime has no network/DNS access, so the raw data could not be fetched here. The attached Python program is designed to run on a genuine derived feature table; it will refuse to run without the required participant, rating, and EEG feature columns. No synthetic EEG or ratings are included.

## References
1. Zhao et al. (2025). *A comprehensive EEG dataset of laser-evoked potentials for pain research*. Scientific Data. https://doi.org/10.1038/s41597-025-05900-1
2. *Exploring electroencephalographic chronic pain biomarkers: a mega-analysis* (2025). eBioMedicine. https://doi.org/10.1016/j.ebiom.2025.105955
3. Mari et al. (2023). *External validation of binary machine learning models for pain intensity perception classification from EEG in healthy individuals*. Scientific Reports. https://doi.org/10.1038/s41598-022-27298-1
4. *External validation of EEG-based machine learning models for continuous pain prediction* (2026). Journal of Pain. https://doi.org/10.1016/j.jpain.2026.106416
5. *Assessing the balance between excitation and inhibition in chronic pain through the aperiodic component of EEG* (2025). eLife. https://doi.org/10.7554/eLife.101727
6. Huang et al. (2018). *Magnitude and Temporal Variability of Inter-stimulus EEG Modulate the Linear Relationship Between Laser-Evoked Potentials and Fast-Pain Perception*. Frontiers in Neuroscience. https://doi.org/10.3389/fnins.2018.00340
