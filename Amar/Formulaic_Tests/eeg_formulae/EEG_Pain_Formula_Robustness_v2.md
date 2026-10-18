# EEG-only pain formula robustness and scale-validity plan
**Version 2 — 16 October 2026**

## Executive conclusion

The strongest current conclusion is not that a universal EEG pain scale has been established. Published work supports a small number of interpretable, protocol-specific relationships between stimulus-evoked EEG and acute pain reports. It does **not** establish that one EEG equation can place people with different pain profiles on a common, calibrated 0–10 scale.

This package expands the candidate equations and implements a stricter empirical test harness. It does not report new raw-EEG estimates because raw recordings could not be retrieved into the analysis runtime in this session (network name resolution failed). No new coefficients, p-values, confidence intervals, or model performance claims are fabricated. The numerical outcomes below are published study results, not results newly calculated by this package.

## What “on their own scales” must mean mathematically

There are three distinct estimands:

1. **Within-person tracking:** does an EEG score rise when this person reports more pain than their own usual level? A person-specific baseline and/or calibration set can be useful here.
2. **Between-person calibration:** if two people have the same reported pain rating, does the formula give comparable scores? This requires calibration that transfers across people; within-person z-scores alone do not do this.
3. **Clinical chronic-pain severity:** does resting EEG estimate a person's usual pain intensity over days or weeks? This is a different target from single-trial evoked pain and cannot be validated by mixing datasets/labels.

A within-person standardized score such as
\[
z_{itk}=\frac{f_{itk}-\mu_{i,\mathrm{baseline},k}}{\sigma_{i,\mathrm{baseline},k}}
\]
is dimensionless and relative to that person's baseline. It is not automatically a pain rating. It cannot be called “6/10” without fitting and independently validating a calibration map to self-report.

A population-standardized score,
\[
z_{itk}^{\mathrm{pop}}=\frac{f_{itk}-\mu_{\mathrm{train},k}}{\sigma_{\mathrm{train},k}},
\]
supports a common feature coordinate system, but still does not make the output a pain scale until the equation is calibrated to observed ratings. Population normalization parameters must be learned from training people only.

## Expanded interpretable formula families

All features below must be computed from genuine EEG, using documented electrodes, reference, filter, epoch, artifact handling and units. Signs/weights are unknown until fitted in training data; do not assume every feature increases with pain.

### A. Single-feature benchmarks

**A1. Evoked RMS**
\[
RMS_i=\sqrt{K^{-1}\sum_{k=1}^{K}x_{ik}^2}
\]
A simple stimulus-locked amplitude benchmark.

**A2. Spontaneous-baseline normalized evoked RMS**
\[
nRMS_i=\frac{RMS_{i,\mathrm{evoked}}-\mu_{i,\mathrm{spont}}}
{\sigma_{i,\mathrm{spont}}}
\]
The baseline mean and SD must come from an independently defined spontaneous/baseline interval. Do not estimate them using the held-out participant's pain labels. If the dataset does not contain an appropriate baseline SD, do not manufacture this feature.

**A3. N2–P2 amplitude**
\[
A_{N2P2}=V_{P2}-V_{N2}
\]
Fix the electrode/reference and latency windows before testing. A peak-to-peak amplitude is not interchangeable with N2 or P2 alone.

**A4. Alpha suppression / reactivity**
\[
S_\alpha=-\log\frac{P_{\alpha,\mathrm{post}}+\epsilon}
{P_{\alpha,\mathrm{base}}+\epsilon}
\]
This is an event-related reactivity feature. It is not a universal resting-state pain marker.

### B. Spectral changes and multiband formula

For band \(b\in\{\theta,\alpha,\beta,\gamma\}\):
\[
\Delta\log P_b=\log(P_{b,\mathrm{post}}+\epsilon)
-\log(P_{b,\mathrm{base}}+\epsilon).
\]
An interpretable multiband equation is
\[
\widehat{NRS}=\beta_0+\beta_\theta\Delta\log P_\theta+
\beta_\alpha\Delta\log P_\alpha+
\beta_\beta\Delta\log P_\beta+
\beta_\gamma\Delta\log P_\gamma+\beta_f PAF.
\]
Use shrinkage or keep predictors few when observations are limited. Gamma requires explicit muscle/EMG artifact checks. PAF is undefined when no reliable alpha peak exists; missingness must be documented, not silently replaced by a typical value.

### C. Complexity and aperiodic features

\[
\widehat{NRS}=\beta_0+\beta_E H_{\mathrm{spectral}}+
\beta_P H_{\mathrm{permutation}}+\beta_\chi\chi+
\beta_o O
\]
where \(H\) denotes the specified entropy estimator, \(\chi\) the aperiodic exponent and \(O\) the aperiodic offset. Fix entropy estimator, frequency range, epoch length and aperiodic-fitting settings in advance. These are candidate features, not established pain-specific quantities.

### D. Connectivity and phase-amplitude coupling

\[
\widehat{NRS}=\beta_0+\beta_\theta C_\theta+
\beta_\alpha C_\alpha+\beta_\beta C_\beta+\beta_{PAC}PAC_{\alpha\to\gamma}
\]
\(C_b\) must name one particular connectivity measure (for example, debiased wPLI) and fixed region pair/network summary. Do not mix connectivity metrics under one label. Address reference/volume-conduction effects, and use surrogate-based PAC significance or bias correction. PAC and connectivity require multiplicity correction if many pairs/bands are screened.

### E. Temporal response and habituation

\[
A_{it}=\gamma_{0i}+\gamma_{1i}\,\mathrm{trialOrder}_{it}+e_{it}
\]
The slope \(\gamma_{1i}\) summarizes habituation/sensitization of an EEG response across repeated trials. It is only defined for a sequence of ordered trials and should be estimated without allowing adjacent trials from the same person to leak across training and test. It may reflect fatigue, attention or adaptation as well as pain.

### F. Microstates and alpha-envelope dynamics

Candidate predictors include class-specific microstate duration/coverage/transition entropy, alpha-envelope autocorrelation time, burst rate and envelope variability. The feature extraction definitions must be fixed. The source drawing's references to altered microstate C/D duration and alpha–gamma coupling are treated as hypotheses to test, not assumed directions.

### G. Broad, interpretable linear score

A broad equation can combine selected, pre-registered features:
\[
\widehat{NRS}=\beta_0+\sum_{k=1}^{p}\beta_k z_k.
\]
For a small sample, use a limited feature set or ridge shrinkage with tuning inside nested participant-grouped folds. It remains a linear statistical formula, not a tree/ensemble/neural-network predictor. A sigmoid or min-max transform alone does not calibrate the score to NRS; calibration must be fitted to observed ratings and evaluated on held-out people.

### H. Person-relative index

\[
I_{it}=\sum_k w_k\frac{f_{itk}-\mu_{i,\mathrm{baseline},k}}
{\sigma_{i,\mathrm{baseline},k}}.
\]
This index answers “how unusual is the current EEG relative to this person's baseline?” The weights must be learned from actual labeled data, and it is only an NRS estimator if separately calibrated. Without such calibration, label it a relative EEG reactivity index.

## Statistical evidence from actual published data

### Bai et al. (2016), acute laser pain
Bai et al. compared evoked RMS with spontaneous-EEG-normalized RMS using leave-one-individual-out validation. Published results:
- MAE on predicted high-pain trials: 1.838 to 1.625 NRS points, paired \(p=0.002\).
- MAE on true high-pain trials: 1.235 to 1.173 points, paired \(p=0.003\).
- Low/high classification accuracy: 68.95% to 70.36%, \(p=0.092\), not statistically significant at 0.05.
These support baseline normalization as a plausible improvement in that specific experimental setting, but not a universal EEG pain scale. Absolute MAE improvement was modest and must be weighed against uncertainty and generalization.

Reference: Bai et al., “Normalization of Pain-Evoked Neural Responses Using Spontaneous EEG Improves the Performance of EEG-Based Cross-Individual Pain Prediction,” *Frontiers in Computational Neuroscience* (2016). https://doi.org/10.3389/fncom.2016.00031

### Huang et al. (2018), inter-stimulus EEG and individual variation
The study supports accounting for individual differences in inter-stimulus EEG when estimating the relationship between laser-evoked potentials and fast-pain ratings. It supports personalized/calibrated linear relationships rather than assuming one slope/intercept is universal.
Reference: https://doi.org/10.3389/fnins.2018.00340

### Bott et al. (2025), chronic pain mega-analysis
A mega-analysis of 614 chronic-pain participants across eight datasets found that conventional resting-state band power and peak alpha frequency had little replicable relationship with chronic pain intensity. Theta-band large-scale connectivity showed associations in pooled analyses, but discovery findings did not consistently generalize to replication datasets. This argues against treating resting alpha/theta/beta power or PAF as a standalone universal chronic-pain meter.
Reference: https://doi.org/10.1016/j.ebiom.2025.105955

### Large public acute-pain dataset
Zhao et al. describe 678 healthy participants across nine experiments, with high-density EEG and single-trial self-reported ratings following laser stimuli. Preprocessed data and MATLAB code are linked by the authors. This is a high-value benchmark for genuine formula testing, but it is still acute experimental pain in healthy participants—not chronic clinical pain.
Reference/data: https://www.nature.com/articles/s41597-025-05900-1 and https://github.com/XiangyueZhao/Supplement-Data

## Thorough test protocol implemented in `eeg_pain_formula_benchmark_v2.py`

The script accepts a CSV of observed EEG-derived features and pain labels and reports:

- Leave-one-participant-out predictions, keeping all trials of each held-out person out of fitting.
- Training-only imputation and scaling.
- Single-feature, ERP, event-related spectral, multiband, complexity/aperiodic, connectivity, PAC, microstate, temporal/habituation and broad linear feature families when columns exist.
- Comparison with a training-fold mean baseline.
- MAE, RMSE, test \(R^2\), Pearson/Spearman correlation, calibration slope/intercept and prediction range.
- Cluster-bootstrap 95% intervals for MAE, resampling people rather than treating trials as independent.
- Per-person audit: observed vs predicted mean and range, bias, SD compression, MAE/RMSE, within-person correlation and calibration slope.
- Within-person feature correlations with Fisher-z aggregation and Benjamini–Hochberg correction.
- Out-of-person prediction tables, fold coefficients, metrics and observed-vs-predicted plots.

Example:
```bash
python eeg_pain_formula_benchmark_v2.py --input observed_eeg_features.csv --out results
```

Required columns: `participant_id`, `pain_nrs`. Optional features include `evoked_rms`, `spontaneous_rms`, `spontaneous_rms_sd`, `n2p2_amp`, `n2_latency`, `p2_latency`, `alpha_base`, `alpha_post`, `theta_base`, `theta_post`, `beta_base`, `beta_post`, `gamma_base`, `gamma_post`, `paf`, `spectral_entropy`, `permutation_entropy`, `aperiodic_exponent`, `aperiodic_offset`, `theta_connectivity`, `alpha_connectivity`, `beta_connectivity`, `alpha_gamma_pac`, microstate variables, `erp_trial_variability`, `habituation_slope`, and alpha-envelope features.

## Does the current evidence show that people are correctly placed on their own scales?

**Not yet, in a generalizable sense.** The published acute-pain study shows modest improvement in NRS prediction error after spontaneous EEG normalization, but not a significant binary accuracy improvement. The chronic-pain mega-analysis undermines the idea that simple resting power/PAF universally orders people by chronic pain intensity. The per-person scale audit in the new script is designed to expose a common failure: good pooled correlation but compressed predictions, systematic bias, or poor calibration within some people.

Interpretation criteria:
- A model that predicts each held-out participant's average rating but fails to track that person's trial-to-trial variation is not a successful within-person pain tracker.
- A formula whose prediction SD is much smaller than the observed SD is over-shrunk and cannot span that person's scale.
- A large within-person correlation does not establish correct absolute calibration.
- A personalized scale that uses a person's pain ratings to calibrate the equation cannot claim zero-calibration performance on that same person; calibration and evaluation trials must be separated.
- If predictions fail to beat the training-mean baseline or stimulus-only baseline, EEG has not demonstrated incremental measurement value.
- Between-person pain severity and within-person evoked pain must be reported separately.

## Data access and limits of this run

I tried to retrieve actual dataset files, but this runtime could not resolve external hosts. The raw EEG therefore was not processed in this session. I have not run the new script on raw recordings, and there are no newly estimated coefficients or new p-values in this report. The included script is ready to run once a genuine feature table is available; raw EEG requires a documented preprocessing/feature-extraction pipeline first.

The public datasets are not interchangeable. Acute laser-evoked pain, chronic clinical pain, resting EEG, and phantom-limb pain must not be pooled as if they measured the same target. A rigorous follow-up should start with one acute-pain dataset, lock the feature definitions and primary formula, then test transport to an independent dataset. Chronic-pain scale validity requires its own clinical dataset and target definition.

## References

1. Bai et al. 2016. https://doi.org/10.3389/fncom.2016.00031
2. Huang et al. 2018. https://doi.org/10.3389/fnins.2018.00340
3. Bott et al. 2025. https://doi.org/10.1016/j.ebiom.2025.105955
4. Zhao et al. 2025. https://doi.org/10.1038/s41597-025-05900-1
5. Public dataset: https://github.com/XiangyueZhao/Supplement-Data
