# EEG pain measurement by explicit formulae — revised analysis
**Version 1.0 | 2026-10-16**  
**Scope:** mathematically interpretable EEG formulas and statistical evidence. No random forest, boosting, neural network, or synthetic observations are used.

## Executive finding
The strongest direct formula-based published evidence located is Bai et al. (2016), which tested equations linking single-trial pain-evoked EEG magnitude to self-reported pain. They compared raw response RMS with spontaneous-EEG-normalized RMS using leave-one-individual-out validation. Normalization improved reported continuous prediction MAE (1.838 to 1.625 NRS points on predicted high-pain trials, p=.002; 1.235 to 1.173 on true high-pain trials, p=.003). The low/high classification accuracy change, 68.95% to 70.36%, was not statistically significant (p=.092). The paper also reports that a piecewise relationship with a breakpoint at NRS=4 fit better than one global linear relation, consistent with weak EEG-rating association at low pain and a stronger association for fast/high pain.
These are **published analyses of real data**, not a fresh reanalysis of the raw EEG by this package. Raw recordings could not be downloaded and independently reprocessed in this execution environment, so I will not claim new coefficients, p-values, confidence intervals, or test-set performance.

A second important finding is the 2025 chronic-pain EEG mega-analysis of 614 participants across eight datasets: conventional resting-state band power and peak alpha frequency showed little replicable association with pain intensity. Large-scale connectivity, especially theta-band network connectivity involving limbic networks, showed stronger associations in the pooled analysis, but discovery effects did not consistently replicate. This suggests that a single universal resting alpha/theta/beta-power equation is not presently supported.

## 1. What counts as a valid formula?
A pain formula should meet all of the following:
1. Its EEG feature is defined reproducibly (electrode/reference, epoch, sampling rate, artifact rejection, frequency limits, units).
2. Coefficients are estimated only from training data or are independently published.
3. Its held-out predictions beat a no-information baseline.
4. Error is measured in rating units (MAE/RMSE), not only correlation.
5. Uncertainty is estimated at participant level, not by treating trials as independent.
6. It is validated on independent people and ideally an independent dataset/site.
7. The intended target is explicit: within-person acute pain, new-person acute pain, chronic pain intensity, or pain-category discrimination. These are not interchangeable.

## 2. Explicit EEG formula families

### F1 — Evoked response RMS
For samples \(x_k\) in a pre-specified post-stimulus epoch:
\[
RMS_P(i)=\sqrt{\frac{1}{K}\sum_{k=1}^{K}x_{i,k}^2}
\]
This is an amplitude measure, not pain itself. In the published laser-pain work, the EEG epoch was Cz, band-pass filtered 1–30 Hz, and the post-stimulus interval was 0–0.5 s. These choices must not be assumed optimal for other datasets.

### F2 — Spontaneous-EEG-normalized evoked RMS
\[
nRMS_P(i)=\frac{RMS_P(i)-\mu(RMS_S)}{\sigma(RMS_S)}
\]
where \(\mu(RMS_S)\) and \(\sigma(RMS_S)\) are estimated from that person's pre-stimulus spontaneous EEG. This is a normalization, not a complete pain equation. The rating equation is:
\[
\widehat{NRS}_i=a\,nRMS_P(i)+c
\]
The intercept \(c\) and slope \(a\) must be trained on labeled participants and tested on held-out participants. A personal baseline is available only if the deployment protocol includes an appropriate baseline recording.

### F3 — Raw-RMS linear calibration
\[
\widehat{NRS}_i=a\,RMS_P(i)+c
\]
Simple and interpretable. It is the essential comparison to normalized RMS. Fit coefficients on training people only. Check residual nonlinearity, heteroscedasticity, calibration slope, and whether predictions extrapolate outside the rating range.

### F4 — Piecewise linear calibration
For a pre-specified knot \(\tau\):
\[
\widehat{NRS}=\beta_0+\beta_1x+\beta_2\max(0,x-\tau)
\]
or, when fitting rating as the predictor in explanatory analysis, use separate slopes below and above a pre-specified pain threshold. In Bai et al., the conceptual knot was NRS=4: the paper reported little relation between RMS and ratings at NRS≤4 and a stronger linear relation above 4. For prediction, a threshold based on the unknown rating cannot be used directly at inference time; instead, use a pre-registered EEG-derived threshold/classification step or fit a continuous spline that uses only EEG at prediction time.

### F5 — N2–P2 evoked-potential amplitude
\[
A_{N2P2}=V_{P2}-V_{N2}
\]
Measure N2 and P2 in fixed time windows and electrode(s), such as Cz, with the same reference and preprocessing across participants. Then test:
\[
\widehat{NRS}=a\,A_{N2P2}+c
\]
This is a candidate formula. N2/P2 responses are also affected by salience, attention, expectation, stimulus properties, and repetition. The rating association should be tested after controlling for stimulus energy and trial order.

### F6 — Baseline-relative alpha suppression
\[
S_\alpha=-\log\frac{P_{\alpha,post}+\epsilon}{P_{\alpha,base}+\epsilon}
\]
Here \(P_\alpha\) is alpha-band power in fixed baseline/post-stimulus windows. This describes change, not pain. The sign convention must be documented. Test it against RMS and N2–P2 using the same held-out folds.

### F7 — Band-power change
\[
\Delta P_b=\log(P_{b,post}+\epsilon)-\log(P_{b,base}+\epsilon), \quad b\in\{\theta,\alpha,\beta,\gamma\}
\]
Fit separate single-feature equations first, then a small pre-specified linear combination. Gamma requires stringent muscle-artifact controls. Relative and absolute power are different quantities and should not be silently substituted.

### F8 — Connectivity-based equation
\[
\widehat{NRS}=\beta_0+\sum_{b\in\{\theta,\alpha,\beta\}}\sum_{r<s}\beta_{b,r,s}\,C_{b,r,s}
\]
\(C\) must be a defined connectivity metric (e.g., debiased wPLI) and defined regions. Avoid raw scalp coherence without addressing volume conduction. A large connectivity equation may be statistically unstable; use a pre-specified small set, or report a regularized linear equation with all coefficients and a locked external test. The 2025 mega-analysis supports further study of theta connectivity, not a universal fixed coefficient set.

### F9 — Spectral complexity and aperiodic background
\[
\widehat{NRS}=\beta_0+\beta_H H_{spectral}+\beta_E E_{aperiodic}+\beta_F PAF
\]
Entropy estimator, spectral range, epoch length, and aperiodic fitting method must be fixed in advance. This is a candidate, not an established biomarker. Include a simple power/ERP baseline and test whether these features add out-of-sample value.

### F10 — Personal baseline deviation index
\[
I_i=\sum_k w_k\frac{f_{i,k}-\mu_{i,base,k}}{\sigma_{i,base,k}}
\]
This yields a relative EEG change score. It is not a universal NRS scale until the weights are calibrated against ratings and independently validated. Handle near-zero baseline SD and session drift explicitly.

## 3. Actual published statistical tests

### Bai et al. (2016): direct comparison of formulae
The study used >100 laser pulses over a wide energy range and single-trial pain ratings. It evaluated equations based on raw evoked RMS and spontaneous-EEG-normalized RMS, with leave-one-individual-out validation.

| Outcome | Raw RMS | Normalized RMS | Reported p-value |
|---|---:|---:|---:|
| Low/high classification accuracy | 68.95% ± 12.91 | 70.36% ± 14.18 | 0.092 |
| MAE, predicted high-pain trials | 1.838 ± 0.602 | 1.625 ± 0.446 | 0.002 |
| MAE, true high-pain trials | 1.235 ± 0.278 | 1.173 ± 0.278 | 0.003 |

Interpretation:
- The binary accuracy improvement was **not statistically significant at 0.05**.
- The reported MAE reductions were statistically significant in that study, but the absolute reductions were 0.213 and 0.062 NRS points respectively. A small p-value does not itself establish clinical usefulness.
- The results are conditional on the paper's paradigm, preprocessing, and validation protocol. Independent reproduction on a different dataset remains necessary.

Source: Bai et al., 2016, *Frontiers in Computational Neuroscience*, https://doi.org/10.3389/fncom.2016.00031

### Huang et al. (2018): response variability and individualized linear equations
This study examined the relation between laser-evoked potential magnitude and fast-pain ratings (NRS>4). It used a linear relation \(y_i=a x_i+b\), then improved cross-person prediction by weighting predictions from training participants with similar inter-stimulus EEG correlates. This is a mathematically interpretable individualized equation, not a universal fixed pain scale. It supports testing personal baseline and signal variability as calibration information.

Source: Huang et al., 2018, https://doi.org/10.3389/fnins.2018.00340

### Bott et al. (2025): chronic-pain resting EEG
The 614-person, eight-dataset mega-analysis found limited replicability for standard band-power and peak-alpha-frequency associations with chronic pain intensity. Large-scale connectivity patterns, especially theta connectivity involving limbic networks, had stronger pooled associations, but discovery/replication differences remained important. This cautions against assuming that an absolute resting-state power formula can provide a reliable chronic-pain intensity score.

Source: Bott et al., 2025, https://doi.org/10.1016/j.ebiom.2025.105955

## 4. Statistical protocol for a fresh raw-data test
1. **Pre-register one primary formula** and a small number of secondary formulas; otherwise, many feature tests inflate false-positive risk.
2. **Keep participants, not trials, as the validation unit.** Use leave-one-participant-out or grouped K-fold validation.
3. **Within-person analysis:** estimate each person's slope/correlation, then use a participant-level mixed model or Fisher-z summary. Do not pool all trials as independent.
4. **Regression metrics:** MAE, RMSE, test \(R^2\), calibration slope/intercept, and rank correlation. Include a training-mean baseline.
5. **Classification metrics:** if categories are required, pre-register thresholds and report balanced accuracy, sensitivity/specificity, confusion matrix, and AUC with participant-aware uncertainty.
6. **Inference:** use participant-cluster bootstrap intervals and a permutation test respecting trial order/blocks. Correct for multiplicity across secondary formulas.
7. **External validation:** lock the equation and evaluate it on a different dataset/site with no re-fitting. Report results separately for acute experimental and chronic clinical pain.
8. **Confound control:** compare EEG with stimulus-only prediction, because stimulus intensity may drive both EEG response and ratings. Check movement/EMG, reference, habituation, expectation, medication and recording differences.

## 5. Reproducibility status
This package contains formula definitions, published real-data results, a workbook, a visualization of the published RMS-normalization comparison, and a Python script for testing formulas on a genuine feature table. It does **not** claim an independent raw-EEG reanalysis: raw EEG files could not be downloaded in this execution environment. No synthetic EEG or pain ratings were generated.

The script `eeg_pain_formula_tests.py` fits only interpretable equations (single-feature and ordinary least-squares linear models), with leave-one-participant-out predictions and participant-level association summaries. It expects real data with `participant_id`, `pain_nrs`, and available EEG feature columns. It should be treated as an analysis starter; for confirmatory work, add design-specific permutation tests, multiplicity correction, and external dataset validation.

## References
1. Bai Y, Huang G, Tu Y, Tan A, Hung YS, Zhang Z (2016). Normalization of Pain-Evoked Neural Responses Using Spontaneous EEG Improves the Performance of EEG-Based Cross-Individual Pain Prediction. https://doi.org/10.3389/fncom.2016.00031
2. Huang G, et al. (2018). Magnitude and Temporal Variability of Inter-stimulus EEG Modulate the Linear Relationship Between Laser-Evoked Potentials and Fast-Pain Perception. https://doi.org/10.3389/fnins.2018.00340
3. Bott SF, et al. (2025). Exploring electroencephalographic chronic pain biomarkers: a mega-analysis. https://doi.org/10.1016/j.ebiom.2025.105955
4. Mari T, et al. (2023). External validation of binary machine learning models for pain intensity perception classification from EEG in healthy individuals. https://doi.org/10.1038/s41598-022-27298-1 (context only; not used as the central approach here)
