# EEG-only pain-scale modelling: evidence, equations, and validation plan

**Version:** 2.0 — 16 October 2026  
**Scope:** EEG signals and EEG-derived features only. No ECG, EMG-derived predictors, skin conductance, facial expression, stimulus intensity, medication, demographic variable, or other non-EEG input may enter the score. Stimulus intensity may be used to stratify a sensitivity analysis, but not as a predictor in the EEG-only score.

## Executive finding

There is not yet a defensible, empirically validated universal EEG-only pain formula in this package. The public evidence establishes that graded laser stimuli can produce graded subjective ratings and laser-evoked EEG responses. It does not establish a fixed coefficient vector that generalizes across people, sessions, EEG systems, pain paradigms, and clinical conditions. The supplied analysis code can estimate candidate models from a genuine trial-level feature file, but **no participant-level feature file was available in the execution environment for this run**. Therefore this release reports real published dataset metadata and published qualitative validation findings, but no invented fitted coefficients, trial counts processed, p-values, confidence intervals, or model accuracy.

The published dataset descriptor reports 678 healthy participants across nine experiments, with single-trial ratings and EEG; raw BIDS EEG and supplementary MATLAB/CSV resources are available from the public repositories. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/ and https://github.com/XiangyueZhao/Supplement-Data.

## Real source-backed values (study design / repository metadata, not this project's re-analysis)

| Dataset | Participants | Channels | Trial design | Pain target / source detail |
|---|---:|---:|---|---|
| ds005293, “95 By BP” | 95 | 60 | 80 trials/person; 8 conditions × 10 trials | Individually calibrated NRS levels 2, 4, 6, 8 crossed with left/right hand; CC0. https://nemar.org/dataset/on005293 |
| ds005285, “29 By ANT” | 29 | 32 | 160 trials/person; four blocks of 40 | Low/high laser intensity, trial-level ratings 0–10. https://nemar.org/dataset/on005285 |
| ds005284, “26 By Biosemi” | 26 | 64 | 16 fixed-intensity trials/person after calibration | Trial-level ratings 0–10; https://nemar.org/dataset/on005284 |
| ds005291, “65 By ANT” | 65 | 32 | 30 laser stimuli/person (as described by dataset record) | Verbal pain ratings 0–10; https://nemar.org/dataset/on005291 |
| Multi-experiment collection | 678 across 9 experiments | varies by experiment | single-trial ratings and EEG | The paper describes EEG LEP responses increasing with stimulus intensity and pain ratings; this is a published technical-validation result, not a fitted model result here. https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/ |

These are published design/metadata values. They must not be reported as the number of valid trials surviving preprocessing or as a validation sample size. Cohorts may overlap across the collection or differ in protocol; never sum participants across datasets without checking overlap and eligibility.

## Candidate model families to compare

Let each trial's EEG-only feature vector be \(x_{it}\), participant \(i\), trial \(t\), and self-reported NRS rating \(y_{it}\in[0,10]\). The NRS is a subjective reference label, not an objective ground truth of nociception.

1. **F0 — alpha-only sigmoid diagnostic:** \(\hat y=10\,\sigma(a_0+a_1z_\alpha)\). This is an intentionally simple comparator, not an assumed truth. It can fail if alpha is non-monotonic or state-dependent.
2. **F1 — regularized linear EEG model:** \(\hat y=\operatorname{clip}_{[0,10]}(b_0+\boldsymbol b^T\boldsymbol z)\). Use ridge and elastic net as separate candidates; report coefficients only from training folds or final refit after model selection.
3. **F2 — ordinal model:** estimate \(P(Y\ge k\mid x)\), for thresholds \(k=1,\ldots,10\), with ordered thresholds; expected score is \(\sum_{k=1}^{10}P(Y\ge k\mid x)\). Check proportional-odds assumptions; if violated, use a flexible ordinal model.
4. **F3 — nonlinear model:** a GAM or gradient-boosted model with prespecified complexity limits. Do not claim interpretability from feature importance alone.
5. **F4 — within-person calibration:** a separate, explicitly personalized score calibrated from a prespecified number of that person's labeled trials. It is a different use case from unseen-person prediction and must be evaluated with chronological holdout.
6. **F5 — temporal smoothing:** \(s_t=\lambda s_{t-1}+(1-\lambda)\hat y_t\), with \(\lambda\) selected on training data only. This smooths noise and can also delay real changes; report lag and unsmoothed performance.
7. **F6 — event-related potential (ERP) model:** use EEG-derived N2/P2 amplitudes, latencies, and prespecified windows around noxious stimulus onset. Event timing and artifact handling must be documented.
8. **F7 — spectral-only ablation:** compare alpha, beta, theta, and cautiously gamma features against ERP-only and combined EEG feature sets. Gamma is particularly vulnerable to muscle contamination; it is included only as an EEG channel-derived candidate and requires sensitivity analysis, not an EMG predictor.
9. **F8 — robust rank/quantile score:** map EEG feature ranks or robust z-scores to an ordinal estimate; compare its stability with absolute-scale regression.

A formula is not “correct” because it correlates with pain ratings in the training data. It is supported only if it outperforms meaningful baselines on held-out participants/sessions, has calibrated errors, replicates across datasets, and passes prespecified robustness checks.

## EEG-only feature groups

- **Spectral power:** log band power for alpha (commonly 8–13 Hz), beta (13–30 Hz), theta (4–8 Hz), and low-frequency/delta only when justified by the paradigm. Frequency bands and reference montage must be fixed before confirmatory analysis.
- **Individual alpha frequency (IAF):** estimate only when signal length and spectral peak quality support it; record failed estimates rather than silently imputing a biologically meaningful value.
- **Alpha envelope dynamics:** amplitude-envelope mean, variability, autocorrelation, and event-related change; avoid feature windows overlapping the rating response if that can introduce movement/motor contamination.
- **Frontal alpha asymmetry:** calculate from a preregistered pair of electrodes and log-power convention; test sensitivity to reference and channel availability. Do not assume a universal direction of association with pain.
- **Theta / midline theta:** predefined midline channel or region and event/rest window; evaluate within-person and between-person effects separately.
- **ERP N2/P2:** baseline-corrected amplitude and latency in preregistered time windows/topography; report peak-picking rules, filtering, and rejected epochs.
- **Connectivity / phase synchrony:** only after a documented leakage/volume-conduction strategy; avoid treating many correlated channel pairs as independent evidence.
- **Spectral entropy / complexity:** document exact estimator, epoch length, frequency range, and sensitivity to artifacts. Incremental value must be tested by ablation.
- **Quality and artifacts:** channel-level quality and EOG/EMG artifact metrics are for exclusion/sensitivity analysis, not model predictors in an EEG-only scale. Do not feed EMG values into the pain formula.

## Required statistical tests and acceptance logic

### Data integrity and preprocessing
1. Verify BIDS metadata, channel types, sample rate, units, event markers, trial-to-rating mapping, and license; retain a provenance manifest and checksums.
2. Count participants, sessions, events, epochs, missing labels, rejected epochs, and usable trials by participant and condition. Report the flow from raw trials to analysis rows.
3. Inspect EEG amplitude/range, flat channels, line noise, clipping, reference choice, filter effects, and artifact rejection. Run sensitivity analyses with/without ICA/automated rejection where justified.
4. Avoid label leakage: fit imputation, scaling, feature selection, baseline normalization, and hyperparameter tuning only within training folds. No trial from a test participant may inform a population model's preprocessing statistics.
5. Keep all epochs from a participant in one outer fold for unseen-person evaluation. If sessions are repeated, also hold out sessions as a separate generalization test.

### Predictive performance
- Primary metrics: MAE and RMSE on out-of-fold trials, aggregated with participant-clustered uncertainty. Report bias and out-of-sample R²; R² may be negative.
- Association metrics: Pearson and Spearman correlations are secondary; high correlation does not imply low absolute error or calibration.
- Ordinal metrics: weighted kappa, ordinal log loss / ranked probability score, and within-1-point accuracy if using an ordinal model.
- Calibration: calibration intercept/slope and observed-vs-predicted curves; examine whether clipping causes edge pile-ups.
- Compare against mean/median rating baselines, alpha-only, spectral-only, ERP-only, and combined EEG-only models.
- Use paired participant-cluster bootstrap confidence intervals for MAE differences; report the number of bootstrap repetitions and seed. Do not bootstrap individual trials independently.
- Permutation test: prespecify a null that respects repeated measures. For within-person association, permute labels within participant and condition/block as appropriate. For between-person effects, permute participant-level labels. Never shuffle all trials indiscriminately.
- Multiplicity: control false discovery rate across feature-wise exploratory tests; reserve a small prespecified feature/model family for confirmatory inference.

### Generalization and falsification
- **Participant-held-out:** primary test of generalization to a new person.
- **Session/time-block-held-out:** within-person personalization; train on earlier labeled trials and test on later trials, with chronological ordering preserved.
- **Cross-dataset:** train on one acquisition/protocol and test on a genuinely independent dataset after harmonizing only EEG feature definitions. A dataset shift is a result, not something to conceal.
- **Condition / hand / stimulus intensity:** use as stratification or sensitivity analyses; they must not be model inputs for an EEG-only pain score.
- **Negative controls:** pre-stimulus EEG should not show the same event-locked effect as post-stimulus ERP; shuffled labels should destroy predictive performance; test whether trial order or block alone explains apparent performance.
- **Artifact sensitivity:** compare clean-channel subsets and feature sets excluding gamma or potentially contaminated windows. A performance collapse under reasonable artifact sensitivity weakens the claim.
- **Demographic subgroup audit:** age/sex categories only where genuinely present and adequately sized in the dataset. Demographics are audit variables, never score inputs. Report per-group MAE, calibration, sample sizes, and uncertainty; do not overinterpret small subgroups.
- **Temporal behavior:** compare unsmoothed vs smoothed scores, abrupt-change response, lag, drift, and habituation-related patterns. Habituation, descending inhibition, sensitization, and central sensitization are hypotheses requiring study-specific evidence; they are not fixed multipliers in the score.

## What this release actually executed

- Software boundary/unit tests passed for score clipping to 0–10, EWMA recurrence, sigmoid bounds, robust z-score handling when MAD is zero, and rejection of invalid smoothing parameters.
- **Not executed:** raw EEG download, raw-to-epoch preprocessing, feature extraction, participant-level model fitting, grouped cross-validation, confidence intervals, p-values, demographic subgroup tests, temporal holdout, or cross-dataset validation. The execution environment could not resolve external hosts from its code runtime, so it could not retrieve the public binary EEG files. The browser research confirmed repository metadata and the paper's stated findings, but is not a substitute for reading the signal arrays.
- Consequently, this report deliberately contains no fabricated coefficient estimates or participant-level performance values. A numerical formula is not endorsed until the runnable pipeline below is executed on genuine feature rows.

## Reproducible model runner

Input CSV schema: `subject_id,trial_id,pain_nrs,<EEG feature columns>`; optional `session_id`, `trial_order`, `condition`, `age_group`, and `sex_group`. Each row must be one real EEG trial with a traceable source epoch. The script writes out-of-fold predictions, grouped-CV metrics, participant-level metrics, participant-cluster bootstrap intervals for MAE difference versus the mean baseline, descriptive feature associations, and subgroup audit tables where metadata exist.

```bash
python -m pip install numpy pandas scipy scikit-learn openpyxl python-docx
python eeg_pain_models.py --self-test
python eeg_pain_models.py --features trial_eeg_features.csv --features-list z_alpha z_beta z_theta z_erp z_n2 z_p2 --outdir results
```

The current script uses fixed prespecified model settings for a first comparison; it is not a nested-CV hyperparameter search. Model selection should be added within inner participant-grouped folds before any final performance estimate is used for publication or deployment. Do not describe the software unit tests as validation of the pain scale.

## Source links

- Zhao et al., *A comprehensive EEG dataset of laser-evoked potentials for pain research*, Scientific Data (2025): https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/
- Supplementary MATLAB/CSV/code repository: https://github.com/XiangyueZhao/Supplement-Data
- ds005293 / 95 By BP: https://nemar.org/dataset/on005293
- ds005285 / 29 By ANT: https://nemar.org/dataset/on005285
- ds005284 / 26 By Biosemi: https://nemar.org/dataset/on005284
- ds005291 / 65 By ANT: https://nemar.org/dataset/on005291
- OpenNeuro documentation: https://docs.openneuro.org/user_guide.html
