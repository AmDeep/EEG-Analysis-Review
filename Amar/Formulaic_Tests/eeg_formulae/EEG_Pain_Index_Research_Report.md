# EEG-only Pain Index: Research Prototype, Formula Set, and Validation Plan

**Version:** 0.1 (research prototype; not a clinical instrument)  
**Date:** 2026-10-16  
**Status:** Formula definitions and reproducible analysis scaffold created. Raw-data model validation has **not** been completed in this environment.

## Executive summary

The Excalibur drawing and supplied EEG feature table motivate an EEG-only feature set: sensorimotor alpha (8–13 Hz), beta (13–30 Hz), theta (4–8 Hz), gamma (>30 Hz), peak alpha frequency (PAF), frontal alpha asymmetry (FAA), alpha-envelope dynamics, spatially defined midline theta, connectivity/phase synchrony, spectral entropy, trial-to-trial response variability, and stimulus-locked evoked responses (including vertex N2–P2). The drawing also names habituation, diffuse noxious inhibitory control (DNIC/conditioned pain modulation), sensitization, central sensitization, and long-term potentiation as mechanisms; these are not directly measured by a single resting EEG spectrum and should not be treated as direct score inputs unless a defined EEG protocol operationalizes them.

**Key scientific limitation:** there is no universally validated EEG-to-pain-intensity mapping. A model must be trained and externally tested against time-matched participant pain ratings, with participant-wise splits to prevent leakage. A number on a 0–10 scale is not justified merely by clipping or rescaling EEG features. Until validated, report a *relative EEG pain-associated index* and uncertainty, not an individual's “true pain level.”

## Source-derived features and interpretation

From the supplied screenshot (Table 2):
- Sensorimotor alpha power, 8–13 Hz: central suppression is reported during experimental pain; direction/magnitude vary, and resting alpha findings vary by chronic-pain cohort.
- Peak alpha frequency: exploratory; absent peaks need explicit handling.
- Alpha-envelope dynamics: potentially distinct from mean power.
- Frontal alpha asymmetry: exploratory and sign convention must be fixed.
- Beta, 13–30 Hz: movement preparation and medication can confound.
- Theta, 4–8 Hz: resting and task recordings must be distinguished; local power differs from connectivity.
- Spatial midline theta: Fz/FCz and Cz should be separate, protocol-defined measures.
- Gamma, >30 Hz: muscle/EMG contamination risk.
- Connectivity/phase synchronization: different measures; reference and volume conduction matter.
- Spectral entropy: candidate input, not established pain-specific measure.
- Response variability/temporal features: require event-locked data and matched comparisons.
- Vertex N2–P2: stimulus-locked, with fixed reference, latency windows, and trial reliability.
- Infant noxious-evoked EEG template: contextual evidence for a separate neonatal pathway, not an adult model feature.

From the Excalidraw text:
- EEG signatures listed include spectral power changes/alpha suppression, gamma oscillations, event-related potentials, phase-amplitude coupling, 1/f slope, and somatosensory evoked potentials.
- The drawing also includes demographic/biological factors (age, sex/gender, estrogen-related effects) and experiential mechanisms (habituation, DNIC, sensitization, central sensitization, long-term potentiation).
- Because the user's later requirement is **EEG-only inputs**, demographics and non-EEG variables are excluded from score formulae. They remain subgroup audit variables where legally/ethically available, not score adjustments by default.

## Candidate formulas (all coefficients below are hypotheses, not fitted/validated coefficients)

All EEG features must be computed from a documented pipeline and standardized using training-set statistics only. Let z(x) be a robust within-person or training-population standardized feature (median/MAD or mean/SD), and clip extreme z-scores only using prespecified training-derived bounds. Do not mix resting and task epochs or different references without harmonization.

### F1 — Within-person spectral deviation (lowest complexity)
`S1 = 50 + 10 * (wA*z(-log(alpha_SM)) + wB*z(beta_SM) + wT*z(theta_midline) + wG*z(gamma_sensorimotor)) / sum(abs(w))`
Use only if the same person has a baseline; feature signs and weights must be learned in training data. Before fitting, no direction is assumed to be universally pain-increasing. In the code, weights are deliberately placeholders and must be supplied from fitted coefficients. Convert to 0–10 only after validation and calibration against self-report.

### F2 — Multiband robust linear predictor
`eta2 = b0 + bA*z(log_alpha) + bB*z(log_beta) + bT*z(log_theta) + bG*z(log_gamma) + bE*z(spectral_entropy) + bP*z(PAF)`
`S2 = 10 * sigmoid(eta2)` only when the target has been fit/calibrated to NRS 0–10; otherwise `eta2` is a model score, not pain units.

### F3 — Evoked-response model
`eta3 = b0 + bN*z(N2P2_amplitude) + bL*z(N2_latency) + bV*z(trial_response_variability) + bH*z(habituation_slope)`
Only for stimulus-locked recordings with event timing and trial-wise pain ratings. Habituation slope must be estimated from ordered trials, not inferred from one epoch.

### F4 — EEG dynamics + spectral model
`eta4 = b0 + sum_j(bj*z(power_band_j)) + bC*z(connectivity) + bQ*z(phase_amplitude_coupling) + bF*z(aperiodic_slope) + bD*z(alpha_envelope_dynamics)`
Connectivity, PAC, and aperiodic slope must use explicitly documented methods and quality control. These features are sensitive to referencing, volume conduction, and artifact processing.

### F5 — Regularized multivariate model (recommended benchmark)
Fit elastic-net regression or ordinal regression to trial-level self-reported pain, with nested participant-grouped cross-validation. This is preferable to hand-assigned weights because coefficient signs and magnitudes are estimated on labeled data. Use a separate external cohort for final validation.

### F6 — Within-person temporal change model
`DeltaS(t) = calibrated_prediction(t) - median(calibrated_prediction in pre-stimulus baseline)`
This estimates change from a person's baseline, not absolute pain. It is only meaningful when acquisition, state, and preprocessing are comparable over time.

### F7 — Latent multi-view model
Fit a supervised latent-factor or partial least-squares model over spectral, evoked, and connectivity feature blocks. Compare to F5; keep only if grouped external validation improves error and calibration. Latent factors are not directly interpretable as mechanisms.

### F8 — Nonlinear benchmark
Random forest or gradient boosting using the same EEG-only features, with nested subject-wise CV and external testing. Treat as a benchmark, not automatically superior. Report calibration, error, and subgroup performance, not just classification accuracy.

## Coefficients and scaling

No numerical coefficient is claimed to be correct before fitting to genuine labeled records. The screenshot supports candidate features and cautions, but it does not provide individual EEG observations or fitted regression coefficients. The Excalidraw references are rationale, not a coefficient table. Any fixed coefficients would be invented. The Python pipeline therefore fits coefficients from a user-supplied, documented feature table and stores them with the split and metrics.

Recommended target: participant-provided 0–10 NRS per trial (or clearly specified VAS conversion). Fit the model to the target directly; do not map arbitrary EEG values to pain categories with arbitrary thresholds. If a categorical scale is required, derive thresholds on training data and preregister them, then evaluate sensitivity/specificity and confusion matrix on untouched test participants.

## Data sources located

1. **OpenNeuro ds005293 / NEMAR on005293 — “95 By BP.”** Healthy participants; painful laser stimuli at individually calibrated intensities; NRS labels 2, 4, 6, 8; 80 trials per participant; left/right hand. Official landing page: https://nemar.org/dataset/on005293 and https://github.com/OpenNeuroDatasets/ds005293. Good trial-level evoked-pain benchmark.
2. **OpenNeuro ds005285 / NEMAR on005285 — “29 By ANT.”** Laser stimuli at low/high intensities, participant ratings 0–10, 160 stimulations per participant. https://nemar.org/dataset/on005285.
3. **OpenNeuro ds005284 / NEMAR on005284 — “26 By Biosemi.”** Calibrated laser stimuli and repeated ratings; 16 fixed-intensity stimuli per participant. https://nemar.org/dataset/on005284.
4. **OpenNeuro ds005289 / NEMAR on005289 — “39 By BP.”** 10 laser trials per participant; pain intensity and unpleasantness ratings 0–10. https://nemar.org/dataset/on005289.
5. **Zenodo 10.5281/zenodo.12570277 — “Unravelling the physiological and psychosocial signatures of pain by machine learning.”** 118 participants (81 chronic-pain participants across CRPS, low-back pain, spinal cord injury neuropathic pain; 37 healthy controls), 4,697 experimental pain trials, EEG and EDA, demographics, medication, sleep/fatigue, and questionnaires. Files include `SubjectAreas.csv`, `Subjects.csv`, and `Trials.pkl`. https://zenodo.org/records/12570277. Because the desired index is EEG-only, exclude EDA, demographics, and questionnaires from predictors; retain them for audit/stratified evaluation where permitted.
6. **OpenNeuro ds006897 — “No effect of rhythmic visual stimulation on experimental pain perception.”** 41 participants; EEG; published in *Pain* (2026). Useful as a distinct experimental context, but confirm labels and task comparability before pooling. https://github.com/OpenNeuroDatasets/ds006897.
7. **OpenNeuro ds006921 — resting-state EEG of phantom-limb pain and controls.** Different chronic pain phenotype and resting protocol; useful for external generalization only if compatible labels and licensing permit. https://github.com/OpenNeuroDatasets/ds006921.
8. A 2026 external-validation paper reports that external EEG regression did not exceed a mean-value baseline, while low/high pain classification reached AUC 0.62–0.68 in its sample. This is a cautionary result, not evidence that any formula here works: https://www.sciencedirect.com/science/article/pii/S1526590026002348.

**Execution note:** public landing pages and dataset descriptions were verified online during preparation. Raw signal files were not downloaded or processed in this environment, so no claim is made that a fitted formula was tested on raw data here. The supplied Python script implements reproducible tests when the dataset has been downloaded and transformed into a trial-level feature table. Do not describe this as completed empirical validation until the pipeline has run on those real files.

## Required feature-table schema

One row per participant × trial/epoch. Required:
- `participant_id` (opaque ID; never include names)
- `session_id`, `trial_id`, `time_seconds`
- `pain_nrs` (numeric 0–10; actual participant report, not stimulus intensity)
- `split_dataset` or source/cohort identifier
- feature columns such as `log_alpha_sm`, `log_beta_sm`, `log_theta_fz`, `log_gamma_sm`, `paf_hz`, `faa`, `alpha_envelope_cv`, `theta_fz_power`, `theta_cz_power`, `spectral_entropy`, `aperiodic_slope`, `n2p2_uv`, `n2_latency_ms`, `habituation_slope`, `connectivity_index`, `pac_index`, `trial_response_variability`
- optional QC columns: `usable_epoch`, `artifact_fraction`, `eyes_condition`, `task_condition`, `reference`, `sampling_rate_hz`, `pipeline_version`

Use real observed values only. Missing features must remain missing and be handled in a documented way; never impute labels. If only condition-average ratings exist, use condition-level observations and do not pretend they are trial-level labels.

## Validation protocol

1. Freeze preprocessing and feature definitions before inspecting test outcomes.
2. Use grouped nested CV: all trials from a participant belong to only one fold. Never random-split individual trials across train/test.
3. Keep one dataset entirely external for transportability. Do not tune thresholds on the external set.
4. Compare every model against a mean/median baseline and, where possible, stimulus-intensity-only and within-person-baseline comparators. Stimulus intensity is not an EEG predictor; it can be a benchmark only.
5. Regression metrics: MAE, RMSE, Spearman/Pearson correlation, calibration slope/intercept, and within-person vs between-person performance separately.
6. Classification metrics if categories are preregistered: balanced accuracy, macro-F1, AUROC, AUPRC, sensitivity, specificity, confusion matrix, calibration. Report confidence intervals bootstrapped by participant, not by trial.
7. Temporal evaluation: chronological split within participants; predict later trials from earlier training trials only, quantify drift, autocorrelation, baseline shifts, and calibration degradation.
8. Subgroup audit: age bands, sex/gender where collected, pain cohort/etiology, medication, sleep/fatigue, and site/device. These are audit variables, not score inputs. Report sample counts and uncertainty; do not apply demographic correction factors without evidence.
9. Robustness: artifact thresholds, reference montage, band boundaries, gamma EMG contamination, missing bands, baseline length, eyes-open/closed, task/rest, and feature normalization.
10. Reproducibility: pin software versions, save raw-data source/version/hash, preprocessing configuration, feature extraction code, split assignments, model coefficients, seed, metrics, and run logs.

## Decision criteria for “validated”

A candidate is not validated merely because its correlation is significant. Minimum proposed gates:
- beats the mean baseline on external participant-held-out MAE with participant-bootstrap confidence interval excluding no improvement;
- acceptable calibration slope/intercept and no systematic clipping;
- stable results across at least two independent datasets and acquisition contexts;
- no severe subgroup degradation hidden by aggregate performance;
- temporal performance better than baseline and stable under realistic drift;
- results reproducible from raw files and a pinned pipeline.
Thresholds must be preregistered with clinical stakeholders. Until then, all scores are exploratory and should not guide treatment or override self-report.

## Formula selection recommendation

Use F5 elastic-net regression as the primary benchmark, F1 as an interpretable within-person baseline-deviation comparator, F3 for event-related protocols, and F8 as a nonlinear comparator. Avoid a single universal EEG-only absolute-pain formula unless external tests show generalizable calibration. Keep all mechanism labels (DNIC, sensitization, LTP) in the conceptual model, not as measured variables unless the protocol specifically measures them.

## References

- Supplied user screenshot: “Table 2. EEG features discussed in the review and their role in the proposed framework,” including feature definitions and cautions summarized above.
- Supplied Excalidraw export in `Pasted markdown(1).md`: EEG signatures, demographic/biological factors, experiential mechanisms, and proposed relationships.
- OpenNeuro ds005293: https://github.com/OpenNeuroDatasets/ds005293
- NEMAR on005293: https://nemar.org/dataset/on005293
- NEMAR on005285: https://nemar.org/dataset/on005285
- NEMAR on005284: https://nemar.org/dataset/on005284
- NEMAR on005289: https://nemar.org/dataset/on005289
- Gozzi et al. dataset, Zenodo DOI 10.5281/zenodo.12570277: https://zenodo.org/records/12570277
- OpenNeuro ds006897: https://github.com/OpenNeuroDatasets/ds006897
- OpenNeuro ds006921: https://github.com/OpenNeuroDatasets/ds006921
- External validation study, *The Journal of Pain* (2026): https://www.sciencedirect.com/science/article/pii/S1526590026002348
