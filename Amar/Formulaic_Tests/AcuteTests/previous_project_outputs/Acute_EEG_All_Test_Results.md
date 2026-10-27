# Comprehensive acute-pain EEG validation ledger

**Run date:** 2026-10-26  
**Status:** Published benchmark arithmetic reproduced; raw trial-level EEG validation not completed.  
**Main published source:** Sun et al., *Detecting acute pain signals from human EEG*, J Neurosci Methods 347 (2020), 108964: https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/  
**Large public laser-evoked potential collection:** Zhao et al. (2025), *A comprehensive EEG dataset of laser-evoked potentials for pain research*: https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/

## Executive result

There are three distinct evidence levels in this analysis:

- **Computed now:** independent calculations on the 16 AUC/accuracy rows printed in Table 2 of Sun et al. The values are rounded to two decimals and represent repeated modality/subject train-test settings on only two chronic-pain participants exposed to acute painful pinprick and non-painful monofilament stimuli.
- **Reported by source, not recalculated:** results the paper reports from its 50-person healthy-volunteer laser dataset, including latency differences and AUC group summaries.
- **Not run:** any of the project's proposed EEG-to-0–10 pain-intensity formulas applied to raw or trial-level features. The raw dataset fetch failed in this runtime with `Temporary failure in name resolution`; zero raw EEG files and zero trial-level feature rows were loaded.

It would be scientifically wrong to claim the proposed intensity formulae are validated on the basis of a published AUC table that evaluates binary onset/detection, not 0–10 intensity regression.

## A. Recalculated published Table 2 summary

| Measure | Mean | Sample SD | Median | Range |
|---|---:|---:|---:|---:|
| SSM unsupervised event-detection AUC | 0.711 | 0.068 | 0.710 | 0.60–0.81 |
| SVM supervised peak AUC | 0.614 | 0.120 | 0.590 | 0.43–0.83 |
| SVM peak accuracy | 0.602 | 0.069 | 0.605 | 0.50–0.73 |
| SSM minus SVM AUC | 0.096 | 0.124 | 0.115 | -0.18–0.28 |

SSM exceeded SVM AUC in 13/16 configurations; SVM exceeded SSM in 3/16; ties: 0. Every SSM row was above AUC 0.5; SVM had 12 rows above, 1 equal to, and 3 below 0.5. AUC >0.5 indicates discrimination across the two classes under these test settings, not calibrated pain intensity.

### Row-level paired tests (exploratory only)

| Test | Statistic | p-value | Limitation |
|---|---:|---:|---|
| Paired t-test | t=3.115, df=15 | 0.0071 | Rows are not independent; configurations recycle the same two patients. |
| Wilcoxon signed-rank, approximate | W=18.5 | 0.0105 | Similar to paper-reported p=0.0113; rounded values explain small difference. Not population-level inference. |
| Exact sign test, two-sided | 13 wins vs 3 losses | 0.0213 | Dependence caveat remains. |
| Exact sign test, one-sided SSM greater | 13 wins vs 3 losses | 0.0106 | Directional sensitivity only. |
| Pearson correlation between method AUCs | r=0.233 | 0.3859 | Cross-row descriptive only. |
| Spearman correlation between method AUCs | rho=0.252 | 0.3467 | Cross-row descriptive only. |

A conventional row-level 95% t interval for the mean AUC difference is [0.030, 0.162], and leave-one-row-out mean differences range from 0.084 to 0.115. **The interval is shown only as a numerical sensitivity calculation. It is not a valid patient-level confidence interval because the 16 rows are dependent repeated settings.**

## B. Stratified analysis by train/test relation

| Group | n | SSM mean AUC | SVM mean AUC | Mean SSM−SVM | SSM wins |
|---|---:|---:|---:|---:|---:|
| within subject | 8 | 0.714 | 0.674 | 0.040 | 5/8 |
| cross subject | 8 | 0.708 | 0.555 | 0.152 | 8/8 |
| same stimulation site modality | 8 | 0.714 | 0.634 | 0.080 | 5/8 |
| cross stimulation site modality | 8 | 0.708 | 0.595 | 0.113 | 8/8 |
| within subject and same site | 4 | 0.720 | 0.767 | -0.047 | 1/4 |
| within subject cross site | 4 | 0.708 | 0.580 | 0.128 | 4/4 |
| cross subject same site | 4 | 0.707 | 0.500 | 0.207 | 4/4 |
| cross subject cross site | 4 | 0.708 | 0.610 | 0.097 | 4/4 |

These strata are descriptive only. Each subgroup has four or eight rows, but not four/eight independent subjects. The factorized row-level CSV preserves the exact train/test labels so these summaries can be audited.

## C. Source-reported findings from the separate 50-volunteer acute laser dataset

The paper describes 51 recruited healthy volunteers and analyzes the motor condition for 50, with 60 laser trials in the condition and Level 3 vs Level 1 as its main binary contrast. It used source-localized responses from eight ROIs (bilateral precentral, postcentral, ACC and insula) and an SSM detection score. It did not fit a 0–10 regression model in the analysis reported here.

| Source-reported result | Number | Status and meaning |
|---|---:|---|
| Group without clear evoked potentials | 18/50 | Reported by authors. |
| Group with clear evoked potentials | 32/50 | Reported by authors. |
| Mean AUC for no-clear-ERP group | 0.49 | Reported by authors. |
| Mean AUC for clear-ERP group | 0.64 | Reported by authors. |
| Group comparison p-value | 1.9×10⁻⁵ | Source-reported rank-sum test; see direction discrepancy below. |
| Median detection latency, Level 3 | 180 ms | Source-reported. |
| Median detection latency, Level 1 | 202 ms | Source-reported. |
| Difference between medians | −22 ms | Arithmetic difference: Level 3 shorter. |
| Latency contrast p-value | 1.5×10⁻⁵ | Source-reported rank-sum test. |
| Detection latency vs reaction time | Spearman rho=0.29, p=0.06 | Weak association; not significant at the conventional 0.05 threshold. |
| Cross-subject SSM average AUC | 0.62 | Source-reported; comparable to within-subject average AUC 0.64. |
| Selected cross-subject model tests | 16 subjects with AUC>0.65 | Reported for best of seven models selected from higher-performing subjects; possible selection optimism. |
| Contralateral ROI latency tests | ACC–M1 p=1.2×10⁻⁵; S1–M1 p=0.02; ACC–insula p=5×10⁻⁴ | Source-reported rank-sum results; not pain-intensity model validation. |

**Source inconsistency requiring attention:** the Results text says the clear-ERP group AUC was higher than the no-clear-ERP group (means 0.64 vs 0.49), but the Figure 3 caption says AUC was greater in Group 1 than Group 2 while also listing means 0.49 and 0.64. I have preserved and flagged the contradiction instead of silently choosing one direction. The paper's figure or source data should resolve it before this result is used.

## D. Published acute-stimulus ratings from two chronic-pain patients

| Patient/site | Painful pinprick median NRS (IQR) | Non-painful monofilament median NRS (IQR) | Difference in published medians |
|---|---:|---:|---:|
| Patient 1, hand | 1 (1,1) | 1 (1,1) | 0.0 |
| Patient 1, back | 1.5 (1,2) | 1 (1,1) | +0.5 |
| Patient 2, hand | 3 (2,4) | 2 (1,2) | +1.0 |
| Patient 2, back | 4 (3,5) | 2 (2,3) | +2.0 |

Across these four aggregate contrasts the mean of the four median differences is 0.875 and the median is 0.75; three contrasts are positive, one is tied and none negative. These calculations summarize published medians only. They are not based on individual trial ratings, and no p-value is appropriate for such tiny, dependent aggregate summaries. Notice that Patient 1's hand median was identical for painful and non-painful stimulation, which cautions against using stimulus category alone as the intensity label.

## E. Larger dataset available for proper 0–10 scale validation — not processed

The 2025 laser-evoked-potential data descriptor reports raw BIDS EEG and single-trial pain ratings across nine experiments and 678 healthy participants. It reports increasing LEP responses with stimulus intensity and perceived pain ratings, and provides data through OpenNeuro plus preprocessed MATLAB derivatives. This is strong rationale for the next validation step but is not a substitute for fitting and testing a formula.

One accessible experiment, ds005293 / “95 By BP”, reports 95 participants, 60 EEG channels, 80 planned trials each, and four target pain-rating levels (NRS 2, 4, 6, 8) crossed with left/right hand, ten trials per condition. That design could test ordinal ordering, within-person trend, held-out-subject performance and calibration, once the recordings and trial-level events are processed. The 80×95=7,600 is a **planned design count**, not a verified post-QC number of usable trials.

## F. Test execution register

See `acute_validation_test_registry.csv` for the row-by-row ledger. Main states are:

- `COMPUTED_FROM_PUBLISHED_TABLE`: calculations run on values printed by the source paper.
- `REPORTED_BY_AUTHORS_NOT_RECOMPUTED`: exact source-reported statistics that cannot be independently reproduced without raw trials.
- `DESCRIPTIVE_ONLY` / `EXPLORATORY_ONLY`: arithmetic on repeated configuration rows or aggregate median contrasts; no population inference.
- `NOT_RUN`: analyses requiring participant-level EEG, trial-level labels or fitted out-of-sample predictions.

## G. Formula-level validations still not run

F0–F14 pain-intensity estimators require feature rows such as alpha/beta/theta power, ERP N2/P2 amplitudes, envelope or spatiotemporal features paired to actual per-trial NRS. None can be assessed using a table of binary detection AUCs. The key tests still pending are:

- Participant-grouped out-of-fold MAE/RMSE/R² and calibration for 0–10 scores.
- Ordinal discrimination and calibration for ordered NRS targets.
- Within-participant Spearman/Kendall trend across actual rating levels, with subject-level uncertainty.
- Participant-cluster bootstrap confidence intervals for model differences.
- Permutation tests shuffling labels within subjects while preserving design structure.
- Temporal holdout and drift checks for repeated trials.
- Cross-dataset locked validation across different EEG systems/montages.
- Feature ablation and artifact/EMG sensitivity.
- Demographic subgroup performance and uncertainty where recorded and adequately powered.
- Test-retest reliability and calibration stability if repeated sessions are available.

**No coefficients, p-values or accuracy scores for these intensity estimators have been fabricated.** The reason they remain unrun is recorded in the access log: this environment failed DNS resolution for external data hosts, and no raw EEG file was available locally. A working data connection or uploaded raw/preprocessed dataset is required to complete these model validations.

## References

- Sun G, Wen Z, Ok D, Doan L, Wang J, Chen ZS. *Detecting acute pain signals from human EEG*. Journal of Neuroscience Methods. 2020;347:108964. https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/
- Zhao X et al. *A comprehensive EEG dataset of laser-evoked potentials for pain research*. Scientific Data. 2025;12:1536. https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/
- OpenNeuro/NEMAR ds005293, “95 By BP”: https://nemar.org/dataset/on005293
- OpenNeuro/NEMAR ds005291, “65 By ANT”: https://nemar.org/dataset/on005291
