# EEG-only acute-pain formula validation: consolidated results

**Version:** 2026-10-26  
**Scope:** Formula application and pain-band categorization using genuine published acute-pain EEG values; reconciliation of earlier results; auditable record of what remains impossible without individual trial-level EEG data.

## Executive conclusion

The proposed equations have **not** been validated as individual-level clinical pain scales. In this run, we applied simple linear calibration formulas to three genuine published *group-average* laser-evoked potential (LEP) values and compared an externally published power law against those same values. Those calculations are real and reproducible, but each pain-band target was represented by the band midpoint (5, 7, 9), not by measured individual-trial ratings. Consequently, the low error from the fitted linear equations is a three-point aggregate calibration check, not evidence of prediction accuracy in individual subjects.

The only defensible ranking from the aggregate calculation is: P2-only linear calibration had the smallest leave-one-band-out error (MAE 0.0386 NRS points); N2–P2 linear was next (0.0492); absolute N2 linear third (0.0606). The differences are tiny, based on three aggregate points, and are not statistically significant. All three features had perfect rank ordering, but exact two-sided permutation p = 0.333 because only 3 points (6 possible permutations) were available. The cross-study power-law equation preserved ordering but had MAE 3.807 NRS points when transferred to these higher rating bands; this is an out-of-range transfer stress test, not a fair refit comparison.

**No real participant-held-out formula leaderboard has been produced.** Raw / trial-level EEG feature rows could not be accessed from the compute container. The full dataset is publicly documented, including single-trial pain ratings in epoch metadata and preprocessed data/code repositories, but the local HTTP requests attempted in this run failed at DNS resolution. No fake participant rows or synthetic trial values were substituted.

## 1. Source data used

Li et al. (2018), *Magnitude and Temporal Variability of Inter-stimulus EEG Modulate the Linear Relationship Between Laser-Evoked Potentials and Fast-Pain Perception*, reports group-average N2 and P2 peak amplitudes at three reported pain-rating bands: 4 < NRS ≤ 6, 6 < NRS ≤ 8, and 8 < NRS ≤ 10. The source reports mean ± SD values:

| Pain-rating band | Representative target used here* | N2 mean ± SD (µV) | P2 mean ± SD (µV) | Derived P2 − N2 (µV) |
|---|---:|---:|---:|---:|
| 4 < NRS ≤ 6 | 5 | −7.58 ± 8.88 | 7.14 ± 8.20 | 14.72 |
| 6 < NRS ≤ 8 | 7 | −19.06 ± 12.72 | 19.38 ± 11.98 | 38.44 |
| 8 < NRS ≤ 10 | 9 | −30.13 ± 18.84 | 31.34 ± 12.70 | 61.47 |

*The values 5, 7 and 9 are assigned band midpoints, not observed means of individual ratings. P2−N2 is computed from published mean peaks; its standard deviation cannot be derived without the covariance between N2 and P2. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC5991169/ and https://doi.org/10.3389/fnins.2018.00340.

### Dispersion check

Approximate mean ± one-SD intervals overlap between neighboring bands:

- P2 band 1: −1.06 to 15.34 µV; band 2: 7.40 to 31.36 µV; band 3: 18.64 to 44.04 µV.
- N2 band 1: −16.46 to 1.30 µV; band 2: −31.78 to −6.34 µV; band 3: −48.97 to −11.29 µV.

These are descriptive mean±SD ranges, not confidence intervals or prediction intervals. Their overlap warns that a monotonic trend between group means does not imply clean individual-trial categorization.

## 2. Formulae actually applied

For the three aggregate points, target y was the representative pain-band midpoint (5, 7, 9). Linear regressions were fit using each one-dimensional signal feature:

1. **F-P2:** ŷ = 3.812263 + 0.165282 × P2(µV)
2. **F-N2P2:** ŷ = 3.730933 + 0.085555 × [P2 − N2](µV)
3. **F-|N2|:** ŷ = 3.643681 + 0.177364 × |N2|(µV)
4. **External power law transfer:** original reported relation P2 = 8.1 × NRS^0.78 (observed in an earlier, lower-rating-range experiment) was inverted as ŷ = (P2 / 8.1)^(1/0.78), then clipped to 0–10.

The three locally fitted linear formulae use the same three points both to estimate their final coefficients and to calculate their in-sample error. Their leave-one-band-out predictions fit on the other two aggregate points and predict the left-out point. That is a useful small numerical check, but the folds are rating bands, not subjects, and two-point training folds cannot estimate uncertainty in a generalizable clinical scale.

## 3. Formula results

| Formula | In-sample MAE | In-sample RMSE | Leave-one-band-out predictions for target 5, 7, 9 | Leave-one-band-out MAE | Leave-one-band-out RMSE | Mean bias (predicted − target) | Nearest-midpoint category accuracy |
|---|---:|---:|---|---:|---:|---:|---:|
| F-P2 | 0.0103 | 0.0109 | 4.953, 7.023, 8.954 | 0.0386 | 0.0401 | −0.0231 | 3/3 (aggregate points only) |
| F-N2P2 | 0.0131 | 0.0139 | 4.940, 7.030, 8.942 | 0.0492 | 0.0511 | −0.0295 | 3/3 (aggregate points only) |
| F-|N2| | 0.0162 | 0.0171 | 4.926, 7.036, 8.929 | 0.0606 | 0.0630 | −0.0364 | 3/3 (aggregate points only) |
| External power law | 3.8074 | 3.8231 | N/A—fixed external coefficients | N/A | N/A | −3.8074 | 1/3 by nearest midpoint |

The simple linear equations produce scores in expected order. Their close ranking is not meaningful given three datapoints and constructed target midpoints. The external power law's scores for the three P2 values were 0.851, 3.060 and 5.667; it remained monotonic but badly under-predicted the 5, 7 and 9 midpoints. Its original relationship was described over a pain-rating range of about 1–5, so using it at higher rating bands is an out-of-range stress check.

## 4. Statistical tests

| Test | Result | What it supports |
|---|---:|---|
| Feature rank ordering vs. target band midpoint: P2 | Spearman ρ = 1.000 | All three aggregate means ordered correctly |
| Feature rank ordering vs. target band midpoint: P2−N2 | Spearman ρ = 1.000 | All three aggregate means ordered correctly |
| Feature rank ordering vs. target band midpoint: |N2| | Spearman ρ = 1.000 | All three aggregate means ordered correctly |
| Exact two-sided rank permutation test, for each feature | p = 0.3333; 6 permutations | Not significant; only 3 aggregate points |
| Adjacent P2 standardized mean differences* | 5→7: 1.192; 7→9: 0.969 | Descriptive standardized separation of reported group means only |
| Extreme-band P2 standardized mean difference* | 5→9: 2.264 | Descriptive separation of endpoints only |

*Standardized differences are calculated as the difference in group means divided by the square-root average of the two reported variances. They are descriptive summaries; no inferential p-values or confidence intervals are calculated because group counts and subject-level distributions are not part of the extract used for this check. The one-SD ranges overlap, so endpoint group separation must not be translated into claimed individual classification accuracy.

The exact permutation p-value of 0.333 is appropriate for the tiny rank-order check: with three points there are 6 possible target label orderings, and two are at least as extreme in absolute Spearman correlation. Reported p-values from the earlier two-patient published classifier benchmark are kept separate; they are not tests of these newly fitted formulas.

## 5. What could not be evaluated from the available values

The source group means do not include the paired raw signals, per-trial features, subject IDs, trial order or individual ratings needed to test the following families. No numerical results are asserted for them:

- Alpha-only sigmoid formula: requires alpha power (and a prespecified baseline/normalization).
- Ordinal cumulative-link model: needs many trial-level examples across rating classes.
- Ridge/elastic-net multifeature regression; GAM; random forest; gradient boosting: need paired feature vectors and ratings, with all preprocessing inside subject-held-out folds.
- Personalized calibration and mixed-effects model: require multiple trials per identified participant.
- Temporal EWMA: requires ordered trial predictions per participant; smoothing cannot fix an uncalibrated raw score.
- Spectral-only, connectivity, entropy, frontal alpha asymmetry or EEG interactions: their trial-level predictors are not included in the group-mean table.
- Participant-held-out MAE/RMSE/R², per-subject confusion matrices, confidence intervals, permutation tests, class-specific sensitivity, and cross-dataset testing: all require paired trial-level EEG-derived features and ratings.
- Clinical acute pain validation: healthy-volunteer laser pain datasets alone do not establish clinical validity in post-operative or injury patients.

## 6. External evidence kept separate from our formula tests

A publicly described analysis of OpenNeuro ds005284 reports a subject-level pain-sensitivity classification task, not trial-level NRS intensity prediction. It reports 26 subjects initially; the dataset did not contain per-trial ratings and the task was changed to median-split `Pain_Threshold_7` sensitivity groups. The repository describes LOOCV balanced accuracy of 0.671 for P2 latency alone after excluding two apparent alpha-power outliers, with a 1000-permutation p-value of 0.032; full seven-feature models were near chance. This was not rerun here and must not be confused with our formulas' results. Source: https://github.com/Sai-sasi/eeg-pain-sensitivity-classifier.

## 7. Raw-data access and reproducibility audit

The 2025 laser-evoked-potential collection documents 678 healthy participants across nine experiments, with single-trial self-reported pain ratings stored in epoch metadata, BIDS EEG data, and supplementary preprocessed data/code. Source: https://www.nature.com/articles/s41597-025-05900-1 and https://github.com/XiangyueZhao/Supplement-Data. OpenNeuro/NEMAR dataset sources include ds005284, ds005285, ds005291 and ds005293. These source records were inspected in the web research environment.

Direct Python HTTP requests from the file-processing container to `raw.githubusercontent.com` and `github.com` failed with `Temporary failure in name resolution`. Thus no raw EEG or trial-level feature matrix was downloaded into the compute environment. The aggregate formula script was run locally and generated the included CSV/JSON outputs; no raw-signal trial validation was run.

## 8. Required full trial-level validation protocol

Once a genuine feature table is present, run `acute_trial_validation_pipeline.py`. Required columns: `subject_id`, `trial_id`, `pain_nrs` and numeric EEG feature columns extracted from the paired actual EEG trial. The script performs participant-grouped cross-validation and outputs out-of-fold trial predictions, regression and category metrics, per-subject errors, cluster-bootstrap comparisons against the training-fold mean baseline, feature association diagnostics and optional temporal holdout if `trial_order` is present.

For final confirmatory claims, additionally:

1. Use a nested participant-grouped CV or independent dataset after model selection.
2. Permute labels using a scheme that preserves subject, condition, intensity and trial structure as appropriate.
3. Report MAE, RMSE, bias, OOF R², calibration slope/intercept, within-1/within-2 NRS point, ordinal weighted kappa, balanced accuracy, macro-F1 and per-class confusion counts.
4. Report performance at each rating band, and compare feature sets on the same outer folds with participant-cluster bootstrap confidence intervals.
5. Run temporal holdout, cross-dataset replication, artifact sensitivity, channel/montage sensitivity, missing data audits, feature ablations, multiple-comparison correction and subgroup checks when sample size supports them.
6. Do not claim the scale is clinical or diagnostic until validated in a separate clinical acute-pain sample.

## 9. Final verdict

- **Aggregate signal trend:** passes descriptively for N2 magnitude, P2 and P2−N2 group means.
- **Formula fit to three aggregate midpoints:** simple linear equations yield near-perfect errors because they are calibrated directly to those same three points; not an independent validation.
- **Cross-study power law:** preserves rank but is badly miscalibrated in this higher-rating dataset.
- **Statistical confirmation:** not established from n=3 aggregates (exact rank permutation p=0.333).
- **Individual trial and subject categorization:** not completed; requires actual paired EEG feature rows and individual pain labels.
- **Clinical usefulness:** undetermined; do not use these candidate equations for clinical decisions.

## References

1. Li X et al. *Magnitude and Temporal Variability of Inter-stimulus EEG Modulate the Linear Relationship Between Laser-Evoked Potentials and Fast-Pain Perception.* Frontiers in Neuroscience. 2018;12:340. https://doi.org/10.3389/fnins.2018.00340
2. Wager TD, Matre D, Casey KL. *Placebo effects in laser-evoked pain potentials.* Brain, Behavior, and Immunity. 2006;20(3):219–230. https://doi.org/10.1016/j.bbi.2006.01.007
3. Zhao X et al. *A comprehensive EEG dataset of laser-evoked potentials for pain research.* Scientific Data. 2025;12:1536. https://doi.org/10.1038/s41597-025-05900-1
4. Sun G et al. *Detecting acute pain signals from human EEG.* Journal of Neuroscience Methods. 2021;347:108964. https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/
