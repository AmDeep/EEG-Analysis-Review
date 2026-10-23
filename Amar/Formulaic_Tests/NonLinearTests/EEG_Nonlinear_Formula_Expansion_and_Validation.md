# EEG-only pain score research: expanded nonlinear formulas and validation

**Run date:** 2026-10-23  
**Primary target:** trial-wise observed NRS, 0–10 (continuous regression)  
**Local development data:** 11,214 unique participant–area–block–trial keys from 117 participants, constructed from the locally available `Trials.pkl` feature table.  
**Independent public raw-EEG validation:** dataset selected and methods audited, but the external EEG files could not be downloaded in this runtime. No performance values in this report are attributed to the external dataset.

## Executive result

An expanded search tested 342 EEG-only formula specifications: every non-empty subset of the five standard band powers (delta, theta, alpha, beta, gamma) across 11 mathematically motivated transform families, excluding five degenerate single-band centered-log-ratio forms, plus composite EEG feature blocks. The fixed-penalty exploratory screen used Ridge alpha=10. The full nested analysis then selected across all 342 candidates inside the training data, using five participant-grouped outer folds and three participant-grouped inner folds, with alpha fixed at 10 for the primary formula-selection analysis.

| Metric | Outer-training-mean baseline | Full 342-formula nested selector |
|---|---:|---:|
| MAE (NRS points) | 2.2101 | **2.2023** |
| RMSE | 2.6342 | 2.6343 |
| R² | -0.0114 | -0.0115 |
| Pearson correlation | not clinically meaningful for fold means | 0.0173 |
| Spearman correlation | not clinically meaningful for fold means | 0.0132 |
| Prediction standard deviation | 0.0936 | 0.3298 |
| Calibration slope | unstable for near-constant fold means | 0.1376 |

The full-search nested model reduces MAE by only 0.0079 NRS points while RMSE is essentially unchanged/slightly worse, R² remains negative, correlation is near zero, and predictions remain highly compressed compared with the observed NRS SD of 2.619. Participant-cluster bootstrap of per-person MAE differences (model minus baseline) gives **-0.0053 points, 95% CI [-0.0290, 0.0191]**, which includes zero. The bootstrap probability that the difference is below zero is 0.672. This does **not** demonstrate a reliably useful pain scale.

The best fixed-alpha formula in the 342-row screen is theta+alpha fractional-power, MAE 2.1850, RMSE 2.6227, R² -0.0026, Pearson r 0.0631 and Spearman rho 0.0457. That is the best-of-342 exploratory minimum and therefore selection-optimistic; it is not the validation estimate. Its MAE improvement of 0.0252 points is tiny and its association with NRS remains near zero.

## 1. Data audit and target discipline

The supplied local `Trials.pkl` had 22,428 rows, 59 columns and 117 participants. There are 11,214 unique keys `(id, Area, B, iTrial)`; each key occurs exactly twice, with the same NRS but differing binary `pain` labels. The main analysis uses one observation per key by averaging EEG feature values within the key and keeping the shared NRS once. This is a transparent working assumption; the original event mapping should be confirmed before interpreting the result as the definitive trial unit. A row-level sensitivity analysis that preserves the paired rows is provided separately.

Predictors were EEG metrics only. Excluded from the NRS formula inputs: `NRS` except as outcome, binary `pain`, skin-conductance columns (`SC`, `SCH`, `SCF` variants), questionnaires, demographic variables, person IDs, block/trial/area metadata and other non-EEG fields. IDs were used only to keep every participant wholly inside one fold or the other. The local file contains precomputed summary EEG features, not the raw channel waveforms; it cannot support retroactive raw-signal ERP/topography/connectivity computation.

Observed local NRS range was 0–10; mean 3.3658, SD 2.6194. Because the paired-row semantics are not fully documented in the source package, results should be considered an analysis of the supplied feature table—not a claim that the original trial data have been completely reconstructed.

## 2. Formula space tested

Let each band power be `P_b > 0`, with `L_b = log(P_b)`. Training-fold standardization is `z_b = (L_b - mu_b) / s_b`, where the means/scales are fit only on training participants. A formula has the scalar form

`NRS_hat = clip(intercept + sum_j beta_j * phi_j(z, EEG_features), 0, 10)`.

The 31 non-empty subsets of delta/theta/alpha/beta/gamma were crossed with these families where mathematically valid:

1. **Log-linear:** linear Ridge combination of log band powers.
2. **Polynomial degree 2:** each term plus squares and pairwise products; all monomials through degree 2.
3. **Polynomial degree 3:** all polynomial monomials through degree 3, including cross-band interactions.
4. **Exponential/inverse-exponential:** `z`, `exp(clip(z,-3,3))`, and `exp(-clip(z,-3,3))` bases.
5. **Exponential interactions:** exponentials of pairwise sums and inverse exponentials.
6. **Sigmoid/saturation:** `z`, `tanh(z)`, logistic sigmoid and `z/sqrt(1+z^2)`.
7. **Signed fractional powers:** `z`, `sign(z)*sqrt(abs(z))`, `sign(z)*abs(z)^1.5`, and `z*abs(z)`.
8. **Reciprocal/saturation:** `z`, `1/(1+abs(z))`, and logistic/inverse-logistic terms.
9. **Additive cubic splines:** knots estimated from training folds only.
10. **Yeo–Johnson power transformation:** transformation fitted on training folds only.
11. **Relative power / CLR log-ratio:** `P_b/sum(P)` and centered log balances `log(P_b) - mean(log(P_subset))`.

Single-band CLR terms are mathematically constant zero, so the five degenerate specifications were skipped. Composite EEG-only linear and quadratic blocks also combine log band powers, relative powers, prespecified log-ratios (including theta/alpha, delta/alpha, beta/alpha, gamma/beta, and slow/fast), normalized spectral entropy, EEG mean/moments, peak frequencies and sample entropy. The result is 342 unique formula specifications.

All formulas are coefficient-based Ridge equations, not RF/AI classifier formulas. The fixed-alpha screen evaluates each formula's five-fold participant-held-out OOF predictions with alpha=10. Because taking the minimum from 342 results introduces selection optimism, the main evaluation is separately nested: within each outer training fold, three grouped inner folds select among all 342 specifications by mean MAE; the outer held-out people are used only for the final prediction. Alpha=10 was fixed for this primary full-space selection to make the exhaustive nested evaluation feasible and avoid adding an additional alpha-selection layer. Alpha sensitivity at 1/10/100 is available for selected formula screens/descriptive diagnostics but was not nested-tuned for all 342 families.

## 3. Ridge penalty sensitivity for leading nonlinear candidates

As a descriptive robustness check, the top 20 candidates ranked by the alpha=10 screen were rerun under Ridge penalties 1, 10 and 100 with the same participant-grouped five-fold splits. The theta+alpha fractional-power candidate stays near the top at all three values; changing regularization barely moves the performance and does not fix the weak association with NRS. Because this shortlist was originally selected at alpha=10, these results are a sensitivity check rather than an independent model-selection estimate.

| Ridge alpha | MAE | RMSE | R² | Pearson r | Spearman rho |
|---:|---:|---:|---:|---:|---:|
| 1 | 2.185650 | 2.623333 | -0.003093 | 0.063609 | 0.046106 |
| 10 | 2.184961 | 2.622745 | -0.002643 | 0.063138 | 0.045656 |
| 100 | 2.185313 | 2.622631 | -0.002556 | 0.057742 | 0.041868 |

Full results for the top 20 candidates x three penalties are in `top20_alpha_sensitivity.csv` and the `Alpha_Sensitivity_Top20` workbook sheet.

## 4. Outer-fold decisions from the full formula search

| Outer fold | Formula selected using inner grouped CV | Inner MAE | Outer MAE | Outer RMSE | Test participants |
|---:|---|---:|---:|---:|---:|
| 1 | gamma, degree-2 polynomial | 2.2268 | 2.0168 | in fold metrics file | 24 |
| 2 | composite EEG block, linear | 2.1214 | 2.3900 | in fold metrics file | 23 |
| 3 | theta, sigmoid/saturation | 2.2097 | 2.2176 | in fold metrics file | 23 |
| 4 | alpha, exponential/inverse-exponential | 2.2436 | 2.0389 | in fold metrics file | 23 |
| 5 | theta+beta+gamma, sigmoid/saturation | 2.2465 | 2.3493 | in fold metrics file | 24 |

Different folds select different bands and functional families. That instability is additional evidence against a fixed universally meaningful EEG score in this dataset. Each fold's fitted weights and predictions are supplied in CSVs.

## 5. Concrete coefficient example (exploratory, not deployable)

For transparency, the full-data descriptive fit of the best exploratory theta+alpha fractional-power family has intercept 3.365837 on the NRS scale. Define `z_theta` and `z_alpha` as standardized log band powers. For each basis column below, also standardize the generated basis using the mean and scale in `descriptive_equations_with_coefficients.md` / `descriptive_nonlinear_coefficients.csv`.

| Basis term | Ridge coefficient on standardized basis |
|---|---:|
| `z(theta)` | -0.160567 |
| `z(alpha)` | -0.642382 |
| `sign(z(theta))*sqrt(abs(z(theta)))` | +0.557302 |
| `sign(z(alpha))*sqrt(abs(z(alpha)))` | +0.366089 |
| `sign(z(theta))*abs(z(theta))^1.5` | -0.143189 |
| `sign(z(alpha))*abs(z(alpha))^1.5` | -0.512519 |
| `z(theta)*abs(z(theta))` | +0.022110 |
| `z(alpha)*abs(z(alpha))` | +0.857351 |

The score is `clip(3.365837 + sum(coef_j * (basis_j - basis_mean_j)/basis_scale_j), 0, 10)`. Exact basis means/scales and the 565-row full coefficient dump are provided. This fit uses the entire local dataset to disclose an example equation; it is **not** an out-of-sample coefficient set and should not be deployed. The actual outer-fold fitted coefficients, which differ by fold, are in `nested_fullspace_selected_coefficients.csv`.

## 6. Public dataset chosen for true independent testing

**Selected target: OpenNeuro `ds005293`, “95 By BP”; NEMAR copy `on005293`, version 1.0.0.** The NEMAR dataset page specifies 95 healthy participants, 80 trials per participant, four individually calibrated laser intensity levels with nominal NRS targets of 2, 4, 6 and 8, left/right-hand stimulation, and participant ratings after each stimulus. The study's observed trial NRS—not nominal target level or laser energy—must be the regression outcome.

Zhao et al. (2025), *Scientific Data*, describe the broader release of 678 healthy participants across nine experiments with high-density raw EEG and single-trial pain ratings. The paper describes BIDS data records and notes that stimulus intensity and subjective rating were saved in the preprocessed EEGLAB `EEG.epoch` structure. This is useful for method review because channel names/units, event tables, acquisition metadata, preprocessing and rating fields are documented. The paper describes filtering, baseline-corrected epochs, visual artifact rejection and ICA eye-artifact correction. Published procedures must be replicated or deviations explicitly documented.

Primary sources:
- Dataset data paper: https://doi.org/10.1038/s41597-025-05900-1
- OpenNeuro `ds005293`: https://openneuro.org/datasets/ds005293
- NEMAR `on005293` experiment/method page: https://nemar.org/dataset/on005293
- Author-provided processed files/code: https://github.com/XiangyueZhao/Supplement-Data

The complete dataset was listed at roughly 300.3 GB on NEMAR when checked. A direct local attempt to fetch the OpenNeuro/NEMAR manifest and GitHub supplement failed because the runtime could not resolve the external hostnames (`Temporary failure in name resolution`). Therefore, the selected public dataset is researched and ready for a reproducible external test, but **no external EEG rows were downloaded and no ds005293 performance values have been produced in this run**. This is a genuine access limitation; local grouped CV results must not be mislabelled external validation.

## 7. External analysis workflow prepared

- `export_lep_trials.m` is a MATLAB/EEGLAB helper to extract single-trial observed NRS, pre-stimulus/response band power, baseline-relative log-power change, entropy and optional Cz/CPz/FCz N2/P2 amplitudes/latencies from the preprocessed epochs. It deliberately stops if the NRS field is ambiguous; nominal intensity is never silently substituted.
- `run_external_ds005293.py` takes the verified trial-feature CSV and evaluates continuous NRS formulas with participant-grouped outer/inner folds. No subjective outcomes or stimulus metadata enter the predictor matrix.
- The downloaded-data manifest/version, exact event field, number of accepted trials, channel units, filter settings, baseline window, epoch window, artifacts and exclusion counts need to be captured when the actual source files are available.

A proper independent study should compare baseline MAE/RMSE and calibration, report participant-cluster intervals, inspect fold/experiment heterogeneity, freeze the family before the final test, and then test other experiments separately. Passing on experimental acute laser-evoked ratings would still not establish validity for chronic clinical pain.

## 8. Files

- `all_subset_formula_screen.csv` — all 342 fixed-alpha formula definitions/results.
- `top20_alpha_sensitivity.csv` — sensitivity analysis for the top 20 alpha=10 screen candidates at alpha 1/10/100.
- `nested_fullspace_oof_predictions.csv` — held-out prediction for every one of 11,214 unique rating keys from the full 342-family nested selector.
- `nested_fullspace_outer_fold_metrics.csv`, `nested_fullspace_selected_formula_per_fold.csv`, `nested_fullspace_selected_coefficients.csv` — fold-level values, selected formulas and weights.
- `participant_cluster_bootstrap.csv` — participant-level uncertainty summary; see `fullspace_nested_metrics.json` for the precise values.
- `descriptive_nonlinear_coefficients.csv` and `descriptive_equations_with_coefficients.md` — exploratory full-data coefficient bases, means and scales.
- `formula_search_results.xlsx` — consolidated workbook with formula screen, nested validation and coefficient sheets.
- `Trials.pkl`, `Subjects.csv`, `SubjectAreas.csv` — local source files for reproducibility; the latter two were not used as formula predictors.
- `run_nonlinear_formula_search.py`, `run_full_nested_formula_search.py`, `aggregate_full_nested_formula_search.py`, `export_descriptive_coefficients.py` — reproducible analysis scripts.
- `dataset_source_and_methods.md`, `EXTERNAL_DATASET_RUNNER.md`, `export_lep_trials.m`, `run_external_ds005293.py` — public-source documentation and external runner.

## Conclusion

The broad nonlinear screen does identify equations with slightly lower local MAE, but the full nested analysis across all 342 formula specifications does not meaningfully beat the mean baseline. With near-zero correlations, negative R², extremely compressed predictions and confidence intervals that include zero, this dataset does **not** validate a formulaic EEG-only continuous pain scale. The most important next test is the independently sourced raw laser-evoked EEG dataset; it remains unscored until its signals are accessible.
