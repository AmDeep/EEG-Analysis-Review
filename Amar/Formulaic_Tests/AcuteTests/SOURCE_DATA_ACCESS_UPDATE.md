# Source data access update

## What was inspected

The GitHub web view exposes `participants_singletrial_26ByBiosemi.csv` from the supplementary repository for the 26-participant Biosemi experiment. The visible table has 417 lines including its header (416 trial rows) and columns `ID`, `Trial_Num`, `laser_power`, and `Painrating`. Its displayed `Painrating` values span 0–10; displayed laser power values span 2.75–4.5. Source URL:

https://github.com/XiangyueZhao/Supplement-Data/blob/main/CSV/participants_singletrial_26ByBiosemi.csv

This is a real trial-level **label/stimulus table**, but it does not contain EEG amplitudes, spectral powers, or other predictors required to apply the proposed EEG score formulas. It therefore cannot serve by itself as the paired input to a formula or validation model.

## What was not obtained

The browser can expose this table's text, but the local processing runtime could not resolve `raw.githubusercontent.com` / `github.com` to download it and could not retrieve the paired EEG epochs or the binary MATLAB files that contain EEG-derived data. No row-level EEG-to-rating pairing was therefore created. No individual trial pain predictions, subject-level score categories, cross-validated errors, or trial-level significance tests are claimed in this package.

## Evidence boundaries

- `aggregate_formula_results/` contains completed calculations from actual published group-average N2/P2 values and explicitly assigned pain-band midpoints.
- `previous_project_outputs/` contains earlier reports/benchmarks; the current authoritative report is at the package root.
- `acute_trial_validation_pipeline.py` accepts genuine trial-level EEG-derived features paired with each actual rating. It must not be run with group means or the label-only CSV as if they contained EEG predictors.
- `trial_level_feature_input_template.csv` contains column names only; it is not a dataset.

## Recommended next input

To finish individual-level validation, provide a local feature table extracted from actual EEG epochs, one row per real trial, with at minimum `subject_id`, `trial_id`, `pain_nrs`, and EEG-derived features such as `n2_uv`, `p2_uv`, `z_alpha`, `z_beta`, plus `trial_order` if available. Alternatively, provide the actual EEG files and the matching event/epoch metadata. The pipeline then runs participant-grouped out-of-fold evaluation and compares formulas on held-out subjects.
