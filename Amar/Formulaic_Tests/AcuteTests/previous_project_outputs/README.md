# EEG-only Pain Scale Project — Evidence and Validation Status

## Current results first

The latest acute-data validation ledger is `Acute_EEG_All_Test_Results.md` and the Word version is `Acute_EEG_All_Test_Results.docx`. They contain the current reanalysis of published acute-pain EEG performance values, source-reported acute EEG findings, rating contrasts, test-by-test execution status and explicit limits.

Key machine-readable files:
- `acute_validation_test_registry.csv`: 30 test/evidence items and their execution statuses.
- `acute_validation_all_results.json`: full compact result object and provenance.
- `acute_validation_reproduced_stats.json`: direct re-run output from the reproducibility script.
- `published_table2_performance_rows.csv`: the 16 performance rows transcribed from Sun et al. Table 2.
- `published_benchmark_factorized_rows.csv`: the same rows with training/test subject and stimulation-site factors.
- `published_benchmark_stratified_results.csv`: descriptive subgroup summaries.
- `published_patient_nrs_summary.csv` and `patient_pain_contrast_descriptive.csv`: published pain-rating medians and arithmetic contrasts.
- `formula_validation_status.csv`: status of the 17 candidate formula/model families.
- `acute_validation_reproduce.py`: code that reproduces the published-table summary and exploratory tests.

## Crucial distinction

No raw EEG recordings or trial-level EEG feature rows were loaded during the latest execution. Public data-host access failed DNS resolution in this runtime. The statistics computed from Sun et al. are a reanalysis of rounded, published AUC/accuracy rows, not a refit of the proposed EEG-to-0–10 intensity formulas. Source-reported latency and group statistics are labeled as source-reported and were not recalculated from raw trials. There are no fabricated participants, predictions, coefficients or significance values in this package.

`EEG_Pain_Scale_Workbook.xlsx` is the earlier project workbook and has **not** been updated with the latest empirical-ledger sheets. Use the current CSV/JSON/Markdown/Word files for the newest results.

## Principal sources

- Sun et al. (2020), *Detecting acute pain signals from human EEG*: https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/
- Zhao et al. (2025), *A comprehensive EEG dataset of laser-evoked potentials for pain research*: https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/
- NEMAR/OpenNeuro ds005293 (95 By BP): https://nemar.org/dataset/on005293
- NEMAR/OpenNeuro ds005291 (65 By ANT): https://nemar.org/dataset/on005291
- OSF source for the healthy-volunteer acute laser dataset cited by Sun et al.: https://osf.io/bsv86/
