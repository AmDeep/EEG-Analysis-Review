# Independent dataset plan: OpenNeuro ds005293 / NEMAR on005293

## Why this dataset was selected

**Primary target:** the “95 By BP” experiment, OpenNeuro dataset `ds005293`, version 1.0.0 (NEMAR copy `on005293`). The dataset page describes 95 healthy participants receiving 80 laser-stimulation trials each across four individually calibrated intensity levels, nominally corresponding to NRS 2, 4, 6 and 8; participants rated perceived pain after each stimulus. The broad companion collection contains nine related experiments and 678 participants overall. This is a stronger first intensity-scoring test than a fixed-intensity experiment because it spans multiple levels and includes observed single-trial ratings.

Sources:
- Dataset landing page: https://openneuro.org/datasets/ds005293/versions/1.0.0
- NEMAR dataset page: https://nemar.org/dataset/on005293
- Dataset paper (data/methods): https://doi.org/10.1038/s41597-025-05900-1
- Published preprocessing and MATLAB supplements: https://github.com/XiangyueZhao/Supplement-Data
- The authors report 1–100 Hz filtering, notch filtering around 50 Hz, epochs from −1 s to +2 s, baseline correction on −1 to 0 s, visual artifact rejection and ICA eye-artifact correction. Single-trial stimulus intensity and subjective rating are available in the EEGLAB `EEG.epoch` structure of preprocessed `.set` files.

## Important methodological caveat

Do not substitute the nominal calibration labels (2, 4, 6, 8) or delivered laser energy for observed NRS. Those can be used in a separately reported stimulus-only diagnostic baseline, but must not enter an EEG-only score formula. Verify that the `EEG.epoch` field exported as `NRS` is the observed rating and is aligned to the same EEG epoch. If an export contains multiple pain/rating-like fields, the MATLAB script halts rather than guessing.

## Files in this package

- `export_lep_trials.m`: EEGLAB/MATLAB feature extraction helper for preprocessed epoched `.set` files. Calculates five-band power in a pre-stimulus window (−800 to −100 ms) and response window (+100 to +1000 ms), log change per band, normalized band entropy, and Cz/CPz/FCz N2/P2 peak amplitude and latency features. It retains stimulus-intensity metadata only as a non-predictor field for diagnostics.
- `run_external_ds005293.py`: Python grouped-nested-CV evaluator for the exported trial feature table. It tests log band power, baseline-relative log changes, power ratios/entropy, LEP component formulas, and combinations using Ridge/Elastic Net coefficient formulas.
- `run_formula_expansion.py`: expanded formula research on the locally mounted feature dataset (included local data copy is `Trials.pkl`).

## How to run externally

1. Download `ds005293` and/or obtain the authors' preprocessed EEGLAB `.set` files. The paper states the complete raw `ds005293` dataset is large (the NEMAR page lists roughly 300 GB for all data); the supplementary repository may be substantially lighter but its file sizes and exact per-file contents should be checked before download.
2. Open EEGLAB and run `export_lep_trials.m` after setting `DATA_ROOT` to the folder containing the preprocessed epoched `.set` files. If the rating field cannot be uniquely inferred, the script prints the fields and stops; set `RATING_FIELD` to the exact field representing *observed NRS*. Do not choose an intensity/dose field instead.
3. Install Python dependencies (`numpy pandas scipy scikit-learn`). Run:

   ```bash
   python run_external_ds005293.py --input ds005293_trial_features.csv --output-dir external_ds005293_results
   ```

4. Freeze formula families and hyperparameter ranges before reading scores from this independent dataset. The runner uses participant-grouped outer CV and participant-grouped inner CV; preprocessing is fit within each training fold. Inspect external metrics, OOF predictions and coefficient signs; do not select the final reported score solely by lowest MAE after searching the external data.
5. For stronger transport validation, repeat leave-one-experiment-out across the nine related experiments. These differ in acquisition systems, stimulus timing, intensity designs and channel montages, so assess harmonization and per-experiment performance rather than pooling and hiding heterogeneity.

## What was and was not run in this environment

Dataset and methods were verified through the public dataset landing page and the Scientific Data paper. A direct attempt to access both the NEMAR manifest and the public supplement from the analysis runtime failed with DNS name-resolution errors (`Temporary failure in name resolution`). Therefore, this report does **not** claim any fitted coefficient, CV metric or statistical significance from `ds005293`. The additional numerical results in the main report are from the precomputed feature dataset already available locally. This is a real access limitation, not evidence that the public dataset is unavailable: the links above remain the authoritative access route.

## Planned external acceptance criteria

- Correct observed NRS labels and event alignment verified before feature extraction.
- Participant-level grouped 5-fold nested CV with MAE/RMSE/R², correlation, calibration intercept/slope and rating-range coverage.
- Participant-cluster uncertainty intervals for all score-vs-baseline differences.
- Baselines: training-fold mean; per-person calibration baseline only if a predeployment calibration phase is explicitly part of the proposed use case; optional stimulus-only baseline reported separately (not an EEG score).
- Primary families frozen before scoring: log band power; baseline-relative log change; band ratios/entropy; N2/P2 amplitudes/latencies; and an interpretable combined equation.
- External transport check: leave-one-experiment-out or train-on-one-experiment/test-on-another, with no participant overlap.
- Report failure rather than recalibration on test people if a formula fails. Recalibration is a separate, explicitly defined use case.
