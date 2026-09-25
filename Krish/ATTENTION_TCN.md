# Attention-TCN laser-intensity model

`09_attention_tcn_loso.py` predicts laser intensity for every EEG feature interval. Laser power is modeled as the 15 discrete experimental set-points from 1.00 to 4.50 J. The output includes the predicted set-point and the probability-weighted expected intensity, which supports error metrics in Joules.

## Model

- Input: 24 EEG-derived features. Identifiers, acquisition metadata, epoch number, and behavioral rating are excluded.
- Temporal context: a causal 16-interval window within one participant. Earlier intervals are used; future intervals are never used.
- Encoder: four residual temporal-convolution blocks with exponentially increasing dilation.
- Attention: a learned weight over the valid encoded intervals, followed by a classification head.
- Missing values and scaling: training-fold median imputation and standardization only.
- Objective: class-weighted cross-entropy to reduce dominance by common laser set-points.

## Validation

The outer evaluation is true leave-one-subject-out cross-validation with `LeaveOneGroupOut`. The group key is `dataset + subject` because subject labels repeat across the nine source datasets. Every interval from the held-out participant remains outside model fitting and preprocessing.

An inner subject-grouped validation split controls early stopping. This validation split is drawn only from the outer training participants.

The source contains 678 dataset-specific participants, so complete LOSO training fits 678 networks and can take substantial time. Fold results are saved independently and an interrupted run resumes by skipping completed folds.

## Setup and run

From the repository root:

```powershell
.\.venv\Scripts\python.exe -m pip install -r .\Krish\requirements_attention_tcn.txt
.\.venv\Scripts\python.exe .\Krish\09_attention_tcn_loso.py
```

Run a small smoke test first:

```powershell
.\.venv\Scripts\python.exe .\Krish\09_attention_tcn_loso.py --fold-start 1 --fold-end 2 --epochs 2 --output-dir .\Krish\attention_tcn_smoke
```

Restrict a run to specified datasets while retaining subject-level LOSO:

```powershell
.\.venv\Scripts\python.exe .\Krish\09_attention_tcn_loso.py --datasets ds005293 ds005280
```

For parallel jobs, assign disjoint fold ranges and distinct output directories, then combine their per-fold CSV and JSON fragments before summarizing. Do not make different jobs write to the same output directory.

## Outputs

- `oof_interval_predictions.csv`: actual, class prediction, expected intensity, and attention peak lag for each completed held-out interval.
- `fold_metrics.csv`: accuracy, balanced accuracy, macro-F1, MAE, RMSE, and tolerance accuracy by participant.
- `overall_metrics.json`: pooled out-of-fold metrics and an `is_complete_loso` flag.
- `run_config.json`: features, class values, hyperparameters, and device.
- `folds/`: resumable prediction and metric fragments for each participant.

The pooled metrics are final only when `is_complete_loso` is `true`.
