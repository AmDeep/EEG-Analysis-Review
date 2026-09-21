# Results

## What actually ran, where

Two separate runs went into this, in two different environments:

1. **My sandbox** (no PyPI/OpenNeuro/OSF access, no torch/scikit-learn/scipy/mne preinstalled) —
   ran `models/numpy_cnn.py`, a real CNN with hand-derived backprop (gradient-checked against
   finite differences, `models/gradcheck.py`, ~1e-7 relative error) but a simplified architecture:
   2-layer dilated Conv1d stack + global pooling + dense head, **no attention, no TCN**.
2. **Your machine** — ran the actual `TabularATCNetRegressor` from `models/atcnet.py` (real
   PyTorch, the full attention + TCN sliding-window architecture) via `run_tabular_baseline.py`.
   This is the result that matters most — it's the architecture the whole project was about.

## Setup

- Data: `Aditya/Feature Extraction/features_combined.csv`, target `laser_power` (Joules, the
  physical stimulus intensity, range ~1.0–4.5J across the 9 sub-experiments, 29,515 trials, 678
  subjects — all 9 Zhao mega-study experiments, already combined by Aditya).
- Split: grouped 80/20 by `dataset_subject` (not just `subject` — the existing baseline scripts
  group by `subject` alone, which silently merges different real people who happen to share a
  `sub-001`-style id across different sub-experiments; both runs below avoid that).
- Preprocessing: median imputation (fit on train only) + standardization (fit on train only),
  matching the existing baseline scripts' approach.
- `run_tabular_baseline.py` additionally min-max normalizes the label to [0,1] (fit on train
  only) before computing MAE/RMSE, so those two numbers are **not on the same scale** as the
  Joules-based baseline numbers below — R² and Pearson r are scale-invariant and compare
  directly; MAE/RMSE for that row are converted to an approximate Joules figure using the ~3.5J
  label range (exact conversion needs the run's actual train min/max, which the script doesn't
  currently print — a nice follow-up fix).

## Results (all include `rating` as a feature, matching the teammates' baseline scripts)

| Model | MAE (J) | RMSE (J) | R² | Pearson r |
|---|---:|---:|---:|---:|
| Linear Regression *(existing baseline)* | 0.5022 | 0.6500 | 0.2036 | – |
| Ridge Regression *(existing baseline)* | 0.5022 | 0.6500 | 0.2036 | – |
| My NumPy CNN (conv only, no attention/TCN) | 0.4611 | 0.6109 | 0.2198 | 0.5030 |
| KNN *(existing baseline)* | 0.4926 | 0.6371 | 0.2349 | – |
| **Real ATCNet — your run, 50 epochs** | **~0.46 (approx)** | **~0.59 (approx)** | **0.2516** | **0.5132** |
| Random Forest *(existing baseline)* | 0.4471 | 0.5989 | 0.3238 | – |
| Gradient Boosting *(existing baseline)* | 0.4524 | 0.5960 | 0.3303 | – |
| Extra Trees *(existing baseline, best)* | 0.4480 | 0.5955 | 0.3315 | – |

(The existing-baseline rows are from `Hrishi/model_comparison.md`, on a similar but not identical
split — same grouped-holdout methodology, different random partition — so treat cross-row
comparisons as "same ballpark," not an exact controlled comparison.)

Raw training log from the real ATCNet run (50 epochs, ~1.6s/epoch on CPU, loss flattened by
~epoch 20 with no divergence):

```
epoch   0: train_loss=0.0661
epoch  10: train_loss=0.0167
epoch  20: train_loss=0.0159
epoch  30: train_loss=0.0157
epoch  40: train_loss=0.0155
epoch  49: train_loss=0.0153
Test metrics (n=5445): MAE=0.1304  RMSE=0.1675  R2=0.2516  Pearson r=0.5132 (p=0.00e+00)
```

## Reading these numbers honestly

- **The real ATCNet (attention + TCN) beat the simplified NumPy CNN** (R² 0.2516 vs 0.2198) — the
  attention/TCN machinery is adding real signal over plain dilated convolution, as hoped.
- **It still doesn't beat the tree ensembles** (Extra Trees/Gradient Boosting/Random Forest,
  R² 0.32–0.33). It's the closest neural net to them so far, but tabular data with this few,
  already-hand-engineered features tends to favor tree ensembles over neural nets — a known
  pattern in the ML literature, not a red flag specific to this run.
- Both CNN runs used `rating` (the subjective pain rating) as an input feature, matching the
  existing baseline scripts exactly, for a fair comparison. That's the easier version of the
  problem (rating→intensity, not brain→intensity) — `--drop-rating` on either script gives the
  more honest "predict the stimulus from brain signal alone" number, which will likely be lower
  for the real ATCNet too (it was for the NumPy CNN: R² dropped from 0.22 to 0.16). Not run yet.
- The NumPy CNN started overfitting past ~40 epochs in my sandbox; the real ATCNet's 50-epoch run
  shows no such divergence (loss is still flat/slightly improving at epoch 49), so it may have a
  bit more headroom with more epochs or a learning-rate schedule — untested.

## What would still add real value

1. `--drop-rating` on the real ATCNet — the more honest EEG-only number.
2. The real `ATCNetRegressor` on **raw EEG** instead of pre-extracted features (only ds005284's
   raw BDF files are in your zip; the other 8 experiments would need
   `data/download_openneuro.py`) — this is the version where a CNN's structural advantage over
   trees should actually show up, since it isn't limited to features a human already decided to
   hand-compute.
3. Tiemann et al. 2018 (OSF) folded in for cross-protocol generalization, as originally asked for.
4. Proper LOSO cross-validation (`train.py --loso`) instead of a single grouped split, and
   printing the exact train min/max in `run_tabular_baseline.py` so the Joules conversion above
   doesn't have to be approximate.