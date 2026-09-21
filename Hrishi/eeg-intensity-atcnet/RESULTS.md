# Results

## What actually ran, and why not the PyTorch ATCNet

This sandbox's outbound network is allowlisted per-domain: `pypi.org`, `download.pytorch.org`,
`osf.io`, and the OpenNeuro S3 bucket are all blocked (`403 blocked-by-allowlist`), and `torch`/
`scikit-learn`/`scipy`/`mne` are not preinstalled — only `numpy`/`pandas`/`matplotlib`. So the
raw-EEG `ATCNetRegressor` (models/atcnet.py, real PyTorch, attention + TCN) could not be run here,
and neither could the OpenNeuro/OSF download scripts.

What I could do, and did: `Aditya/Feature Extraction/features_combined.csv` turned out to
**already contain features from all 9 Zhao mega-study experiments** (29,515 trials, 678
subjects) — Aditya had already done the multi-dataset combination I was planning to do myself.
So I wrote `models/numpy_cnn.py`, a real CNN (2-layer dilated Conv1d stack + global pooling +
dense head, ATCNet-inspired but without the multi-head attention / TCN sliding-window branches,
implemented with hand-derived backprop and Adam, gradient-checked against finite differences
before trusting it — see `models/gradcheck.py`, passed with ~1e-7 relative error) and trained it
for real, in this sandbox, on the real 29,515-trial dataset.

**Everyone reading this should treat the numbers below as real but from a simplified
architecture** — the full ATCNet in `models/atcnet.py` is the intended model; run it on a machine
with normal PyPI access (yours almost certainly has one) for the actual result.

## Setup

- Data: `Aditya/Feature Extraction/features_combined.csv`, target `laser_power` (Joules, the
  physical stimulus intensity, range ~1.0–4.5J across the 9 sub-experiments).
- Split: grouped 80/20 by `dataset_subject` (not just `subject` — the existing baseline scripts
  group by `subject` alone, which silently merges different real people who happen to share a
  `sub-001`-style id across different sub-experiments; my split avoids that).
- Preprocessing: median imputation (fit on train only) + standardization (fit on train only),
  matching the existing baseline scripts' approach.
- Two feature sets: **EEG-only** (26 features, drops `rating`) and **+rating** (27 features,
  matches the teammates' baseline scripts exactly, which include the subjective pain rating as an
  input feature alongside the EEG features).

## Results

| Model | Features | MAE | RMSE | R² | Pearson r |
|---|---|---:|---:|---:|---:|
| Linear Regression *(existing baseline)* | +rating | 0.5022 | 0.6500 | 0.2036 | – |
| Ridge Regression *(existing baseline)* | +rating | 0.5022 | 0.6500 | 0.2036 | – |
| KNN *(existing baseline)* | +rating | 0.4926 | 0.6371 | 0.2349 | – |
| **NumPy CNN (this run, 40 epochs)** | **+rating** | **0.4611** | **0.6109** | **0.2198** | **0.5030** |
| Random Forest *(existing baseline)* | +rating | 0.4471 | 0.5989 | 0.3238 | – |
| Gradient Boosting *(existing baseline)* | +rating | 0.4524 | 0.5960 | 0.3303 | – |
| Extra Trees *(existing baseline, best)* | +rating | 0.4480 | 0.5955 | 0.3315 | – |
| NumPy CNN (this run, 40 epochs) | EEG-only, no rating | 0.4773 | 0.6321 | 0.1646 | 0.4330 |

(The existing-baseline rows are from `Hrishi/model_comparison.md`, on a similar but not identical
split — same grouped-holdout methodology, different random partition — so treat this as "same
ballpark," not an exact controlled comparison.)

## Reading these numbers honestly

- The CNN beats plain Linear/Ridge regression by a real margin, and lands close to KNN — a
  believable result for a small non-attention CNN on 27 tabular features.
- **It does not beat the tree ensembles** (Extra Trees/Gradient Boosting/Random Forest), which
  remain the strongest models found so far on this feature set. Tabular data with this few,
  already-hand-engineered features tends to favor tree ensembles over neural nets — this is a
  known pattern in the ML literature, not a red flag specific to this run. It's a real signal,
  but not yet a case for switching your team off the tree-ensemble baseline.
- Training longer (150 vs 40 epochs) made training loss keep dropping (0.41 → 0.39) but made test
  performance *worse* (test R² 0.22 → 0.19) — the model overfits past ~40 epochs. Reported numbers
  above are the 40-epoch run; a version with real early stopping would be a needed next step, not
  just more epochs.
- Including `rating` as a feature helps noticeably (R² 0.16 → 0.22) — worth flagging since `rating`
  is a separately-measured human judgment, not something you'd have at prediction time in most
  real applications (you're trying to predict the stimulus from the brain response, and having the
  person's own pain rating as an input is a different, easier problem: rating→intensity, not
  brain→intensity). The EEG-only row is the more honest number for the task as originally framed.

## What a full run (your machine, with PyPI access) would add

1. The real `ATCNetRegressor` on raw EEG (only ds005284's raw BDF files are in your zip; the other
   8 experiments would need `data/download_openneuro.py`) — attention + TCN, not just conv, and
   operating on the actual waveform instead of 27 pre-computed scalars. This is the version that
   can plausibly beat the tree ensembles, since a CNN's advantage over trees shows up on raw
   signal, not on already-summarized tabular features.
2. Tiemann et al. 2018 (OSF) folded in for cross-protocol generalization, as you asked for.
3. Proper LOSO cross-validation (`train.py --loso`) instead of a single grouped split, and early
   stopping on a validation set instead of a fixed epoch count.
