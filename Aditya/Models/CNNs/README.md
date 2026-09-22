# NUROVO EEG Pipeline

## Pipeline

```text
EEGLAB .set + label CSV
        |
        v
build_epochs.py
        |
        +--> nurovo_epochs.npz
        |
        +--> dwt_features.py
        |        |
        |        +--> nurovo_dwt.npz
        |
        +--> train_eegnet.py
                 |
                 +--> subject-grouped CV
                 +--> train-only normalization
                 +--> EEGNet / EEG regression
                 +--> OOF predictions
                 +--> evaluation metrics
```

## Important leakage rule

For every fold, all preprocessing statistics are calculated using the
training subjects only and then applied unchanged to validation subjects.

`StratifiedGroupKFold` uses `global_subject` as the group, so a subject cannot
occur in both training and validation.

For the original classification pipeline, `q33` and `q67` are also calculated
from training subjects only and then applied unchanged to validation subjects.

The current laser-power regression pipeline does **not** use protocol, dataset,
or epoch position as model inputs.

## DWT

`db4`, level 5, using:

* A5
* D5
* D4
* D3
* D2
* D1

For 250 Hz sampling, the approximate ranges are:

* A5: 0–3.9 Hz
* D5: 3.9–7.8 Hz
* D4: 7.8–15.6 Hz
* D3: 15.6–31.3 Hz
* D2: 31.3–62.5 Hz
* D1: 62.5–125 Hz

DWT features are computed for Fz, Cz, Pz, C3, C4 by default.

## EEGNet

Input:

`N x 1 x channels x time`

Block 1:

temporal convolution → batch normalization → depthwise spatial convolution
→ batch normalization → ELU → average pooling → dropout

Block 2:

depthwise temporal convolution → pointwise convolution → batch normalization
→ ELU → average pooling → dropout

The current regression model can optionally add:

* an additional depthwise-separable temporal block
* an EEG-derived signal-feature branch
* an ensemble of EEG-only models
* output quantization

---

# Current Laser-Power Regression

The current objective is to predict the exact `laser_power` value from each
single-trial EEG epoch.

The primary clinical metric is the percentage of predictions within:

`±0.5 laser-power units`

### Validated result

The original dataset contains:

* 10,984 epochs
* 124 subjects
* 28 EEG channels
* 250 time samples

The fully evaluated EEG-only ensemble achieved:

**67.72% within ±0.5**

with:

* MAE: 0.5392
* RMSE: 0.6841
* 5-fold subject-grouped CV
* 50 epochs
* batch size 64
* Huber loss
* half-step output quantization
* fold-held-out ensemble-weight selection

No protocol or trial-order information is used.

### Expanded-data result

An expanded tensor was created containing:

* **29,515 epochs**
* **678 subjects**
* **9 datasets**

The strongest result so far is:

**75.85% within ±0.5**

with:

* MAE: 0.4555
* RMSE: 0.6077
* 3-fold subject-grouped CV
* 20 epochs
* batch size 256
* wider/deeper EEG CNN
* EEG-derived feature branch
* EEG-only ensemble
* fold-held-out ensemble weighting

This result is **preliminary** because the planned 5-fold confirmation has not
yet been completed.

The expanded tensor is:

`CNN/nurovo_epochs_all.npz`

---

# Important Finding

A protocol/dataset/epoch prior reached 70.40%, but it was rejected for the
clinical model because it relies on trial-order/protocol information rather
than patient EEG.

The current model therefore uses **EEG only**.

---

# Next Steps

1. **Complete 5-fold CV on the 29,515-epoch expanded dataset.**
   This is the most important next experiment and will determine whether the
   75.85% result holds under the same evaluation standard as the validated
   67.72% result.

2. **Move expanded-data training to GPU.**
   CPU training became prohibitively slow for the full 5-fold/50-epoch run.

3. **Run the 2-second EEG experiment on GPU.**
   `nurovo_epochs_all_2s.npz` is already available, but the first run was
   stopped because of CPU runtime.

4. **Keep ensemble evaluation fold-held-out.**
   Do not select ensemble weights using the same OOF predictions used for the
   final reported score.

5. **Compare generalization across subjects and datasets.**
   Report subject-level performance and investigate whether performance is
   consistent across the nine expanded datasets.

---

# Smoke Test

Before a full run, create synthetic data with the same shape as the real
tensor and run one or two epochs.

Verify:

1. model forward pass
2. loss decreases without NaNs
3. fold splits contain disjoint subjects
4. train-derived preprocessing is applied to validation only
5. normalization statistics come only from training
6. output shapes and filenames are stable
7. OOF predictions contain every held-out subject exactly once
