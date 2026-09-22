# NUROVO EEG pipeline

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
                 +--> train-only tertile thresholds
                 +--> train-only channel normalization
                 +--> z clipping
                 +--> weighted CE
                 +--> EEGNet
                 +--> OOF predictions
                 +--> subject bootstrap CI
```

## Important leakage rule

The final classifier labels are based on laser power. For every fold,
`q33` and `q67` are calculated from the training subjects only and then
applied unchanged to validation subjects.

Normalization is also fitted only on training epochs.

`StratifiedGroupKFold` uses `global_subject` as the group, so a subject
cannot occur in both training and validation.

## DWT

`db4`, level 5, using:

- A5
- D5
- D4
- D3
- D2
- D1

For 250 Hz sampling, the approximate ranges are:

- A5: 0–3.9 Hz
- D5: 3.9–7.8 Hz
- D4: 7.8–15.6 Hz
- D3: 15.6–31.3 Hz
- D2: 31.3–62.5 Hz
- D1: 62.5–125 Hz

DWT features are computed for Fz, Cz, Pz, C3, C4 by default.

## EEGNet

Input:

`N x 1 x channels x time`

Block 1:
temporal convolution -> batch normalization -> depthwise spatial
convolution -> batch normalization -> ELU -> average pooling -> dropout

Block 2:
depthwise temporal convolution -> pointwise convolution -> batch
normalization -> ELU -> average pooling -> dropout

Then a linear classifier.

## Smoke test

Before a full run, create synthetic data with the same shape as the real
tensor and run one or two epochs. The key things to verify are:

1. model forward pass
2. loss decreases without NaNs
3. fold splits contain disjoint subjects
4. train-derived thresholds are applied to validation
5. normalization statistics come only from training
6. output shapes and filenames are stable
