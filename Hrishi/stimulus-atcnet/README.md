# Zhao + Tiemann: continuous stimulus regression

**Target: physical laser energy (joules) from poststimulus EEG.** This predicts delivered stimulus intensity after a stimulus; it does not predict future pain or use pain ratings as intensity labels.

## Status and results

This package implements a new raw-EEG experiment. **No Zhao–Tiemann training results are available yet, and R² ≥0.75 is not established.** In the execution environment, OSF/PyPI requests returned HTTP 403, PyTorch and MNE were absent, and available uploads contained feature tables rather than the required raw EEG. The synthetic tabular smoke test passed; syntax compilation passed. Downloads, raw EEG preprocessing, and ATCNet training remain unverified end to end. Synthetic tests are not research results.

Previous Zhao feature-table experiments reached approximately **R² 0.44** on held-out participants. The earlier Gozzi result (~0.73) predicted calibrated subjective pain, a different target; it is not evidence for stimulus prediction. Gozzi is excluded here because the supplied table lacks the required physical-intensity labels.

## What is implemented

- ATCNet scalar regression using Braindecode's implementation, with a linear output and MSE; shortened temporal kernels/pooling for brief evoked-response epochs.
- Spline + Ridge using common EEG band power and evoked-response features.
- ATCNet + Spline/Ridge weighted ensemble, with weights learned from development out-of-fold predictions.
- Extra Trees comparator and a training-mean baseline.

The default benchmark **requires both Zhao and Tiemann Munich** and trains on participants from both. All sessions of a person stay together. Hyperparameters use three participant-grouped development folds; neural early stopping uses a further participant split inside the training fold. Final participants are held out before tuning. Output includes pooled and per-dataset R², MAE/RMSE in joules, and participant-bootstrap pooled R² intervals. Report per-dataset results: pooled R² can reflect study differences. This is a candidate comparison, not proof these are universally the best two models.

## Run

Use Python 3.11 in a fresh environment. Commands below assume this folder is the working directory. Dependencies are specified but installation was not verified here. A CUDA GPU is recommended for training; CPU is supported but slower.

```bash
python -m venv .venv
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python pipeline.py smoke --neural --device cpu

python pipeline.py download --source ds005473 --dest data/zhao
python pipeline.py download --source tiemann --dest data/tiemann
python pipeline.py inventory --root data/zhao --out zhao_inventory.json
python pipeline.py inventory --root data/tiemann --out tiemann_inventory.json
```

If the public download helper fails, download through the source websites below. Extract archives first. The OSF helper traverses project file folders, not linked child projects; confirm the downloaded collection actually contains Munich raw EEG and its event metadata. Do not proceed on an empty or incomplete download.

### One required data-mapping step

Copy `sources.example.json` to `sources.json`. Inspect the inventories, dataset documentation, and event JSON descriptions. The exact event schemas could not be retrieved here, so **the example intentionally refuses to run until verified**. Set:

1. `root` to the correct extracted BIDS root, `energy_column` to physical energy, and `event_filter` to laser-stimulus rows only. If values are event codes, supply `energy_map` mapping their exact strings to documented joules; do not infer energies from trial order or pain ratings. Numeric joules need no map. Confirm onset is stimulus onset in seconds relative to raw recording start; use `onset_offset_s` only for documented offsets.
2. Tiemann selection to the **internal Munich cohort**, excluding its Beijing replication sample unless participant overlap with Zhao has been resolved. Use a cohort-specific root, `events_glob`, or `subjects` whitelist. The paper describes four Munich energies (2.5, 3, 3.5, 4 J); verify what the actual raw release contains. Prefer all four, not a derived subset selected for a different analysis.
3. Audit participant overlap across sources. Use `subject_map` to give the same person the same global ID across datasets. A different dataset prefix does not establish independence. Set `cohort_verified` and `schema_verified` to true only after these checks.

Supported raw formats beside each BIDS events file: BrainVision `.vhdr` (with `.eeg`/`.vmrk` companions), EDF, BDF, EEGLAB `.set` (with `.fdt` if needed), and FIF. Non-BIDS releases need an explicit canonical trial CSV with columns `dataset,subject,participant_uid,session,run,raw_file,onset_s,energy_j,trial_id,source_url`; use the same verification requirements. Dataset names must start with `zhao` and equal `tiemann_munich` respectively.

```bash
python pipeline.py manifest --config sources.json --out trials.csv
python pipeline.py prepare --manifest trials.csv --out prepared
python pipeline.py benchmark --data prepared --out run_joint --device cuda --epochs 100
# Replace cuda with cpu if necessary.
```

Shared defaults: Fz/Cz/Pz/C3/C4, selected-channel average reference, 1–95 Hz filtering, 250 Hz, −0.2–0 s baseline, +0.02–0.8 s input, and 150 µV peak-to-peak rejection. Missing/bad common electrodes fail explicitly. Inspect retained-trial counts and energy distributions before training; revise preprocessing only using development data. Five electrodes prioritize cross-study compatibility, not proven optimality. All models use the same retained trials. Check source timing and reference differences before interpreting transfer performance.

`run_joint/results.json` contains test results; `development.json` contains tuning results; `selection_locked.json` records choices before test prediction; `predictions.csv` contains test predictions; `splits.csv` records all partitions. Saved models support prediction via:

```bash
python pipeline.py predict --data prepared --models run_joint --out predictions_all.csv --device cuda
```

That example includes training participants and **must not be scored as independent validation**. For new data, prepare with exactly the same preprocessing. Prediction itself ignores labels, though the manifest workflow is designed for labeled research datasets. Load only your own trusted model files.

### Optional external validation

Predeclare this separate protocol before inspecting results:

```bash
python pipeline.py benchmark --data prepared --out run_external --external tiemann_munich --device cuda --epochs 100
```

This trains on Zhao only and holds out all Tiemann people, including any people shared across sources. It tests study transfer; it is distinct from the joint-training experiment. Do not tune either protocol from its test results or report only whichever scores better. Joules across different lasers, spot sizes, skin sites, and protocols do not guarantee equivalent exposure. An additional Zhao source (`--source ds005293`) can be added as a third config entry only after an overlap audit.

## Presentation talking points

- We predict continuous physical stimulus intensity from EEG, separately from subjective pain.
- The proposed comparison combines Zhao and an independent Tiemann Munich cohort using shared EEG inputs.
- ATCNet learns temporal patterns; Spline + Ridge models nonlinear feature relationships; the ensemble is evaluated against both and Extra Trees.
- Participant-held-out evaluation prevents repeated trials or sessions from inflating performance.
- New model scores are pending local execution; 0.75 is an aspiration, not a demonstrated result.

## Primary sources

- Tiemann study and data availability: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003948
- Raw EEG collection: https://osf.io/z2h86/
- Zhao/OpenNeuro candidates: https://openneuro.org/datasets/ds005473 and https://openneuro.org/datasets/ds005293
- ATCNet original implementation: https://github.com/Altaheri/EEG-ATCNet
- Braindecode ATCNet API: https://braindecode.org/stable/generated/braindecode.models.ATCNet.html

The source names and manual cohort audit must establish suitability; this package does not claim to have independently inspected inaccessible raw files. Keep the downloaded source versions and all companion files. Preparation hashes raw entry files; for multifile formats, those hashes do not cover every companion file.
