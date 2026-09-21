# Datasets

All datasets below use **laser-evoked nociceptive stimuli with EEG recorded simultaneously**, and all
have a genuine, per-trial stimulus-intensity label (not just a subjective rating) — the requirement for
predicting *stimulus intensity* rather than *perceived pain*.

## 1. Zhao et al. mega-study (primary — this is "the Zhao dataset")

Hu, Li et al. "A comprehensive EEG dataset of laser-evoked potentials for pain research."
*Scientific Data* 12, 1536 (2025). https://doi.org/10.1038/s41597-025-05900-1 (open access)

678 healthy participants across **9 independently-recorded experiments** (different EEG systems / sites,
same lab, same protocol family): laser stimuli varying continuously from **2.5 J to 4.5 J** on the dorsum
of the hand, single-trial self-reported pain rating *and* the physical stimulus energy recorded per trial.
CC0 / OpenNeuro, no access request needed.

| Experiment | OpenNeuro ID | N subjects | EEG system | Features already in `features_combined.csv`? | Raw EEG in your zip? |
|---|---|---|---|---|---|
| 1 | [ds005284](https://doi.org/10.18112/openneuro.ds005284.v1.0.0) | 26 | Biosemi | ✅ (348 trials) | ✅ `Amar/ds005284/` |
| 2 | [ds005285](https://doi.org/10.18112/openneuro.ds005285.v1.0.0) | 29 | ANT | ✅ (4,618 trials) | ❌ |
| 3 | [ds005289](https://doi.org/10.18112/openneuro.ds005289.v1.0.0) | 39 | BrainProducts | ✅ (372 trials) | ❌ |
| 4 | [ds005286](https://doi.org/10.18112/openneuro.ds005286.v1.0.0) | 30 | ANT | ✅ (880 trials) | ❌ |
| 5 | [ds005291](https://doi.org/10.18112/openneuro.ds005291.v1.0.0) | 65 | ANT | ✅ (1,885 trials) | ❌ |
| 6 | [ds005293](https://doi.org/10.18112/openneuro.ds005293.v1.0.0) | 95 | BrainProducts | ✅ (7,291 trials) | ❌ |
| 7 | [ds005292](https://doi.org/10.18112/openneuro.ds005292.v1.0.0) | 142 | Biosemi | ✅ (4,153 trials) | ❌ |
| 8 | [ds005280](https://doi.org/10.18112/openneuro.ds005280.v1.0.0) | 223 | BrainProducts | ✅ (6,275 trials) | ❌ |
| 9 | [ds005473](https://doi.org/10.18112/openneuro.ds005473.v1.0.0) | 29 | BrainProducts | ✅ (3,693 trials) | ❌ |

**Correction from my first pass:** I originally assumed Aditya's CSV only covered ds005284 and planned to
download 2 more sibling experiments myself. Once the zip was actually opened, `features_combined.csv`
turned out to already have all 9 experiments combined — 678 subjects, 29,515 trials total, target
`laser_power`. So the "add more Zhao sibling data" step you asked for is already done, at the feature
level. `data/download_openneuro.py` is still useful for one specific reason: getting **raw EEG** (not
just extracted features) for the 8 experiments beyond ds005284, if you want to run the real ATCNet on the
full 678-subject pool instead of just the 26 subjects whose raw BDF files are already in your zip.

## 2. Tiemann et al. 2018 (outside dataset — different protocol, for generalization)

Tiemann, L. et al. "Distinct patterns of brain activity mediate perceptual and motor and autonomic
responses to noxious stimuli." *Nature Communications* 9, 4487 (2018).
https://doi.org/10.1038/s41467-018-06875-x

Public data: **OSF project https://osf.io/bsv86/**

51 healthy right-handed participants (25 F, mean age 27), laser stimuli at **3 individually-calibrated
intensity levels** (low / medium / high, calibrated per subject to their pain threshold) rather than a
continuous Joule value. Different lab, different EEG hardware than the Zhao studies — good for testing
whether the model generalizes across acquisition systems, at the cost of a coarser (3-level, relative)
intensity label instead of continuous Joules.

## Datasets considered and not used (for the record)

- **BioVid Heat Pain Database** — very commonly cited pain-intensity dataset, but its "Part A/B" biosignals
  are SCL/EMG/ECG, not EEG (Part C has fNIRS+EEG but only for pain/no-pain, not graded intensity) — doesn't
  fit the EEG + stimulus-intensity requirement well.
- **OpenNeuro ds005307** ("Laser-evoked potentials in the human spinal cord and cortex") — laser LEPs but
  aimed at spinal/cortical comparison, not a graded-intensity design; skipped for this first pass.
- **PainMonit** — multimodal pain dataset (thermal, several biosignals) but EEG is not part of its
  standard released modality set; would need verification before relying on it.

## If you want me to paste something instead of fetching it

If your network/OpenNeuro access from wherever you run this is restricted, the raw BIDS data for the Zhao
sibling experiments and Tiemann's OSF project are both large (each experiment is several GB of raw EEG).
If direct download fails, tell me and I can walk through `openneuro-py`/`aws s3` command lines for you to
run and paste output back, rather than relying on my sandbox reaching them.
