# Independent EDF EEG State Validation

## Dataset acquisition

PhysioNet EEG Motor Movement/Imagery v1.0.0 was checked on 2026-09-15 before
acquisition.

- Access: public, subject to the dataset license
- License: Open Data Commons Attribution License v1.0
- DOI: 10.13026/C28G6P
- Full size: 3.4 GB uncompressed; 1.9 GB ZIP
- Local subset: S001-S010, runs 01/02/03/04/07/08/11/12
- Local files: 80 EDF+ recordings, 167,230,368 bytes
- Comparison subset: ds005284 sub-001 through sub-010, balancing subject counts

The EDF subset includes eyes-open rest, eyes-closed rest, motor execution, and
motor imagery. Its 64 EEG channels are sampled at 160 Hz. The complete acquisition
record is in `notebooks/data/preprocessed/eegmmidb_acquisition_audit.json`.

## Analysis

Both datasets were filtered to 1-40 Hz, resampled to 160 Hz, and average-referenced
over the full EEG montage. Portable Tier 1 summaries were extracted from
non-overlapping 2, 4, and 8 second windows:

- relative delta, theta, alpha, beta, and low-gamma power;
- spectral entropy;
- Hjorth mobility and complexity;
- robust RMS and peak-to-peak amplitude;
- global field power.

Each channel-level measure was summarized by its median and interquartile range.
QC variables were retained separately and excluded from clustering. Subject and
dataset groups were balanced before discovery.

Each leave-one-dataset-out direction performed state-count selection using only the
training dataset. Gaussian-mixture candidates with two to four states were compared
using held-out-subject log likelihood, held-out-subject coverage, and subject plus
temporal-block bootstrap ARI. K-Means silhouette and HDBSCAN cluster/noise results
were retained as geometry checks. HMM fitting was not available in the installed
environment, so occupancy, dwell, and transition metrics were computed directly
from the assigned sequences. Temporal blocks are defined from actual recording time
as `floor(start_seconds / 24)`, not from retained-row order.

The selected training-only scaler and GMM were applied to the other dataset without
refitting. Windows with posterior confidence below 0.60 or likelihood below the
training fifth percentile were reported as low-confidence unknowns. The two outer
LODO directions provide the available dataset-resampling test; the EDF dataset never
participates in model selection when it is held out.

## Results

| Window | States trained on ds005284 / EDF | Confound gate | Minimum bidirectional LODO coverage |
|---:|---:|---|---:|
| 2 s | 2 / 3 | Pass | 0.235 |
| 4 s | 4 / 4 | Reject: subject/dataset/device/QC dominance | 0.165 |
| 8 s | 3 / 2 | Reject: subject or dataset/device dominance | 0.180 |

The 2-second candidate was not dominated by subject, dataset/device, or QC:
across the two LODO directions, maximum subject share was at most 0.147, maximum
dataset share was at most 0.655, dataset/device Cramer's V was at most 0.244, and
maximum QC normalized mutual information was at most 0.119.
However, its bidirectional transfer failed the prespecified 0.50 coverage gate:
training on ds005284 covered 23.5% of held-out EDF windows, while training on the
EDF subset covered 66.9% of held-out ds005284 windows.

The 4-second solutions contained nuisance-dominated states and one direction also
exceeded the QC threshold. The 8-second solutions exceeded subject or
dataset/device dominance limits. Both were rejected before task interpretation.
Sequence-level occupancy, median dwell, transition counts, and unknown fractions
are reported per subject and recording in
`notebooks/data/preprocessed/multidataset_lodo_temporal_metrics.csv`. Because feature
extraction retains at most 48 windows per recording, dwell runs and transitions are
explicitly broken whenever adjacent retained rows are separated by more than one
window length. The output records 2,830 such sampling-gap breaks.

Task/run annotations were excluded from feature construction, scaling, clustering,
model selection, confound checks, and LODO fitting. They were joined only after
assignment. The resulting counts are retained as an exploratory audit, not as
evidence for a reproducible task-related state.

## Conclusion

No portable EEG state is confirmed on this independent EDF dataset. The only
candidate family that passed the confound screen, at 2 seconds, did not transfer
with adequate coverage in both directions.

This is a negative but informative external validation of the new portable Tier 1
representation. It is separate from the earlier ds005284-only 135-feature state
definition and does not claim to have directly backfit that incompatible fixed-ROI
model. A general EEG-state claim remains additionally gated on recurrence in a third
dataset.