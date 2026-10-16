# EEG stimulus-response feature extraction pipeline

**Status: reproducible prototype; not a validated EEG pain biomarker.** The `make_synthetic_demo.py` output is entirely simulated and must not be interpreted as experimental evidence. This project ingests **continuous event-marked EEG** and produces one wide numeric CSV row per accepted stimulus trial for subsequent within-subject or subject-held-out modeling.

## Installation

Python 3.10+ recommended:

```bash
python -m pip install -r requirements.txt
```

## Run a full smoke test (no external data required)

```bash
python make_synthetic_demo.py --out demo_input
python eeg_pain_features.py \
  --raw demo_input/synthetic_continuous_raw.fif \
  --events demo_input/synthetic_events.tsv \
  --event-col trial_type --event-value laser \
  --subject sub-SYNTHETIC --session ses-01 --out demo_results
```

## Run on real EEG

```bash
python eeg_pain_features.py \
  --raw path/to/sub-001_task-pain_eeg.set \
  --events path/to/sub-001_task-pain_events.tsv \
  --event-col trial_type --event-value laser \
  --subject sub-001 --session ses-01 --out results/sub-001
```

**IMPORTANT:** Adapt the paths, event columns, stimulus `trial_type`, units, and alignment to the **specific dataset**. Mixed event files containing rating prompts, response markers, and fixation onsets must be filtered so the `onset` values correspond to the **actual noxious stimulus**. Event `onset` must be expressed in seconds from the start of this same EEG recording, not global/session time or a sample index. The script only accepts **continuous** EEGLAB `.set` files; if a dataset supplies pre-epoched `.set`, use the epoch-specific adapter described below before attempting analysis. This prototype is **not yet verified on OpenNeuro ds005293 files**.

The optional `--bad-channels Fp1,T8`, `--mains 50`, `--reject-ptp-uv 120`, `--reject-eog-ptp-uv 120`, and `--skip-ica` flags allow input adaptation. With EOG channels absent, ICA ocular correction is skipped and must be addressed by visual inspection or an explicitly validated automated component classifier.

## Input events

Minimum: `onset` in **seconds**, optionally `duration`, `trial_type`, `trial_id`, `stimulus_intensity`, `pain_rating`, `stimulus_site`, and other task metadata.

```tsv
onset\tduration\ttrial_type\ttrial_id\tstimulus_intensity\tpain_rating
5.0\t0.0\tlaser\ttrial-0001\t2\t3.0
10.5\t0.0\tlaser\ttrial-0002\t4\t5.0
```

All extra event metadata except `onset`, `duration` and `sample` is preserved unmodified in the feature CSV. Labels are **never used in feature computation**. Keep physical energy (e.g. J) and individually calibrated intensity categories distinguishable; don't treat an NRS calibration *target* as an actual trial report.

## Processing flow

1. Load continuous raw EEG (MNE) and identify named EOG channels.
2. Flag channels with extremely low or outlying variance; merge user-specified bad-channel list. Visual inspection is required for final bad-channel selection.
3. Optional 50/60 Hz notch before final filtering, only if below Nyquist; filter EEG from **0.3–45 Hz** with zero-phase FIR. **30–45 Hz** gamma does not include high gamma and remains highly sensitive to EMG.
4. When EOG is present, fit ICA on a **separate 1 Hz high-pass copy**; detect EOG components and apply removal to the 0.3–45 Hz signal. Screen EOG amplitude in every trial even after ICA. Without EOG, automatic ICA identification is *skipped*.
5. Apply average reference across remaining EEG channels (bad channels excluded).
6. Epoch around each stimulus, **−2 to +2 s**. Exclude epochs intersecting MNE `BAD*` annotations, with any EEG channel peak-to-peak amplitude over **150 µV**, under **0.5 µV**, EOG peak-to-peak excursions over **150 µV** (when available), nonfinite values, or unavailable time boundaries. Record *all* exclusions in `trial_qc.csv`.
7. Spectral baseline: **[−2.0, −0.2) s**. Spectral response: **[+0.2, +2.0) s**, two matched 1.8-second windows. Ignore the immediate 200 ms around onset for these stationary spectral summaries. The response window includes ongoing/background activity as well as evoked effects — no additive separation is asserted.
8. N2/P2: separate, stimulus-locked **0.18–0.35 s** (negative peak) and **0.30–0.60 s** (positive peak) on Cz, with local **[−0.2, −0.02) s** DC mean correction and 15 Hz low-pass. Inspect averages before interpreting single-trial amplitudes/latencies.
9. Compute pre/post/delta features. For `*_ln_power`, the difference equals **natural log(post-power / pre-power)**; other deltas are arithmetic differences.
10. Write CSV files and a `run_metadata.json` file containing settings, bad channels, ICA decisions and coverage.

## Features from the supplied table

| Feature | Output mapping | Intended interpretation / caveat |
|---|---|---|
| Sensorimotor alpha 8–13 Hz | `*_sensorimotor_alpha_ln_power` on C3/C4 | Log integrated Welch power; task response vs pre-stim baseline |
| Peak alpha frequency | `*_sensorimotor_peak_alpha_hz`; also global PAF | Short windows yield noisy PAF; `NaN` if no weakly identifiable peak |
| Alpha envelope dynamics | `*_sensorimotor_alpha_envelope_cv` | Coefficient of variation of Hilbert envelope, exploratory in 1.8 s |
| Frontal alpha asymmetry | `*_frontal_alpha_asymmetry_ln_f4_over_f3` | log F4 alpha minus log F3 alpha (explicit sign convention) |
| Beta 13–30 Hz | `*_sensorimotor_beta_ln_power`, global | Log power, check movement contamination |
| Theta 4–8 Hz | `*_global_theta_ln_power` | Global theta; frontal/midline reported separately |
| Midline theta | `*_frontal_midline_theta_ln_power` for Fz/FCz; `*_centro_midline_theta_ln_power` for Cz | Do not pool Fz/FCz with Cz |
| Gamma 30–45 Hz | `*_gamma_central_gamma_ln_power`, global | Sensorimotor and Cz, assess facial EMG |
| Inter-electrode coherence and phase sync | `*_central_c3_c4_*`, `*_frontal_f3_f4_*`, `*_midline_fz_cz_*` with each band coherence and PLI | Sensor-space coherence and phase-lag index are exploratory; volume conduction and short-window bias |
| Spectral entropy | `*_sensorimotor_spectral_entropy`, global | Normalized Shannon entropy of mean PSD over 4–45 Hz |
| N2/P2 evoked response | `erp_cz_n2_amp_uv`, `erp_cz_p2_amp_uv`, `erp_cz_n2_to_p2_uv`, latency/means | Single-trial estimates; electrode/window choice should be prespecified |
| Response variability | `subject_feature_variability.csv` | Across-trial SD of ERP metrics for QC; **never directly join into trial-level cross-validation without proper nesting** |
| Infant noxious-evoked EEG template | **Excluded** | Neonatal-specific; inappropriate as an adult pain predictor |

All feature columns use prefixes `pre_`, `post_`, and `delta_` wherever appropriate. If required named electrodes are missing, outputs are `NaN` rather than substituting other electrodes.

## Outputs

- `eeg_trial_features.csv` — trial-level table: `subject`, `session`, `trial_id`, `onset_s`, event labels, feature columns.
- `trial_qc.csv` — one row for **every selected stimulus**; reason for rejection and peak-to-peak metrics.
- `subject_feature_variability.csv` — subject/session trial-to-trial ERP variability summary; QC only.
- `run_metadata.json` — cleaning configuration, identified EOG/EEG/bad channels, missing features, accepted/rejected counts.
- `feature_dictionary.csv` (distributed in package) — generated column-by-column feature glossary.

For multiple recordings/subjects, run once per recording, then concatenate feature CSVs with pandas. Keep all sessions/trials for a subject in the same training/test partition to prevent leakage. Fit imputation, scaling, selection and any EEG score **only on training subjects**, and score held-out subjects afterward.

## Critical limitations and verification checklist

- **Event timing:** verify laser trigger delay and recorded response prompt timing; otherwise N2/P2 estimates can be meaningless. Inspect stimulus-aligned grand averages.
- **Stimulus artifacts and muscle:** automatic amplitude rejection does not remove all EMG, blinks, eye movements, high-frequency bursts, or electrode drifts. More rigorous pipelines should consider ICLabel with manual component review, robust interpolation and muscle-specific QC. Reject rather than interpolate when an individual epoch is badly contaminated.
- **Short 1.8 s windows:** adequate to begin theta/alpha/beta power analysis but not ideal for PAF, alpha envelope dynamics or short-epoch connectivity. These should remain exploratory; consider longer pre/post windows **only when interstimulus spacing, subject responses and recording duration permit**. Inspect event timing before widening.
- **False independence:** PLI values within a short trial are autocorrelated and biased, and coherence can be driven by common sources; neither alone proves neural communication.
- **Spectral endpoints:** 0.3–45 Hz is a deliberate compromise that sacrifices higher gamma. Do not treat absent >45 Hz as a negative finding.
- **Validation:** study-level thresholds, bad-channel QC, ERP selection, frequency ranges and sampling rate should be reviewed by someone with EEG expertise and registered before hypothesis testing where feasible.
- **Stimulus vs pain:** later models should preserve both *measured* physical intensity and *actual trial-by-trial* subjective rating as distinct labels. This pipeline deliberately does not calculate a pain score.
- **No real EEG recordings were supplied here.** The included result CSV is synthetic demonstration data only.

## Notes on other formats

Continuous `.fif`, `.edf`, `.bdf`, `.vhdr`, and EEGLAB continuous `.set` supported. Additional loaders can be added for BrainVision markers or pre-epoched `.set`; do **not** reinterpret pre-epoched timestamps as seconds from a continuous session.

Source software documentation: [MNE ICA](https://mne.tools/stable/generated/mne.preprocessing.ICA.html), [MNE ICA tutorial](https://mne.tools/stable/auto_tutorials/preprocessing/40_artifact_correction_ica.html). Candidate public dataset: [OpenNeuro ds005293](https://openneuro.org/datasets/ds005293).
