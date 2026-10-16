# EEG pain model: all 13 categories from Table 2 (v2)

**A runnable research prototype, not a clinically validated pain score.** This is the revised companion to the original raw-EEG pipeline and stimulus → neural-score → reported-pain statistical mediation model.

## What changed

The original mediation model selected **7** columns. This version defines and supports **all 13 feature categories from your image**. For a study of **adult laser-evoked pain**, the new default is **12 candidate categories**: the 13th (a noxious-evoked *infant* EEG template) is not a defensible adult EEG marker. It is now available **only through an explicit neonatal experiment + externally supplied and validated template**. `--feature-set all-13` will refuse adult data. Do not fabricate template values or silently median-impute an entirely unmeasured feature.

The adult default is *broader*, not necessarily *better*. A nested-CV comparison with the original 7-feature model remains important, because a larger set of candidates can overfit or represent artifacts.

## Table 2: mapping of every category

| # | Category in your table | Compact model input | How extracted / qualifications |
|---|---|---|---|
| 1 | Sensorimotor alpha power (8–13 Hz) | `delta_sensorimotor_alpha_ln_power` | C3/C4 Welch 8–13 Hz log-power post minus pre |
| 2 | Peak alpha frequency | `delta_sensorimotor_peak_alpha_hz` | Dominant alpha peak frequency, post minus pre; unstable in short windows |
| 3 | Alpha envelope dynamics | `delta_sensorimotor_alpha_envelope_cv` | Change in Hilbert alpha envelope coefficient of variation; exploratory |
| 4 | Frontal alpha asymmetry | `delta_frontal_alpha_asymmetry_ln_f4_over_f3` | Change in ln(F4 alpha power) − ln(F3 alpha power) |
| 5 | Beta power (13–30 Hz) | `delta_sensorimotor_beta_ln_power` | C3/C4 log-power change; check movement / EMG |
| 6 | Theta power (4–8 Hz) | `delta_global_theta_ln_power` | Global log theta power change; separate from local midline theta |
| 7 | Spatially defined midline theta | `delta_frontal_midline_theta_ln_power` | Fz/FCz log theta power change; Cz separate in CSV; exploratory and short-window |
| 8 | Gamma power (>30 Hz) | `delta_gamma_central_gamma_ln_power` | C3/C4/Cz 30–45 Hz log power change only; no higher gamma above 45 Hz |
| 9 | Connectivity and phase synchronization | `delta_central_c3_c4_alpha_pli` | C3–C4 alpha phase-lag index change. Other coherence/PLI bands available; susceptible to short-epoch bias and volume conduction |
| 10 | Spectral entropy | `delta_sensorimotor_spectral_entropy` | PSD Shannon entropy (4–45 Hz) change |
| 11 | Response variability / temporal features | `delta_cz_response_temporal_sd_uv` | Change in Cz waveform SD, baseline −0.75 to −0.15 s vs response +0.15 to +0.75 s. In addition, `erp_cz_n2_to_p2_previous5_sd_uv` is a **past-only** (min 3 prior trials) across-trial metric; `subject_feature_variability.csv` is for QC only |
| 12 | Vertex N2–P2 amplitude | `erp_cz_n2_to_p2_uv` | Event-locked baseline-corrected Cz N2 trough to P2 peak, single trial; suited to discrete stimuli |
| 13 | Infant noxious-evoked EEG template | `neonatal_noxious_template_corr` | **Neonatal data only**, correlation with an independent, scientifically validated, protocol- and electrode-matched infant evoked template; unavailable (`NaN`) by design in adults |

The input CSV also retains pre/post values and secondary ERP, connectivity, and temporal features. The **12 adult model inputs** are the one-per-category columns in the table; `--expanded` adds prespecified secondary features (it is not the default).

## Install

```bash
python -m pip install -r requirements.txt
```

## Step 1 — From raw EEG to trial features

For continuously recorded EEG with stimulus onsets in a tab-delimited/CSV file:

```bash
python eeg_pain_features.py \
  --raw sub-001_task-pain_eeg.set \
  --events sub-001_task-pain_events.tsv \
  --event-col trial_type --event-value laser \
  --subject sub-001 --out eeg_results/sub-001
```

The `.set` loader requires **continuous** EEGLAB files, not pre-epoched `.set`. Other supported formats: `.fif`, `.edf`, `.bdf`, `.vhdr`. The events file must include an `onset` column in seconds relative to the raw recording; physical stimulus `stimulus_intensity` and **actual reported** `pain_rating` may be included and are passed through as labels without touching feature extraction.

Default windows: matched spectral analysis **[−2, −0.2) s** and **[+0.2, +2) s**, 0.3–45 Hz filtering, optional named EOG-ICA cleaning, event-specific Cz N2/P2 around 0.18–0.60 s. Temporal SD category 11 uses matched [−0.75,−0.15) and [+0.15,+0.75) s. Match these to the real study design before analysis: another stimulus, a pain-rating prompt, or an ocular artifact inside the post window may invalidate the comparison. The `run_metadata.json` and `trial_qc.csv` preserve QC and excluded trials.

### Neonatal-specific template feature (13th category)

Use **only** when studying neonates with an independent, scientifically justified noxious-evoked template. Supply your own waveform in microvolts:

```csv
time_s,amplitude_uv
0.000,0.11
0.005,0.15
...
0.800,-0.09
```

It must span at least +0.05 to +0.70 s with >=20 increasing samples and match the stimulus modality, EEG electrode, age, filtering and timing; do not derive it from the held-out test set. Then run:

```bash
python eeg_pain_features.py \
  --raw infant_recording.fif --events infant_events.tsv \
  --subject sub-infant001 --out eeg_results/sub-infant001 \
  --neonatal-cohort \
  --neonatal-template independent_infant_template.csv \
  --neonatal-template-channel Cz
```

This creates `neonatal_noxious_template_corr` and `neonatal_noxious_template_projection` per accepted trial, and marks the CSV cohort as neonatal. The above is **a code interface, not a claim that a Cz infant template or adult preprocessing choices are validated**. Neonatal EEG requires its own filtering, epoch, reference, artifact rejection, stimulus protocol and scientific validation. Do not apply the provided adult-default settings unchanged without expert review.

## Step 2 — Train the neural mediation index

Concatenate extracted CSVs (keep `subject` identifiers stable):

```python
from pathlib import Path
import pandas as pd
paths = sorted(Path('eeg_results').glob('sub-*/eeg_trial_features.csv'))
assert paths, 'No feature files found'
pd.concat([pd.read_csv(f) for f in paths], ignore_index=True).to_csv('all_eeg_trials.csv', index=False)
```

Default **12 applicable adult features**:

```bash
python neural_mediation.py --csv all_eeg_trials.csv --out model_results
```

Original **7-feature** comparison:

```bash
python neural_mediation.py --csv all_eeg_trials.csv --feature-set core --out core_results
```

Opt-in **all 13**, only with scientifically appropriate neonatal recordings and an actually measured template similarity column:

```bash
python neural_mediation.py --csv all_neonatal_trials.csv \
  --feature-set all-13 --out neonatal_results
```

You can use `--exclude-feature COLUMN` for channels/metrics that cannot be measured, and `--extra-feature COLUMN` to add a justified secondary feature. Missing *columns* and wholly absent features raise clear errors rather than being silently dropped or filled with zero. Partial missing measurements are imputed from **training-fold medians only**, and >25% missingness triggers a warning.

**Score learning:** a Ridge-regression EEG-only index learns from within-subject variability in reported pain, without first removing all stimulus-related variance from the features. Separate pooled prediction heads estimate `pain ~ intensity`, `pain ~ EEG_score`, and `pain ~ intensity + EEG_score`. After grouped, nested-validation score estimation, within-subject linear `a`, `b`, `c'`, `a*b` associations are reported. These are **statistical associations, not causal mediation effects**. The bootstrap uses fixed cross-fitted scores and omits score-training uncertainty. Repeated sessions, crossover designs, and covariates may require hierarchical mediation and more thorough tests.

**At deployment, the saved score needs EEG features only**:

```bash
python apply_neural_score.py --model model_results/eeg_neural_score_model.json \
  --csv new_trial_features.csv --out predicted_scores.csv
```

## Test the complete code without human EEG recordings

```bash
python make_synthetic_demo.py --out demo_raw
python eeg_pain_features.py --raw demo_raw/synthetic_continuous_raw.fif \
  --events demo_raw/synthetic_events.tsv --event-col trial_type \
  --event-value laser --subject sub-FAKE --out demo_features --skip-ica
python make_demo_data.py --subjects 16 --trials 20 \
  --out demo_model_input/fake_multisubject_features.csv
python neural_mediation.py --csv demo_model_input/fake_multisubject_features.csv \
  --out demo_results --folds 4 --bootstraps 20
python test_pipeline.py
```

The last command also runs synthetic tests for the guarded **13-feature neonatal path**. All demonstration signals, template waveforms and labels are synthetic; nothing here constitutes evidence of a clinically or biologically meaningful score.

## What is saved

- Preprocessing: `eeg_trial_features.csv`, `trial_qc.csv`, `subject_feature_variability.csv`, `run_metadata.json`.
- Model: `eeg_neural_score_model.json`, `feature_weights.csv`, `learned_equation.txt`, `heldout_metrics.json`, `mediation_oof.json`, `heldout_trial_predictions.csv`, `run_summary.json`.
- `feature_catalog.csv` lists the 13 image categories, one representative column each, and adult/neonatal applicability.

## Limitations to resolve before publication

1. **Feature count ≠ validity.** Peak alpha, alpha envelope, frontal asymmetry, gamma/EMG, short-window phase synchrony, temporal variability and neonatal templates are especially exploratory. Compare pre-registered feature subsets on held-out *subjects*, not just individual trials.
2. **Window suitability:** 1.8-second PSD windows have limited frequency resolution, especially for peak alpha and connectivity. Avoid contamination from response prompts or close-together stimuli. Gamma 30–45 Hz is not high-gamma.
3. **ERP reliability:** single-trial N2/P2 amplitudes can be noisy and reflect salience/attention rather than pain alone. Check peak morphology, latency and trialwise QC on real data.
4. **Baseline vs response:** log-power differences and entropy deltas represent pre/post associations, not physiological isolation of pain from background EEG. Event-locked ERP and template correlations do not have a meaningful pre-stimulus *ERP* counterpart, so they use baseline-corrected waveforms.
5. **Repeated-trial leakage:** the past-five-trial SD uses only earlier accepted trials, but within-subject chronological state may not generalize to a new subject. Whole-subject summary SD is NOT included in model training.
6. **Mediation limitations:** learned mediators, confounding (expectation, salience, attention, pain sensitivity), conditioning on outcomes, and measurement error prevent interpreting a nonzero `a*b` as a causal brain mechanism. For causal claims, design and analyze a more rigorous experiment.
7. **Adult vs infant:** infant evoked templates cannot reasonably be treated as adult pain signatures. The code deliberately will not enable the thirteenth without neonatal evidence and explicit user action.
