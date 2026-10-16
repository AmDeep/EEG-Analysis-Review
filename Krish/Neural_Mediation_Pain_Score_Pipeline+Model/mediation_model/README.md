# EEG neural mediation score: research prototype

This package trains a **single EEG-only latent index** from trial-level EEG features and tests whether that index statistically accounts for *part* of the relationship between measured stimulus intensity and trial-level **reported** pain. It works with `eeg_trial_features.csv` produced by the companion EEG preprocessing package.

**Not validated clinically. Not a diagnostic pain meter. Statistical mediation is not proof that a measured scalp-EEG signal caused pain.** Real-data results require verification of recording protocols, labels, artifact rejection and appropriate replication.

## 1. Installation

```bash
python -m pip install -r requirements.txt
```

## 2. Run the fully synthetic demo (works without any EEG files)

```bash
python make_demo_data.py --out demo_input/synthetic_multisubject_features.csv
python neural_mediation.py \
  --csv demo_input/synthetic_multisubject_features.csv \
  --out demo_results
python apply_neural_score.py \
  --model demo_results/eeg_neural_score_model.json \
  --csv demo_input/synthetic_multisubject_features.csv \
  --out demo_results/applied_scores.csv
```

**All files under `demo_input/` and `demo_results/` are SIMULATED**. The generator intentionally creates a known indirect association so code paths can be checked; model scores from it do not demonstrate any real EEG pain biomarker or effect size.

## 3. Train with real extracted EEG features

1. Run the supplied `eeg_pain_features.py` on *each continuous EEG recording* and its accurately aligned stimulus-event file. The event TSV must contain actual physical intensity, as well as **trial-level observed pain ratings**, not pre-calibration targets. Validate electrode geometry, labels, baseline window, event timing, subject/session IDs, and recording coverage for your dataset.
2. Concatenate accepted trial feature CSV files across subjects (do not combine rejected epochs):

```python
from pathlib import Path
import pandas as pd
files = sorted(Path("eeg_results").glob("sub-*/eeg_trial_features.csv"))
assert files, "No EEG feature CSVs found"
all_trials = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
all_trials.to_csv("all_eeg_trials.csv", index=False)
```

3. Fit and evaluate on held-out subjects:

```bash
python neural_mediation.py --csv all_eeg_trials.csv --out model_output
```

The script expects numeric columns **`stimulus_intensity`** and **`pain_rating`** along with a **`subject`** identifier and EEG feature columns. At least 3 separate subjects are required for subject-held-out evaluation; preferably far more. If you have only one subject for pipeline debugging, use `--fit-only`; this produces an unvalidated fitted formula only, not a held-out comparison or mediation estimates.

4. Run the *saved EEG-only score* on an independent trial feature CSV, even with stimulus and pain labels absent:

```bash
python apply_neural_score.py \
  --model model_output/eeg_neural_score_model.json \
  --csv unlabeled_trial_features.csv \
  --out eeg_only_scores.csv
```

## 4. Model equations

For EEG feature vector `X`, the single-index neural score is

\[
S = \frac{X_{\mathrm{standardized}}w-\mu_S}{\sigma_S}.
\]

To learn the weights, the training algorithm removes the linear contributions of stimulus intensity **and training-subject intercepts** from both the EEG feature matrix and reported pain, then fits Ridge regression:

\[
\hat w=\arg\min_w \frac1n\|P_{res}-X_{res}w\|^2+\lambda\|w\|_2^2.
\]

This explicitly targets EEG variation associated with **reported pain above and beyond physical stimulus intensity**. The score uses only EEG at inference time: *no intensity or pain labels are fed into its equation*. Feature imputation, normalization, alpha selection, and pain calibration are all learned using training subjects only.

For mediation, we fit **within-subject** de-meaned linear associations using subject-held-out/cross-fitted neural scores:

\[
S_{it} = aI_{it} + u_i+\epsilon^S_{it},\qquad
P_{it}=c'I_{it}+bS_{it}+v_i+\epsilon^P_{it}.
\]

`a*b` is the *indirect association* in units of pain-rating points per physical-intensity unit; `c'` is the *adjusted direct association*, and `c` is the total association. The actual code uses within-subject demeaning equivalent to **subject fixed intercepts** in a linear model, not a mixed/random-effects model. It is **not a causal mediation estimator**, as it does not ensure mediator-outcome ignorability or measure all neural and psychological influences. Score supervision on pain labels additionally complicates conventional mediator interpretation.

To judge usefulness, compare held-out MAE/RMSE/R² for three independently calibrated models: `P~I` (stimulus-only), `P~I+S` (does EEG add information?) and `P~S` (EEG-only application). **If joint prediction is no better than the intensity-only benchmark, do not claim the EEG feature score adds information.** If within-person `a` or `b` is weak or inconsistent, do not claim evidence for the proposed pathway.

The EEG-only displayed `score_0_100` is calibrated from the EEG-only predicted 0–10 pain rating; it is a convenience scale, not an objective physiological pain measurement.

## 5. EEG inputs derived from the supplied feature table

Default feature set (7 trial-level columns):

| Category in image | Input column | Notes |
|---|---|---|
| Sensorimotor alpha 8–13 Hz | `delta_sensorimotor_alpha_ln_power` | Difference of log power (log ratio) |
| Beta 13–30 Hz | `delta_sensorimotor_beta_ln_power` | Change from pre-stimulus |
| Theta 4–8 Hz | `delta_global_theta_ln_power` | Globally pooled |
| Gamma 30–45 Hz | `delta_gamma_central_gamma_ln_power` | Muscle-artifact sensitive |
| Spectral entropy | `delta_sensorimotor_spectral_entropy` | Difference, not ratio |
| Connectivity/phase synchronization | `delta_central_c3_c4_alpha_pli` | Exploratory; short-epoch uncertainty |
| Vertex N2–P2 amplitude | `erp_cz_n2_to_p2_uv` | **Discrete-stimulus only**; not baseline difference |

Run `--expanded` for optional PAF, alpha envelope CV, frontal alpha asymmetry, frontal/centro-midline theta, coherence, midline phase-locking, and N2/P2 latencies. *Infant EEG evoked templates are excluded as inapplicable.* Cross-trial variability is a separate group-level property, not independently extractable from one trial. For an ablation excluding exploratory connectivity and ERP, run:

```bash
python neural_mediation.py --csv all_eeg_trials.csv --out model_spectral_only \
  --exclude-feature delta_central_c3_c4_alpha_pli \
  --exclude-feature erp_cz_n2_to_p2_uv
```

To add a baseline feature rather than just its change: `--extra-feature pre_sensorimotor_alpha_ln_power`. Use sensitivity analyses for different frequency ranges, preprocessing choices and ERP definition.

## 6. Outputs

- `heldout_trial_predictions.csv`: per-trial *out-of-subject-fold* EEG score and predictions for all three benchmarks, true pain rating and stimulus intensity for comparisons.
- `heldout_metrics.json`: full held-out aggregate predictive metrics and MAE change of joint vs stimulus-only baseline.
- `mediation_oof.json`: estimated within-subject `a`, `b`, `c'`, total `c`, indirect `a*b` and *descriptive* subject-cluster-bootstrap intervals with **fixed cross-fitted scores**. These CIs do **not** refit the score on each bootstrap and therefore omit feature-learning uncertainty; do not use them as formal significance tests.
- `eeg_neural_score_model.json`: feature list, train-only imputation/standardization parameters, learned feature weights and three calibration heads.
- `learned_equation.txt`: readable equation corresponding to the saved final model.
- `feature_weights.csv`: learned coefficients and preprocessing scalers.
- `outer_cv_folds.json`: inner-selected ridge penalty and subject counts for each outer validation fold.
- `run_summary.json`: feature presence, subject/trial counts, flags.

## 7. Important design limitations

- **Input reality:** `ds005293` and other datasets must actually contain *observed* trial-level self-reports and physically meaningful intensity labels. Some intensity fields are **calibration targets**, not reported outcomes. Do not mislabel these; for mixed units or individually calibrated designs, fit each experimental protocol separately or define a scientifically valid intensity normalization strategy first.
- **Mediation limitations:** Randomizing the noxious stimulus supports more credible stimulus effect estimates, but it does *not* randomize EEG mediators. Attention, expectancy, habituation, brain state, movement/EMG and latent neural processes can confound mediator-pain associations. Conditioning on a noisy, outcome-trained EEG score can also distort indirect effects. Interpretation requires preregistration, independent replication and more rigorous causal sensitivity analysis.
- **Cross-validation:** GroupKFold keeps all trials/sessions from a held-out subject together. Inner alpha-selection uses only outer-training subjects. Do **not** fit, prune, or adjust EEG feature parameters based on the full dataset or use OOF scores for post-hoc feature shopping without separate confirmation.
- **Data size and missingness:** Too few subjects or invalid EEG channels make fitted weights unreliable. Missing feature values are median-imputed using outer-training subjects; absent columns are reported. Investigate missingness and perform robustness analysis rather than trusting imputation blindly.
- **Temporal leakage:** Event-locked ERP peaks must precede pain ratings and rating prompts. No post-rating data should enter an outcome model.
- **N2/P2 and connectivity:** Trial-level ERP peaks and 1.8-second connectivity estimates are inherently noisy, and the N2/P2 extension was explicitly not a tonic-state core feature in the table. Test without them.
- **Uncertainty:** The fixed-score bootstrap does not capture model-refitting uncertainty. A full nested subject-bootstrap/retraining analysis or genuinely independent cohort is necessary for defensible inference. Do not equate descriptive intervals with causal CIs.
- **Input feature generation:** The raw preprocessing package's window/filter settings are research-prototype choices, not optimal for every paradigm. Auditing EEG artifacts, EMG/ocular sources, baselines, stimulus timing and trial counts is required.

## 8. Suggested next steps

1. Run the preprocessing on a few genuine EEG recordings and inspect both the time-aligned EEG waveforms and the trial features before fitting any supervised model.
2. Collect actual physical stimulus and experienced pain labels, multiple levels and many subjects.
3. Fit the seven-feature model and compare EEG-only/joint/stimulus-only performance on held-out participants.
4. Run ablations: spectral-only, without gamma, without connectivity, without ERP.
5. If stable, explore symbolic regression to replace the linear learned `S` with a small nonlinear expression and evaluate it on *new* subjects.

## 9. Smoke tests

```bash
python smoke_tests.py
```

This generates temporary synthetic data and verifies subject-held-out predictions, output schema, score bounds, and most importantly, that an EEG score is **identical with or without intensity/pain labels provided at inference time**. The test does not establish external validity.

The within-subject mediation analysis also requires **stimulus intensities to vary within participants**. If every person only receives a single intensity, the `I → EEG` slope is not identifiable using the within-person design.
