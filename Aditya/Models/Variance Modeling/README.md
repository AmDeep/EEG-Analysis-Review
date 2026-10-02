# Baseline-normalized EEG response, the Nurovo Neural Response Score, and variance decomposition

Separate from the EEGNet/DWT classifier pipeline in `../CNN/`. This is a
scientific analysis, not a predictive model: for a small, interpretable EEG
feature set, does the baseline-normalized change track laser stimulus
intensity, after accounting for subject -- and if there's leftover variance,
what explains it?

## Project summary

16 stages, in four arcs:

1. **Does it work at all? (Stages 1-2)** Baseline-normalized EEG change
   (response minus each trial's own pre-stimulus window) tracks laser
   intensity -- yes, for several interpretable features (relative alpha/beta
   power, entropy, Hjorth parameters), with subject-consistency mostly >0.9.
   Those features combine into one learned composite, the Nurovo Neural
   Response Score: cross-validated r=0.49 within-subject, r=0.33 pooled
   across all 9 Zhao et al. datasets.
2. **How consistent is it across people, and what explains the leftover
   variance? (Stages 3-6, 9-14)** People at the same intensity cluster only
   moderately (eta^2=0.25-0.48 -- intensity explains under half the
   variance), and that spread genuinely grows with intensity (not noise --
   Levene p=1e-74). A subject's own resting theta/alpha balance predicts
   ~21% of who-differs-from-whom (Stage 6). Raw baseline EEG doesn't explain
   the rest at this data size (Stage 5), a second, independently-extracted
   ERP feature set barely moves it (+0.010 R^2, Stage 9), and six different
   dimensionality-reduction methods (PCA, t-SNE, UMAP, Kernel PCA, Isomap,
   LLE -- Stages 10-14) all agree resting-state EEG structure essentially
   doesn't predict upcoming intensity (<4% R^2, every method). The leftover
   variance looks like a property of the data/question, not a blind spot of
   any one method.
3. **Can this generalize to the clinic, where intensity is never known?
   (Stages 7-8)** Yes, two ways that need no calibrated stimulus and no
   per-patient history: baseline EEG alone predicts an independently
   measured pain threshold (r=0.21, non-circular, Stage 7), and scoring a
   response against a fixed population-average baseline works just as well
   as each subject's own baseline (r=0.320 vs. 0.327, Stage 8) -- producing
   an actual SD-band -> pain-intensity calibration table.
4. **How far can raw accuracy be pushed? (Stages 15-16, since dropped)** A
   separate push explicitly targeted R>0.97 for a per-patient clinical pain
   meter. Adding Stage 6's sensitivity covariate or a nonlinear model barely
   moved Stage 8's composite (r=0.314-0.377, and the higher number came with
   an untrustworthy, non-monotonic calibration table -- Stage 15). A raw
   28-channel CNN at full data scale did meaningfully better (r=0.483,
   reversing Stage 5's smaller-scale negative result -- Stage 16), but a
   bigger architecture and more training didn't improve on that further
   (r=0.477), and r=0.97 looks physiologically unreachable from one EEG
   reading given how much of the response is genuine individual variation
   (Stages 3-4). This direction was abandoned in favor of the honestly
   validated, deployable scores from Stages 7-8.

## Overview: what each stage answers

| Stage | Script | Question | Headline result |
|---|---|---|---|
| 1 | `stage1_extract_features.py`, `stage1_baseline_normalization.py`, `stage1_intensity_analysis.py` | Does baseline-normalized EEG change track laser intensity? | Yes for several features (e.g. relative alpha/beta power, entropy, Hjorth params), subject-consistency mostly >0.9 |
| 2 | `stage2_neural_response_score.py` | Can those features combine into one score, with weights learned (not eyeballed) on held-out subjects? | Cross-validated r=0.49 (within-subject) / 0.33 (pooled 9 datasets) vs. laser power |
| 3 | `stage3_consistency_analysis.py` | Do people at the SAME intensity respond similarly? Can we bucket into an SD scale? | eta^2=0.48 (within-subject); spread grows ~4-5x from lowest to highest intensity |
| 4 | `stage4_variance_decomposition.py` | Is that growing spread real, and is it different slopes, plateaus, or noise? | Real (Levene p=1e-74); real slope heterogeneity, not noise; baseline theta/alpha ratio predicts a subject's sensitivity |
| 5 | `stage5_train_residual_cnn.py` | Can raw EEG (a CNN) explain the leftover trial-level residual better than hand-crafted features? | No, not at this data size (ridge r=0.33 beats CNN r=0.27) |
| 6 | `stage6_standardized_score.py` | Use the baseline-state finding as an actual covariate; build a fairer, heteroscedasticity-corrected SD scale | Explains ~21% of previously-unexplained slope variance; SD-band composition now roughly flat across intensity (was wildly uneven) |
| 7 | `stage7_healthy_norm_score.py` | Clinical deployment has no known laser intensity at all -- can a score be built from baseline EEG alone, validated against something OUTSIDE the EEG data? | Baseline EEG predicts an independently-measured psychophysical pain threshold, r=0.21 (p=0.001, n=255, cross-validated) -- modest but real and non-circular. Establishes a "healthy" reference distribution (all 9 datasets, n=678) to score new patients against with no stimulus needed |
| 8 | `stage8_population_deviation_score.py` | Score a RESPONSE by its deviation from the population-average baseline (not each person's own baseline) -- does that still track intensity, and can it become an SD-band -> pain-group lookup table? | Yes -- r=0.320 (vs. 0.327 for the subject-own-baseline version): a fixed population reference works just as well, no per-patient baseline needed. Calibration table's band means climb monotonically (3.28J to 3.83J), but only the 3+ SD band is clearly separated -- 0-1/1-2/2-3 SD overlap heavily in practice |
| 9 | `stage9_ancova_covariates.py` | Bring in an independently-extracted feature set (KNNs folder's Cz ERP features, incl. classic N2/P2 amplitude) as ANCOVA covariates on top of `laser_power` -- does anything close the 52% gap from Stage 3's eta^2=0.48? | Barely: R^2 rises from 0.482 to 0.493 (+0.010). Three covariates individually significant (skewness, max\_abs, the N2P2 peak-to-peak amplitude) but all with tiny partial eta^2 (<0.01 each) -- real but nowhere near closing the gap |
| 10 | `stage10_pca_variance.py` | How much do the 9 chosen composite-score features overlap/duplicate each other? | PC1 alone = 50% of their combined variance; needs 6 of 9 components for 90% -- real but not extreme redundancy. Rerun on all 20 baseline-only features (no intensity involved): LESS redundant, PC1 only 26%, needs 9 of 20 |
| 11 | `stage10_pca_variance.py --target-col` | Map those same baseline-only PCA components against laser_power -- do resting-state components predict upcoming stimulus intensity? | No -- only ~3% of intensity's variance explained, essentially nothing. Expected and reassuring: pre-stimulus brain state shouldn't predict what intensity comes next |
| 12 | `stage12_pca_regression_intensity.py` | Same question as Stage 11, but as an honest subject-grouped cross-validated regression (linear + Random Forest) instead of an in-sample check | Confirms Stage 11: tops out at R^2=1.4% (linear) / 3.7% (Random Forest) at 20 components. A little real nonlinear structure, but still negligible either way |
| 13 | `stage13_manifold_intensity.py` | Same question with t-SNE and UMAP instead of PCA | Both agree with Stages 11-12: t-SNE in-sample R^2=0.3%, UMAP cross-validated R^2=0.4% -- if anything, worse than PCA's 3.7%, since neither method is built to maximize predictability of an external variable |
| 14 | `stage14_manifold_intensity_v2.py` | Three more nonlinear methods that (unlike t-SNE) support proper train/test: Kernel PCA, Isomap, LLE | Best is Isomap at 0.8%, still far below PCA's 3.7%. Six methods now agree: baseline EEG doesn't predict upcoming intensity |
| 15 | `stage15_nurovo_pain_score.py` | A separate, since-dropped push for a much higher R: does adding Stage 6's baseline-predicted sensitivity, or a nonlinear model, beat Stage 8's r=0.320 population-referenced score? | Barely -- ridge+sensitivity_hat: r=0.314 (no real change). A gradient-boosted model scored higher in CV (r=0.377) but inverted the top SD band's calibration (small-n tail overfitting), so ridge was kept as the trustworthy model |
| 16 | `stage16_full_scale_cnn.py` | Same dropped push: does a raw 28-channel EEG CNN at the FULL 678-subject/29,515-trial scale beat hand-crafted features, reversing Stage 5's negative result? | Yes, meaningfully -- r=0.483 (vs. Stage 15's r=0.314), confirming Stage 5's negative result was a data-size artifact, not a fundamental one. A bigger architecture + more epochs (`stage16_..._b` run) didn't improve on this (r=0.477), suggesting ~0.48 is close to this approach's real ceiling, not a tuning gap. Pursuit of R>0.97 was abandoned here as physiologically unreachable for single-reading EEG-only pain decoding |

Each stage's script produces its own `stageN[_all]_*.csv`/`.png` outputs in
this directory and `plots/`, referenced in that stage's section below.

## Pipeline (Stages 1-2)

```text
../CNN/build_epochs.py --tmin -1.0 --tmax 1.0
        |
        +--> nurovo_epochs_baseline.npz   (baseline [-1,0), response [0,1))
        |
        v
stage1_extract_features.py
        |
        +--> features_baseline_response.csv
        |
        v
stage1_baseline_normalization.py
        |
        +--> features_normalized.csv
        |
        v
stage1_intensity_analysis.py
        |
        +--> stage1_results.csv
        |
        v
plots.py
        |
        +--> plots/*.png
        |
        v
stage2_neural_response_score.py   (Stage 2, reads stage1_results.csv + features_normalized.csv)
        |
        +--> stage2_weights.csv        (final deployable weight vector)
        +--> stage2_cv_results.csv     (per-fold held-out r)
        +--> stage2_scored_trials.csv  (features_normalized.csv + neural_response_score column)
        +--> plots/neural_response_score.png
```

## Data

`.set` files under `../data/<dataset>/derivatives/rerefer/` for `ds005293` and
`ds005473` have a real pre-stimulus baseline: confirmed by loading a sample
subject from each and checking `tmin`/`tmax`/shape directly (`tmin=-1.0,
tmax=1.999, sfreq=1000 Hz`). Other datasets under `../data/` have not been
checked the same way yet — extend `--datasets` in the `build_epochs.py` call
below only after doing that check, since a dataset without a real pre-stim
window would silently produce a bogus baseline.

Labels come from `../../Data Analysis/lep_eda_output/labels_combined.csv`
(`dataset, subject, epoch, laser_power, rating`), matching `build_epochs.py`'s
default `COLS` mapping.

## Design choices

- **Epoch window**: `nurovo_epochs_baseline.npz` is built by calling the
  existing `build_epochs.py` with `--tmin -1.0 --tmax 1.0` (baseline =
  `[-1, 0)`, response = `[0, 1)`). The existing `nurovo_epochs.npz` (response
  only, `tmin=0`) used by the CNN pipeline is untouched.
- **Channel**: a single vertex channel, `Cz` by default (falls back through
  `CPz, FCz, Fz, Pz` — same priority order `../../Data
  Analysis/lep_feature_extraction.py` already uses for N2/P2/gamma/ERD on this
  same data). Kept deliberately small/interpretable rather than 28
  channels x 20 features; override with `stage1_extract_features.py --channels`.
- **Baseline is per-trial, not per-subject**: each trial's own `[-1, 0)`
  window is its baseline (there's no separate baseline session in this data).
  So for a feature X:
  - `X_delta_abs = X_response - X_baseline` (this trial's own baseline)
  - `X_delta_rel = X_delta_abs / |X_baseline|`
  - `X_z = X_delta_abs / sigma_baseline(subject)`, where
    `sigma_baseline(subject)` is the std of `X_baseline` pooled across all of
    that subject's trials (a single trial's 1s baseline window is too short
    to give a stable variance on its own). Subjects with fewer than
    `--min-baseline-trials` (default 5) usable baseline values get `X_z =
    NaN` for that feature.
- **Baseline quality/drift check**: each trial's baseline window is split into
  4 x 250ms quarters; RMS per quarter and the linear trend across them are
  saved as `baseline_rms_q1..q4` / `baseline_drift_slope`. This is a
  diagnostic only — nothing is filtered out based on it. `stage1_extract_features.py`
  prints what fraction of trials have a drift slope more than 3x the median
  magnitude.
- **Feature set** (computed on both baseline and response windows of the
  vertex channel): theta/alpha/beta/gamma band power (Welch), absolute +
  relative to broadband; beta/alpha, gamma/alpha, theta/alpha ratios; spectral
  entropy; peak frequency; RMS, variance, std; Hjorth activity/mobility/
  complexity; sample entropy. Band edges match `../../Data
  Analysis/lep_feature_extraction.py`'s `BANDS`.
- **Statistical model**: `statsmodels` `MixedLM`, fixed effect = laser power,
  random intercept per `global_subject`, fit by ML (not REML) so the linear
  and `+ power^2` models are comparable by likelihood-ratio test (that's the
  `nonlinear` column: `p_nonlinear < 0.05`). `r2_pooled` is a naive OLS R^2
  ignoring subject grouping — a descriptive number reported alongside the
  mixed-model beta/p, not a replacement for it. `subject_consistency` is the
  fraction of subjects with >= 3 distinct power levels whose own per-subject
  slope has the same sign as the population fixed effect.
- **Plots**: `plots.py` uses the Nurovo dataviz palette — a single muted hue
  (`#898781`) for the many per-subject lines (too many series for categorical
  identity), with the first two categorical slots (`#2a78d6` blue, `#eb6834`
  orange) reserved for the linear/quadratic fit curves, the only place two
  distinct series appear on one axis.

## Stage 2: Nurovo Neural Response Score

`R = w1*Z_1 + w2*Z_2 + ... + intercept` over the Stage 1 features that actually
survived scrutiny -- not hand-picked weights:

- **Feature selection**: from `stage1_results.csv`, keep `normalization ==
  "z"` rows with `p_linear < 1e-3` and `subject_consistency >= 0.8`.
- **Redundancy pruning**: `rms`, `variance`, `std`, and `hjorth_activity` are
  near-duplicate measures of the same broadband-amplitude signal (by
  construction `hjorth_activity = variance`, and `rms`/`std` are monotonic
  transforms of it on baseline-corrected data). Greedily drop one of any pair
  with `|correlation| > 0.95`, keeping the stronger (lower `p_linear`) of the
  two, so the score doesn't triple-count one underlying effect. This dropped
  `rms_z`/`variance_z`/`std_z` in favor of `hjorth_activity_z` on the first run.
- **Weights**: ridge regression (`alpha=1.0`) predicting `laser_power` from the
  surviving Z-scored features, per-fold standardized on the training subjects
  only. Sign and magnitude come from the regression, not from eyeballing --
  this also sidesteps PCA's loading-sign ambiguity, and directly targets what
  the proposal asked the score to track (intensity).
- **Validation**: `GroupKFold` on `global_subject` (a subject's trials never
  split across train/test, verified by checking the fold indices directly).
  The reported cross-validated `r` is the generalization estimate; the
  `stage2_weights.csv` coefficients come from a separate refit on *all* data
  and are the deployable formula, not the performance number.

First run (same 124 subjects / 10,984 trials as Stage 1): 9 features kept,
cross-validated `r = 0.494 +/- 0.031` across 5 subject-grouped folds between
`R` and laser power on held-out subjects. `hjorth_activity_z` dominates the
weight vector (~0.33, very stable across folds), with `sample_entropy_z`
second (~-0.11); the rest contribute small, fold-stable corrections. See
`stage2_weights.csv` / `stage2_cv_results.csv` for the full numbers and
`plots/neural_response_score.png` for the fit.

## Full-dataset run (all 9 datasets)

All 9 datasets under `../data/` were checked the same way as ds005293/ds005473
(load one subject, inspect `tmin`/`tmax`) and all have a genuine pre-stimulus
baseline (`tmin` between -1.0 and -1.1, `tmax` between 1.9 and 2.0). Laser
power is on the same scale across all 9 (range roughly 1.0-4.5, all maxing out
at 4.0 or 4.5), so pooling `laser_power` across datasets is valid -- it's not
mixing incompatible units.

Rebuilding with `--datasets ds005280 ds005284 ds005285 ds005286 ds005289
ds005291 ds005292 ds005293 ds005473` picks up **29,515 trials across 678
subjects** (vs. 10,984 / 124 before), zero skipped -- every dataset's montage
contains the required 28 channels. Files from this run carry an `_all` suffix
(`nurovo_epochs_baseline_all.npz`, `features_normalized_all.csv`,
`stage1_results_all.csv`, `stage2_all_*`), so the original 2-dataset run stays
intact for comparison.

**Important caveat surfaced by the expansion**: only 124 of the 678 subjects
(exactly the original ds005293 + ds005473 cohort) actually received more than
2 distinct laser power levels *within* that subject -- the other 7 datasets
are between-subjects designs, where a given subject was tested at one (or
occasionally two) fixed power level(s) and different subjects got different
levels. That means for those 554 subjects, the "does EEG change track
intensity" question is answered by comparing *across* subjects/studies rather
than *within* one person -- weaker evidence, since it's confounded with
whatever else differs between those subjects and studies (equipment, montage,
population). The random-intercept-per-subject mixed model still runs and the
fixed-effect direction/significance mostly holds up, but effect sizes shrink
noticeably once the between-subjects-only data is pooled in (see table below)
-- report the within-subject (2-dataset) and pooled (9-dataset) numbers
side by side rather than only the pooled one.

Stage 1, Z-normalized features, pooled-R^2 for the strongest features:

| feature | 2-dataset R^2 (within-subject) | 9-dataset R^2 (pooled) |
|---|---|---|
| std / rms / variance / hjorth_activity | 0.23-0.24 | 0.09 |
| sample_entropy | 0.17 | 0.07 |
| hjorth_mobility | 0.13 | 0.06 |
| spectral_entropy | 0.09 | 0.03 |
| hjorth_complexity | 0.09 | 0.04 |

Direction, sign, and per-subject consistency (computed only over the 124
subjects with enough within-subject power variation to define a slope) are
essentially unchanged -- so the pooled result isn't contradicting the
within-subject one, it's just diluted by subjects who can't speak to a
dose-response question at all.

Stage 2 score, same pattern: cross-validated `r` (score vs. laser power on
held-out subjects) drops from **0.494 +/- 0.031** (2 datasets) to **0.327 +/-
0.047** (9 datasets) -- still a real, leakage-free, highly significant signal,
just weaker once diluted by the between-subjects-only cohort. Compare
`plots/stage2_neural_response_score.png` (2-dataset) against
`plots/stage2_all_neural_response_score.png` (9-dataset) to see this directly.

## Stage 3: consistency across subjects, and a standardized SD scale

Stage 1/2 ask "does the response track intensity" (a regression question).
That's different from: *do people who got the same intensity respond
similarly to each other*, and if so, can that be turned into a "you're N SDs
from typical" score? `stage3_consistency_analysis.py` answers both, on one row per
(subject, laser_power) -- averaging a subject's repeat trials at the same
exact level first, since all 9 datasets share one 1.0-4.5-by-0.25 power grid
(no binning needed).

- **Consistency**: one-way ANOVA of the composite score across laser_power
  levels, reported as eta^2 (share of total variance explained by *which
  intensity level a person got*, agnostic to a linear/monotonic shape --
  the ANOVA analogue of R^2). On the clean within-subject cohort (2 datasets,
  124 subjects), **eta^2 = 0.48**: about half the variance in the score is
  intensity, the other half is person-to-person spread at the *same*
  intensity. Pooling in the 7 between-subjects datasets dilutes this to
  **eta^2 = 0.25** (678 subjects) -- both are enormously significant
  (p < 1e-90) given the sample size, but the effect size tells the real
  story: intensity explains a real, substantial share of the response, not
  all of it.
- **The spread is NOT constant across intensity** -- this is the more
  actionable finding. Per-level std of the composite score rises roughly
  4-5x from the lowest to the highest laser power (2-dataset: 0.08 at
  power=1.0 up to 0.41 at power=4.5; see `plots/stage3_score_spread_by_level.png`,
  boxes colored light-to-dark blue by intensity). People are very consistent
  at mild stimulation and diverge substantially at strong stimulation --
  plausibly real individual differences in pain/stimulus sensitivity that
  only show up once the stimulus is strong enough to matter.
- **SD scale**: re-standardize the composite score against its own
  population distribution (`score_z = (score - mean) / std`) and bucket every
  observation into `0-1 / 1-2 / 2-3 / 3+` SD bands. The bands do climb with
  intensity -- at power=1.0 essentially nobody exceeds 1 SD; at power=4.5,
  ~20-25% of people exceed 2 SD -- but the relationship is fairly weak taken
  as a monotone ordinal trend (Spearman rho=0.29 within-subject, 0.18 pooled).
  That's expected: collapsing a continuous score into 4 bands throws away the
  finer gradient the continuous Stage 2 score already captures (CV r=0.49 /
  0.33) -- the band scale is a coarser, more interpretable readout of the same
  signal, not a stronger one. Useful for a human-facing "score," not for
  further statistical modeling.

Outputs: `stage3[_all]_anova.csv` (eta^2/F/p per feature),
`stage3[_all]_score_per_level.csv` (mean/std per intensity level --
this is the number to read for "how similar were people at this intensity"),
`stage3[_all]_sd_band_by_level.csv` (band composition per level),
`stage3[_all]_sd_scale.csv` (every subject-level observation with its band).

## Stage 4: is the growing spread real, and is it slopes, plateaus, or noise?

`stage4_variance_decomposition.py` formalizes what Stage 3's boxplot showed and asks
what's driving it. Run on the within-subject cohort (`stage2_scored_trials.csv`,
124 subjects) since a per-subject slope is only identifiable for subjects with
their own intensity variation.

- **Heteroscedasticity is real, not visual**: Breusch-Pagan (regressing squared
  residuals of a random-intercept model on `laser_power`) gives p=3e-51;
  Levene's test across the 15 exact intensity levels gives p=1e-74.
- **It's slope heterogeneity, not just noise**: a random-slopes model
  (`score ~ laser_power + (1 + laser_power | subject)`) beats random-intercept-
  only by a likelihood-ratio test (p~0). But slope variance is only 17% of
  total between-subject variance -- most of "who's different from whom" is a
  stable offset (intercept), with a smaller, real difference in *how much
  someone's response climbs per unit intensity* (slope). See
  `plots/stage4_trajectories_by_slope.png`: lines are colored by their BLUP
  slope deviation (red=below-average, blue=above-average) and visibly fan out
  in a color-ordered way at high intensity -- if it were pure noise, the color
  order wouldn't track the fan-out.
- **Most subjects accelerate, they don't plateau**: 84% of per-subject
  quadratic fits curve upward (positive curvature); steeper linear responders
  curve upward *even more* (r=0.475, p=2.5e-8). A ceiling/saturation
  hypothesis is not supported by this data.
- **Individual differences**: 8/20 of a subject's own (non-normalized)
  baseline features predict their fitted sensitivity slope after FDR
  correction -- strongest are baseline theta/alpha ratio (r=0.38) and
  baseline relative alpha power (r=-0.33). People who rest with more
  alpha-dominant, less theta-heavy baselines respond less to increasing
  intensity. This is a real, interpretable, low-cost answer to "why do people
  differ" -- not everything left over is unexplainable noise.

Also exports `stage4_trial_residuals.csv`: for every trial, the composite
score minus what the random-slopes model predicts (population trend + that
subject's own intercept and slope) -- i.e., what's left after Stage 1-4's
tabular modeling. That's the target for Stage 5.

## Stage 5: does raw baseline EEG explain any of what's left?

`stage5_train_residual_cnn.py` asks the harder question directly, on the SAME
target and SAME subject-grouped folds for a fair comparison: predict
`stage4_trial_residuals.csv`'s `residual` column from (a) ridge regression on
the ~20 hand-crafted baseline features, vs. (b) a compact EEGNet-style CNN
(reusing `../CNN/train_eegnet.py`'s architecture/training loop directly)
reading the raw 28-channel x 1s baseline waveform.

**Result: ridge r=0.330 beats the CNN's r=0.268.** The hand-crafted features
already capture more of the residual than a from-scratch CNN trained on
~11,000 trials can. This is a fair, if slightly disappointing, negative
result for the "train a deep model on raw EEG" idea *at this data size* --
not evidence deep learning can't help here, just that a compact model with no
pretraining and ~11k single-subject-pool trials doesn't have enough data to
beat well-chosen features. Deep learning on raw physiological signals
typically needs far more data (or pretraining) to out-compete domain feature
engineering; this negative result is itself informative.

**Important caveat on the ridge number itself**: `residual` is derived from
the composite score, which is built from `ΔX = X_response - X_baseline`
computed with *that same trial's own baseline value*. So a trial's baseline
feature values are mechanically, not just biologically, coupled to that
trial's own delta/Z (a low baseline value mechanically inflates the delta,
independent of any real "sensitivity" effect). That means ridge r=0.330
shouldn't be read as "0.33 of previously-unknown variance, newly explained" --
some of it is the arithmetic of how the score was constructed showing back up.
Both models see the same trial's baseline data, though, so the *relative*
comparison (ridge beats CNN) is still fair; it's the absolute r that needs
this caveat. Stage 4's individual-differences result (subject-AVERAGED
baseline features vs. subject-level fitted slope, not a single trial's own
value) is the cleaner, less-confounded version of "does baseline state
predict response."

**Bottom line for "how do we explain more of the variance"**: the strongest,
cleanest lever found so far is Stage 4's individual-differences result
(baseline theta/alpha ratio and relative alpha power predict sensitivity).
Raw-EEG deep learning doesn't yet beat that at this sample size. The most
likely further gains: (1) more within-subject-intensity-varying data (more
subjects like ds005293/ds005473, or more trials per subject/level, which
would both give a CNN more to learn from and shrink single-trial noise via
averaging), or (2) accepting that some fraction of single-trial EEG variance
is just irreducible -- well documented in the ERP literature, which is
exactly why studies average many trials per condition rather than trusting
single trials.

## Stage 6: using the baseline-state covariate to build a fairer standardized score

Stage 4 found *correlations* between baseline state and sensitivity. Stage 6
(`stage6_standardized_score.py`) turns that into an actual predictive covariate and
a properly heteroscedasticity-corrected standardized score:

1. **Predict sensitivity from baseline state, honestly.** Ridge-regress each
   subject's Stage 4 fitted slope on all ~20 baseline features, with
   leave-one-subject-out CV so every subject's `sensitivity_hat` is
   out-of-fold. **r=0.439** (vs. the best single univariate feature at
   r=0.38) -- combining features multivariately does meaningfully better than
   any one alone.
2. **Moderated mixed model**: `score ~ laser_power * sensitivity_hat`, random
   intercept + slope per subject. The interaction term is highly significant
   (coef=0.79, p<0.001) -- `sensitivity_hat` genuinely moderates the
   intensity slope, it's not a spurious correlation. Random slope variance
   drops from 0.0220 (Stage 4, no covariate) to 0.0175 -- **this one baseline-
   derived covariate explains ~21% of the previously "unexplained" between-
   subject slope variance.**
3. **Heteroscedasticity-corrected standardization.** Rather than one global
   SD (wrong -- variance grows ~4-5x with intensity), fit
   `log(deviation^2) ~ laser_power` and back-transform with **Duan's smearing
   estimator**, not a naive `exp()` of the fitted log -- Jensen's inequality
   means that naive back-transform systematically underestimates the true
   variance (verified empirically: the first attempt at this underestimated
   sigma by 1.4-2.2x at every single intensity level, which is why the very
   first band table looked backwards, worse at low intensity than high).
   With the smearing correction, Levene's test statistic for remaining
   heteroscedasticity drops from 24.6 to 11.2 (both still p<0.05 -- not
   perfectly flat, particularly at the lowest intensity level, but a real,
   substantial improvement). See `plots/stage6_band_by_level.png`: SD-band
   composition is now roughly 65-85% "0-1 SD" at nearly every intensity level,
   instead of ranging from ~20% to ~90% before.

The deployable score is `deviation = score - population_expected`, where
`population_expected` uses ONLY population-level fixed effects (intensity +
predicted sensitivity) -- deliberately NOT each subject's own fitted random
intercept/slope, since a new trial wouldn't have those available yet. Divide
by `predict_sigma(laser_power)` for the final standardized value
(`score_z_corrected` in `stage6_scored_trials.csv`), then bucket into the same
0-1/1-2/2-3/3+ SD bands as Stage 3.

**Where this leaves the original question**: baseline-normalized EEG change
tracks laser intensity (Stage 1-2), people at the same intensity cluster
moderately tightly (Stage 3, eta^2 0.25-0.48), a real chunk of who-differs-
from-whom is explained by a subject's own resting theta/alpha balance (Stage
4 + 6, ~21% of slope variance), raw EEG deep learning doesn't yet beat that
at this sample size (Stage 5), and the standardized SD scale is now
substantially fairer across the intensity range after correcting for both of
those (Stage 6) -- though not perfectly flat, so treat "N SDs" as an
approximate, continuously-improvable readout rather than an exact scale yet.

## Stage 7: toward clinical deployment -- a score with no stimulus, no known intensity

Everything above assumes `laser_power` is known, which won't be true
clinically (a patient gets a headset, not a calibrated research laser). Two
things needed fixing before this generalizes:

- Stage 2's composite score already doesn't need intensity as an *input* at
  inference time (it was only the training target) -- given a new patient's
  baseline/response Z-features from some real event, it scores them fine.
- Stage 6's standardized score does NOT generalize -- `population_expected`
  explicitly requires `laser_power` as an input. Not usable as-is if intensity
  is unknown.
- The bigger problem: every validation so far checked "does the score
  correlate with known intensity" -- which stops being available at all once
  intensity is gone, and was checked against the SAME data the score was
  built from.

Per instruction, every subject across all 9 Zhao et al. datasets is treated
as the HEALTHY reference population for now (a chronic-pain comparison group
is a separate, future data-acquisition step -- none of the 9 datasets contain
diagnosed chronic pain patients; a `Group` column exists in 3 of them but is
undocumented in their BIDS metadata and doesn't look like a diagnosis).

`stage7_healthy_norm_score.py` uses a genuinely external anchor: 5 of the 9 datasets
(`ds005284`, `ds005286`, `ds005289`, `ds005291`, `ds005293`) include a
per-subject `Pain_Threshold` in `participants.tsv` -- the laser energy that
produces a 7/10 pain rating for that person, from psychophysical testing, not
derived from any EEG trial. That's real ground truth, independent of
everything Stage 1-6 built.

- Ridge-regress `Pain_Threshold` on subject-averaged BASELINE features (no
  stimulus, no response window, no intensity -- purely resting-state
  characteristics), 5-fold CV: **r=0.21, p=0.001, n=255**. Modest, but real
  and non-circular -- baseline EEG alone predicts an independently-measured
  sensitivity trait. See `plots/stage7_validation.png`.
- Refit on all 255 labeled subjects, apply to the full 678-subject population
  to get `predicted_sensitivity` for everyone (labeled or not) -- this is the
  **healthy reference distribution**: mean=3.11, std=0.315 (see
  `plots/stage7_healthy_distribution.png`). `stage7_healthy_norm_reference.csv`
  stores this distribution's mean/std per baseline feature plus the composite
  score, and `stage7_model_weights.csv` stores the fitted ridge weights +
  scaler -- together these are what's needed to score a brand-new patient:
  compute their baseline features, apply the saved weights, compare to the
  saved healthy mean/std, done. No stimulus, no intensity, no per-patient
  historical data required.

**What's still needed for real clinical use** (in rough priority order):
1. When a chronic-pain (or other diagnosed) patient cohort becomes available,
   score them against this same healthy norm and check whether they land at a
   different point on the distribution -- that's the actual clinical
   discrimination test, not yet possible with current data.
2. r=0.21 is real but modest -- worth checking whether a larger/better-chosen
   baseline feature set, or more Pain_Threshold-labeled subjects, improves it
   before treating the score as clinically trustworthy on its own.
3. Confirm the target EEG headset's channel montage actually includes a
   usable vertex-region channel (Cz/CPz/FCz/Fz/Pz) -- everything here assumes
   that.
4. Decide how much resting baseline is clinically practical to record (this
   analysis used the 1s pre-stimulus window each trial already had -- a
   deployed protocol may want a longer resting recording for more stable
   spectral estimates, which would need its own validation, not assumed to
   transfer automatically from the 1s-window numbers above).

## Stage 8: population-average-baseline deviation, and the SD-band -> pain-group table

This is a different, simpler reference scheme than Stage 1-2/6, matching a
specific ask: instead of normalizing each trial against *that subject's own*
pre-stimulus baseline, build ONE reference "what does a calm/resting healthy
person's EEG look like" from the whole population (already computed in Stage
7's `stage7_healthy_norm_reference.csv` -- the population mean/std of every
baseline feature, subject-averaged, over all 678 "healthy" subjects). Then for
any trial, at any intensity:

```
Z_population = (response - population_baseline_mean) / population_baseline_std
```

-- how many population-baseline-SDs this response sits from "normal calm."
`stage8_population_deviation_score.py` combines the same 9 features Stage 2 already
found predictive (reused directly from `stage2_all_weights.csv`, now computed
against the population reference instead of each subject's own) into one
composite via the identical ridge + subject-grouped-CV recipe.

**Result: cross-validated r=0.320 +/- 0.026 vs. laser power** -- statistically
indistinguishable from Stage 2's subject-own-baseline version (r=0.327) on the
same 9 datasets. This is the important finding: a **fixed population
reference works just as well as requiring each patient's own historical
baseline**, which matters a lot for deployment -- no need to bank a clean
per-patient calibration recording ahead of time, just compare their current
reading to one reference established here.

The composite score is then re-standardized against its own population
distribution and bucketed into the same 0-1/1-2/2-3/3+ SD bands as Stage 3,
producing the actual clinical lookup table (`stage8_band_calibration.csv`):
what does each SD band correspond to in real (measured) laser power?

| SD band | mean laser power | n trials |
|---|---|---|
| 0-1 SD | 3.28 J | 21,249 |
| 1-2 SD | 3.35 J | 7,174 |
| 2-3 SD | 3.50 J | 963 |
| 3+ SD | 3.83 J | 129 |

**Honest caveat**: the table's *means* climb monotonically, but
`plots/stage8_band_calibration.png` (a boxplot of actual power per band)
shows the 0-1/1-2/2-3 SD bands have nearly identical medians and heavily
overlapping IQRs -- the real, reliable separation is "3+ SD clearly indicates
a stronger response" vs. everything below that, not four cleanly graded pain
levels. If a small number of discrete groups is the deliverable, the data
supports something closer to "3+ SD = elevated response" vs. "below 3 SD =
not clearly elevated" rather than 4 finely-spaced bands. The continuous
`score_sd_units` column in `stage8_scored_trials.csv` carries more
information than any of the discretizations and should be preferred over the
bands themselves wherever the downstream use can consume a continuous number.

## Stage 9: ANCOVA with an independent feature set -- does anything close the 52% gap?

`stage9_ancova_covariates.py` brings in `../KNNs/features_cz_erp_merged.csv` -- a
completely independent Cz-channel feature extraction, for the exact same
10,984 trials, that includes things Stage 1-8's 20-feature set never
computed: classic N2/P2 laser-evoked-potential amplitude/latency (the
standard single-trial pain-intensity biomarker in the LEP literature),
skewness, kurtosis, and permutation entropy.

This is a real ANCOVA: same one-way design as Stage 3
(`score ~ C(laser_power)`, R^2=0.482, matching Stage 3's eta^2=0.48), plus
the new covariates added to the same OLS model (Type II sums of squares via
`statsmodels.anova_lm`), on the subject-per-level aggregated table (760 rows,
124 subjects x up to 15 levels).

**Two data problems found and fixed before the result means anything:**
- `permutation_entropy` is a constant 0 for all 10,984 rows (a bug in
  whatever produced that file, not a real feature) -- dropped, since a
  constant column makes the design matrix rank-deficient.
- `n2p2_peaktopeak_uv` is *exactly* `p2_amplitude_uv - n2_amplitude_uv` (max
  discrepancy ~1e-16) -- an exact linear dependency if all three are in the
  model together. Kept the peak-to-peak (the metric actually used in the LEP
  literature) and dropped the two raw components.

**Result: R^2 = 0.482 -> 0.493 (+0.010).** Three covariates individually
significant at p<0.05 -- skewness (partial eta^2=0.0097), `max_abs`
(0.0079), and the N2P2 peak-to-peak amplitude itself (0.0054) -- but all
small. This feature set does **not** meaningfully close the ~52% gap; it
nudges it to ~51%. Full table in `stage9_ancova_table.csv`.

## Stage 10: how much do the 9 chosen features actually overlap?

`stage10_pca_variance.py` runs PCA on the 9 features Stage 2 selected for the
composite score, purely to visualize redundancy: `plots/stage10_pca_variance.png`
(bar = variance per component, cumulative line on the same 0-100% axis --
not a dual-axis chart, since both series are percentages of the same total).

**Result: the 1st component alone explains 50% of the variance across those
9 features; it takes 6 of the 9 to reach 90%.** So there's real redundancy
(roughly a third of the raw dimensionality is duplicate information -- no
surprise, e.g. `hjorth_activity`, `sample_entropy`, `spectral_entropy`, and
`hjorth_mobility`/`complexity` are all measuring related aspects of signal
complexity/amplitude), but the features aren't wildly redundant either -- it
genuinely takes most of them to capture the full picture. `PC1`'s loadings
(`stage10_pca_loadings.csv`) read like a general "response magnitude/
complexity" axis: mobility, sample/spectral entropy, and beta power load one
way, Hjorth complexity and activity load the other.

## Stage 10b: the same PCA, but with no laser intensity involved at all

Stage 10's 9 features were *selected* by correlating with `laser_power`
(Stage 2), so that PCA still indirectly depends on intensity being known.
`stage10_pca_variance.py --all-baseline-features` reruns the identical analysis on
all 20 raw baseline features instead -- the set Stage 7 used, computed and
selected with zero reference to any stimulus or intensity.

**Result: PC1 only explains 26% (vs. 50% for the intensity-selected 9), and
it takes 9 of 20 components for 90% of the variance** (`plots/
stage10_baseline_only_pca_variance.png`). The baseline-only feature set is
*less* redundant, not more -- makes sense, since Stage 2's 9 features were
partly chosen because they behave similarly (all track intensity), while
these 20 baseline features span genuinely different things (raw band power,
band ratios, entropy, Hjorth parameters) with less reason to move together.
PC1 here reads as a general amplitude/complexity axis again (sample entropy,
Hjorth mobility, RMS/std with opposite sign), but it takes much more of the
feature set to describe someone's resting EEG than to describe their
composite intensity-response score.

## Stage 11: mapping the baseline-only PCA components against laser intensity

`stage10_pca_variance.py --target-col laser_power` reuses the exact same 20
baseline-only features and PCA fit as Stage 10b, but instead of asking "how
much of the FEATURES' own variance does each component explain," asks "how
much of `laser_power`'s variance does each component explain." Since PCA
components are orthogonal by construction, each component's own R^2 with the
target sums exactly to the multiple-regression R^2 using all of them together
-- the same bar+cumulative-line chart works for this question too, just with
a different quantity on the y-axis (and no reason to expect it hits 90%, so
no threshold line this time).

**Result: all 20 components together explain only ~3% of laser intensity's
variance** (`plots/stage11_baseline_vs_intensity_pca_variance.png` --
essentially flat, no PC contributes meaningfully). This is expected, and
actually a good sign, not a disappointing one: these are PRE-stimulus
baseline features, recorded before the laser fires. There's no reason a
patient's resting brain state should predict what intensity the experimenter
is about to choose next -- if it did, that would suggest a confound in how
stimuli were ordered/assigned. Contrast with Stage 10's original PCA (on
features computed FROM the post-stimulus response), where the same kind of
components very much do relate to intensity, because that's what they were
built to measure.

(Minor note: the sanity check printed by the script flags a small mismatch,
3.02% vs. 2.32%, between the summed per-PC R^2 and the direct multiple-
regression R^2. That's a numerical artifact of `hjorth_activity_baseline`
being an exact duplicate of `variance_baseline` -- same redundancy noted back
in Stage 2/8/9 -- which makes one PCA direction have essentially zero
variance and the regression on it ill-conditioned. Both numbers still say the
same thing: negligible.)

## Stage 12: proper cross-validated regression, PCA components -> laser intensity

Stage 11's ~3% number was an in-sample sanity check on the whole pooled
dataset (no train/test split, no subject grouping). `stage12_pca_regression_intensity.py`
redoes it properly: for 1/2/3/5/9/15/20 retained components x 2 model types
(plain linear regression, and a Random Forest to catch anything nonlinear a
linear model would miss), fits with `GroupKFold` on `global_subject` (a
subject's trials never split across train/test) and reports held-out R^2.
PCA itself is refit on the training fold only each time, same discipline as
every other cross-validated result in this project.

**Result** (`plots/stage12_cv_results.png`, `stage12_cv_results.csv`):
cross-validated R^2 tops out at **1.4% (linear, 20 components)** and **3.7%
(Random Forest, 20 components)**. The Random Forest finding ~2.5x more than
linear at 20 components says there's a little genuine nonlinear structure in
how baseline features relate to intensity -- but "a little" is the operative
phrase; even the best result here explains under 4% of intensity's variance,
confirming Stage 11's finding under honest cross-validation rather than
overturning it. Baseline (pre-stimulus) EEG essentially doesn't predict what
intensity is about to be applied, linearly or not.

## Stage 13: the same question with t-SNE and UMAP instead of PCA

`stage13_manifold_intensity.py` repeats Stage 10b/11/12's question (do dimensionality-
reduced baseline-EEG axes relate to laser intensity?) with two nonlinear
methods. They're treated differently on purpose, and that difference is the
main methodological point of this stage:

- **t-SNE is descriptive only.** scikit-learn's `TSNE` has no `.transform()`
  for new data -- it jointly optimizes positions for exactly the points it's
  given, with no way to place a held-out point afterward. That makes a
  leak-free cross-validated regression like Stage 12's impossible with it.
  So this fits one t-SNE embedding on an 8,000-trial subsample (t-SNE doesn't
  scale to the full 29,515 in reasonable time either), plots it colored by
  laser power, and reports an **in-sample** (not cross-validated, and stated
  as such) R^2 for the record.
- **UMAP gets the full Stage 12 treatment** -- it can `.transform()` new
  points after fitting, so it's fit on training folds and evaluated on held-
  out folds, subject-grouped, across 2/5/10/20 embedding dimensions, with
  both linear and Random Forest regressors (on a 12,000-trial subsample, for
  runtime).

**Results, and they land in the same place as Stages 11-12:**
- t-SNE: in-sample R^2 = **0.3%** (`plots/stage13_tsne_scatter.png` -- no
  visible color structure, laser power is scattered uniformly across the
  embedding).
- UMAP: best cross-validated R^2 = **0.4%** (linear, 20 components;
  `plots/stage13_umap_cv_results.png`), and the 2D visualization
  (`plots/stage13_umap_scatter.png`) shows the same thing -- whatever
  structure UMAP finds in baseline EEG, it isn't organized by laser
  intensity.

Both nonlinear methods actually do a bit *worse* than PCA at this (PCA's best
was R^2=3.7%, Stage 12). That's not a contradiction -- t-SNE/UMAP are built to
preserve local neighborhood structure and produce good-looking clusters for
visualization, not to maximize linear predictability of some external
variable, so there's no reason to expect them to beat a variance-maximizing
linear method like PCA at this particular job. Combined with Stages 10b-13,
four different dimensionality-reduction approaches now agree: baseline
(pre-stimulus) EEG structure just doesn't relate to what intensity comes
next, by any method tried.

## Stage 14: three more nonlinear methods -- Kernel PCA, Isomap, LLE

`stage14_manifold_intensity_v2.py` tries three more nonlinear dimensionality
reductions, picked specifically because -- unlike t-SNE, Spectral Embedding,
or MDS -- scikit-learn's Kernel PCA, Isomap, and Locally Linear Embedding
(LLE) all support `.transform()` on new, unseen data. That means all three
get the full proper treatment (no descriptive-only shortcut needed): fit on a
training fold, transform the held-out fold, subject-grouped CV, linear +
Random Forest regressors, across 2/5/10/20 components. Isomap and LLE build a
k-nearest-neighbor graph per fold, which gets expensive at scale, so they run
on a 5,000-trial subsample; Kernel PCA runs on 12,000.

**Results** (`plots/stage14_cv_results.png`, `stage14_cv_results.csv`): best
across all three is **Isomap at 5 components, R^2=0.8%** -- Kernel PCA tops
out at 0.4%, LLE at 0.35%. All three land well below PCA's 3.7% (Stage 12)
and in the same negligible range as t-SNE (0.3%) and UMAP (0.4%, Stage 13).

**Running tally across every method tried**: PCA (3.7%) > Isomap (0.8%) >
UMAP (0.4%) ~ Kernel PCA (0.4%) ~ LLE (0.35%) > t-SNE (0.3%, in-sample only).
Six dimensionality-reduction methods, two of them genuinely different
nonlinear families (manifold-learning graph methods and kernel methods) on
top of neighborhood-embedding methods, all agree: baseline (pre-stimulus) EEG
structure does not meaningfully predict upcoming laser intensity, by any
method tried so far. At this point further exotic dimensionality-reduction
methods are unlikely to change that conclusion -- the signal (or lack of one)
looks like a property of the data/question, not a blind spot of any one
algorithm.

## Stage 15-16: a separate push for R>0.97, and why it was dropped

Stages 1-14 above answer "does baseline-normalized EEG track intensity, and
what explains the variance that's left" -- a scientific question, deliberately
not optimized past what the data honestly supports. Stages 15-16 are a
different, later push: explicitly targeting R>0.97 for a per-patient clinical
pain meter (walk in, headset on, get a number), tried in good faith before
being dropped as physiologically unreachable.

`stage15_nurovo_pain_score.py` tried the cheapest lever first: add Stage 6's
baseline-predicted sensitivity (computed from resting EEG alone, so it's
still deployable with no known intensity) as an extra predictor on top of
Stage 8's 9-feature population-referenced composite, and tried a nonlinear
model (gradient boosting) in case the true relationship isn't linear.
Ridge+sensitivity_hat: r=0.314, statistically the same as Stage 8's r=0.320
-- the covariate added nothing once folded into this trial-level score.
Gradient boosting scored higher in cross-validation (r=0.377), but its SD-band
calibration table inverted at the top (the rare "3+ SD" band, only 61 trials,
averaged the *lowest* laser power instead of the highest -- classic small-n
tail overfitting), so ridge was kept as the trustworthy model despite the
lower headline number: a calibration table you can trust beats a correlation
you can't.

`stage16_full_scale_cnn.py` tried the one genuinely untried lever: a raw
28-channel EEGNet-style CNN (reusing `../CNN/train_eegnet.py`'s architecture)
on the FULL 678-subject, 29,515-trial pool, predicting laser_power directly
from the whole baseline+response window -- not just Stage 5's single Cz
channel / baseline-only / 11k-trial residual target. Result: **r=0.483**,
clearly beating Stage 15's hand-crafted r=0.314 on the identical trials, and
reversing Stage 5's earlier negative verdict -- that result was a data-size
artifact, not a fundamental "deep learning can't help here" finding. A
stacked ensemble of the CNN with the hand-crafted composite barely moved the
number (r=0.490), meaning the CNN already captures nearly everything the
hand-crafted features knew, plus more.

A second run with a bigger architecture (temporal_filters 8->12,
depth_multiplier 2->3, matching EEGNet's own class defaults rather than
Stage 5's deliberately-shrunk version) and more training epochs (20->30, so
every fold actually converges instead of hitting an epoch cap) was tried to
see if r=0.483 was a model-capacity ceiling. It wasn't: r=0.477, no real
change despite ~2.4x the compute. Two honest attempts to give the model more
room produced no improvement -- good evidence that r~0.48 is close to the
actual ceiling of what one EEG reading can say about laser intensity in this
dataset, not a tuning gap.

**Why R>0.97 was dropped, not just under-achieved**: Stage 3's own finding
(eta^2=0.25-0.48 -- at a FIXED, calibrated intensity, up to half the variance
in response is real person-to-person difference, not noise) puts a hard
ceiling on how well any single EEG reading, however modeled, can recover an
exact stimulus/pain level across a population. r=0.97 would require pain to
be almost perfectly determined by a momentary EEG snapshot, which contradicts
evidence from this project's own data, not just outside literature. Chasing
it further would mean overfitting this dataset rather than building something
that generalizes to a new patient -- not something to ship as a clinical
metric. Stages 7-8's r=0.21-0.33, honestly validated and requiring no known
stimulus, remain the deployable result.

## Running it

```bash
# 1. Build the baseline-inclusive epoch tensor (from ../CNN/)
python ../CNN/build_epochs.py --root ../data --labels "../../Data Analysis/lep_eda_output/labels_combined.csv" \
    --tmin -1.0 --tmax 1.0 --out nurovo_epochs_baseline.npz

# 2. Extract features
python stage1_extract_features.py --input nurovo_epochs_baseline.npz --output features_baseline_response.csv

# 3. Normalize
python stage1_baseline_normalization.py --input features_baseline_response.csv --output features_normalized.csv

# 4. Fit the mixed-effects models
python stage1_intensity_analysis.py --input features_normalized.csv --output stage1_results.csv

# 5. Plots for the top features
python plots.py --features features_normalized.csv --results stage1_results.csv --outdir plots

# 6. Stage 2: derive the candidate Neural Response Score
python stage2_neural_response_score.py --features features_normalized.csv --results stage1_results.csv --output-prefix stage2

# 7. Stage 3: cross-subject consistency + standardized SD scale
python stage3_consistency_analysis.py --scored stage2_scored_trials.csv --results stage1_results.csv --output-prefix stage3

# 8. Stage 4: heteroscedasticity test, random-slopes model, individual differences
python stage4_variance_decomposition.py --scored stage2_scored_trials.csv --output-prefix stage4

# 9. Stage 5: does raw baseline EEG explain the residual better than hand-crafted features?
python stage5_train_residual_cnn.py --epochs-npz nurovo_epochs_baseline.npz --residuals stage4_trial_residuals.csv \
    --features features_normalized.csv --output-prefix stage5

# 10. Stage 6: baseline-state-adjusted, heteroscedasticity-corrected standardized score
python stage6_standardized_score.py --scored stage2_scored_trials.csv --blups stage4_subject_blups.csv --output-prefix stage6

# 11. Stage 7: healthy-norm score from baseline EEG alone, validated against measured pain threshold
python stage7_healthy_norm_score.py --data-root ../data --features features_normalized_all.csv --output-prefix stage7

# 12. Stage 8: population-baseline-deviation score + SD-band -> pain-group calibration table
python stage8_population_deviation_score.py --features features_normalized_all.csv --healthy-norm stage7_healthy_norm_reference.csv \
    --feature-list-from stage2_all_weights.csv --output-prefix stage8

# 13. Stage 9: ANCOVA with independent KNNs-folder ERP covariates
python stage9_ancova_covariates.py --scored stage2_scored_trials.csv --erp-features ../KNNs/features_cz_erp_merged.csv --output-prefix stage9

# 14. Stage 10: PCA variance graph on the chosen composite-score features
python stage10_pca_variance.py --features features_normalized_all.csv --feature-list-from stage2_all_weights.csv --output-prefix stage10

# 14b. Stage 10b: same, but on baseline-only features (no intensity involved anywhere)
python stage10_pca_variance.py --features features_normalized_all.csv --all-baseline-features --output-prefix stage10_baseline_only

# 15. Stage 11: map those baseline-only PCA components against laser_power
python stage10_pca_variance.py --features features_normalized_all.csv --all-baseline-features --target-col laser_power \
    --output-prefix stage11_baseline_vs_intensity

# 16. Stage 12: cross-validated regression, PCA components -> laser intensity
python stage12_pca_regression_intensity.py --features features_normalized_all.csv --output-prefix stage12

# 17. Stage 13: same question with t-SNE (descriptive) and UMAP (cross-validated)
python stage13_manifold_intensity.py --features features_normalized_all.csv --output-prefix stage13

# 18. Stage 14: Kernel PCA / Isomap / LLE, all cross-validated
python stage14_manifold_intensity_v2.py --features features_normalized_all.csv --output-prefix stage14

# 19. Stage 15 (dropped R>0.97 push): sensitivity covariate + nonlinear model on Stage 8's composite
python stage15_nurovo_pain_score.py --features features_normalized_all.csv --healthy-norm stage7_healthy_norm_reference.csv \
    --feature-list-from stage2_all_weights.csv --blups stage4_subject_blups.csv --sensitivity-hat stage6_sensitivity_hat.csv \
    --output-prefix stage15

# 20. Stage 16 (dropped R>0.97 push): raw 28-channel CNN, full 678-subject pool, vs laser_power
python stage16_full_scale_cnn.py --epochs-npz nurovo_epochs_baseline_all.npz --hand-crafted-scored stage15_scored_trials.csv \
    --output-prefix stage16
```

## First-run result (ds005293 + ds005473, 124 subjects, 10,984 trials)

Directionally consistent with the laser-evoked-potential literature: relative
alpha/beta/gamma power fall with intensity (desynchronization), broadband
RMS/variance/std rise, spectral/sample entropy and Hjorth mobility fall while
Hjorth complexity rises, mostly with high subject consistency (>0.9) and a
significant nonlinear (quadratic) component. See `stage1_results.csv` for the
full table.
