# Independently sourced EEG feature validation

**Partial empirical result.** This is a new supervised physical-stimulus experiment,
not a modification of the previous unsupervised state-discovery analyses.
Multiple participants are not multiple datasets.

## What was actually scored

- OSF **f2mua**, DOI **10.7554/eLife.62809**: ten presampled people, 2520 heat trials.
- Physical temperatures **42/46/48°C**, mapped from the author's source codebook.
- Four-fold **participant-held-out** prediction. No individual contributes training
  and test trials to the same fold.
- Native temperature regression: fixed mean baseline, Ridge, Extra Trees.
- Secondary classification: **42°C versus 46/48°C**, not a pain diagnosis.
- Power-only, all-compatible candidates and prestimulus-only, with no test-driven selection.
- 505 cached numeric predictors plus two training-only CSP features: **507 full-model predictors**.
- **21/23 feature families** have candidate implementations; high gamma and CVUE/Piqβ
  are not tested or silently approximated under their original names.

The downloaded second cohort **w25k4** is *not scored*: its deposited cue epochs
end about 100 ms after shock. The planned poststimulus windows cannot be obtained.
Other candidate sources were excluded, inaccessible or lacked verified dose labels.
Therefore **the requested multiple-independent-dataset blind test is incomplete**.
No cross-dataset accuracy is fabricated. The original two-cohort “great accuracy”
criterion is unmet/untestable with these data, regardless of single-cohort results.

## Open first

| File | Contents |
|---|---|
| `blind_stimulus_validation.ipynb` | Executed notebook, fitted-model source and measured tables; extraction source displayed |
| `measured_results.html` | Self-contained readable report with actual tables |
| `blind_stimulus_results.xlsx` | Regression, classification, gains, ablations, coverage and provenance tables |
| `accuracy_results.csv` | All fixed-model scores, native-unit errors and clustered intervals |
| `heldout_predictions.csv` | Every held-out trial prediction and fold |
| `participant_accuracy_results.csv` | Per-person scores |
| `feature_set_gains.csv` | Paired all-compatible minus power-only balanced-accuracy differences |
| `family_ablation_results.csv` | Fixed logistic, participant-held-out exploratory family removals |
| `permutation_controls.csv` | Every training-label permutation result |
| `permutation_summary.csv` | Twenty-permutation null summaries; minimum attainable p=1/21 |
| `generated_feature_catalogue.csv` | All 505 cached columns, family and missingness |
| `feature_family_coverage.csv` | All 23 original families, implemented/untestable status and qualifications |
| `original_feature_catalogue.csv` | Unchanged snapshot of the supplied feature-source list |
| `dataset_eligibility.csv` | Successful, rejected and blocked cohorts |
| `dataset_exclusion_audit.json` | Feature-source exclusion records and catalogue hash |
| `target_mapping.json` | Source-defined targets, columns, units and alignment |
| `selected_files.json`, `download_manifest.json` | Prescored sampling, public URLs, byte sizes and signal SHA256 |
| `extraction_audit.json` | Per-person usable trials and reasons for rejection |
| `participant_target_counts.csv` | Actual held-out target counts |
| `protocol.json`, `protocol_frozen.json` | Original model/feature/sampling design and hash |
| `protocol_availability_amendment.json` | Prescore restriction to the only feasible cohort |
| `analysis_frozen.json` | Source-code hashes before the first model fit |
| `split_leakage_audit.json` | Training/testing subject lists with no overlap |
| `features.npz`, `trial_metadata.csv` | Actual feature/covariance cache and aligned target rows |
| `verdict.json` | Explicit validation scope, limits and software versions |
| `checksums.csv` | Delivered file SHA256 checksums |
| `blind_stimulus_notebook_and_results.zip` | Notebook, executable code, cache and all result files together |

## Reproduce

Unzip the bundle and use the included cache to open/run the notebook in this folder.
The notebook executes the actual evaluation code rather than loading imagined scores.
Raw signal files are deliberately not included in the bundle.

Python dependencies: NumPy, SciPy, pandas, scikit-learn, **fooof==1.1.1**,
nbformat, nbclient, ipykernel and xlsxwriter. FOOOF 1.1.1 is functional but
deprecated in favor of specparam; it is pinned to reproduce this analysis.
Measured installed versions are in `verdict.json`. Package management on Replit
should use the workspace package manager rather than a separate virtual environment.

To acquire and recompute signals:

```text
python acquire.py
python build_features.py
```

The acquisition script uses public OSF listings and prespecified hash sampling;
it also retrieves the rejected VR cohort to reproduce the timing audit. It writes
approximately 3.4 GB outside the result folder to `generated-artifacts/blind-stimulus-eeg`.
This directory is ignored by Git and publishing. Re-running extraction overwrites the
derived cache, not the previous project analyses. The project folder structure in
the original workspace is expected by the acquisition/extraction source; for a
standalone clone keep the copied literature catalogue and exclusion records or adapt
paths explicitly. Evaluation with the included cache runs directly in this folder.

## Interpretation and caveats

1. **Physical input, not pain:** temperatures are experimenter-controlled doses.
   No subjective ratings or missing current values were invented.
2. **Source map:** MATLAB column 2 is the condition combination. Conditions
   1–3/19–21 map to 42°C; 4–6/22–24 to 46°C; 7–9/25–27 to 48°C.
   Column 4 contains cue codes 37–42, not stimulus temperature. The OSF wiki has
   an older four-column schema description; deposited files actually have six.
3. **Alignment:** the fixed deposited stimulus trigger is used. No shift based
   on the true temperature’s ramp-up time is applied.
4. **Signal processing:** prespecified nine common electrodes, average reference,
   fixed −10 to +30 ms trigger interpolation for every class, causal fourth-order
   1–40 Hz filter, 500→250 Hz resampling. Amplitude and completeness QC use no labels.
   Heat MAT voltage values assume BrainAmp microvolts because no unit field is supplied.
5. **No exact-algorithm overclaim:** short-window spectral/entropy/connectivity/PAC
   estimates and component-like extrema are exploratory implementations.
   Generic Hilbert-envelope CV is not Piqβ; 30–40 Hz is not high gamma.
6. **Baseline:** majority classification reaches 66.7% ordinary accuracy but only
   50% balanced accuracy. Continuous errors are in Celsius, not a misleading
   percentage “regression accuracy”. Exact-level accuracy rounds to the three
   source-defined temperatures without fitting a cutoff.
7. **Uncertainty:** 1000 participant-cluster resamples, conditional on fitted models.
   Only ten people, one cohort, retrospective validation; cue expectation,
   arousal, movement and stimulation artifacts are not ruled out.
8. **Feature conclusions:** family ablations are exploratory and correlated,
   not a license to select on held-out results and then claim a new blind score.
9. **Second cohort needed:** usable independent EEG with physical dose labels
   and adequate poststimulus duration is required before the cross-dataset claim.

Sources: https://osf.io/f2mua/ ;
https://elifesciences.org/articles/62809 ;
https://api.osf.io/v2/wikis/xc3vs/content/ ;
https://osf.io/w25k4/ ;
https://github.com/dhewitt16/PainAnticipationVRStudy .
