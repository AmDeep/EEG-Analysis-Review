# EEG variance, signal strength and pain: research tables

Research/access-check date: **2026-10-09**.

The catalogue contains **172 distinct screened research items**, including **145 paper entries**. **33 items** have both open publication/resource access and verified public EEG data; **14 of those are research papers**, including **10 direct pain papers**. A further **4 items** have public derived data only. The requested minimum of 100 open-access original papers with public EEG data was NOT established: 14 were verified under that strict criterion. No inaccessible/unverified paper or dataset/software resource is counted as an eligible original paper.

Of the catalogue entries, **67 were reviewed for study/access evidence**, while **105 are explicitly supplementary bibliographic/full-text-access screens**. The latter provide real open-paper metadata and saved source excerpts, but their datasets, metrics and validation designs have not been independently reviewed. They do not satisfy the public-data requirement.

The literature table distinguishes direct pain evidence from transferable EEG methodology. General EEG decoding, noise-reduction or reliability results are not demonstrated pain-prediction gains. Reusing an existing dataset does not create an independent cohort. Prediction rationale is a proposed use, not a guaranteed effect, and literature results do not establish an unsupervised physiological-state count.

## Files

| File | Contents |
|---|---|
| eeg_research_tables.xlsx | Filterable workbook with summary, catalogue, public-EEG subset, derived-only subset, feature table, claim audit and nonqualifying access records |
| literature_table.csv | Complete screened table: citation, paper/data links, feature, potential use, verified metric, target, validation design, access and limitations |
| verified_public_eeg_items.csv | Only rows assessed as open text/resource + verified public EEG |
| feature_prediction_table.csv | 23 feature families, definitions, qualified prediction rationale, source IDs and current-pipeline limitations |
| supplied_claims_audit.csv | Checks of the supplied examples and strong claims, with primary sources and decisions |
| summary.json | Machine-readable counts; separates papers from other items and qualifying from unverified access |
| research/sources.json | Source registry keyed by the table's mechanically assigned source IDs |
| research/sources/ | Saved page/statement/snippet evidence; consult the path in each row |
| research/all_records.json | Deduplicated structured research records |
| build_tables.py | Rebuilds tables from the saved research records without inventing rows |
| screen_open_methods.py | Documents the automated supplementary screen; no dataset-access eligibility is inferred from mentions |

## Interpretation and safeguards

Explained variation in pain ratings requires a stated outcome, evaluation set and R² definition. Correlation, binary accuracy, explained variance of EEG signals and variance retained by a representation are not interchangeable. Reported source-specific figures must be interpreted in their original design; no cross-study benchmark equivalence is inferred.

The current C3/C4/Cz pipeline at 128 Hz with 1–40 Hz filtering cannot extract frontal/occipital network effects or 60–100 Hz gamma from its retained representation. Event-related amplitudes, latencies and ERD require correctly aligned events and pre-stimulus baselines. The feature table documents these requirements; this research does not modify the existing analysis.

Publication access and data access were assessed separately from saved evidence. “On request”, publicly readable results, supplementary summary tables and open code are not silently promoted to public raw EEG. Public derived data are labelled separately. A verified public dataset link documents accessibility, not independent verification of every signal file, pain rating or license condition.

## Limitations

This is an evidence-screened research catalogue, not an exhaustive systematic review or a clinical recommendation. Any strict quota shortfall is disclosed above. Entries with unverified access remain visible for follow-up but do not satisfy the requested public-data criterion. Metrics not verified from available source text are left explicitly unverified rather than fabricated.

## New physical-stimulus prediction experiment

`blind_stimulus_validation/` contains the executable notebook, actual measured
participant-held-out temperature results, generated-feature catalogue, code and
dataset/exclusion audits. It uses an independently sourced heat cohort not in the
feature-source papers. The requested multiple-cohort blind validation remains
incomplete: the downloaded second cohort lacks adequate postshock EEG, and no
cross-dataset accuracy is fabricated. These scores concern physical stimulus
temperature, not subjective pain. Earlier literature tables and analyses are unchanged.
