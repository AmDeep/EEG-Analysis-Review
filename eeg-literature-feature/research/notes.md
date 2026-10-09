# Research Notes: EEG variance, signal strength and pain

**Status:** tables assembled; verification complete; requested access quota not achieved
**Depth:** Deep
**Requested output:** At least 100 papers/research items with links, accessible papers and data, and a table explaining features and their potential use in prediction.

## Plan

- **Question:** Which open EEG research items support feature selection for signal strength, explained variation or pain prediction, and what public data can reproduce or test them?
- **Scope:** Direct pain evidence plus clearly labelled transferable EEG variance/signal-quality methods and data resources. Do not represent general EEG evidence as demonstrated pain prediction.
- **Audience:** Technical EEG researcher.
- **Deliverable:** New-folder research table in CSV and Excel, feature/evidence table, claim audit, and source/access verification records.
- **Integrity:** Deduplicate papers by DOI/title. Count datasets/software separately from papers. Separate public data from data available only on request. Never invent R², accuracy, sample size or accessible data.
- **Prior-study boundary:** Literature research only; no changes to EEG preprocessing, state discovery or clustering outputs.

## Focus Areas

| Area | Status | Target |
|---|---|---|
| Direct pain, evoked responses and oscillatory prediction | reviewed | 21 initial screened records; not all access-qualified |
| Phase connectivity and cross-frequency coupling | reviewed | 11 initial screened records; not all access-qualified |
| Spectral power, alpha frequency and asymmetry | reviewed | 11 initial screened records; not all access-qualified |
| Envelopes, variability and nonlinear complexity | reviewed | 8 initial screened records; not all access-qualified |
| Signal quality, variance, prediction validation and open data | reviewed | 23 initial screened records; includes data/software |
| Public-data gap fills | reviewed | Two targeted gap fills; 5 distinct additions, with nonqualifiers disclosed |
| Supplementary open-paper bibliography | access screened only | 105 additional real open-text entries; public data NOT verified |

## Coverage Checklist

- [x] Distinct research items with source-backed titles and links; paper/data/software types separated.
- [x] Open full-text/access status assessed separately from data availability; unknowns remain explicit.
- [x] Public data links and label/modality limitations.
- [x] Feature definitions and qualified prediction rationale.
- [x] Explained variance distinguished from correlation and classification accuracy.
- [x] Supplied examples and strong claims checked; 13-row audit.
- [x] Current C3/C4/Cz, 128 Hz, 1–40 Hz and five-second-window limitations flagged where relevant.
- [x] Count shortfalls and unverified entries explicitly disclosed; screened bibliography not counted as eligible.

## Findings Log

Final deduplicated catalogue: 172 items, including 145 paper entries. Of these, 67 received study/access review and 105 are supplementary automated bibliographic/full-text-access screens. Only 33 items qualify as open text/resource plus verified public EEG; 14 are research papers, including 10 direct pain papers. Four items provide public derived data only. The requested minimum of 100 eligible items/papers is not verified.

Tables contain 23 candidate feature families and a 13-row claim audit. Candidate rationales are not promised predictive effects; source-specific metrics retain their target and validation limitations.

## Conflicts & Open Questions

- Piqβ 2024 explicitly excludes public raw-data sharing; pilot association and threshold claims are not universal external validation.
- PhysioPain raw/derived descriptions conflict. The catalogue conservatively labels accessible 1-Hz band exports as derived data; raw voltage EEG was not verified.
- 2021 Zolezzi proceedings and 2024 Ryu journal research are distinct; do not merge citations by similar topic.
- Public OSF metadata or a sign-in screen alone does not establish available raw files. Mari's public project had zero files; Han's file access was unverified.
- Reanalyses of the same 36-person Mendeley cohort are distinct papers, not independent cohorts.

## Gaps

- The strict 100-item/public-data requirement remains unmet. Supplementary candidates have real open text but unverified dataset access and study results.
- Public-data verification documents landing pages/files and article linkage, not downloads or independent reconstruction of every dataset.
- No uniform held-out pain-rating R² benchmark exists in these heterogeneous sources; classification accuracy, correlation and preserved signal variance remain separate.
- No preprocessing, dimensionality reduction, clustering or state-count analysis was rerun as part of this literature-table request.
