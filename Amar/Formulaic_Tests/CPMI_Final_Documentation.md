# Composite Pain Modulation Index (CPMI) – Final Comprehensive Documentation

## Purpose
Transparent, literature-directed formulae that map clinical (and optionally EEG) inputs onto a 0–10 scale, operationalising the Multidimensional Pain Modulation Framework and the EEG feature roles in Table 2.

## Glossary
- **NRS**: Numerical Rating Scale (0–10 self-reported pain intensity)
- **PCS**: Pain Catastrophizing Scale
- **CPM**: Conditioned Pain Modulation (human equivalent of DNIC – endogenous “pain-inhibits-pain” inhibition)
- **PAF**: Peak Alpha Frequency (frequency of maximum alpha-band power)
- **EEG**: Electroencephalography
- **HADS**: Hospital Anxiety and Depression Scale
- **BPI**: Brief Pain Inventory
- **CRPS / LBP / SCI-NP / HC**: Complex Regional Pain Syndrome / chronic Low Back Pain / Spinal-Cord-Injury neuropathic pain / Healthy Controls
- **Z-score**: (value − reference mean) / reference standard deviation

## Formulae (gender coded 1 = female/woman, 0 = male/man)

**A – Clinical Only**  
0.70·NRS + 0.40·Gender_F + 0.25·Age_norm + 0.45·PCS_z − 0.40·CPM_z + 0.35·CentralSens_z + 0.15·(Gender_F × Age_norm)

**B – Core EEG**  
0.60·NRS + 0.25·Gender_F + (−0.50·Z_α − 0.45·Z_PAF + 0.35·Z_β + 0.45·Z_θ + 0.30·Z_γ)

**C – Full Multidimensional (recommended)**  
0.55·NRS + 0.30·Gender_F + 0.18·Age_norm + 0.35·PCS_z − 0.32·CPM_z + 0.28·CentralSens_z + EEG_mod + interactions  
EEG_mod = −0.40·Z_α − 0.35·Z_PAF + 0.28·Z_β + 0.38·Z_θ + 0.22·Z_γ + 0.55·Z_connectivity + 0.18·Z_entropy

**D – Connectivity-Heavy**  
0.50·NRS + 0.25·Gender_F + 0.30·PCS_z − 0.30·CPM_z + (0.70·Z_connectivity + 0.30·Z_entropy − 0.25·Z_PAF + 0.25·Z_θ)

**E – Conservative Power**  
0.75·NRS + 0.20·Gender_F + (−0.30·Z_α + 0.30·Z_θ + 0.20·Z_β)

All scores clamped to [0, 10].

## Real-data validation (Formula A only)
Source: Gozzi et al. Subjects.csv (Zenodo 12570277). Usable n = 140 (37 healthy, 103 chronic).

| Group | n | Mean CPMI_A | Median | Max |
|-------|---|-------------|--------|-----|
| Healthy | 37 | 0.15 | 0.00 | 1.36 |
| Chronic | 103 | 3.15 | 2.87 | 8.78 |

r(CPMI_A, NRS_now) = 0.96; r(CPMI_A, PCS) = 0.68.

No public processed EEG feature table matching Table 2 variables could be joined to the same subjects; therefore Formulae B–E were not scored on real individuals. No values were fabricated.

## How the remaining gaps can be closed
1. Obtain or compute subject-aligned Table-2 EEG features from already-public raw recordings.
2. Re-estimate coefficients by regularised regression on the enlarged multi-modal sample.
3. Validate change scores on any longitudinal clinical+EEG releases.
4. Perform calibration and decision-curve analysis versus NRS alone.

## Files
- cpmi_multi_formulas.py
- CPMI_MultiFormula_Validation.xlsx
- real_data/subjects_gozzi.csv
- real_data/gozzi_cpmi_A_results.csv
- CPMI_Final_Comprehensive_Report.docx (this content in Word form)
