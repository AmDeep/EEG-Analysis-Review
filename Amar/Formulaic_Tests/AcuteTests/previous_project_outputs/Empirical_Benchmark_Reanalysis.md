# Empirical benchmark reanalysis — actual acute-pain EEG study

**Status:** reproducible reanalysis of published numerical outputs, not raw EEG validation of the project's 0–10 score formulae.

## Source and provenance

Sun et al., *Detecting acute pain signals from human EEG*, J Neurosci Methods 347 (2020), 108964. Open full text: https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/. Public first-cohort source cited by that paper: https://osf.io/bsv86/.

The analysis transcribes the 16 published Table 2 rows (SSM unsupervised AUC, SVM supervised peak AUC and peak accuracy). Table values were rounded to two decimal places in the article. The same table's 16 lines are multiple train/test modality configurations arising from **two chronic-back-pain participants**, not 16 independent patients. Therefore row-based inference is exploratory only.

## Results reproduced from Table 2

| Metric | Mean | SD | Median | Range |
|---|---:|---:|---:|---:|
| SSM unsupervised AUC | 0.711 | 0.068 | 0.71 | 0.60–0.81 |
| SVM supervised peak AUC | 0.614 | 0.120 | 0.59 | 0.43–0.83 |
| SVM peak accuracy | 0.602 | 0.069 | 0.605 | 0.50–0.73 |
| Paired AUC difference (SSM−SVM) | 0.096 | 0.124 | 0.115 | -0.18–0.28 |

SSM had higher AUC in **13/16** rows; SVM had higher AUC in **3/16** rows. Nominal mean difference was 0.096 AUC points, with a conventional paired-t 95% interval of [0.030, 0.162]. Because the rows are dependent and only two patients underpin the table, this interval is not a valid population-level confidence interval.

### Sensitivity tests on rounded row values (exploratory only)

| Test | Estimate / statistic | p-value | Interpretation limit |
|---|---:|---:|---|
| Paired t-test, SSM vs SVM AUC | t=3.115, df=15 | 0.0071 | Depends on treating configurations as paired independent units; not population-valid here |
| Wilcoxon signed-rank, two-sided | W=18.5 | 0.0105 | Similar direction to paper's reported p=0.0113; slight difference expected from rounded table values and implementation |
| Exact sign test, two-sided | 13 wins vs 3 losses | 0.0213 | Directional consistency check; still dependent rows |
| Pearson association of SSM vs SVM AUC across rows | r=0.233 | 0.3859 | Cross-row descriptive only |
| Spearman association of SSM vs SVM AUC across rows | ρ=0.252 | 0.3467 | Cross-row descriptive only |

These tests do **not** show that the new pain-intensity formulas work. They show that on this paper's rounded benchmark outputs the SSM detector's AUC tends to exceed the supervised SVM AUC across the listed configurations. The publication itself reports a signed-rank p-value of 0.0113 for the comparison (source, Table 2).

## Additional findings reported by the original authors (not recalculated)

- The public acute-pain EEG cohort included 50 analyzed healthy participants (the source study recruited 51; the motor condition analysis used 50).
- Among the 50, 32 were reported to have clear localized evoked potentials and 18 were not; group AUC differed at rank-sum p=1.9×10⁻⁵. This is a reminder that signal quality/ERP detectability varies appreciably across individuals.
- High-intensity pain detection latency had a median of 180 ms versus 202 ms for low intensity, with rank-sum p=1.5×10⁻⁵. Detection latency and reaction time showed a weak, nonsignificant Spearman correlation (ρ=0.29, p=0.06).
- Mean cross-subject SSM AUC was reported as 0.62; best single-subject SSM AUC was 0.79, while best supervised SVM AUC was 0.95. Best-case results should not be treated as typical performance.
- In the chronic-back-pain demonstration, the authors analyzed two patients (ages 69 and 48). The table's reported NRS summaries show overlap between painful pinprick and non-painful monofilament, especially for subject 1 (hand medians both 1; back 1.5 vs 1). Thus stimulus condition is not a perfect proxy for experienced pain intensity.

## Which project formulae were applied?

**None of the proposed 0–10 EEG intensity formulae were fitted or applied to these records.** The paper's table contains model AUC/accuracy summaries, not individual EEG samples, spectral features, LEP amplitudes paired to each trial rating, or trial-level predicted intensities. It would be scientifically incorrect to call an AUC-to-0–10 mapping a pain score. Formula families remain unvalidated until actual trial-level EEG and labels are read.

## Blocker observed and reproducibility

The runtime attempted raw-file access to `raw.githubusercontent.com`, `api.github.com`, and `openneuro.org`, but DNS name resolution failed. No raw EEG or extracted feature file was present in `/mnt/data/EEG_Pain_Scale_Project`. The script deliberately does not fabricate trials or estimate coefficients from the aggregate benchmark table.

Run the aggregate reanalysis: `python published_benchmark_reanalysis.py --outdir .`

Run the intended genuine-trial EEG pipeline when a real preprocessed feature CSV is available: `python eeg_pain_models.py --features /path/to/real_trial_features.csv --outdir results`

## References

1. Sun G, Wen Z, Ok D, et al. Detecting acute pain signals from human EEG. *Journal of Neuroscience Methods*. 2020;347:108964. https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/
2. Tiemann L, et al. Public EEG dataset referenced by Sun et al. Open Science Framework record: https://osf.io/bsv86/.
3. Zhao X, et al. A comprehensive EEG dataset of laser-evoked potentials for pain research. *Scientific Data*. 2025;12:1536. https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/ (independent larger dataset; raw per-trial EEG/ratings are needed for the target regression analysis).

## Further robustness observations

- Leave-one-row-out diagnostic: mean SSM-minus-SVM AUC difference stayed positive, ranging from 0.084 to 0.115; SSM retained 12–13 wins among the remaining 15 rows. This shows the descriptive advantage is not driven by only one table line, but is **not** a population-level robustness test because rows are dependent.
- Exact one-sided sign test for SSM greater than SVM: p=0.0106; two-sided p=0.0213. The direction is consistent with the two-sided Wilcoxon result.
- Naive one-sample tests of the 16 SSM AUC values vs 0.5 (t=12.378, p=2.83e-9), SVM AUC vs 0.5 (t=3.806, p=0.00172), and SVM accuracy vs 0.5 (t=5.930, p=2.76e-5) were also computed from the printed table, but **must not be treated as inferential results**: the 16 rows repeat overlapping train/test configurations and only two patient datasets. AUC is also not a pain-intensity score; it measures discrimination.
- The table-based Pearson association between SSM and SVM AUC was r=0.233 (nominal p=0.386), and Spearman rho=0.252 (nominal p=0.347). AUCs across methods did not move in close lockstep over these rounded entries; this is only a two-patient descriptive comparison.
- The original paper's patient NRS summary implies painful-minus-non-painful median differences of 0, 0.5, 1, and 2 points across the subject/site combinations. There are only four aggregated contrasts; there are no trial-level NRS observations here, so no paired test or uncertainty estimate is justified.

## Extra formula families added to the candidate registry

1. **State-space latent EEG change-point detector (SSM/Kalman):** latent state evolves as `z[k] = a*z[k-1] + epsilon[k]`, while multiregion EEG amplitudes are modeled as `y[k] = c*z[k] + d + noise`; baseline z-score and uncertainty limits can detect an acute-event onset. This estimates detection/time, not a calibrated 0–10 pain intensity. The referenced study used 100 ms bins and 1–100 Hz band-passed time-averaged ROI amplitudes.
2. **Spectral linear SVM benchmark:** uses EEG spectral power at 4–12 Hz, 12–45 Hz, and 60–100 Hz for binary high-vs-low acute-pain stimulus classification. It is a classifier, not an ordinal intensity scale; any continuous 0–10 mapping would need separately trained calibration and external validation.

Both are literature-informed candidate comparators, not newly fitted formulas in this project. Do not interpret their published benchmark as validating the other proposed score equations.
