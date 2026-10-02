# Peak alpha frequency (PAF) from the existing C3/C4/Cz EEG

## What the supplied paper actually measures

[Furman et al. (2019), *Cerebral peak alpha frequency reflects average pain severity in a human model of sustained, musculoskeletal pain*](https://pmc.ncbi.nlm.nih.gov/articles/PMC6843105/), DOI [10.1152/jn.00279.2019](https://doi.org/10.1152/jn.00279.2019), measures sensorimotor PAF at **C3, Cz and C4**. Although the introduction describes an alpha peak, the operational estimator is **center of gravity (CoG)**:

`PAF = sum(frequency × spectral power) / sum(spectral power)` over **9–11 Hz, inclusive**.

They use five-second epochs, a Hanning taper, and **0.2-Hz frequency bins**. Each channel's epoch-level CoGs are averaged, then the three channel means are averaged to give a participant/visit ROI feature. Averaging CoGs is not the same as calculating a single CoG from the pooled spectrum. The paper calls its weights spectral “amplitude” but describes the procedure as PSD and center of spectral power; this implementation explicitly uses **linear power**, not log power or square-root power.

The published association is between **pain-free resting PAF before NGF injection and subsequent average musculoskeletal pain across days**, not an instantaneous pain-state classifier. The reported combined Spearman correlation is **−0.47, p=0.01**; these are **the authors' findings, not results from our recordings**. The authors did not detect PAF slowing after NGF, and PAF shifts were not significantly associated with average pain. A lower PAF must not simply be translated into “more pain right now.”

## Extraction performed here

I reread the **eight accessible, SHA-256-verified raw binaries**, rather than trying to recover PAF from the old five-band DE features. Those features aggregate away the within-alpha frequency distribution. The previous two-second windows also do not supply the paper's five-second spectral spacing.

The new extraction uses the existing selected excerpts (up to 300 seconds), scalp-channel average reference, 128-Hz resampling and 1–40 Hz filtering. It then forms **nonoverlapping five-second windows** from the continuous preprocessed signal; it does not join previously retained two-second windows across rejection gaps. Screening uses all retained scalp channels with the same amplitude, flatline and finite-data criteria as before. Different window length can change which windows are rejected.

Each retained C3/C4/Cz window receives a **symmetric Hann-tapered periodogram**, constant detrending, no zero padding and linear PSD weighting. At 128 Hz, 640 samples give 0.2-Hz frequency spacing. This matches the paper's epoch duration, taper type, band and averaging scheme, but is **not a complete replication**: the paper uses 3-minute eyes-closed resting recordings, 500-Hz processing, a different filter and montage, visual channel inspection and PCA-based ocular rejection. We retain the project's **no-PCA rule** and automated screening; residual ocular/muscle contamination remains possible. Native device/reference differences remain.

No new dimensionality reduction or pain regression is needed to calculate this feature. Output units are **Hz**, not alpha power or DE. Uniform amplitude scaling cancels in the CoG ratio.

## Actual extracted values

These values are measurements from the selected excerpts, **not** the paper's participants. The OSF rows are two conditions from the **same** subject. C3/C4/Cz means are averages of per-epoch CoGs; ROI is their equally weighted mean. Rounded display does not alter the full-precision CSV/JSON.

| Recording | Retained 5-s epochs | C3 (Hz) | C4 (Hz) | Cz (Hz) | ROI CoG (Hz) |
|---|---:|---:|---:|---:|---:|
| ds005292 | 60 | 9.976 | 9.978 | 9.992 | **9.982** |
| ds005293 | 60 | 10.069 | 10.117 | 9.992 | **10.060** |
| ds005289 | 41 | 10.159 | 10.226 | 10.229 | **10.205** |
| ds005286 | 60 | 10.158 | 10.080 | 10.084 | **10.108** |
| ds005284 | 59 | 10.131 | 10.061 | 10.145 | **10.112** |
| ds005307 | 13 | 10.025 | 9.984 | 9.997 | **10.002** |
| OSF eyes closed | 60 | 9.798 | 9.694 | 9.761 | **9.751** |
| OSF eyes open | 58 | 9.929 | 9.826 | 9.938 | **9.898** |
| ds008115 | — | — | — | — | Raw download unavailable; no estimate |

**411 epochs** were retained. Four five-second windows were rejected: one each in ds005284 and ds005307, and two in OSF eyes-open. Incomplete trailing samples are not padded into epochs. The half-excerpt ROI changes range from **−0.121 to +0.139 Hz**; these are descriptive within-excerpt checks, not test–retest or cross-subject reliability.

### Important alpha-quality limitation

A narrow-band centroid exists even for a flat or sloped spectrum. A flat 9–11 Hz spectrum yields exactly 10 Hz **without containing an alpha oscillation**. Also, several mean spectra have local maxima outside 9–11 Hz: e.g., ds005293 C3 at 11.8 Hz and ds005307 C3 at 12.0 Hz. The narrow centroid can therefore be a truncated summary rather than the true dominant alpha frequency.

To make that visible, the output also saves **8–12 Hz CoG**, the raw 8–12 Hz maximum, and an exploratory local-peak estimate from the **mean spectrum** (≥1 dB prominence within a 7–13 Hz search neighborhood). This prominence threshold is an explicit **heuristic**, not taken from the paper and not a validated physiological quality cutoff. All 24 channel mean spectra happen to pass it; that does not prove all individual epochs have clear alpha peaks. The 8–12 Hz maximum and local-peak frequency differ conceptually from the paper's epoch-averaged 9–11 Hz CoG. Weak or missing qualifying peaks would be left missing, not replaced with 10 Hz.

## Interpretation and files

**Yes, the feature is extractable.** It is a promising additional, amplitude-normalized frequency feature. These excerpts are not all eyes-closed rest or established pain-free baselines, however, so differences between datasets cannot be attributed to pain sensitivity. This extraction alone does not validate the paper's association, a state-only pain predictor, or a cross-person clinical biomarker. It does not fit a correlation using the nine summary rows or treat two resting conditions as independent subjects.

- `paf_features.py`: raw extraction, centroid and peak functions, aggregation.
- `paf_outputs.py`: actual-data spectral inspection figure and executed notebook.
- `test_paf_features.py`: analytical/synthetic **unit tests only**, not fabricated study results.
- `paf_results.ipynb`: executed tables and embedded spectrum figure.
- `summary.csv`, `results.json`: full-precision recording/channel summaries, screening counts, hashes and limitations.
- `*_epochs.csv`: per-epoch C3/C4/Cz and ROI CoG with original epoch index and onset time.
- `*_spectra.npz`: mean spectra, per-epoch narrow CoGs and channel order.
- `alpha_spectra.png`: measured mean spectra and shaded centroid band.

All the files above are together in `eeg-analysis/paf_results/`. From that folder,
using the existing Python dependencies:

```bash
python paf_features.py
python paf_outputs.py
python -m unittest test_paf_features.py
```

Shared preprocessing, the raw-data manifest and dependencies remain in the parent
`eeg-analysis/` folder. Raw data are resolved from the parent analysis folder or
the preserved hidden download area, as in the DE study. OSF-derived outputs are
covered by `../OSF_LICENSE_NOTICE.md`. Existing two-second analyses and their saved
results are not overwritten.