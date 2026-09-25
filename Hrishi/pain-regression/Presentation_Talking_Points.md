# Continuous Stimulus and Pain Prediction — Presentation Notes

## Goal and approach

- Predict **continuous stimulus intensity from EEG**, and study how stimulus relates to reported pain.
- Use **Zhao** for laser-energy and pain labels, plus independent **Gozzi** data for pain ratings, EEG and skin conductance. Gozzi has no physical-intensity labels.
- Compare **seven model families**, select using development cross-validation, and keep participants separate between global training and testing.

## Main results

| Task | Selected model | Test R² | Mean absolute error |
|---|---|---:|---:|
| EEG → stimulus, Zhao | Extra Trees | 0.438–0.446 | 0.437–0.602 J |
| EEG → stimulus, Zhao | Random Forest | 0.429–0.439 | 0.445–0.600 J |
| Personalized pain, Gozzi | Spline + Ridge | **0.726** | **0.87 pain points** |
| Personalized pain, Gozzi | Ridge | 0.698 | 0.97 pain points |

Zhao ranges cover ds005293 and ds005473. Pain scores use a 0–10 scale. **R² is a regression score, not classification accuracy.**

## What to emphasize aloud

- **“Stimulus intensity and experienced pain are different targets.”** The original intensity task reached about R² 0.44; the near-0.75 result concerns personalized pain.
- **“Calibration made the biggest difference.”** The Gozzi result uses EEG, skin conductance and up to 10 earlier labeled trials per body area. It was evaluated on 1,103 separate trials from 24 participants; the calibration-mean baseline scored R² 0.566.
- **“More data did not automatically improve prediction.”** The tested joint Zhao–Gozzi band-power training approach worsened results.
- **“The next step is fresh validation.”** The 0.726 result is exploratory because that test cohort had already been viewed; its 95% interval is wide: 0.483–0.853. Confirm it on new participants/sessions and obtain matched physical-intensity data for the original goal.
- Gozzi provided an independent dataset for testing whether physiological signals and personal calibration could predict reported pain.
**Closing line:** “We found promising personalized pain prediction, while EEG-based stimulus-intensity prediction remains the harder, unfinished goal.”
