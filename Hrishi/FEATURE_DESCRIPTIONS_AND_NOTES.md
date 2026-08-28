# Feature Descriptions and Experimental Design Notes

## Experimental Design Questions & Answers

### 1. Is epoch self-extracted, or 3-second bursts?
**3-second bursts** — the source data comes pre-epoched into ~3-second windows around each laser pulse by the original dataset authors. Epoch column indexes through those already-segmented trials.

### 2. Time between epochs?
Varies by sub-experiment, roughly 3–8 seconds of jittered inter-trial interval, plus a few extra seconds for fixation and rating in between. So the real gap between stimulus onsets is longer than the ITI number alone suggests.

### 3. Interaction between trials?
Yes — repeated stimulation causes both physical skin sensitization/fatigue and neural habituation (shrinking N2-P2 amplitude over the session), even when the person's reported pain doesn't change. Therefore, some sub-experiments rotate the stimulation site instead of hitting the same spot repeatedly.

### 4. Does this make it resemble chronic pain?
Partially — repeated-trial sensitization is mechanistically related to central sensitization, one driver of acute-to-chronic transition. But it's still a single-session, healthy-volunteer paradigm, not a real stand-in for actual chronic pain data (not too much data to assume).

---

## Feature Descriptions

### Metadata Features
**sfreq** — The sampling frequency (Hz) the EEG was recorded at for that trial/dataset (e.g., 1000Hz or 1024Hz). It's metadata rather than a physiological feature — it just tells you the time resolution of the recording, and different sub-experiments in the collection use slightly different values depending on which EEG system recorded them.

**gamma_band_hz** — The exact frequency range (e.g., "30-80") used to compute gamma power for that row. Not a feature itself — it's metadata documenting which gamma sub-band was used, since "gamma" isn't strictly standardized across studies.

### Event-Related Potential (ERP) Features
**n2_amp** — The amplitude (µV) of the N2 wave, a negative voltage deflection ~150–350ms after a painful stimulus. Reflects how strongly the sensory cortex reacted, and is one half of the classic laser-evoked potential pair.

**n2_lat** — The timing (seconds) of that N2 peak. Faster latency generally reflects quicker A-delta fiber conduction speed.

**p2_amp** — The amplitude of the P2 wave, a positive deflection ~300–550ms post-stimulus. The most consistently pain-intensity-correlated single ERP component in the literature.

**p2_lat** — The timing of the P2 peak, used alongside p2_amp to describe the full late cortical response shape.

**n2p2_amp** — The peak-to-peak amplitude between N2 and P2 (p2_amp minus n2_amp). The single most commonly cited pain-intensity biomarker in laser-EEG research.

### Spectral Power Features
**gamma_power** — Power in the gamma band after the stimulus, computed over the specific frequency range given in gamma_band_hz. Thought to reflect moment-to-moment pain salience, though vulnerable to muscle-artifact contamination.

**alpha_erd_pct** — Percent reduction in alpha power after the stimulus vs. a pre-stimulus baseline. Bigger drops mean the brain is more actively engaged; currently 100% missing in your extracted data.

**beta_erd_pct** — Same idea as alpha ERD, but for the beta band — linked to disengaging the current motor state as the brain prepares a protective response.

**psd_delta** — Delta-band (1–4Hz) power via Welch's method. More associated with drowsiness generally, but appears as a secondary marker in some chronic-pain resting-state work.

**psd_theta** — Theta-band (4–8Hz) power. Rises with pain intensity in acute settings, and is one of the strongest chronic-pain resting-state markers (Sarnthein et al.).

**psd_alpha** — Alpha-band raw power via Welch PSD (distinct from the ERD percentage — this is absolute power, not percent-change). Useful both acutely and at rest.

**psd_beta** — Beta-band raw power via Welch PSD. Raw-power counterpart to beta_erd_pct.

**psd_gamma** — Gamma-band raw power via Welch PSD, the raw-power counterpart to gamma power. Frequently one of the most informative features in pain-classification ML work.

### Connectivity Features (Phase-Locking Value)
**plv_Fz-Cz** — Phase-locking value between frontal (Fz) and central (Cz) electrodes — how consistently their oscillation timing stays aligned. Phase-connectivity features have outperformed simple power features in recent pain-classification work.

**plv_Cz-Pz** — Same phase-locking concept between central (Cz) and parietal (Pz) electrodes, capturing front-back connectivity.

**plv_C3-C4** — Phase-locking between left (C3) and right (C4) central electrodes — tests interhemispheric connectivity and laterality, since pain responses are typically stronger contralateral to the stimulated hand.

**plv_FCz-CPz** — Phase-locking between FCz and CPz, electrodes just anterior/posterior to the vertex (Cz) — another connectivity angle centered on the primary sensorimotor region. Currently 85% missing since not every montage includes both electrodes.

### Signal Complexity & Entropy Features
**perm_entropy** — Permutation entropy — how unpredictable the signal's ordering pattern is. Higher values mean a more irregular, less repetitive signal.

**spectral_entropy** — Entropy computed on the frequency spectrum rather than the raw waveform — how spread out vs. concentrated the power is across frequencies.

**sample_entropy** — Measures how likely similar patterns in the signal are to repeat. Tied to trial-to-trial variability, which has independently predicted pain perception in prior work.

### Fractal & Scaling Features
**higuchi_fd** — Higuchi fractal dimension — a geometric complexity measure of the waveform (how much fine detail/roughness it has across scales).

**dfa** — Detrended fluctuation analysis exponent — measures long-range temporal correlations, i.e., whether fluctuations early in the signal relate statistically to fluctuations much later.

### Hjorth Parameters
**hjorth_mobility** — One of the three Hjorth parameters; approximates the signal's mean frequency, computed cheaply from variance ratios without a full frequency transform.

**hjorth_complexity** — The second Hjorth parameter; measures how much the signal's frequency content changes over time — a proxy for how irregular vs. sine-wave-like the waveform is.

### Target/Label Features
**rating** — The subjective pain rating (typically 0–10 NRS) the participant gave for that specific trial. This is your ground-truth label for "how much did it hurt."

**laser_power** — The physical stimulus intensity (in Joules) delivered on that trial. This is the objective stimulus-strength label, used in the notebook as the comparison target against rating.

---

## Summary

This comprehensive feature set combines:
- **Neurophysiological markers** (ERPs, spectral features) reflecting immediate pain processing
- **Connectivity measures** capturing brain-region dialogue during pain perception
- **Signal complexity metrics** reflecting irregularity and scaling properties
- **Subjective and objective labels** for model training and validation

The data structure reflects a well-designed single-session, repeated-measures pain-perception paradigm with known confounds (habituation, sensitization) that need to be addressed in analysis and modeling.
