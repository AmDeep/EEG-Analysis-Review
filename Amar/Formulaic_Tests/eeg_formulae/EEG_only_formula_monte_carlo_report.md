# EEG-only formula Monte Carlo benchmark

## Status and key limitation
This is a **synthetic Monte Carlo stress test**, not analysis of raw EEG. No subjective pain ratings, PAS/NRS, demographic variables, or personalization variables were used as formula inputs. The three labels are simulated EEG-pattern classes, not clinical pain profiles. Therefore results test mathematical behavior only; they do not establish real-world pain classification.

## Formula construction
Each candidate feature set was standardized and compressed to a single scalar using its first principal component (PC1). PC1 is an unsupervised mathematical scalar, not a validated pain-intensity axis. The sign is oriented deterministically for reproducibility, but the meaning of the sign is not biologically established.

## EEG metrics included
- Sensorimotor alpha suppression (`alpha_suppression`) — Spectral; Core candidate.
- Beta power change (`beta_change`) — Spectral; Core candidate.
- Theta power change (`theta_change`) — Spectral; Core candidate.
- Gamma power change (`gamma_change`) — Spectral; EMG-sensitive candidate.
- Peak alpha frequency (`peak_alpha_freq`) — Spectral; Exploratory.
- Frontal alpha asymmetry (`frontal_alpha_asymmetry`) — Spatial/spectral; Exploratory.
- Spatially defined midline theta (`midline_theta`) — Spatial/spectral; Exploratory.
- Predefined theta connectivity summary (`theta_connectivity`) — Connectivity; Exploratory.
- Predefined alpha connectivity summary (`alpha_connectivity`) — Connectivity; Exploratory.
- Normalized spectral entropy (`spectral_entropy`) — Complexity; Candidate.
- Aperiodic spectral exponent (`aperiodic_exponent`) — Aperiodic; Exploratory.
- Alpha-envelope autocorrelation (`alpha_envelope_autocorr`) — Temporal; Exploratory.
- Alpha burst rate (`alpha_burst_rate`) — Temporal; Exploratory.
- Stimulus-locked N2–P2 amplitude (`erp_n2p2`) — ERP; Discrete stimuli only.
- ERP trial-to-trial variability (`erp_trial_variability`) — Temporal/ERP; Exploratory.
- ERP habituation slope (`erp_habituation_slope`) — Temporal/ERP; Exploratory.

## Candidate combinations
- **{fn}** ({len(cols)} features): alpha_suppression
- **{fn}** ({len(cols)} features): alpha_suppression, beta_change, theta_change
- **{fn}** ({len(cols)} features): alpha_suppression, beta_change, theta_change, gamma_change, peak_alpha_freq
- **{fn}** ({len(cols)} features): alpha_suppression, beta_change, theta_change, spectral_entropy
- **{fn}** ({len(cols)} features): theta_connectivity, alpha_connectivity, midline_theta
- **{fn}** ({len(cols)} features): alpha_envelope_autocorr, alpha_burst_rate, erp_trial_variability, erp_habituation_slope
- **{fn}** ({len(cols)} features): erp_n2p2, erp_trial_variability, erp_habituation_slope
- **{fn}** ({len(cols)} features): alpha_suppression, beta_change, theta_change, theta_connectivity, alpha_connectivity
- **{fn}** ({len(cols)} features): alpha_suppression, beta_change, theta_change, gamma_change, spectral_entropy, aperiodic_exponent
- **{fn}** ({len(cols)} features): alpha_suppression, beta_change, theta_change, spectral_entropy, theta_connectivity, erp_trial_variability
- **{fn}** ({len(cols)} features): alpha_suppression, beta_change, theta_change, gamma_change, peak_alpha_freq, frontal_alpha_asymmetry, midline_theta, theta_connectivity, alpha_connectivity, spectral_entropy, aperiodic_exponent, alpha_envelope_autocorr, alpha_burst_rate, erp_n2p2, erp_trial_variability, erp_habituation_slope
- **{fn}** ({len(cols)} features): alpha_suppression, theta_change, spectral_entropy, theta_connectivity, alpha_envelope_autocorr, erp_n2p2

## Monte Carlo design
- 200 independent simulated datasets, each with 70 observations in each of 3 assumed profile classes (210 observations per replicate).
- 16 EEG-derived feature metrics with hand-specified latent-pattern loadings and correlated residual noise.
- Scalar diagnostics: between-profile eta-squared, mean pairwise standardized mean separation, scalar SD, and Kruskal–Wallis omnibus p-values against the *simulated* class labels.
- Noise stress test: residual noise scales 0.6, 1.0, 1.5, 2.0 and signal scales 0.5, 1.0, 1.5; 45 replicates per combination and five selected feature sets.
- Null check: shuffled labels in 200 replicates for five selected feature sets.

## Main results
Top formula by mean simulated eta-squared: **F07_ERP** (mean eta-squared 0.099, SD 0.035).
Top formula by pairwise standardized separation: **F07_ERP** (mean separation 0.503).
These rankings are conditional on the simulated feature loadings, noise covariance, and class definitions. They are not evidence that the winning feature set distinguishes real people with different pain states.

## Statistical significance: correct interpretation
The Kruskal–Wallis tests compare scores across **simulated** classes. A high fraction of p < .05 is expected when the assumed data-generating process intentionally creates between-class differences. It is not evidence of clinical significance. The shuffled-label null is a sanity check; its nominal false-positive fraction should be near 0.05, subject to Monte Carlo variation and repeated testing. Formula multiplicity would require correction in real research.

## Mathematical-validity findings
- A scalar can always be formed from multiple EEG metrics, but a scalar is not automatically a valid pain score.
- PCA is useful for an unsupervised scalar but maximizes variance, not clinical pain relevance.
- Broad combinations can be redundant and may produce unstable weights when features are correlated.
- Standardization is required because EEG metrics have different units and ranges.
- Gamma needs EMG artifact controls; connectivity needs fixed estimator, reference, electrode pairs and frequency band.
- N2–P2 and habituation features are stimulus-locked and should not be forced into tonic/continuous-pain models.
- Profile separation requires independent labels. Without such labels, the defensible output is an EEG-pattern phenotype, not a pain profile.

## Recommendation
Treat this benchmark as a formula-screening design exercise only. For empirical validation, obtain real EEG with independently established clinical profile labels, define feature extraction in advance, fit any weights on training participants only, use subject-wise nested cross-validation and an external dataset, and report bootstrap confidence intervals, permutation tests, calibration, test–retest reliability, and comparison against simple baselines. If independent pain-profile labels are unavailable, do not call EEG clusters pain profiles.
