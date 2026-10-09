# Time-resolved parameterization of aperiodic and periodic brain activity

Source URL: https://elifesciences.org/articles/77348
Source title: Time-resolved parameterization of aperiodic and periodic brain activity
Date: 2022; DOI 10.7554/eLife.77348
Capture note: Fetched during this research run; quoted publication/date metadata is recorded below.

## Exact source excerpt 1

Macroscopic neural dynamics comprise both aperiodic and periodic signal components. Recent advances in parameterizing neural power spectra offer practical tools for evaluating these features separately. Although neural signals vary dynamically and express non-stationarity in relation to ongoing behaviour and perception, current methods yield static spectral decompositions. Here, we introduce Spectral Parameterization Resolved in Time (SPRiNT) as a novel method for decomposing complex neural dynamics into periodic and aperiodic spectral elements in a time-resolved manner. First, we demonstrate, with naturalistic synthetic data, SPRiNT’s capacity to reliably recover time-varying spectral features. We emphasize SPRiNT’s specific strengths compared to other time-frequency parameterization approaches based on wavelets. Second, we use SPRiNT to illustrate how aperiodic spectral features fluctuate across time in empirical resting-state EEG data (n=178) and relate the observed changes in aperiodic parameters over time to participants’ demographics and behaviour. Lastly, we use SPRiNT to demonstrate how aperiodic dynamics relate to movement behaviour in intracranial recordings in rodents. We foresee SPRiNT responding to growing neuroscientific interests in the parameterization of time-varying neural power spectra and advancing the quantitation of complex neural dynamics at the natural time scales of behaviour.

The paper addresses the highly timely question of how to quantify aperiodic and periodic neural activity. This was done by extending previous work by embracing time-resolved parametrization of both simulated, noninvasive EEG and intracranial data. The new approach is termed Spectral Parametrization Resolved in Time (SPRiNT) and the paper shows that the slope of aperiodic activity is linked with both behavior and age. The method thus demonstrates the importance of evalua

## Exact source excerpt 2

in the lower (3–30 Hz) and respectively, higher (30–80 Hz) frequency ranges: MAEs for centre frequency (0.32, resp. 0.32), amplitude (0.27, resp. 0.22), and SD (0.35, resp. 0.29).

We applied SPRiNT and _specparam_ to resting-state EEG data from the openly available Leipzig Study on Mind-Body-Emotion Interactions (LEMON) dataset ( [Babayan et al., 2019](https://elifesciences.org/articles/77348#bib2)). Participants (n=178) were instructed to open and close their eyes (alternating every 60 s). We used _Brainstorm_ ( [Tadel et al., 2011](https://elifesciences.org/articles/77348#bib45)) to preprocess EEG time series from electrode Oz and obtained parameterized spectra with _specparam_ and parameterized spectrograms with SPRiNT in both behavioural conditions (eyes open or closed). We also generated time-frequency decompositions of the same preprocessed EEG time series using Morlet wavelets (with default parameters; see Methods and Supplemental materials).

As expected, the group-averaged periodograms showed increased Oz signal power in the alpha range (6–14 Hz) in the eyes-closed behavioural condition with respect to eyes-open ( [Figure 4a](https://elifesciences.org/articles/77348#fig4)