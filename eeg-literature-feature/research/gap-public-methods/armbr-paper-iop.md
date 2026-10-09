Fetched from https://doi.org/10.1088/1741-2552/ade566

### # Artifact-reference multivariate backward regression
ing.org/) is a coalition of three not-for-profit publishers in the field of physical sciences: AIP Publishing, the American Physical Society and IOP Publishing.

Together, as publishers that will always put purpose above profit, we have defined a set of industry standards that underpin high-quality, ethical scholarly communications.

**We are proudly declaring that science is our only shareholder.**

Paper
•
The following article is Open access

# Artifact-reference multivariate backward regression (ARMBR): a novel method for EEG blink artifact removal with minimal data requirements

L Alkhoury\*, G Scanavini, S Louviot, A Radanovic, S A Shah and N J Hill

Published 26 June 2025 •
© 2025 The Author(s). Published by IOP Publishing Ltd


[Journal of Neural Engineering](https://iopscience.iop.org/journal/1741-2552),
[Volume 22](https://iopscience.iop.org/volume/1741-2552/22),
[Number 3](https://iopscience.iop.org/issue/1741-2552/22/3)**Citation** L Alkhoury _et al_ 2025 _J. Neural Eng._ **22** 036048**DOI** 10.1088/1741-2552/ade566

[PDF Opens in a new tab.](https://iopscience.iop.org/article/10.1088/1741-2552/ade566/pdf)

[PDF Opens in a new tab.](https://iopscience.iop.org/article/10.1088/1741-2552/ade566/pdf)

## Authors

L Alkhoury

AFFILIATIONS

Department of Radiology, Weill Cornell Medicine, New York, NY 10065, United States of America

EMAIL

[Lua4006@med.cornell.edu](mailto:Lua4006@med.cornell.edu)

## Author notes

\* Author to whom any correspondence should be addressed.

[https://orcid.org/0000-0003-4317-701X](https://orcid.org/0000-0003-4317-701X)

G Scanavini

AFFILIATIONS

Department of Radiology, Weill Cornell Medicine, New York, NY 10065, United States of America

[https://orcid.org/0000-0001-9336-8792](https://orcid.org/0000-0001-9336-8792)

S Louviot

AFFILIATIONS

Department of Radiology, Weill Cornell Medicine, New York, NY 10065, United States of America

A Radanovic

AFFILIATIONS

Department of Radiology, Weill Cornell Medicine, New York, NY 10065, United States of America

S A Shah

AFFILIATIONS

Department of Radiology, Weill Cornell Medicine, New York, NY 10065, United States of America

[https://orcid.org/0000-0003-2720-4479](https://orcid.org/0000-0003-2720-4479)

N J Hill

AFFILIATIONS

National Center for Adaptive Neurotechnologies, Stratton VA Medical Center, Albany, NY, United States of America

Electrical and Computer Engineering Department, State University of New York at Albany, Albany, NY, United States of America

[https://orcid.org/0000-0002-4253-6976](https://orcid.org/

---

### ## Abstract
.iop.org/questions/referencing-citation-novelty/) Yes


[Buy this article in print](https://pod-iopscience.org/?doi=10.1088/1741-2552/ade566&UTCDate=09102026_211048)

[Journal RSS](https://iopscience.iop.org/journal/rss/1741-2552)

[Sign up for new issue notifications](https://myiopscience.iop.org/signin?origin=a0&idhub=true&return=https%3A%2F%2Fiopscience.iop.org%2Fmyiopscience%2Falerts%2Fsubscribe%3Fjournal%3D1741-2552)

1741-2552/22/3/036048

## Abstract

_Objective_. We present a novel and lightweight method that removes ocular artifacts from electroencephalography (EEG) recordings while demanding minimal training data. _Approach_. A robust, cross-validated thresholding procedure automatically detects the times at which eye blinks occur, then a linear scalp projection is estimated by regressing a simplified time-locked reference signal against the multi-channel EEG. _Main results_. Performance was compared against four commonly-used and readily available blink removal methods: signal subspace projection and forward regression (Reg) from the MNE toolbox, EEGLab’s independent component analysis (ICA) combined with ICLabel for automated component identification, and Artifact Subspace Reconstruction (ASR) Python implementation compatible with MNE. On semi-synthetic blink-contaminated EEG data, our method exhibited better reconstruction of the ground truth than the two MNE native methods, and comparable (or better in some scenarios) performance to ASR algorithm and ICA+IClabel. We also examined a real EEG dataset from 16 human participants, where the ground truth was unknown. Our method affected contaminated channels in blink intervals more than the two MNE native methods and ASR, while having a smaller impact on non-blink intervals, uncontaminated channels, and higher-frequency amplitudes, than the two MNE methods; its performance was again similar to ICA+ICLabel. On a second real dataset from 42 human participants, we showed that ARMBR removed the unwanted blink artifacts while successfully preserving the desired event-related-potential signals. _Significance_. The proposed algorithm exhibited comparable, and in some scenarios better performance relative to readily-available implementations of other widely-used methods. Another feature of our method is its potential as method for online applications. Therefore, it stands to make valuable contributions towards the automation of neural-engineering technologies and their translation from laboratory to clinical and other real-world usage.

Export citation an

Verified excerpt on derived signal deposit and real EEG links:
### A Python and MATLAB implementation
sight on how to utilize this framework on other artifacts. In section [8](https://iopscience.iop.org/article/10.1088/1741-2552/ade566#jneade566s8), we conclude that the ARMBR method is desirable since (a) it exhibits, in many scenarios, high quantitative performance when compared to alternative blink-removal methods and (b) the required user intervention is limited to identifying the channels that are expected to be heavily influenced by blinks. A Python and MATLAB implementation of the ARMBR method as well as the semi-synthetic and real signals used for validation are available at: [https://github.com/S-Shah-Lab/ARMBR.git](https://github.com/S-Shah-Lab/ARMBR.git).

## 2. EEG datasets

To test the performance of ARMBR and compare it to the other tested methods, we employed two strategies: a semi-synthetic-data approach and a real-data approach. For both approaches, we used the corpus of Broderick _et al_ \[ [7](https://iopscience.iop.org/article/10.1088/1741-2552/ade566#jneade566bib7), [8](https://iopscience.iop.org/article/10.1088/1741-2552/ade566#jneade566bib8)\] in which 19 participants were listening to recordings of natural speech. All signals were collected from a 128-channel ActiveTwo system (BioSemi) at 512 Hz, down-sampled to 128 Hz, and reference to the average of the right and left mastoids. The data were then bandpass-filtered from 1 Hz to 40 Hz using a 4th order Butterworth filter.

In addition, our real-data approach incorporated an event-related potential (ERP) analysis, on data from the OpenNeuro _Nencki-Symfonia EEG/ERP dataset_ \[ [16](https://iopscience.iop.org/article/10.1088/1741-2552/ade566#jneade566bib16)\]. This dataset contains real EEG data from 42 participants performing a visual oddball task. In the oddball paradigm, three stimuli (rare target, rare distractor, frequent standard) were presented in a pseudorandom order with the restriction that two rare stimuli could not appear in a row. Participants were instructed to push a button in response to the target stimulus and inhibit responses to other stimuli. The session contained 660 stimuli, which included approximately 12% targets, 12% 

---

### The comparison we provided
1088/1741-2552/ade566#jneade566bib13)\]. These methods were selected as being readily available for downloadable and usable with minimal parameter tuning or expert judgment; we also emphasize non-ideal or online use-cases, in which limited training data are available (in this aspect, the ICA-based method is being used somewhat outside its usual scope—but, as we shall see, its performance is nonetheless competitive under some of the conditions).

The comparison we provided, in the case of the semi-synthetic dataset, is in terms of root mean square error (RMSE), SNR, and Pearson correlation (![$\rho_\mathrm{pearson}$](https://content.cld.iop.org/journals/1741-2552/22/3/036048/revision3/jneade566ieqn1.gif)) between the clean brain wave signals and the blink-suppressed signals. As for the real dataset, since the blink-suppressed ground truth is unavailable, we computed a semi-normalized correlation coefficient, _R_, for blink-epochs and non-blink-epochs between the blink-contaminated EEG signals and the blink-suppressed signals. We also computed relative mean absolute error (RMAE) of power-spectral-density (PSD) features in three frequency bands (8–12 Hz, 12–25 Hz, and 25–40 Hz) that typically contain EEG signals of interest while being less affected by blink artifacts. Lastly, we evaluated whether the blink-suppressed EEG signals retain task-relevant responses from an ERP-based oddball dataset. There, we computed the average ERP before and after blink removal (using ARMBR). When tested on the semi-synthetically generated blink-contaminated EEG signal, ARMBR exhibited smaller RMSE and greater SNR and ![$\rho_\mathrm{pearson}$](https://content.cld.iop.org/journals/1741-2552/22/3/036048/revision3/jneade566ieqn2.gif) as compared to the two MNE method and comparable (or better in some scenarios) performance to ASR and ICA+IClabel. When tested on the real dataset, ARMBR exhibited comparable _R_ levels to the ICA+ICLabel method (when trained on long-enough data segments) and better _R_ levels than MNE SSP and MNE Reg methods. As for ASR, _R_ levels where higher for non-blink intervals. The same conclusion holds for RMAE va