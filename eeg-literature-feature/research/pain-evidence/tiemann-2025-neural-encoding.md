Title: Neural encoding of pain is robust within but unstable between individuals | PLOS Biology
Source URL: https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003948
Publisher page title: Neural encoding of pain is robust within but unstable between individuals | PLOS Biology
Primary-source excerpts follow (verbatim Markdown as returned by webFetch):

## Abstract

The perception of pain varies both within and between individuals, even when sensory input remains constant. Understanding how the brain encodes these intra- and inter-individual variations is central to elucidating the neural mechanisms of pain and to developing reliable brain-based markers for clinical use. Yet, previous findings have been inconsistent, and their robustness across time and populations remains unclear. Here, we used electroencephalography (EEG) in 161 healthy participants to re-investigate the neural patterns explaining intra- and inter-individual variations in the perception of brief painful stimuli independent of stimulus intensity. Using Bayesian multivariate multi-model regression, we related pain ratings to canonical EEG responses. To directly assess robustness, the experiment was repeated after 4 weeks in the same participants and replicated in an independent cohort ( _n_ = 111). Neural patterns associated with inter-individual differences in pain perception were repeatable over time but not replicable across cohorts. In contrast, neural patterns underlying intra-individual fluctuations were robust both over time and across cohorts. These findings indicate that within-person and between-person variability in pain perception is encoded by distinct neural patterns that differ fundamentally in their robustness. Furthermore, they show that brain-based markers are particularly suited for tracking intra-individual fluctuations of pain, while being less sensitive to inter-individual differences. More broadly, they highlight the importance of within-person approaches for advancing both mechanistic models of pain and the development of clinically useful biomarkers.

## Figures

![Fig 7](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g007)

![Fig 1](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g001)

![Fig 2](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g002)

![Fig 3](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g003)

![Fig 4](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g004)

![Fig 5](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g005)

![Fig 6](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g006)

![Fig 7](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g007)

![Fig 1](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g001)

**Citation:** Tiemann L, Bott FS, May ES, Nickel MM, Hohn VD, Gil Ávila C, et al. (2026) Neural encoding of pain is robust within but unstable between individuals. PLoS Biol 24(8):
e3003948.

https://doi.org/10.1371/journal.pbio.3003948

**Academic Editor:** Choong-Wan Woo, Sungkyunkwan University, KOREA, REPUBLIC OF

**Received:** April 28, 2026; **Accepted:** July 29, 2026; **Published:** August 10, 2026

**Copyright:** © 2026 Tiemann et al. This is an open access article distributed under the terms of the [Creative Commons Attribution License](http://creativecommons.org/licenses/by/4.0/), which permits unrestricted use, distribution, and reproduction in any medium, provided the original author and source are credited.

**Data Availability:** All data in standardized EEG-BIDS format \[ [44](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003948#pbio.3003948.ref044)\] are available at [https://osf.io/z2h86](https://osf.io/z2h86). Code used for the current manuscript and data underlying the figures are publicly available at [https://doi.org/10.17605/OSF.IO/BS3YJ](https://doi.org/10.17605/OSF.IO/BS3YJ).

**Funding:** The study was supported by the Deutsche Forschungsgemeinschaft (PL 321/14-1, PL321/16-1, SFB1158 to MP; [https://www.dfg.de/en](https://www.dfg.de/en)) and the Technical University of Munich (TUM Innovation Network Neurotechnology in Men

---

## Results

### Overview

In the present study, we investigated how brain responses to noxious stimuli relate to inter-individual and intra-individual variability in pain perception ( [Fig 1A](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003948#pbio-3003948-g001)). Intra-individual variability was assessed at both short (moment-to-moment) and long (day-to-day) timescales. The experiment consisted of two sessions ~1 month apart. In each session ( [Fig 1B](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003948#pbio-3003948-g001)), participants received 80 brief painful stimuli to the dorsum of the left hand and provided verbal pain ratings for each stimulus. Stimuli were delivered at four different intensities to reduce stimulus predictability and to increase the range of pain intensity ratings. To optimize the signal-to-noise ratio, we only included trials with stimuli of the two highest intensities. We considered the resulting reduction in stimulus intensity-related variability acceptable because our primary aim was to investigate variability in pain perception beyond that explained by stimulus intensity.

[![thumbnail](https://journals.plos.org/plosbiology/article/figure/image?size=inline&id=10.1371/journal.pbio.3003948.g001)](https://journals.plos.org/plosbiology/article/figure/image?size=medium&id=10.1371/journal.pbio.3003948.g001 "Click for larger image")

Download:


- [PNG\\
\\
larger image](https://journals.plos.org/plosbiology/article/figure/image?download&size=large&id=10.1371/journal.pbio.3003948.g001)
- [TIFF\\
\\
original image](https://journals.plos.org/plosbiology/article/figure/image?download&size=original&id=10.1371/journal.pbio.3003948.g001)

Fig 1.  Study overview.

**A)** Research questions and corresponding result figures. **B)** During the experiment, participants received 80 brief painful laser stimuli to the dorsum of their left hand at four different fixed intensities (2.5, 3.0, 3.5, 4.0 J). An auditory cue prompted participants to verbally rate their pain intensity on a numerical rating scale ranging from 0 (“no pain”) to 100 (“maximally tolerable pain”). Concurrently, EEG was recorded with a 64-channel system. Throughout the experiment, participants sat in a comfortable chair with eyes closed, wore protective goggles, and listened to white noise via headphones to mask ambient sounds.

[https://doi.org/10.1371/journal.pbio.3003948.g001](https://doi.org/10.1371/journal.pbio.3003948.g001)

Brain responses were quantified using EEG. We extracted six commonly analyzed features of event-related brain activity: the amplitudes of the N1, N2, and P2 evoked responses, and the power of induced oscillatory responses at alpha, beta, and gamma frequencies. Pain perception was measured using verbal pain ratings on a numerical rating scale ranging from 0 (no pain) to 100 (maximally tolerable pain). To isolate variability beyond that explained by stimulus intensity, analyses were performed on intensity-adjusted brain responses and pain ratings.

To quantify associations between brain responses and pain ratings, we used Bayesian multi-model regression, treating the six EEG features as predictors and the pain ratings as the outcome variable. Bayesian multi-model inference enabled variable selection and parameter estimation within a unified framework by averaging over models with all plausible combinations of predictors. This approach accounts for model uncertainty and yields more stable and less biased inferences about relationships between brain responses and pain ratings \[ [20](https://journals.plos.org/plosbiology/article?id=10.1371/journal.pbio.3003948#pbio.3003948.ref020)\].

The study and its analysis were preregistered at ClinicalTrials.gov ( [https://clinicaltrials.gov/study/NCT05616091](https://clinicaltrials.gov/study/NCT05616091)) and the OSF Registries ( [https://osf.io/wfjhq](https://osf.io/wfjhq)).

### Brain responses to noxious stimuli

To confirm that the EEG responses in the present study reproduced the well-established patterns elicited by brief noxious laser stimuli, we computed grand averages of evoked potentials and TFRs ( [Fig 2](https://journa