Fetched from https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268

Abstract and open code evidence:
### # Test–retest reliability
iley.com/journal/14698986 "Psychophysiology homepage")

[Volume 60, Issue 7](https://onlinelibrary.wiley.com/toc/14698986/2023/60/7) e14268 [![Psychophysiology](https://onlinelibrary.wiley.com/pb-assets/journal-banners/14698986-1501384714697.jpg)](https://onlinelibrary.wiley.com/journal/14698986 "Psychophysiology homepage")

REGISTERED REPORT

Open Access

[Creative Commons license information](http://creativecommons.org/licenses/by-nc-nd/4.0/)

# Test–retest reliability of resting-state EEG in young and older adults

[Tzvetan Popov](https://onlinelibrary.wiley.com/authored-by/Popov/Tzvetan),

Corresponding Author

Tzvetan Popov

- [tzvetan.popov@uzh.ch](mailto:tzvetan.popov@uzh.ch "Link to email address")

- [orcid.org/0000-0002-5495-1376](https://orcid.org/0000-0002-5495-1376)

Methods of Plasticity Research Laboratory, Department of Psychology, University of Zurich, Zürich, Switzerland

University Research Priority Program (URPP) Dynamic of Healthy Aging, Zurich, Switzerland

**Correspondence**

Tzvetan Popov, Methods of Plasticity Research Laboratory, Department of Psychology, University of Zurich, Zurich, Switzerland.

Email: [tzvetan.popov@uzh.ch](mailto:tzvetan.popov@uzh.ch "Link to email address")

Contribution: Formal analysis, Methodology, Validation, Visualization, Writing - original draft, Writing - review & editing

[Search for more papers by this author](https://onlinelibrary.wiley.com/authored-by/Popov/Tzvetan)
[Marius Tröndle](https://onlinelibrary.wiley.com/authored-by/Tr%C3%B6ndle/Marius),

Marius Tröndle

Methods of Plasticity Research Laboratory, Department of Psychology, University of Zurich, Zürich, Switzerland

University Research Priority Program (URPP) Dynamic of Healthy Aging, Zurich, Switzerland

Neuroscience Center Zurich (ZNZ), Zurich, Switzerland

Contribution: ​Investigation, Writing - original draft, Writing - review & editing

[Search for more papers by this author](https://onlinelibrary.wiley.com/authored-by/Tr%C3%B6ndle/Marius)
[Zofia Baranczuk-Turska](https://onlinelibrary.wiley.com/authored-by/Baranczuk%E2%80%90Turska/Zofia),

Zofia Baranczuk-Turska

Methods of Plasticity Research Laboratory, Department of Psychology, University of Zurich, Zürich, Switzerland

University Research Priority Program (URPP) Dynamic of Healthy Aging, Zurich, Switzerland

Institute of Mathematic

---

### The present registered report
l indicators of age-related cognitive decline. The reliability of the utilized metrics is a prerequisite for establishing robust brain–behavior relationships and clinically relevant indicators of cognitive decline. To date, however, test–retest reliability examination of measures derived from resting human EEG, comparing different resting-state measures between young and older participants, within the same adequately powered dataset, is lacking. The present registered report examined test–retest reliability in a sample of 95 young (age range: 20–35 years) and 93 older (age range: 60–80 years) participants. A good-to-excellent test–retest reliability was confirmed in both age groups for power estimates on both scalp and source levels as well as for the individual alpha peak power and frequency. Partial confirmation was observed for hypotheses stating good-to-excellent reliability of microstates measures and connectivity. Equal levels of reliability between the age groups were confirmed for scalp-level power estimates and partially so for source-level power and connectivity. In total, five out of the nine postulated hypotheses were empirically supported and confirmed good-to-excellent reliability of the most commonly reported resting-state EEG metrics.

## 1 INTRODUCTION

Spontaneous noninvasive recordings of electrical neural activity reflect the dynamics of large-scale brain networks (Biswal et al., [1995](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0007); Buzsaki & Draguhn, [2004](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0009)). The fluctuation of brain activity during awake resting conditions is considered to be a functional architecture that organizes brain function and supports cognitive tasks/activations (Cole et al., [2016](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0015); Raichle, [2015](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0096)). Resting-state neuroelectric indicators of normal and abnormal cognition offer great promise in advancing not only a fundamental understanding of brain function but also effective and personalized bench-to-bedside transfer (Insel, [2014](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0047); Olbrich et al., [2015](https:/

---

### All code is available
 30 Hz) and gamma (30 Hz to 40 Hz). The resulting power estimates for 105 channels, for individual frequency bands, were further extended by computing averages for six anatomically-defined electrode clusters (see Figure [S1a](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#support-information-section), Supplementary material): fronto-central, posterior-central, left-frontal, right-frontal, left-posterior, and right-posterior cluster. All code is available at [https://osf.io/a759y/](https://osf.io/a759y/).

#### 2.2.4 EEG microstates

Microstate analysis aims to identify spatiotemporal features of spontaneous EEG activity as topographic prototypes that are indicative of large-scale underlying neural network activity (Koenig et al., [2002](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0056); Koenig & Lehmann, [1996](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0054); Michel & Koenig, [2018](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0074)). In line with previous work, microstate analysis was performed with the MST toolbox (Poulsen et al., [2018](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0094)) applied to the eyes-closed condition. The present study utilized a two-step clustering approach similar to that described in Khanna et al. ([2014](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0051)). Segments of EC data were identified from the continuous recording only from the 105 electrodes as described above and further segmented into 2-sec epochs. Epochs exceeding the amplitude threshold of ±90μV were excluded from further analysis. After a common reference transformation, the epoched data were concatenated. An initial cluster analysis was conducted with individual-subject data and then a second clustering was performed on the derived individual prototypes. The eyes-closed EEG data were initially band-pass filtered between 1 Hz and 30 Hz. The first step of the individual-subject prototype derivation served to identify and extract maps at local maxima of the global field power (GFP). GFP peaks are thought to reflect the time points of stable network activity (Michel & Koenig, [2018](https://onlinelibrary.wiley.com/doi/full/10.1111/psyp.14268#psyp14268-bib-0074

OSF a759y is cited in article as scripts/code; no evidence in the reviewed landing page of raw EEG files. This candidate is not included in qualifying items because raw EEG was not verified public.