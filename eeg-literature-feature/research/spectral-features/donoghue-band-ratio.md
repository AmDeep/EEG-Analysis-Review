# Electrophysiological Frequency Band Ratio Measures Conflate Periodic and Aperiodic Neural Activity

Source URL: https://www.eneuro.org/content/7/6/ENEURO.0192-20.2020
Source title: Electrophysiological Frequency Band Ratio Measures Conflate Periodic and Aperiodic Neural Activity
Date: 2020; DOI 10.1523/ENEURO.0192-20.2020
Capture note: Fetched during this research run; quoted publication/date metadata is recorded below.

## Exact source excerpt 1

# Electrophysiological Frequency Band Ratio Measures Conflate Periodic and Aperiodic Neural Activity

Thomas Donoghue, Julio Dominguez and Bradley Voytek

eNeuro 25 September 2020, 7 (6) ENEURO.0192-20.2020; https://doi.org/10.1523/ENEURO.0192-20.2020

Thomas Donoghue

1Department of Cognitive Science, University of California, San Diego

- [Find this author on Google Scholar](https://www.eneuro.org/lookup/google-scholar?link_type=googlescholar&gs_type=author&author%5B0%5D=Thomas%2BDonoghue%2B "Open in new tab")
- [Find this author on PubMed](https://www.eneuro.org/lookup/external-ref?access_num=Donoghue%20T&link_type=AUTHORSEARCH "Open in new tab")
- [Search for this author on this site](https://www.eneuro.org/search/author1%3AThomas%2BDonoghue%2B)
- [ORCID record for Thomas Donoghue](https://orcid.org/0000-0001-5911-0472 "Open in new tab")

Julio Dominguez

1Department of Cognitive Science, University of California, San Diego

- [Find this author on Google Scholar](https://www.eneuro.org/lookup/google-scholar?link_type=googlescholar&gs_type=author&author%5B0%5D=Julio%2BDominguez%2B "Open in new tab")
- [Find this author on PubMed](https://www.eneuro.org/lookup/external-ref?access_num=Dominguez%20J&link_type=AUTHORSEARCH "Open in new tab")
- [Search for this author on this site](https://www.eneuro.org/search/author1%3AJulio%2BDominguez%2B)

Bradley Voytek

1Department of Cogni

## Exact source excerpt 2

Band ratio measures, computed as the ratio of power between two frequency bands, are a common analysis measure in neuroelectrophysiological recordings. Band ratio measures are typically interpreted as reflecting quantitative measures of periodic, or oscillatory, activity. This assumes that the measure reflects relative powers of distinct periodic components that are well captured by predefined frequency ranges. However, electrophysiological signals contain periodic components and a 1/f-like aperiodic component, the latter of which contributes power across all frequencies. Here, we investigate whether band ratio measures truly reflect oscillatory power differences, and/or to what extent ratios may instead reflect other periodic changes, such as in center frequency or bandwidth, and/or aperiodic activity. In simulation, we investigate how band ratio measures relate to changes in multiple spectral features, and show how multiple periodic and aperiodic features influence band ratio measures. We validate these findings in human electroencephalography (EEG) data, comparing band ratio measures to parameterizations of power spectral features and find that multiple disparate features influence ratio measures. For example, the commonly applied θ/β ratio is most reflective of differences in aperiodic activity, and not oscillatory θ or β power. Collectively, we show that periodic and aperiodic features can create the same observed changes in band ratio measures, and that this is inconsistent with their typical interpretations as measures of periodic power. We conclude that band ratio measures are a non-specific measure, conflating multiple possible underlying spectral changes, and recommend explicit parameterization of neural power

## Exact source excerpt 3

ibrary was used for managing and processing EEG data ( [Gramfort et al., 2014](https://www.eneuro.org/content/7/6/ENEURO.0192-20.2020#ref-21)). Custom code was used to calculate band ratio measures and perform analyses. All code for this project is openly available in the project repository ( [https://github.com/voytekresearch/BandRatios](https://github.com/voytekresearch/BandRatios)) and in [Extended Data 1](https://www.eneuro.org/content/7/6/ENEURO.0192-20.2020#DC1).

### Extended Data 1

Project code. Supplementary package of code used for simulations and analysis. Download [Extended Data 1, ZIP file](https://www.eneuro.org/content/eneuro/7/6/ENEURO.0192-20.2020/DC1/embed/inline-supplementary-material-1.zip?download=true).

### Literature analysis

The literature analysis was done using the Literature Scanner (LISC) Python toolbox ( [Donoghue, 2019](https

## Exact source excerpt 4

g/content/7/6/ENEURO.0192-20.2020#T1), [2](https://www.eneuro.org/content/7/6/ENEURO.0192-20.2020#T2)).

### EEG data analysis

To further examine how various spectral parameters affect band ratio measures, in real data, we used the openly available Multimodal Resource for Studying Information Processing in the Developing Brain (MIPDB) dataset of human EEG data released by the Child Mind Institute ( [Langer et al., 2017](https://www.eneuro.org/content/7/6/ENEURO.0192-20.2020#ref-30)). The study population is a community sample of children and adults ( _n_ =126, age range = 6–44, age mean = 15.79, age SD = 8.03, number of males = 69). Data for each subject includes resting state and task EEG data, behavioral measures, and eye tracking data. EEG data were collected on a 128 channel Geodesic Hydrocel system, from which the outermost channels, around the chin and neck, were excluded, leaving