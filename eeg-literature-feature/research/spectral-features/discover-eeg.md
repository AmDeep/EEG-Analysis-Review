# DISCOVER-EEG: an open, fully automated EEG pipeline for biomarker discovery in clinical neuroscience

Source URL: https://www.nature.com/articles/s41597-023-02525-0
Source title: DISCOVER-EEG: an open, fully automated EEG pipeline for biomarker discovery in clinical neuroscience
Date: 2023; DOI 10.1038/s41597-023-02525-0
Capture note: Fetched during this research run; quoted publication/date metadata is recorded below.

## Exact source excerpt 1

## Abstract

Biomarker discovery in neurological and psychiatric disorders critically depends on reproducible and transparent methods applied to large-scale datasets. Electroencephalography (EEG) is a promising tool for identifying biomarkers. However, recording, preprocessing, and analysis of EEG data is time-consuming and researcher-dependent. Therefore, we developed DISCOVER-EEG, an open and fully automated pipeline that enables easy and fast preprocessing, analysis, and visualization of resting state EEG data. Data in the Brain Imaging Data Structure (BIDS) standard are automatically preprocessed, and physiologically meaningful features of brain function (including oscillatory power, connectivity, and network characteristics) are extracted and visualized using two open-source and widely used Matlab toolboxes (EEGLAB and FieldTrip). We tested the pipeline in two large, openly available datasets containing EEG recordings of healthy participants and patients with a psychiatric condition. Additionally, we performed an exploratory analysis that could inspire the development of biomarkers for healthy aging. Thus, the DISCOVER-EEG pipeline facilitates the aggregation, reuse, and analysis of large EEG datasets, promoting open and repr

## Exact source excerpt 2

w.nature.com/articles/s41597-023-02525-0#ref-CR29 "Barker, M. et al. Introducing the FAIR Principles for research software. Sci Data 9, 622,                    https://doi.org/10.1038/s41597-022-01710-x                                     (2022)."). The code of the DISCOVER-EEG pipeline is published on GitHub and co-deposited at Zenodo, where it is uniquely referenced by a DOI[30](https://www.nature.com/articles/s41597-023-02525-0#ref-CR30 "Gil Ávila, C., Bott, F. S., Gross, J. & Ploner, M. crisglav/discover-eeg: 1.0.0 v. 1.0.0. Zenodo                    https://doi.org/10.5281/zenodo.8207523                                     (2023).") ( _Findability_). The code can be easily downloaded ( _Accessibility_) and receive contributions (please refer to the section _Code availability_). To ensure its _Interoperability_ and _Reusability_, the pipeline is based on two open-source Matlab toolbo

## Exact source excerpt 3

14, 610388,                    https://doi.org/10.3389/fnins.2020.610388                                     (2020).").

#### 7\. Data segmentation into epochs

Lastly, the continuous data are segmented into epochs with the function _pop\_epoch()_. By default, data are segmented into 2-second epochs with a 50% overlap. Although longer epochs might be desirable for the Alpha Peak Frequency estimation to increase frequency resolution, short epochs favor the reliability of functional connectivity measures[40](https://www.nature.com/articles/s41597-023-02525-0#ref-CR40 "Fraschini, M. et al. The effect of epoch length on estimated EEG functional connectivity and brain network organisation. Journal of neural engineering 13, 036015,                    https://doi.org/10.1088/1741-2560/13/3/036015                                     (2016)."), [41](https://www.nature.com/articles/s41597-023-02525-0#ref-CR41 "van Diessen, E. et al. Opportunities and methodological challenges in EEG and MEG resting state functional brain network research. Clinical neurophysiology: official journal of the Inte

## Exact source excerpt 4

[No matching source text in captured page.]