# A comprehensive EEG dataset of laser-evoked potentials for pain research

Primary source title: A comprehensive EEG dataset of laser-evoked potentials for pain research
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/
Fetched by webFetch/search; retrieval date was not supplied in callback metadata.

## Abstract

Understanding the neural mechanisms of pain is key to the development of novel pain diagnostic and treatment strategies. Here, we present a large-scale, comprehensive electroencephalogram (EEG) dataset of laser-evoked potentials (LEPs) from 678 healthy participants. This dataset comprises high-density EEG recordings and single-trial self-reported pain ratings in response to nociceptive laser stimuli of varying intensities (from 2.5 J to 4.5 J) delivered to either the left or right dorsum of the hand, without any physical or psychological interventions. As the largest nociceptive-evoked EEG dataset to date, it can serve as a valuable resource for mechanistic pain studies such as investigating electrophysiological underpinnings of within-individual and between-individual pain variations, for clinical studies such as providing normal values of LEPs to assess the possible lesions of pain pathways, and for methodological innovations in EEG signal processing such as optimizing preprocessing pipelines and developing new analytical tools.

**Subject terms:** Neuroscience, Psychology



---

### Single trial behavioral data

During the preprocessing process, the stimulus intensity and subjective rating of pain perception for each single trial were recorded in each participant’s **.set** file. When the data is opened in EEGLAB, these values can be found in the **EEG.epoch** structure.

### EEG data

All data are named according to the experiments and identified as follows: 26\_ByBiosemi, 29\_ByANT, 39\_ByBP, 30\_ByANT, 65\_ByANT, 95\_ByBP, 142\_ByBiosemi, 223\_ByBP, and 29\_ByBP. The raw data are organized by subject participation and named as sub-xxx (where xxx is the subject number, starting from 001). For each subject, data are arranged by session and include original EEG data files (formats: **\_eeg.eeg/\_eeg.bdf**), along with the associated **\_channels.tsv,** **\_electrodes.tsv,** **\_events.tsv**, and **\_eeg.json** files. The **\_channels.tsv** file contains the names, types, and units of electrode channels; the **\_electrodes.tsv** file contains the electrode positions; the **\_events.tsv** file records the start and duration of each event, the sampling rate, and the type of markers; the **\_eeg.json** file provides details on the data collection institution, address, department, equipment used, EEG reference electrode, project name, number of electrodes, sampling type, sampling time, and sampling frequency.

## Technical Validation

Our technical validation includes quality control of behavioral data and EEG responses to nociceptive laser stimuli. The relationship between stimulus intensities (in unit of Joule \[J\]) and pain ratings is illustrated in Fig. [3](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/#Fig3), showing an increasing trend of higher pain ratings corresponding to higher stimulus intensities. These behavioral patterns offer valuable insights into how varying stimulus intensities influence subjective pain perception, supporting the validity of graded laser stimulus intensities in eliciting graded pain perception.

### Fig. 3.

![Fig. 3](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/b9cb/12405539/d6c4863b0a46/41597_2025_5900_Fig3_HTML.jpg)

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/figure/Fig3/)

The relationship between stimulus intensities and pain ratings.

For the EEG data, we calculated LEP responses for each stimulus intensity (Fig. [4](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/#Fig4), the code and the **.mat** data file are available in the **supplement** folder, directly accessible at: [https://github.com/XiangyueZhao/Supplement-Data](https://github.com/XiangyueZhao/Supplement-Data)). As expected, LEP responses remarkedly increased with stimulus intensities of nociceptive laser stimuli. A similar relationship was observed between LEP responses and subjective pain ratings, confirming the strong relationship among nociceptive stimulus intensities, subjective pain ratings, and LEP responses in our dataset.

### Fig. 4.

[![Fig. 4](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/b9cb/12405539/c3bf26661616/41597_2025_5900_Fig4_HTML.jpg)](https://www.ncbi.nlm.nih.gov/core/lw/2.0/html/tileshop_pmc/tileshop_pmc_inline.html?title=Click%20on%20image%20to%20zoom&p=PMC3&id=12405539_41597_2025_5900_Fig4_HTML.jpg)

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/figure/Fig4/)

Technical validations. ( **a**) The average LEP responses at the Cz electrode for different stimulus intensities (2.5 J to 4.5 J). Each curve represents the average response for a specific stimulus intensity, highlighting variations in peak amplitudes and latencies. ( **b**) The average LEP responses at the Cz electrode for different pain ratings (1 to 10), with each line color representing a specific pain rating. The results illustrate how EEG responses change with increasing ratings of perceived pain. ( **c**) Scalp topographies of N2 and P2 waves for different stimulus intensities. ( **d**) Scalp topographies of N2 and P2 waves for different ratings of perceived pain. The timestamp on each map indicates the peak latency of N2 or P2 wave.

## Acknowledgements

This work was supported by the Beijing Natural Science Foundation (JQ22018, 5252019), National Key Research and Development Program of China (2023YFC2508702), and National Natural Science Foundation of China (32071061).

## Author contributions

Xiangyue Zhao, Jingyao Zhou, Libo Zhang, Yun Zhuang, Xuejing Lu, Yanzhi Bi, and Li Hu conceived and conducted this research, and drafted the manuscript. Haoqing Duan, Shiyu Wei, and Suchen Yao provided support for data processing.



---

## Data availability

All research data has been consented to by participants, with explicit agreement for public sharing. Data sharing follows the appropriate ethical approval procedures to ensure the privacy of participants is fully protected.



---

Public repository linked by article: https://github.com/XiangyueZhao/Supplement-Data
