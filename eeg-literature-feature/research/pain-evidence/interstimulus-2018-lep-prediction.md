Title: Magnitude and Temporal Variability of Inter-stimulus EEG Modulate the Linear Relationship Between Laser-Evoked Potentials and Fast-Pain Perception
Source URL: https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00340/full
Publisher page title: Frontiers | Magnitude and Temporal Variability of Inter-stimulus EEG Modulate the Linear Relationship Between Laser-Evoked Potentials and Fast-Pain Perception
Primary-source excerpts follow (verbatim Markdown as returned by webFetch):

## Abstract

The level of pain perception is correlated with the magnitude of pain-evoked brain responses, such as laser-evoked potentials (LEP), across trials. The positive LEP-pain relationship lays the foundation for pain prediction based on single-trial LEP, but cross-individual pain prediction does not have a good performance because the LEP-pain relationship exhibits substantial cross-individual difference. In this study, we aim to explain the cross-individual difference in the LEP-pain relationship using inter-stimulus EEG (isEEG) features. The isEEG features (root mean square as magnitude and mean square successive difference as temporal variability) were estimated from isEEG data (at full band and five frequency bands) recorded between painful stimuli. A linear model was fitted to investigate the relationship between pain ratings and LEP response for fast-pain trials on a trial-by-trial basis. Then the correlation between isEEG features and the parameters of LEP-pain model (slope and intercept) was evaluated. We found that the magnitude and temporal variability of isEEG could modulate the parameters of an individual's linear LEP-pain model for fast-pain trials. Based on this, we further developed a new individualized fast-pain prediction scheme, which only used training individuals with similar isEEG features as the test individual to train the fast-pain prediction model, and obtained improved accuracy in cross-individual fast-pain prediction. The findings could help elucidate the neural mechanism of cross-individual difference in pain experience and the proposed fast-pain prediction scheme could be potentially used as a practical and feasible pain prediction method in clinical practice.

## Introduction

Pain is a subjective perception and is primarily assessed by means of self-report. Because the capacity to effectively report pain is limited in vulnerable population groups (e.g., babies and people with cognitive or communicative impairments), assessment of pain levels based on physiological signals has attracted a growing interest (Wager et al., 2013). For example, several pain prediction models based on functional magnetic resonance imaging (fMRI) have been developed to assess pain in healthy persons and patients with chronic pain (Wager et al., 2013; López-Solà et al., 2017). As compared to fMRI, electroencephalography (EEG) could capture a wealth of pain-related brain activities in a cheap and easy-to-use manner, so EEG-based pain assessment is a promising technique in clinical settings. The most commonly used pain-related EEG activities in research are laser-evoked potentials (LEP), and the strong relationship between the LEP amplitudes and the subjective pain intensity on a trial-by-trial basis has been well characterized (Garcíalarrea et al., 1997; Iannetti et al., 2005; Hu et al., 2014). Based on the singe-trial LEP-pain relationship, our previous study has been able to predict subjective pain ratings from single-trial LEP features and achieved an accuracy greater than 80% (Huang et al., 2013a). The relationship between EEG activities and pain rating in other pain paradigms has also been identified and been used to develop EEG-based pain prediction models. For example, based on the spectral characteristics of EEG before drug treatment, one study classified responders and non-responders and presented the application of EEG-based pain models in the prediction of analgesic effect (Gram et al., 2015).

An EEG-based pain prediction model can be achieved at within-individual level (the model is trained on and applied to the same individual) or at cross-individual level (the model is trained on a group of individuals but applied to different individuals) (Huang et al., 2013b). Cross-individual pain prediction is more desired in clinical practice but it normally has a lower accuracy than within-individual prediction (Huang et al., 2013b), because the relationship between brain activity and pain ratings could vary largely among individuals. Actually, each individual has a unique prediction model that links his/her neural activities and pain perception. The substantial cross-individual variability in pain experien

---

## Results

### Relationship between LEP responses and pain ratings

Two participants were discarded from the further analysis because of he/she only had fast-pain trials with NRS < 7. Figure [2](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00340/full#F2) shows the group-averaged waveforms of the LEP responses elicited at Cz with three different NRS levels (4<NRS≤6, 6<NRS≤8, 8<NRS≤10), and the scalp topographies at the peak latencies of the N2 and P2 waves. When only fast-pain trials were included, linear correlation was observed between pain perception and corresponding brain responses (as shown in Figure [3](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00340/full#F3)).

Figure 2

![Figure 2](https://www.frontiersin.org/files/Articles/359383/xml-images/fnins-12-00340-g0002.webp)

Group averages and scalp topographies of LEP responses at different NRS levels. LEP waveforms were recorded from the vertex (Cz-nose). The group mean value for N2 peak is −7.58 ± 8.88 μV, −19.06 ± 12.72 μV and −30.13 ± 18.84 μV for three levels respectively. The group mean value for P2 peak is 7.14 ± 8.20 μV, 19.38 ± 11.98 μV and 31.34 ± 12.70 μV for three levels respectively. The scalp topographies of N2 and P2 waves are displayed at their peak latencies.

Figure 3

![Figure 3](https://www.frontiersin.org/files/Articles/359383/xml-images/fnins-12-00340-g0003.webp)

Relationship between pain intensity and N2-P2 amplitude at Cz. Colored dots represent the N2-P2 amplitudes of fast-pain trials averaged across participants, and error bars denote the SD values across participants. The blue line represents the fitted linear model for low-pain trials and fast-pain trials separately. Note that, when NRS = 10, the N2-P2 amplitudes are slightly lower than those of NRS = 9. The possible reason is that only 18 participants had fast-pain trials of NRS = 10, which might make the estimation have large variance and not reliable.

### Relationship between isEEG features and LEP-pain model parameters

The isEEG of alpha-2 band (11–13 Hz) across almost the whole brain showed significant negative correlation between nMSSD and the model slope of (4), as shown in Figure [4](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00340/full#F4). Also, we observed that the model intercept of (4) showed significant positive correlation with RMS of isEEG across almost the whole brain in all frequency bands, as shown in Figure [5](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00340/full#F5). Non-significant results are only reported in the Supplementary Material, as shown in Figure [S1](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00340/full#SM1) and Figure [S2](https://www.frontiersin.org/journals/neuroscience/articles/10.3389/fnins.2018.00340/full#SM2).

Figure 4

![Figure 4](https://www.frontiersin.org/files/Articles/359383/xml-images/fnins-12-00340-g0004.webp)

Significant correlation between temporal variability (nMSSD) of isEEG in alpha-2 band (11–13 Hz) and the fitting slope of the linear LEP-pain model (4).

Figure 5

![Figure 5](https://www.frontiersin.org/files/Articles/359383/xml-images/fnins-12-00340-g0005.webp)

Significant correlation between isEEG magnitude (RMS) and the fitting intercept of the linear LEP-pain model (4).

### Individualized LEP-pain prediction model

The individualized LEP-based fast-pain prediction scheme can achieve a prediction error (MAE) of 1.19 ± 0.35 (mean ± SD), which is significantly smaller than the MAE, 1.36 ± 0.51, of the conventional cross-individual LEP-pain prediction ( _p_ = 0.002; two-sample paired _t_-test).

## Discussion

In this study, by analyzing isEEG data recorded during pain experiments from healthy participants, we obtained two main findings. First, both the magnitude (measured by RMS) and temporal variability (measured by nMSSD) of isEEG are related to cross-individual difference in the parameters of LEP-pain model for fast-pain prediction. Second, by selecting and weighting training individuals according to isEEG correlates of LEP-pain model, cross-individual fa