Title: Detecting acute pain signals from human EEG
Source URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/
Publisher page title: Detecting acute pain signals from human EEG - PMC
Primary-source excerpts follow (verbatim Markdown as returned by webFetch):

## Abstract

### Background:

Advances in human neuroimaging has enabled us to study functional connections among various brain regions in pain states. Despite a wealth of studies at high anatomic resolution, the exact neural signals for the timing of pain remain little known. Identifying the onset of pain signals from distributed cortical circuits may reveal the temporal dynamics of pain responses and subsequently provide important feedback for closed-loop neuromodulation for pain.

### New method:

Here we developed an unsupervised learning method for sequential detection of acute pain signals based on multichannel human EEG recordings. Following EEG source localization, we used a state-space model (SSM) to detect the onset of acute pain signals based on the localized regions of interest (ROIs).

### Results:

We validated the SSM-based detection strategy using two human EEG datasets, including one public EEG recordings of 50 subjects. We found that the detection accuracy varied across tested subjects and detection methods. We also demonstrated the feasibility for cross-subject and cross-modality prediction of detecting the acute pain signals.

### Comparison with existing methods:

In contrast to the batch supervised learning analysis based on a support vector machine (SVM) classifier, the unsupervised learning method requires fewer number of training trials in the online experiment, and shows comparable or improved performance than the supervised method.

### Conclusions:

Our unsupervised SSM-based method combined with EEG source localization showed robust performance in detecting the onset of acute pain signals.

**Keywords:** State-space model, source localization, event-related potential, acute pain

## 1\. Introduction

Pain is a highly dynamic process, and is constantly shaped by sensory inputs and internal affective or cognitive variables ( [Kucyi and Davis, 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R25)). A distributed network of cortical circuits, rather than a single “pain cortex”, regulates pain ( [Apkarian et al., 2005](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R1); [Perl, 2007](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R34)). Human neuroimaging studies have advanced the understanding of brain mechanisms in pain. Studies using functional magnetic resonance imaging (fMRI) and positron emission tomography (PET) have identified regions involved in affective and sensory processing of acute pain, including the primary somatosensory cortex (S1), anterior cingulate cortex (ACC), and insular cortex ( [Hutchison et al., 1999](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R19); [Isnard et al., 2011](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R22); [Downar et al., 2003](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R9); [Vierck et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R51); [Liberati et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R28); [van der Miesen et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R50); [Mouraux and Iannetti, 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R32)). However, despite high spatial resolution, fMRI and PET cannot measure brain activity directly, and have limited temporal resolution ( [Wager et al., 2013](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R52)). This lack of temporal specificity in fMRI or PET data makes it difficult to fully capture the dynamic nature of pain perception and experience. Meanwhile, non-invasive electroencephalography (EEG) has several significant advantages compared to other imaging modalities. First, EEG provides high temporal resolution and allows the assessment of oscillatory activity related to pain processing ( [Ploner et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R38)). Studies have shown that acute pain triggers evoked potentials (or event-related potentials, ERPs) in the EEG, as well as elevated EEG theta and gamma powers (e.g., [Stern et al., 2006](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R44); [Ploner et al., 2017](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R38); [Liberati et al., 2018](https://pmc.ncbi.nlm.n

---

#### Public dataset.

The first human EEG dataset was obtained from a published study ( [Tiemann et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R47)), in which data were collected in Germany and then in made in public ( [https://osf.io/bsv86/](https://osf.io/bsv86/)). Fifty-one right-handed healthy participants (25 females) with a mean age of 27 years (range 20–37) participated in the study. Participants were recruited via advertisements on bulletin boards of local universities. The study was approved by the local ethics committee and carried out in accordance with the relevant guidelines and regulations. The detailed protocol is referred to ( [Tiemann et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R47)).

Briefly, the experiment included three core conditions (motor, perceptual and autonomic), which were presented in randomized order. In each condition, 60 painful stimuli were applied to the dorsum of the left hand. We have focused on the motor condition (with 50 participated subjects) in our current study. Stimulus intensity was varied between three individually adjusted levels (Leve-1 \[ _n_ = 20\], Level-2 \[ _n_ = 20\], and Level-3 \[ _n_ = 20\]) in a pseudorandomized sequence. Painful stimuli were applied by means of cutaneous laser stimulation (160 nm wavelength, 1-ms pulse duration, 5-mm spot diameter), which induces a pinprick-like sensation. Stimulation sites were slightly changed after each stimulation to avoid tissue damage and habituation/sensitization effects. In the motor condition, reaction times (RTs) were measured in the millisecond range. Subjects were instructed to close eyes during the course of experiment. Here we focused on the classification between Level-3 and Level-1 trials.

EEG data were recorded with 65 scalp electrodes consisting of all electrodes of the International 10–20 system ( [Fig. 1A](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#F1)). Two additional electrodes were fixed below the outer canthus of each eye. During the recording, the EEG was referenced to the FCz electrode, grounded at AFz, sampled at 1000 Hz, high-pass filtered at 0.015 Hz, and low-pass filtered at 250 Hz. EEG data were preprocessed using the BrainVision Analyzer software (Brain Products, Munich, Germany) and then downsampled to 500 Hz. For artifact detection, a high-pass filter of 1 and a 50 Hz notch filter for line noise removal were applied to the EEG data. Independent component analysis (ICA) was performed to identify components representing eye movements and muscle artifacts based on the energy at spatial topographic map ( [Fig. 1B](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#F1)).

##### Figure 1. Schematic of EEG signal processing and source localization.

![Figure 1.](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/878d/7744433/315bca6ff142/nihms-1636346-f0001.jpg)

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/figure/F1/)

**A:** Illustration of high-density EEG data acquisition. **B**: Independent component analysis (ICA) for rejection of eyeblink artifacts (two upper right components related to eye movement are indicated in the box). The warm (cold) color in scalp topography indicates high (low) energy in the EEG. **C:** Illustration of localized sources on the cortical surfaces in two hemispheres. Warm color indicates high activation activity.

#### Chronic pain patient dataset.

In the second EEG dataset, 32-channel EEG recordings (Enobio, Neuroelectrics) were collected (sampling rate: 500 Hz; high-pass filtered at 0.1 Hz, and low-pass filtered at 250 Hz) at the NYU Langone Health Comprehensive Epilepsy Center. The study was performed in accordance with the National Institutes of Health (NIH) policy for human subjects, and were approved by the Institutional Review Board (IRB) at the NYU School of Medicine. Two right-handed subjects (ages: 69 and 48, female and male, respectively) with chronic back pain first underwent resting EEG with eyes closed and eyes open for 5 minutes each. The testing protocol consisted of 40 trials of mechanical stimulation (noxious painful pinprick (PP) with a 32 gauge blunt needle vs non-noxious stimulus (NP) with a 10 g 

---

### Public dataset.

The first human EEG dataset was obtained from a published study ( [Tiemann et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R47)), in which data were collected in Germany and then in made in public ( [https://osf.io/bsv86/](https://osf.io/bsv86/)). Fifty-one right-handed healthy participants (25 females) with a mean age of 27 years (range 20–37) participated in the study. Participants were recruited via advertisements on bulletin boards of local universities. The study was approved by the local ethics committee and carried out in accordance with the relevant guidelines and regulations. The detailed protocol is referred to ( [Tiemann et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#R47)).

Briefly, the experiment included three core conditions (motor, perceptual and autonomic), which were presented in randomized order. In each condition, 60 painful stimuli were applied to the dorsum of the left hand. We have focused on the motor condition (with 50 participated subjects) in our current study. Stimulus intensity was varied between three individually adjusted levels (Leve-1 \[ _n_ = 20\], Level-2 \[ _n_ = 20\], and Level-3 \[ _n_ = 20\]) in a pseudorandomized sequence. Painful stimuli were applied by means of cutaneous laser stimulation (160 nm wavelength, 1-ms pulse duration, 5-mm spot diameter), which induces a pinprick-like sensation. Stimulation sites were slightly changed after each stimulation to avoid tissue damage and habituation/sensitization effects. In the motor condition, reaction times (RTs) were measured in the millisecond range. Subjects were instructed to close eyes during the course of experiment. Here we focused on the classification between Level-3 and Level-1 trials.

EEG data were recorded with 65 scalp electrodes consisting of all electrodes of the International 10–20 system ( [Fig. 1A](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#F1)). Two additional electrodes were fixed below the outer canthus of each eye. During the recording, the EEG was referenced to the FCz electrode, grounded at AFz, sampled at 1000 Hz, high-pass filtered at 0.015 Hz, and low-pass filtered at 250 Hz. EEG data were preprocessed using the BrainVision Analyzer software (Brain Products, Munich, Germany) and then downsampled to 500 Hz. For artifact detection, a high-pass filter of 1 and a 50 Hz notch filter for line noise removal were applied to the EEG data. Independent component analysis (ICA) was performed to identify components representing eye movements and muscle artifacts based on the energy at spatial topographic map ( [Fig. 1B](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/#F1)).

##### Figure 1. Schematic of EEG signal processing and source localization.

![Figure 1.](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/878d/7744433/315bca6ff142/nihms-1636346-f0001.jpg)

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/figure/F1/)

**A:** Illustration of high-density EEG data acquisition. **B**: Independent component analysis (ICA) for rejection of eyeblink artifacts (two upper right components related to eye movement are indicated in the box). The warm (cold) color in scalp topography indicates high (low) energy in the EEG. **C:** Illustration of localized sources on the cortical surfaces in two hemispheres. Warm color indicates high activation activity.

#### Chronic pain patient dataset.

In the second EEG dataset, 32-channel EEG recordings (Enobio, Neuroelectrics) were collected (sampling rate: 500 Hz; high-pass filtered at 0.1 Hz, and low-pass filtered at 250 Hz) at the NYU Langone Health Comprehensive Epilepsy Center. The study was performed in accordance with the National Institutes of Health (NIH) policy for human subjects, and were approved by the Institutional Review Board (IRB) at the NYU School of Medicine. Two right-handed subjects (ages: 69 and 48, female and male, respectively) with chronic back pain first underwent resting EEG with eyes closed and eyes open for 5 minutes each. The testing protocol consisted of 40 trials of mechanical stimulation (noxious painful pinprick (PP) with a 32 gauge blunt needle vs non-noxious stimulus (NP) with a 10 g m