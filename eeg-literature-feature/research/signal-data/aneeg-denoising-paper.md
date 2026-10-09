# Source capture

- Title: AnEEG: leveraging deep learning for effective artifact removal in EEG data | Scientific Reports
- URL: https://www.nature.com/articles/s41598-024-75091-z
- Fetch date: 2026-10-09
- Capture: webFetch markdown as returned (source output truncated by webFetch; returned text preserved below).

[Skip to main content](https://www.nature.com/articles/s41598-024-75091-z#content)

Thank you for visiting nature.com. You are using a browser version with limited support for CSS. To obtain
the best experience, we recommend you use a more up to date browser (or turn off compatibility mode in
Internet Explorer). In the meantime, to ensure continued support, we are displaying the site without styles
and JavaScript.

AnEEG: leveraging deep learning for effective artifact removal in EEG data


[Download PDF](https://www.nature.com/articles/s41598-024-75091-z.pdf)

[Download PDF](https://www.nature.com/articles/s41598-024-75091-z.pdf)

## Abstract

In neuroscience and clinical diagnostics, electroencephalography (EEG) is a crucial instrument for capturing neural activity. However, this signal is polluted by different artifacts like muscle activity, eye blinks, environmental interference, etc., which makes it more difficult to retrieve important information from the signal. Deep learning methods have demonstrated the potential to lower these artifacts and enhance the EEG’s quality in recent years. In this work, a novel deep learning method,“AnEEG” is presented for eliminating artifacts from EEG signal. The quantitative matrices NMSE, RMSE, CC, SNR and SAR are calculated to confirm the effectiveness of the proposed model. Through this process, it was found that the suggested model outperformed wavelet decomposition techniques. The model achieves lower NMSE and RMSE values, which indicates better agreement with the original signal. Achieving higher CC values means stronger linear agreement with the ground truth signals. Additionally, the model shows improvements in both SNR and SAR values. Overall, this suggested approach showcases promising results in improving the quality of EEG data by utilizing deep learning.

### Similar content being viewed by others

![](https://media.springernature.com/w215h120/springer-static/image/art%3A10.1038%2Fs41598-025-98653-1/MediaObjects/41598_2025_98653_Fig1_HTML.png)

### [A novel EEG artifact removal algorithm based on an advanced attention mechanism](https://www.nature.com/articles/s41598-025-98653-1?fromPaywallRec=false)

ArticleOpen access03 June 2025

![](https://media.springernature.com/w215h120/springer-static/image/art%3A10.1038%2Fs41598-024-61832-7/MediaObjects/41598_2024_61832_Fig1_HTML.png)

### [Insights from EEG analysis of evoked memory recalls using deep learning for emotion charting](https://www.nature.com/articles/s41598-024-61832-7?fromPaywallRec=false)

ArticleOpen access24 July 2024

![](https://media.springernature.com/w215h120/springer-static/image/art%3A10.1038%2Fs41598-023-30864-w/MediaObjects/41598_2023_30864_Fig1_HTML.png)

### [Removing artefacts and periodically retraining improve performance of neural network-based seizure prediction models](https://www.nature.com/articles/s41598-023-30864-w?fromPaywallRec=false)

ArticleOpen access11 April 2023

### Explore related subjects

Discover the latest articles and news in related subjects.

- [Cognitive neuroscience](https://www.nature.com/subjects/cognitive-neuroscience)
- [Computational biology and bioinformatics](https://www.nature.com/subjects/computational-biology-and-bioinformatics)
- [Computer science](https://www.nature.com/subjects/computer-science)
- [Information technology](https://www.nature.com/subjects/information-technology)
- [Neuroscience](https://www.nature.com/subjects/neuroscience)
- [Deep Learning Techniques for Epileptic Seizure Detection](https://www.nature.com/subjects/deep-learning-techniques-for-epileptic-seizure-detection)

## Introduction

One of the most important foundations of neuroscience studies is electroencephalography (EEG), which has produced invaluable insights regarding the vital functions of the human brain. EEG is recorded by capturing the electrical fluctuations of neurons, it does so by using a series of electrodes applied to the scalp to measure voltage fluctuations[1](https://www.nature.com/articles/s41598-024-75091-z#ref-CR1 "Hirsch, L. J. & Brenner, R. P. (eds) Atlas of EEG in Critical Care (Wiley, 2011)."). The versatility of EEG extends not only to BCI applications but also to clinical diagnostics in neurology, whereby it helps solve mysteries about the brain and diagnose neurological disorders, among many others. Through EEG, people have been able to investigate brain function, detect neurologic disorders, and even delve into human cognition depths[2](https://www.nature.com/articles/s41598-024-75091-z#ref-CR2 "Jiang, X., Bian, G. B. & Tian, Z. Removal of artifacts from EEG signals: a review. Sensors 19(5), 987 (2019)."). However, the occurrence of unwanted artifacts makes the usefulness of EEG data highly problematic. cost-effective, simple, and exceptional temporal resolution is the main advantage of EEG[3](https://www.nature.com/articles/s41598-024-75091-z#ref-CR3 "Reddy, A. G. & Narava, S. Artifact removal from EEG signals. Int. J. Comput. Appl. 77(13) (2013).") from the abstract. This explains why it’s employed in several medical and scientific domains. ocular, muscle, powerline noise, and other environmental factors polluted the EEG signal during recordings from the abstract.

Artifacts are mainly separated in 2 categories: biological sources and environmental sources. Biological artifacts are generated by the human body and serve functions like ocular movements, cardiac rhythms, muscular contractions, etc. These biological noise sources can distort the recorded EEG signal, making it difficult to accurately interpret or analyze them. On the other hand, environmental noise originating from outside includes power line interference as well as inadvertent electrode movement. The second source is environmental artifacts generated from external sources like powerline interference, the movement of electrodes, etc. These artifacts also corrupt the structure of EEG data, adding to the complexity of the artifact removal task[4](https://www.nature.com/articles/s41598-024-75091-z#ref-CR4 "Niedermeyer, E. & da Silva, F. L. (eds) Electroencephalography: Basic Principles, Clinical Applications, and Related Fields (Lippincott Williams & Wilkins, 2005).").

These artifacts render EEG incomprehensible and unreliable but also add complexity to the process of meaningful analysis and diagnosis. Unable to address these artifacts not only complicates the interpretation of brain activity but also has important implications in clinical practice since their wrong presence can lead to misdiagnosis and treatment. Consequently, substantial efforts have been made in the field of EEG signal processing to create reliable artifact removal methods. Typically, analyzing EEG data involves considering signals within the range of 1–100 Hz. This range includes both neuronal signals that are the main and artifact signals, thus making their separation a daunting task. These include eye blinks, which typically have low-frequency components below 4 Hertz, and muscular activity with high-frequency characteristics above 13 Hz, thereby overlapping with genuine frequencies associated with EEGs[5](https://www.nature.com/articles/s41598-024-75091-z#ref-CR5 "Muthukumaraswamy, S. D. High-frequency brain activity and muscle artifacts in MEG/EEG: a review and recommendations. Front. Hum. Neurosci. 7, 138 (2013)."). As such, it is extremely difficult to separate these artifacts from their sources to maintain the rich neural information contained in an EEG dataset.

Several methods have been proposed to solve this problem and have been improved over the years, but sometimes they may not be effective in solving it[6](https://www.nature.com/articles/s41598-024-75091-z#ref-CR6 "Jafarifarmand, A. & Badamchizadeh, M. A. Artifacts removal in EEG signal using a new neural network enhanced adaptive filter. Neurocomputing 103, 222–231 (2013)."). These methods are mainly classified into two main approaches: those that estimate and remove artifacts from reference channels, and those that decompose EEG signals into alternative regions for artifact extraction. These methods include regression-based methods[7](https://www.nature.com/articles/s41598-024-75091-z#ref-CR7 "Bigdely-Shamlo, N., Mullen, T., Kothe, C., Su, K. M. & Robbins, K. A. The PREP pipeline: standardized preprocessing for large-scale EEG analysis. Front. Neuroinform. 9, 16 (2015)."), Blind Source Separation(BSS)[8](https://www.nature.com/articles/s41598-024-75091-z#ref-CR8 "Jung, T. P. et al. Removal of eye activity artifacts from visual event-related potentials in normal and clinical subjects. Clin. Neurophysiol. 111(10), 1745–1758 (2000)."), Wavelet Transform, EMD (empirical mode decomposition)[9](https://www.nature.com/articles/s41598-024-75091-z#ref-CR9 "Safieddine, D. et al. Removal of muscle artifact from EEG data: comparison between stochastic (ICA and CCA) and deterministic (EMD and wavelet-based) approaches. EURASIP J. Adv. Signal Process. 2012, 1–15 (2012)."), and their hybrid variants. Successfully identifying and eliminating artifacts from EEG-signals is a very important task in both clinical and practical applications. Manually visualizing segments, identifying, and removing the artifact causes significant loss in valuable neural data. Therefore, research needs to develop more sophisticated artifact removal methods, particularly when it comes to physiological artifacts.

Deep learning, a constantly growing technique, has shown its ability to overcome these challenges in recent years[10](https://www.nature.com/articles/s41598-024-75091-z#ref-CR10 "Correa, A. G., Laciar, E., Patiño, H. D. & Valentinuzzi, M. E. Artifact removal from EEG signals using adaptive filters in cascade. In Journal of Physics: Conference Series (Vol. 90, No. 1, p. 012081) (IOP Publishing, 2007)."). As a type of neural network, Generative Adversarial Networks (GANs) have demonstrated success in creating data that is nearly devoid of artifacts. GAN consists of two main components: a discriminator and a generator. The main job of the discriminator is to differentiate genuine from generated data, and on the other hand, the generator attempts to provide data that is identical to the genuine data[11](https://www.nature.com/articles/s41598-024-75091-z#ref-CR11 "Krishna, G., Tran, C., Carnahan, M., & Tewfik, A. Improving EEG based continuous speech recognition using GAN. arXiv preprint                    arXiv:2006.01260                                     (2020)."). This adversarial learning approach has demonstrated remarkable effectiveness in generating artifact-free EEG signals. This study takes an important step by proposing a new approach to artifact removal techniques that enhances the capabilities of GANs by integrating long-term and short-term memory-LSTM layers. LSTM, a type of recurrent neural network, is effectively capturing temporal dependencies and contextual information, making it well suited for EEG data processing tasks[12](https://www.nature.com/articles/s41598-024-75091-z#ref-CR12 "Roy, S., Kiral-Kornek, I. & Harrer, S. ChronoNet: A deep recurrent neural network for abnormal EEG identification. In Artificial Intelligence in Medicine: 17th Conference on Artificial Intelligence in Medicine, AIME 2019, Poznan, Poland, June 26–29, 2019, Proceedings 17, 47–56 (Springer International Publishing, 2019)."), where temporal dynamics are important.

The primary objective of this work is to remove artifacts from EEG using AnEEG, a LSTM-based GAN deep model. The goal was to enable the GAN to generate pure EEG signals that maintain the original neural activity data by training the model on a variety of datasets containing EEG recordings with various artefacts. In a GAN architecture, the discriminator examines the quality, compares the generated data with ground-truth data, and guides the generator to produce more accurate signals without artifacts.

In the following sections of this paper, the methodology, experiments, and results of the proposed AnEEG model are explained. The effectiveness of the model in removing artifacts under different types of artifacts and recording conditions will also be demonstrated in the following parts. Ultimately, the proposed effort will significantly improve the quality of EEG data, which will increase the reach and significance of neurobiological research and therapeutic applications based on EEG.

## Literature survey

In neuroscience research, removing artifacts from EEG-signals is a very essential task, because artifacts seriously affect the accuracy and reliability of the recorded data. Various approaches have been put forth to deal with this issue, with an emphasis on the creation of sophisticated techniques and clever algorithms for artifact removal. Literature shows that GANs are useful not only as generative models but also as a denoising technique for artifacts. Yang An et al.[13](https://www.nature.com/articles/s41598-024-75091-z#ref-CR13 "An, Y., Lam, H. K. & Ling, S. H. Auto-denoising for EEG signals using generative adversarial network. Sensors 22(5), 1750 (2022).") described an automatic denoising method for multichannel-EEG signal using a GAN-Generative Adversarial Network. To determine whether or not the filtered EEG can preserve as much of the original effective information, they suggested a new loss function. EEG signal’s range limitation was achieved by utilizing energy threshold-based normalization and sample entropy to identify anomalous signals. For removing noise, they defined a method of using GAN-based blind denoising and the discriminator was used to judge whether the noise was filtered out or not. In this work, they used the HaLT dataset that recorded 12 subjects and 960 trials per participant with six imaginary motion states. Data is collected using 22 electrodes and the data sampling frequency is 200 Hz.

For practical BCI applications, Eoin Brophy et al.[14](https://www.nature.com/articles/s41598-024-75091-z#ref-CR14 "Brophy, E. et al. Denoising EEG signals for real-world BCI applications using GANs. Front. Neuroergon. 2, 805573 (2022).")proposed a GAN-supervised technique for EEG signal denoising. They created a pipeline that removed artifacts from EEG time series data. In this experiment, the generator was fed noisy EEG data, and the resulting clean EEG signal was compared to it in the discriminator. With one layer fully connected to the output, the generator composed of a two-layered array of long-term short-term memory (LSTM) with 50 units hidden in each layer. The discriminator is a four-layer, one-dimensional convolutional neural network with a sigmoid function. This experiment made use of the open-source PhysioNet motor/imaging dataset. A 64-channel BCI2000 system with a 160 Hz data sample rate is used to record data.

Tian-Jian Luo et al.[15](https://www.nature.com/articles/s41598-024-75091-z#ref-CR15 "Luo, T. J., Fan, Y., Chen, L., Guo, G. & Zhou, C. EEG signal reconstruction using a generative adversarial network with wasserstein distance and temporal-spatial-frequency loss. Front. Neuroinform. 14, 15 (2020).") developed a technique for reconstruction of EEG signals using GAN with temporal-spatial-frequency loss function and Wasserstein distance. By calculating the mean-squared error from time series and power spectral density features, the loss function recreated the signal. Three distinct datasets with varying sampling rates were used in this experiment to train and assess the networks. The 2018-published Action Observation Dataset was the first dataset, and it was sampled at 250 Hz and 64 channels. The Grasp and Lift (GAL) dataset was the second one. A “Brain Amp” device with thirty two channels, sensitivity, and sampled at 500Hz and 0.1 mv is used to collect the data. The third dataset was the Motor Imagery dataset. Data was collected from 12 subjects using 32 channels, and the sampling rate was 250Hz.

Jin Yin et al.[16](https://www.nature.com/articles/s41598-024-75091-z#ref-CR16 "Yin, J., Liu, A., Li, C., Qian, R. & Chen, X. A GAN guided parallel CNN and transformer network for EEG denoising. IEEE J. Biomed. Health Inform. (2023).") presented a technique for denoising EEG signals dubbed GCTNet, which included a transformer network and a GAN-guided parallel CNN. The discriminator identified and rectified holistic discrepancies between clean and denoised EEG signals, while the generator composed of parallel CNN blocks and transformer blocks, captured both global and temporal dependencies. To assess their approach, the researchers used one genuine dataset and three semi-simulated datasets. The MIT-BIH Arrhythmia Dataset combined clean ECG signals with EEG segments; the EEG DenoiseNet dataset combined clean EEG segments with EMG and EOG segments; and the semi-simulated EEG/EOG dataset involved linearly mixing EOG artefacts with clean EEG signals. The actual dataset is made up of recordings from a patient with epilepsy that were made for 10 s at a sampling rate of 250 Hz and included 21 channels. Compared to other existing approaches, GCTNet showed a considerable performance improvement, achieving a reduction of 11.15% in relative root mean square error-RRMSE and an improvement of 9.81 in signal-to-noise ratio-SNR.

Phattarapong Sawangjai et al.[17](https://www.nature.com/articles/s41598-024-75091-z#ref-CR17 "Sawangjai, P. et al. EEGANet: Removal of ocular artifacts from the EEG signal using generative adversarial networks. IEEE J. Biomed. Health Inform. 26(10), 4913–4924 (2021).") developed an Ocular artifact removal framework ‘EEGENet’, based on generative adversarial networks. The algorithm runs under various conditions, i.e., no Eye-movement, vertical Eye-movement, horizontal Eye-movement, and eye blinking. For the training set, state-of-the-art EOG suppression techniques were used to create a clean target EEG signal that could be used as the original truth. The cleaned dataset is used to train the discriminator, and raw EEG data is passed into the Generator. For the data, they used 3 open-source datasets. The first one was the EEG Eye Artefact Dataset, which was pre-processed using a notch filter after data was gathered from 50 subjects. The second dataset, BCI Competition IV2b, has three channels and a 250Hz sampling rate. The Multimodal Signal Dataset was the third dataset. It featured sixty electrodes and a 2500 Hz sampling rate.

Sandhyalati Behera et al.[18](https://www.nature.com/articles/s41598-024-75091-z#ref-CR18 "Behera, S. & Mohanty, M. N. A machine learning approach for artifact removal from brain signal. Comput. Syst. Sci. Eng.  45(2) (2023).") proposed a machine-learning based approach for eliminating artifacts from EEG signals. To address the challenges posed by nonlinear and non-stationary artifactual signals, they utilize a RVFLN(Random Vector Functional Link Network) model. An exponentially weighted RLS(Recursive Least Squares) algorithm is employed to build the adaptive filter within this novel RVFLN model. To build and validate their model, clean EEG data is sourced from the openly available Mendeley database, while ECG signals, used to create artifacts in the EEG signals, are collected from the Physionet database. This approach aims to effectively remove artifacts and verify the efficacy of the proposed algorithm.

A novel approach to artifact removal was put forth by Zainab Jamil et al.[19](https://www.nature.com/articles/s41598-024-75091-z#ref-CR19 "Jamil, Z., Jamil, A. & Majid, M. Artifact removal from EEG signals recorded in non-restricted environment. Biocybern. Biomed. Eng. 41(2), 503–515 (2021)."), where they classified EEG fragments based on Eye-movement and then used discrete wavelet transform and independent component analysis to eliminate artifacts. The dataset was obtained under unrestricted conditions, from 29 subjects who engaged in activities such as walking, watching videos, and expressing gestures and facial expressions. EEG signals were processed by extracting thirteen morphological features to identify segments containing Eye-movements. These segments are then denoised without distorting the signal morphology.

Rajdeep Ghosh et al.[20](https://www.nature.com/articles/s41598-024-75091-z#ref-CR20 "Ghosh, R. et al. Automatic eyeblink and muscular artifact detection and removal from EEG signals using k-nearest neighbor classifier and long short-term memory networks. IEEE Sens. J. 23(5), 5422–5436 (2023).") provided a dependable technique for automatically identifying and eliminating muscular artifacts and eye blinks from EEG recordings. This method combines a long short-term memory-LSTM with a k closest neighbour (KNN) classifier. A sliding-window method lasting 0.5 s is utilized to find and eliminate artifacts. For every EEG segment, the properties of peak-to-peak amplitude, average rectified value, and variance are determined. Bed segments were identified by KNN classifier and then processed by the LSTM network to eliminate artifacts. The suggested techniques showed outstanding performance in terms of the signal-to-artifact ratio,correlation coefficient, structural similarity, and normalized RMS error, with an accuracy of 97.4% in identifying damaged segments.

Shalini Stalin et al.[21](https://www.nature.com/articles/s41598-024-75091-z#ref-CR21 "Stalin, S. et al. A machine learning-based big EEG data artifact detection and wavelet-based removal: an empirical approach. Math. Probl. Eng. 2021, 1–11 (2021).") suggested an approach for identifying and eliminating major movement artifacts from one channel EEG. “Support Vector Machine(SVM)” was utilized in the detection procedure, and “Ensemble Empirical Mode Decomposition (EEMD)” was utilized to extract signal features. Motion artifacts can be eliminated with the use of canonical-correlation-analysis-CCA. And the remaining motion artifacts were eliminated using the WT-(Wavelet Transform) technique, they employed the “Hawks optimization algorithm (HHO)” to optimize the outcomes.

Sakib Mahmud et al.[22](https://www.nature.com/articles/s41598-024-75091-z#ref-CR22 "Mahmud, S. et al. Restoration of motion-corrupted EEG signals using attention-guided operational CycleGAN. Eng. Appl. Artif. Intell. 128, 107514 (2024).") proposed the Attention Guided Operational CycleGAN (AGO-CycleGAN), to remove motion artifacts and enhance the quality of corrupted EEG signals. This approach combines PatchGAN-based discriminators with attention-guided Feature Pyramid Network generators that are adjusted bottlenecks, as well as self-generative operational neurons. The model was tested and trained on a single-channel EEG dataset of 23 subjects, using a subject-independent Jackknife cross-validation approach. The method outperformed other techniques and was evaluated through both qualitative and quantitative analyses, employing robust metrics in the temporal and frequency domains.

**Table 1 Summary of literature survey.**

[Full size table](https://www.nature.com/articles/s41598-024-75091-z/tables/1)

Abhay B. Nayak et al.[23](https://www.nature.com/articles/s41598-024-75091-z#ref-CR23 "Nayak, A. B., Shah, A., Maheshwari, S., Anand, V., Chakraborty, S. & Kumar, T. S. An empirical wavelet transform-based approach for motion artifact removal in electroencephalogram signals. Decis. Anal. J., 100420 (2024).") developed an effective approach for eliminating motion artifacts from EEG signals using the empirical wavelet transform (EWT) technique. Initially, the EEG signals are decomposed into narrowband signals known as intrinsic mode functions (IMFs). The first step involves employing principal component analysis (PCA) to suppress noise from the decomposed IMFs. In the second step, they identify IMFs with noisy components using a variance measure and subsequently remove the identified sections. For their experiments, they utilize publicly available EEG data from the Physionet dataset. Their IMF-variance-based method proves to be more efficient than PCA-based approaches.

A discrete wavelet transform-based technique for eliminating artifacts from EEG signals-blink artefacts in particular-is presented by Wenjia Gao et al.[24](https://www.nature.com/articles/s41598-024-75091-z#ref-CR24 "Gao, W., Liu, D., Wang, Q., Zhao, Y. & Sun, J. FBLPF-ABOW: an effective method for blink artifact removal in single-channel EEG signal. IEEE J. Biomed. Health Inform. (2023)."). The technique consists of multiple steps: first, intervals containing blink artifacts are detected using a fixed-length window and a forward-backward low-pass filter (FB-LPF). The best representative blink signal is then used to build an adaptive bi-orthogonal wavelet (ABOW). Lastly, discrete wavelet transform-(DWT) and ABOW are used to filter the discovered signals. The DWT’s decomposition depth is automatically determined by comparing the signals that contain artefacts. They use a semi-simulated EEG dataset for their research, in which they capture 200 Hz signals from 54 healthy people without blinking, normalise, scale, and add ten various amplitude blink signals taken from the dataset.

Yuanzhe Dong et al.[25](https://www.nature.com/articles/s41598-024-75091-z#ref-CR25 "Dong, Y., Tang, X., Li, Q., Wang, Y., Jiang, N., Tian, L. et al. An approach for EEG denoising based on wasserstein generative adversarial network. IEEE Trans. Neural Syst. Rehab. Eng. (2023).") proposed an EEG denoising technique called Artefact Removal WGAN (AR-WGAN), which used the Wasserstein Generative Adversarial Network (WGAN). Its efficacy is assessed using a dataset that is made available to the public as well as one that was acquired on its own. The main contribution is providing a WGAN-based end-to-end denoising model that can effectively filter massive amounts of unprocessed EEG data. The generator, which consists of 1D convolutional and transposed convolutional layers, produced fake data, which produced output close to 0, which the discriminator discriminated from genuine data, which was produced around 1. The public EEGdenoiseNet Benchmark Dataset, which comprises semi-synthetic data with clean EEG segments along with segments of ocular and muscular artifacts, is used by the authors for experimental validation. They also use a self-assembled dataset that includes the raw EEG data from four healthy controls.

The literature reviews are detailed in Table [1](https://www.nature.com/articles/s41598-024-75091-z#Tab1), describing each work’s dataset, methodologies, performance, and limitations.

## Generative adversarial network

Proposed in 2014[26](https://www.nature.com/articles/s41598-024-75091-z#ref-CR26 "Creswell, A. et al. Generative adversarial networks: An overview. IIEEE Signal Process. Mag. 35(1), 53–65 (2018)."), Generative Adversarial Networks (GANs) is a new method for semi-supervised and unsupervised learning. They accomplish this by modeling high-dimensional data distributions informally. The two primary parts of a GAN are the discriminator and the generator. The discriminator contrasts the bogus data produced by the generator with the genuine data. Most importantly, the generator can only learn through interacting with the discriminator; it does not have direct access to actual images. The discriminator can work with samples taken from a stack of real photos as well as generated samples. By identifying if the image originated from the generator or the real stack, the discriminator receives an error signal. This same error signal, through the discriminator, is used to train the generator, guiding it toward producing higher-quality forgeries. Using Equation  [1](https://www.nature.com/articles/s41598-024-75091-z#Equ1), the cost of training the GAN is calculated.[27](https://www.nature.com/articles/s41598-024-75091-z#ref-CR27 "Abdelfattah, S. M., Abdelrahman, G. M. & Wang, M. Augmenting the size of EEG datasets using generative adversarial networks. In 2018 International joint conference on neural networks (IJCNN), 1–6 (IEEE, 2018)."):

$$\\begin{aligned} \\begin{aligned} \\min \_{Gen} \\max \_{Dis} F(Dis,Gen)&= {\\mathbb {E}}\_{x \\sim P(X)} \[\\log Dis(x; \\theta \_D)\] \\\&\\quad + {\\mathbb {E}}\_{x \\sim P(Z)} \[\\log (1 - Dis(z; \\theta \_{\\text {Gen}}))\] \\end{aligned} \\end{aligned}$$

(1)


Additionally, the loss function of GANs is expressed as:

$$\\begin{aligned} \\begin{aligned} L = \\min \_{Gen} \\max \_{Dis} \[\\log Dis(x) + \\log (1 - Dis(Gen(z)))\] \\end{aligned} \\end{aligned}$$

(2)


Three guiding concepts form the basis of GAN’s operational framework[28](https://www.nature.com/articles/s41598-024-75091-z#ref-CR28 "Aggarwal, A., Mittal, M. & Battineni, G. Generative adversarial network: An overview of theory and applications. Int. J. Inf. Manag. Data Insights 1(1), 100004 (2021)."):

- Enable the learning of the generative model so that data can be generated using random representation.

- Training the model in a conflicting situation where the discriminator and generator are in opposition.

- Training the system as a whole with artificial intelligence algorithms and deep learning neural networks.


Granted that they function well for semi-supervised and reinforcement learning, GAN networks are essentially used for unsupervised machine learning techniques. Together, these elements provide GANs with end-to-end solutions across a range of industries, including finance, healthcare, and mechanical.

## Materials and methods

EEG data is prone to artifacts, and there is currently no algorithm in the literature that can completely eliminate all artifacts. This work seeks to employ GAN as a method of artifact removal covering most artifacts because of its reconstructive approach. This study aims to denoize EEG data for sophisticated Brain-Computer Interface (BCI) analysis. The proposed method entails feeding noisy signals through a generator to generate clean versions, which are then compared to actual clean signals by a discriminator. This iterative process facilitates the effective denoising of EEG signals. Figure 3 illustrates the proposed models architecture, providing insight into the proposed approach to denoising EEG signals.

### Data recording

Building a computational dataset is crucial to training a deep learning model. A novel dataset with nine different artifacts for training and evaluating the model performance was created. These data were gathered from five subjects studying at Gauhati University, Assam, India. The subjects signed written consent for the EEG recording. Subjects performed several tasks while their EEG signals were recorded. These tasks included blinking, eye-movements, chewing, and others. Every participant took part in multiple recording sessions, providing us with different samples for efficient training and testing of the proposed model.

The data recording session began with a 2-s relaxing period during relaxation the subject was directed to remain motionless and relax. Following the relaxing phase, the subject was guided to blink continuously for 11 s. The process is mentioned in Fig. [1](https://www.nature.com/articles/s41598-024-75091-z#Fig1). The primary goal of the given task was to capture the electrical activity that goes along with blinking the eyes. Subsequently, another 2-s relaxation period ensued. Following the previous phase, the process was repeated, this time directing the participant to intentionally move their eyes for 11 s. The motive of the task is to capture electrical signals of the brain related to eye-movement. This was followed by another 2 s relaxation period, allowing the subject to recover and go back to their initial state before moving on to the next phase.

**Fig. 1**

![Fig. 1](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-024-75091-z/MediaObjects/41598_2024_75091_Fig1_HTML.png)

[Full size image](https://www.nature.com/articles/s41598-024-75091-z/figures/1)

Data recording session paradigm, first and last 2-s is relaxing period.

During the data collection session, this pattern is alternating between specific activities and relaxation intervals. Data was gathered for nine different artifacts including eye blinking, eye-movement, chewing, clenching teeth, swallowing movement, tongue movement, jaw movement, head movement, and shoulder movement.

However, during the recording process, subjects naturally blinked their eyes, which introduced an additional artifact. Therefore, the EEG segments contain the main artifact of interest along with the eye blink artifact. This was considered in the artifact removal process.

The data were recorded from thirty two channels using the international 10–20 system for electrode placement, at a sampling frequency of 128 Hz. A wearable EMOTIV-EPOC Flex Gel Kit with thirty two channels was used to collect data. A total of 117 s of data were recorded for each subject, resulting in a total data length of 14,976 samples per subject.

### Data pre-processing

Preprocessing procedures are very essential part for EEG data analysis to improve the data quality and prepare them for more effective analysis. Given that the frequency range of 0.5 to 40 Hz is often where the main characteristics of EEG signals can be discovered, so, the raw EEG data was filtered within this range. This filtering process was effective in removing high-frequency noise, like linear noise at 50 or 60 Hz, while also correcting baseline drift[29](https://www.nature.com/articles/s41598-024-75091-z#ref-CR29 "Zhang, S., McCane, B., Neo, P. S. & McNaughton, N. Removing mains power artefacts from EEG-a novel template-based method. bioRxiv, 2020-01 (2020).").

Subsequently, the original signals were normalized to limit the signal range. During activities such as eye blinking, facial muscle movement, or other artifacts, the amplitude values of EEG channels can experience sudden high fluctuations. Uncontrolled EEG values have the potential to cause instability in the GAN model because different subjects may show different ranges of uncertainty in their data. To address this issue, min-max normalization is used[30](https://www.nature.com/articles/s41598-024-75091-z#ref-CR30 "Jeong, D., Yoo, S. & Yun, J. Cybersickness analysis with eeg using deep learning algorithms. In 2019 IEEE Conference on Virtual Reality and 3D User Interfaces (VR), 827–835 (IEEE, 2019)."). This method scales the EEG within a specified range, typically from − 1 to 1, ensuring consistency across all subjects.

After the normalization, each artifact is clipped from the raw normalized signal. This separation makes efficient management and independent analysis of artifacts while preserving their characteristics, temporal patterns, and associated features.

In used dataset, each artifact spans 11 s and is preceded and followed by a 2-s relaxation period, with the final 2 s overlapping with the relaxation period of the subsequent artifact, resulting in a total artifact duration of 15 s.

The EEG was captured from thirty two channels at a 128 Hz sampling rate. To extract all artifacts, the number of samples is calculated by multiplying the sampling rate with the length of the artifact plus the relaxation period (11 + 2 + 2). Mathematically, the total number of frames to be clipped (F) is determined as follows:

$$\\begin{aligned} F = T \\times 128 \\end{aligned}$$

(3)


Where F represents the total frames to be clipped and T represents the duration of the activity.

**Fig. 2**

![Fig. 2](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-024-75091-z/MediaObjects/41598_2024_75091_Fig2_HTML.png)

[Full size image](https://www.nature.com/articles/s41598-024-75091-z/figures/2)

5-s EEG segments of blinking, eye-movements, chewing and clenching teeth from sub1 sub2 sub3 and sub4.

After that, a 5-s window size was adopted to segment and each 15-s artifact was divided into non-overlapping 5-s segments. This results in the extraction of three segments from each channel, totaling 96 segments from all 32 channels. Considering data collection from five subjects, a total of 480 segments are collected. Similarly, 480 segments are collected for each artifact type. Additionally, Fig. [2](https://www.nature.com/articles/s41598-024-75091-z#Fig2) illustrates a segment of eye blink, eye-movement, chewing, and clench teeth artifacts from the first four subjects.

The segments are divided into two groups, adopting a ratio of 80:20 for training and testing purposes. Specifically, portions from four subjects were employed for training, and the segments from the fifth subject were saved for testing. A diverse dataset is ensured by this allocation for the model evaluation. Eyeblink, eye-movement, chewing, and clench teeth artifact segments from the four subjects are used for training, while the same artifact segments from the fifth subject are reserved for testing. Overall, 1536 EEG segments are allocated for training, with an additional 384 segments designated for testing.

### Model architecture

The proposed investigation used a recurrent neural network architecture in addition to the Generative Adversarial Network (GAN) concept first presented by Ian Goodfellow[31](https://www.nature.com/articles/s41598-024-75091-z#ref-CR31 "Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S. et al. Generative adversarial nets. Adv. Neural Inf. Process. Syst. 27 (2014)."). The high-frequency noise, such as linear noise at 50 or 60 Hz, was successfully eliminated by this filtering procedure. As demonstrated in Fig. [3](https://www.nature.com/articles/s41598-024-75091-z#Fig3) A, the design consists of two primary parts: the discriminator and the generator. Figure [3](https://www.nature.com/articles/s41598-024-75091-z#Fig3) B,C show the architecture of the generator and the discriminator, respectively.

**Fig. 3**

![Fig. 3](https://media.springernature.com/lw685/springer-static/image/art%3A10.1038%2Fs41598-024-75091-z/MediaObjects/41598_2024_75091_Fig3_HTML.png)

[Full size image](https://www.nature.com/articles/s41598-024-75091-z/figures/3)

( **A**) Architectural Design of the Model B. Generator Architecture with Layered Model, C. Discriminator Architecture with Layered Model.

The generator’s main goal was to generate clean EEG signals from raw input data that had been distorted by things like teeth clenching, eye blinking, and movement. On the other hand, the discriminator receives training to discern between the clean ground truth signals and the produced signals.

From the artifacted EEG signals \\(S^\\text {artifacted}\\) in the training dataset, wavelet decomposition (DWT) artifact removal method was applied to generate the clean target EEG signal _S_. These clean signals served as the reference baseline. Subsequently, \\(S^\\text {artifacted}\\) was utilized along with their corresponding clean signal _S_ as inputs, to train the generator. The focus was to produce artifact-free signal \\(S'\\).

A two-step cleaning technique was employed, in the wavelet denoising method. First, each row of the data matrix was selected for subtracting the average trend from signal using Golay filtering. This initial step aids in eliminating any underlying trends that might obscure the EEG data. Then a thresholding method for the wavelet coefficients of each line was used to solve the noise removal problem. This involved decomposing the signal using the wavelet transform, determining the threshold of detail factors, and correspondingly reconstructing the signal. The thresholding process was determined by the third detail coefficient’s standard deviation, multiplied by a factor of 0.8.

To analyse serial data, the generator model was constructed using LSTM. When data sequences are followed by an activation function called a rectified parametric linear unit (PReLU), it can identify long-term relationships in those sequences. The architecture is described in Table  [2](https://www.nature.com/articles/s41598-024-75091-z#Tab2). Using both the input and hidden states at each timestep, the LSTM produces the final output in a recurrent feedback loop. To enable the activation function to adaptively learn the rectification parameter for each neuron during training, PReLU introduce learnable parameters to it. This helps to reduce the vanishing gradient problem that deep networks frequently face[32](https://www.nature.com/articles/s41598-024-75091-z#ref-CR32 "Brophy, E., Wang, Z., She, Q. & Ward, T. Generative adversarial networks in time series: A systematic literature review. ACM Comput. Surv. 55(10), 1–31 (2023)."). With a ’tanh’ activation function, the Dense layer is the last in the model. The dense layer is used to convert the characteristics that the LSTM layers have learned into the final output format. The “tanh” activation function overwrites the output in the range \[− 1, 1\], which is typically used in GANs to create generator outputs. The generator output is then compared with the real data to calculate the RMSE(root mean square error).

LSTM layers make up the majority of the discriminator’s architecture when it comes to designing discriminator blocks. The LSTM layer is followed by a LeakyReLU activation function with a slightly decline in slope (alpha=0.2). In the LSTM layer, the number of units decreases to 64 from 1028. A Flatten layer is used after the LSTM layer. Next, a single neuron Dense layer is employed. The sigmoid activation function is applied to the Dense layer’s output. Figure [3](https://www.nature.com/articles/s41598-024-75091-z#Fig3) describes the model architecture.

### Model training and implementation

In the phase of model training, the generator was fed with training data, from which the generator generated fake samples. These generated data were then sent to the discriminator block, where clean data of the corresponding artifacted segments were already present.

Specifically, 80% of the data was set aside for training, and this data was filtered using the wavelet decomposition technique beforehand. Thus, the generated signal is compared with its respective clean signal. The discriminator loss-function was set to binary cross-entropy, while the generator loss function was set to mean squared error (MSE). Both loss functions were optimized using the Adam optimizer, configured with a learning rate of 1e−3, \\(\\beta \_1\\) at 0.9, \\(\\beta \_2\\) at 0.999, and \\(\\epsilon\\) at 1e−08.

To prevent the discriminator from being updated during the generator training phase, it is set to non-trainable. The discriminator compares the generated data with the clean data at the end of each epoch to determine the MSE loss. The generator weights are then updated based on this loss. Additionally, the GAN loss was weighted with \[1.0, 5e−4\], applied to MSE and binary cross-entropy, respectively. This process helped the generator produce a signal \\(S'\\) that closely resembled the ground truth signal _S_. The model was trained with a batch size of 128 over 500 epochs.

**Table 2 Parameters for the Generator and Discriminator models.**

[Full size table](https://www.nature.com/articles/s41598-024-75091-z/tables/2)

Once the training process is completed, the trained generator and discriminator models are obtained. However, the generator is needed to purify the signal. The discriminator’s main job is to deliver loss, which directs the generator to adjust its weights to produce more precise signals. After successful training, it was required to send the 5-s segmented normalized data, which was stored for testing purposes. Specifically, 20% of the data was reserved for testing. This test data is then fed into the trained generator to produce cleaned EEG signals, enabling us to evaluate the performance of the model on unseen data.

### Evaluation methods

Five metrics are primarily utilised to assess the performance of the signal produced by the suggested model: CC (Correlation Coefficient), RMSE (Root Mean Square Error), NMSE (Normalised Mean Square Error), Signal to Noise Ratio (SNR) and Signal to Artifact Ratio (SAR).

#### Normalized mean squared error (NMSE)

It is a metric, used to confirm whether the generated signal by the model is similar to the actual EEG signal. In addition to visual examination, the Normalised Mean Squared Error (NMSE) measures the average squared difference between the Raw and Generated signals, normalised by the original signal’s variance. Mathematically, it is presented as:

$$\\begin{aligned} \\text {MSE}= & \\frac{{\\Vert \\text {orgSig}(:) - \\text {recSig}(:) \\Vert }\_{2}^2}{\\text {length}(\\text {orgSig}(:))} \\end{aligned}$$

(4)


$$\\begin{aligned} \\text {NMSE}= & \\frac{{\\text {MSE}}}{\\Vert \\text {orgSig}(:) \\Vert \_{2}^2} \\end{aligned}$$

(5)


Here, orgSig(:) represents the original signal, recSig(:) represents the reconstructed signal.

NMSE provides a measurement of the relative error between the original (raw signal) and generated signals, considering the scale of the signals. NMSE values close to zero indicate a good match between the original (raw signal) and generated signals, while higher values indicate larger discrepancies.

#### Root mean squared error (RMSE)

Without normalization, the RMSE offers a metric comparable to the NMSE. It represents the square root of the average squared difference between corresponding samples of the original and reconstructed signals. Mathematically, it is expressed as[33](https://www.nature.com/articles/s41598-024-75091-z#ref-CR33 "Wang, W. & Lu, Y. Analysis of the mean absolute error (MAE) and the root mean square error (RMSE) in assessing rounding model. In IOP Conference Series: Materials Science and Engineering (Vol. 324, p. 012049) (IOP Publishing, 2018)."):

$$\\begin{aligned} \\text {MSE}= & \\frac{\\sum \_{i=1}^{N} (data\_i - \\text {estimate}\_i)^2}{\\text {numel}(data)} \\end{aligned}$$

(6)


$$\\begin{aligned} \\text {RMSE}= & \\sqrt{\\text {MSE}} \\end{aligned}$$

(7)


Here, \\(\\text {data}\_i\\) and \\(\\text {estimate}\_i\\) represent the \\(\\text {i}\\)-th elements of the data and estimate vectors, respectively. The total number of elements in the data vectors are defined by N. _numel_( _data_) indicates the number of elements in the _data_ vector.

RMSE quantifies the average size of the errors between the generated and original signals. It presents an absolute measure of the error, regardless of the scale of the signals. Lower RMSE (Root-Mean-Square Error) values define better agreement between the original (raw signal) and reconstructed signals.

#### Correlation coefficient (CC)

Correlation Coefficient (CC) is employed to evaluate the model’s capability to remove artifacts while maintaining the brain signal data. It measures how closely relate original (raw signal) and reconstructed signals to each other linearly. Mathematically, CC is expressed as:

$$\\text {CC} = \\frac{{\\sum \_{i=1}^{N}}({{Sig}\_{i}} - {\\bar{Sig}}) ({{Sig}^{\\prime}\_{i}} - {\\bar{Sig}^{\\prime}})}{\\sqrt{{\\sum \_{i=1}^{N}}({{Sig}\_{i}} - {\\bar{Sig}})^2} {\\sqrt{{\\sum \_{i=1}^{N}}({{Sig}^{\\prime}\_{i}} - {\\bar{Sig}^{\\prime}})^2}}}$$

(8)


Here, _Sig_ and \\(Sig^{\\prime}\\) represent the \\(i-\\)th sample of the original and generated signals, respectively, and \\({\\bar{Sig}}\\) and \\(Sig^{\\prime}\\) represent the mean of the original (raw signal) and generated signals. Respectively, Total number of samples is represented by _N_.

How well the variation of one signal predicts the variation of the others is measured by CC. Its values range from − 1 to 1, where 0 denotes no correlation, − 1 represents a perfect negative correlation, and 1 represents a perfect positive correlation. Higher CC values indicate stronger linear agreement between the original (raw signal) and reconstructed signals. Thus, CC helps assess the model’s ability to preserve essential brain signal data while removing artifacts.

#### Signal to noise ratio (SNR)

SNR is employed to evaluate the effectiveness of artifact removal by comparing the signal power within a specific frequency band to the noise power. The formula used for calculating the SNR in decibels (dB) is:

$$\\begin{aligned} \\text {SNR}\_{\\text {band}} = 10 \\times \\log \_{10}\\left( \\frac{P\_{\\text {signal}}}{P\_{\\text {noise}}}\\right) \\end{aligned}$$

(9)


\\(P\_{\\text {signal}}\\) is the power of the EEG signal within a particular frequency band, calculated by summing the power spectral density (PSD) values within the frequency band of interest.

$$\\begin{aligned} P\_{\\text {signal}} = \\sum \_{k \\in \\text {Band}} \|Y(k)\|^2 \\end{aligned}$$

(10)


_Y_( _k_) represents the FFT coefficient at the \\(k^{\\text {th}}\\) frequency bin.

The power of the noise is then calculated by subtracting the signal power from the total power of the entire signal:

$$\\begin{aligned} P\_{\\text {noise}} = \\sum \_{k=0}^{N-1} \|Y(k)\|^2 - P\_{\\text {signal}} \\end{aligned}$$

(11)


in the equation, \\(\\sum \_{k=0}^{N-1} \\left\| Y(k) \\right\| ^2\\) is the total power of the signal, summed across all frequency.

A higher SNR indicates a cleaner signal with less residual noise, confirming the model’s effectiveness in separating the desired EEG signal from unwanted artifacts[35](https://www.nature.com/articles/s41598-024-75091-z#ref-CR35 "Väisänen, O. & Malmivuo, J. Improving the SNR of EEG generated by deep sources with weighted multielectrode leads. J. Physiol. Paris 103(6), 306–314 (2009)."). This metric demonstrates how well the artifact removal process enhances the signal quality, ensuring that the EEG data is more representative of the true neural activity, with minimal distortion from external noise sources.

#### Signal to artifact ratio (SAR)

Signal-to-Artifact Ratio (SAR) measures the effectiveness of a methodology by comparing the variance of the artifact signal to the variance of the residual artifact after applying a given methodology. Mathematically, SAR is expressed as:

$$\\begin
…[truncated]