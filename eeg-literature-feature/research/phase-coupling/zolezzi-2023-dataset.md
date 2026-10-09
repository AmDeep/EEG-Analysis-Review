# Chronic neuropathic pain: EEG data in eyes open and eyes closed with painDETECT and brief pain inventory reports

Primary source title: Chronic neuropathic pain: EEG data in eyes open and eyes closed with painDETECT and brief pain inventory reports
URL: https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/
Fetched by webFetch/search; retrieval date was not supplied in callback metadata.

## Abstract

Thirty-six chronic neuropathic pain patients (8 men and 28 women) of Mexican nationality with a mean age of 44±13.98 were recruited for EEG signal recording in eyes open and eyes closed resting state condition. Each condition was recorded for 5 min, with a total recording session time of 10 min. An ID number was given to each patient after signing up for the study, with which they answered the painDETECT questionnaire as a screening process for neuropathic pain alongside their clinical history. The day of the recording, the patients answered the Brief Pain Inventory, as an evaluation questionnaire for the interference of the pain with their daily life. Twenty-two EEG channels positioned in accordance with the 10/20 international system were registered with Smarting mBrain device. EEG signals were sampled at 250 Hz with a bandwidth between 0.1 and 100 Hz. The article provides two types of data: (1) raw EEG data in resting state and (2) the report of patients for two validated pain questionnaires. The data described in this article can be used for classifier algorithms considering stratifying chronic neuropathic pain patients with EEG data alongside their pain scores. In sum, this data is of extreme relevance for the pain field, where researchers have been seeking to integrate the pain experience with objective physiological data, such as the EEG.

**Keywords:** EEG raw data, EEG resting state, chronic neuropathic pain, pain classification, Brief Pain Inventory, painDETECT, pain severity, pain scores

* * *

**Specifications Table**

|     |     |
| --- | --- |
| Subject | Biological Sciences: Neuroscience |
| Specific subject area | Neuroscience: Sensory Systems, Electrophysiology |
| Type of data | Electrophysiological recordings – signals recorded through electroencephalography<br>Text files – electrode positions <br>Excel file – ID with demographics, clinical history, and questionnaire results |
| How the data were acquired | Electroencephalographic recordings <br>24-channel EASYCAP electrode cap by the 10/20 international system<br>SMARTING EEG Bluetooth amplifier<br>Sampling frequency: 250 Hz<br>Resolution: 24 bits<br>Bandwidth: 0.1 – 100 Hz<br>Openvibe for signal acquisition <br>Electrode impedances were kept below 5 kΩ. <br>Right (M2) and left mastoid (M1) electrodes as offline reference<br>Questionnaires painDETECT and Brief Pain Inventory were used in their Spanish validated version. The English version is uploaded in the repository |
| Data format | Questionnaire results and demographics– excel<br>Electrophysiological signals – raw<br>Electrode positions – text |
| Description of data collection | All patients had: age above 18 years old, chronic neuropathic pain for more than 3 months, long-term pharmacological treatment for at least 4 weeks prior to the EEG recording, absence of a major psychiatric disorder (i.e., schizophrenia, major depressive disorder, bipolar disorder), absence of a neurological disorder (i.e., epilepsy, tinnitus), and Total Score >12 points of painDETECT(the questionnaire outcome was confirmed by the clinical history of the patient). |
| Data source location | • Institution: Tecnológico de Monterrey<br>• City/Town/Region: Monterrey, Nuevo León<br>• Country: México<br>• Latitude and longitude: 25.647903503278236,−100.28928530251193 |
| Data accessibility | _Electrophysiological recordings, text file and excel file are available in a public repository._<br>Repository name: Mendeley Data<br>Data identification number: doi: [10.17632/yj52xrfgtz.4](https://doi.org/10.17632/yj52xrfgtz.4)<br>Direct URL to data: [https://data.mendeley.com/datasets/yj52xrfgtz/4](https://data.mendeley.com/datasets/yj52xrfgtz/4)<br> Version 4<br>_M. Zolezzi, Daniela; Naal-Ruiz, Norberto E.; Alonso Valerdi, Luz María; Ibarra Zárate, David Isaac (2023), “Chronic Neuropathic Pain: EEG data in eyes open (5 min) and eyes closed (5 min) with questionnaire reports”, Mendeley Data, V4_, doi: [10.17632/yj52xrfgtz.4](https://doi.org/10.17632/yj52xrfgtz.4) |
| Related research article | D.M. Zolezzi, L.M. Alonso-Valerdi, D.I. Ibarra-Zarate, EEG frequency band analysis in chronic neuropathic pain: A linear and nonlinear approach to classify pain severity, Comput Methods Programs Biomed. 230 (2023) 107349. [https://doi.org/10.1016/j.cmpb.2023.107349](https://doi.org/10.1016/j.cmpb.2023.107349). |

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/table/utbl0001/)

## Value of the Data

- •
To our knowledge, this is the first available dataset containing the electroencephalographic activity of chronic neuropathic pain patients alongside the patient's pain reports from two validated pain questionnaires. This information is useful for characterization and classification for better pain management.

- •
Pain researchers, neuroscientists, data scientists, or clinicians would benefit from exploring different pain characterization and stratification methods.

- •
This data can be reused to understand the relation of electrophysiological data, such as neuronal brain oscillations, with the patient's experience of pain. For example, creating new algorithms for online monitor systems that assess pain through objective measures. Also, it could be used to train supervised and unsupervised classifiers with different psychological factors that are assessed through the questionnaires.


## 1\. Objective

Chronic neuropathic pain (NP) is a challenge for clinicians and patients. It is present in about 7-10% of the general population in adults [\[1\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0001) and up to 6% of infants [\[2\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0002). Chronic NP is a direct consequence of a lesion or disease affecting the nervous system that is persistent or recurrent for more than 3 months [\[3\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0003). Maladaptive changes in the nervous system allow for unbearable pain described as stabbing, throbbing, tingling,or electrical [\[4\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0004). Despite increased pharmacological trials for chronic NP, efficacy has failed due to a lack of proper characterization and stratification. The electroencephalogram (EEG) offers advantages as a non-invasive assessment tool with a lower cost and simpler methodology than other imaging techniques [\[5\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0005). However, much research is still needed to develop methods for reliable pain characterization using EEG. Thus, this dataset which contains physiological (i.e., brain activity) and psychological (i.e., questionnaire reports), complements the collective approach of the pain field to improve the assessment and management of chronic NP. This data article adds value to the original research article by allowing researchers to explore the data with other linear and nonlinear methods as proposed in [\[6\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0006).

## 2\. Data Description

This section describes the data in the Mendeley Data repository.

### 2.1. Electrophysiological recordings

The raw EEG data is provided for every patient ID in .gdf format. A General Data Format (GDF) file can be read and written by BioSig, an open-source software library for biomedical signal processing implemented in Octave/Matlab, C/C++, Python, Java and R [\[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0007), [8\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0008). The open-source software for opening GDF files can be downloaded in [\[7\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0007), and detailed information on GDF file type can be found in [\[8\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0008). Electrophysiological signals correspond to the spontaneous activity of 36 chronic NP patients at resting state for 10 min in two conditions: eyes open (first 5 min) and eyes closed (last 5 min). The datasets contain an extra five seconds at the beginning of each recording, it is recommended to use the signals starting from the second five. The causes for NP in the studied sample and the pharmacological treatment of the studied population are described in [Fig. 1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#fig0001). Other details concerning their clinical history and area of pain can be revised in the Excel file.

#### Fig. 1.

[![Fig 1](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/aa71/10050461/980d6ca8c98f/gr1.jpg)](https://www.ncbi.nlm.nih.gov/core/lw/2.0/html/tileshop_pmc/tileshop_pmc_inline.html?title=Click%20on%20image%20to%20zoom&p=PMC3&id=10050461_gr1.jpg)

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/figure/fig0001/)

The etiologies of the NP sample were spinal cord injury (31%), painful peripheral neuropathy (23%), diabetes (17%), trigeminal neuralgia (8%), CNS disorder (8%), and other (14%). The type and frequency of the pharmacological treatment were: eighteen patients (n=18) taking centrally acting drugs for over a year, twelve patients (n=12) were not taking medication, three patients (n=3) were on cannabidiol derivatives (CBD), and three (n=3) took nonsteroidal anti-inflammatory (NSAID) drugs for pain attacks.

### 2.2. Excel file

The excel file contains three sheets. The first contains the demographics of each patient: age, gender, etiology, and clinical history. The second and third sheets contain the answers to the pain questionnaires, Brief Pain Inventory (BPI) and painDETECT questionnaire (PDQ), which were used in their validated Spanish version [\[9](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0009), [10\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0010). The answers have been translated according to their English version [\[11](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0011), [12\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0012) and can be found in the data repository. An overview of the questionnaire results is described in the Supplementary Material of [\[6\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0006).

#### 2.2.1. Brief Pain Inventory answers

The second sheet contains the answers to the BPI. The _functionality_ category of the BPI is divided into seven items: general activity, mood, ability to walk, normal work, relationships with other people, rest, and enjoyment of life. Each item is measured on an 11-point scale from 0 to 10, where 0 is no pain or no interference, and 10 is worst pain or complete interference.

#### 2.2.2. painDETECT answers

The third sheet contains the answers to the nine items of the PDQ. Two questions had to be acquired differently for dataset purposes since they required for patients to _draw_ and _mark_. When the questionnaire asks patients to “Please _mark_ your main area of pain”, we asked them instead to _write_ the main area of pain as specifically as possible. Also, for the question “If yes, please _draw_ the direction in which the pain radiates”, we asked them to _write_ the direction in which it irradiates and to be as explicit as possible (e.g., from the right arm to the head, from the left leg towards the back, etc.). For the PDQ, a total score between -1 and 38 can be calculated from the nine items, with a greater probability of having NP with higher scores. A score less than or equal to 12 indicates pain with a modest probability of being NP, greater than or equal to 19 means that there is a 90% probability that NP exists. The final score of each patient can be calculated with the uploaded questionnaire in the repository.

### 2.3. Text files

Two .txt files of channel locations (mBrain 24 ch locations and rereferenced 22 ch, without M1 and M2) are available. The rereferenced 22 ch text file can be used when preprocessing the data after referencing to M1 and M2.

## 3\. Experimental Design, Materials and Methods

### 3.1. Sample size calculation

The sample size was calculated in line with [Eq. (1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#eqn0001), with a two-tailed Z distribution to evaluate the similarity between a control group (null hypothesis) and the NP group (alternate hypothesis). Calculations were performed in MATLAB R2020a (The Mathworks, Inc., Natick, MA, USA) with the Statistics and Machine Learning Toolbox [\[13\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0013).

|     |     |
| --- | --- |
| Power=ϕ\[z∝+(μ0−μ1)\*n\]σ\] | (1) |

Note that for [Eq. (1)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#eqn0001), the power depends on four factors: ∝, \|μ0+μ1\|, σ, and _n._ Thus, to compute the power with this formula, an expected value from the control group (μ0withitsvarianceσ) and from the NP group (μ1) is necessary. To solve for _n,_ see [Eq. (2)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#eqn0002), where 1-ß is the power and ∝, is the significance level:

|     |     |
| --- | --- |
| n=(z1−∝/2+z1−ß)2σ2(μ0−μ1)2 | (2) |

The parameter used to estimate significant differences between the groups was the individual alpha frequency (IAF) value, an EEG biomarker taken from the analysis of the alpha band, which is modulated by lifestyle, diet, exercise, sex, age, and other factors. Moreover, it can be affected by neuropathologies such as tinnitus and NP due to the mechanism of thalamocortical dysrhythmia. It has been reported that the IAF of tinnitus and NP patientsis equivalent, given the same thalamocortical dysrhythmia as the background neuronal activity [\[14\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0014). Thus, to approximate the unknown value of the IAF of NP patients, EEG data from tinnitus patients and the control group was requested from the authors of a previous study [\[15\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0015). For the control group, a mean of 10 Hz and a standard deviation of 1 Hz were obtained from 18 healthy subjects (11 men and 7 women, with a mean age of 38+-10.2). These results support those found in another group of healthy patients from ages 20-80 years reported in [\[16\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0016) and in [\[17\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0017). On this basis, μ0 (the mean of the null hypothesis) was set to 10 Hz, and the σ to 1 Hz. For patients with tinnitus, the resulting IAF was calculated from 65 patients with tinnitus (40 women and 25 men; mean age 54.92+-11.79) and resulted in 9.5 Hz (μ1, mean of the alternate hypothesis), which is also supported by the results of another NP study [\[18\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0018). Considering a power between 0.8 and 0.9, the following parameters were applied in [Eq. (2)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#eqn0002), μ0=10,μ1=9.5,∝=0.05,andσ=1. According to [Eq. (2)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#eqn0002), for a power of 0.8, ß=0.2, and ∝= 0.05, n=32, meaning that 32 NP patients are needed to reach significant differences between NP and a control group. For a power of 0.9, ß=0.1 and ∝= 0.05, n=43. Thus, the objective was to recruit 32 to 43 patients with chronic NP.

### 3.2. Questionnaire election

As stated previously, two validated questionnaires in Spanish \[ [\[9\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0009), [\[10\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0010)\] were used to characterize NP [\[19\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0019). The reasons for selecting these questionnaires were the following:

- 1.
The PDQ [\[12\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0012) is an NP diagnostic tool and was used as an inclusion criterion. Additionally, this questionnaire specifies four different patterns of pain fluctuation (see attached questionnaire). These patterns can be important information for subsequent analysis of EEG signals. In addition, pain qualities are used with various adjectives that can help the patient respond according to his or her symptoms (e.g., burning, tingling, sudden pains, electric shocks, numbness).

- 2.
The BPI isan NP assessment tool [\[11\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0011). This second questionnaire was chosen because the patient reports the impact that NP causes in his or her daily life. For example, the patient responds the degree of pain relief relative to the individual treatment and the effect of pain on work, mood, ability to walk, or relationships.


### 3.3. Justification for allowing patients to have medication

For ethical reasons, patients were allowed to continue their medication if they had been receiving it at least 4 weeks before the study, which is supported by a prior EEG study where patients received co-analgesic treatment (anticonvulsants and antidepressants) [\[20\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0020). Moreover, the sub-chronic dose (25 mg for more than 15 days) of amitriptyline (an antidepressant given for NP) did not affect the P3 component (EEG evoked potential) in patients with NP [\[21\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0021). Finally, no significant differences were found between NP patients with or without central drugs (opiates and antidepressants) in EEG studies at resting state andat a bandwidth of 2-18 Hz [\[22\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0022).

### 3.4. EEG recording equipment

Ten minutes of spontaneous EEG data were recorded using 22 electrodes (Fp1, Fp2, AFz, F7, F3, Fz, F4, F8, T7, C3, Cz, C4, T8, CPz, P7, P3, Pz, P4, P8, POz, O1, O2) positioned according to the International 10/20 System with two earlobe references (M1 and M2). The channel positions for the data are shown in [Fig. 2](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#fig0002), which has also been uploaded to the dataset files. [Fig. 3A](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#fig0003) and [B](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#fig0003) shows the Smarting mBrain cap and amplifier used. The input impedance of the system is 1 GΩ. The input referred noise is less than 1 µV. It has a resolution of 24 bits. The sampling frequency used was 250Hz. The bandwidth is 0-250 Hz, with a flat frequency response of 0-133 Hz. The communication type is wireless Bluetooth v2.1 (See [Fig. 3B](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#fig0003)). OpenViBe software \[ [24](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib24)\] was used to implement the experimental paradigms and record the EEG signals. The sampling frequency was 250 Hz, and the bandwidth was between 0.1 and 100 Hz. Electrode impedances were kept below 5 kΩ.

#### Fig. 2.

[![Fig 2](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/aa71/10050461/18581aa7303e/gr2.jpg)](https://www.ncbi.nlm.nih.gov/core/lw/2.0/html/tileshop_pmc/tileshop_pmc_inline.html?title=Click%20on%20image%20to%20zoom&p=PMC3&id=10050461_gr2.jpg)

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/figure/fig0002/)

Topographic map for the 24 channels positioned according to the 10/20 system for the Smarting mBrain cap. M1 and M;2 correspond to the earlobe (i.e., mastoid) electrodes for offline reference.

#### Fig. 3.

[![Fig 3](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/aa71/10050461/70fddf7f8c8f/gr3.jpg)](https://www.ncbi.nlm.nih.gov/core/lw/2.0/html/tileshop_pmc/tileshop_pmc_inline.html?title=Click%20on%20image%20to%20zoom&p=PMC3&id=10050461_gr3.jpg)

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/figure/fig0003/)

Laboratory Resources of the Neuroengineering and Neuroacoustics Research Group. (A) MBrainTrain cap with 24 Ag/AgCl electrodes. (B) Smarting device used for communication between electrodes and recording device by means of Bluethooth.

### 3.5. Experimental procedure

The experimental procedure was as follows [\[6](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0006), [23\]](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#bib0023). First, the recording procedure was described to each patient. If the patient agreed to participate, they signed the informed consent. Afterward, patients answered the BPI questionnaire, and the EEG was installed. For the EEG recording, patients sat in an upright position. For the first five minutes, they were asked to keep their eyes opened and fixed on a white cross in a dark background of a monitor 50 cm away. At the end of the first 5 minutes, the cross disappeared, and patients closed their eyes for the last 5 minutes until an auditory beep marked the end of the recording. A simplified diagram of the experimental procedure is depicted in [Fig. 4](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/#fig0004).

#### Fig. 4.

[![Fig 4](https://cdn.ncbi.nlm.nih.gov/pmc/blobs/aa71/10050461/16efd4cb8096/gr4.jpg)](https://www.ncbi.nlm.nih.gov/core/lw/2.0/html/tileshop_pmc/tileshop_pmc_inline.html?title=Click%20on%20image%20to%20zoom&p=PMC3&id=10050461_gr4.jpg)

[Open in a new tab](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/figure/fig0004/)

Experimental Procedure for the Recording Session. First, the procedure was explained to the patient thoroughly. Second, the informed consent was signed. Third, the BPI questionnaire was answered. Fourth, the EEG equipment was installed. Finally, the spontaneous EEG recording paradigm started and the session ended in approximately 55 min.

## Ethics Statements

Prior to the start of the experiment, all patients provided written informed consent according to the World Association Declaration of Helsinki. This study was approved by the Clinical Investigation Ethics Committee of Tecnológico de Monterrey (number: P000369-DN-RespElectro-CI-CR005).

## CRediT Author Statement

**Daniela M. Zolezzi:** Literature Research, Data Recording, Writing – original draft; **Norberto E. Naal-Ruiz:** Data Recording, Supervision; **Luz María Alonso-Valerdi:** Supervision, Conceptualization – review & editing; **David I. Ibarra-Zarate:** Data Supervision, Writing – review & editing.

## Declaration of Competing Interest

The authors declare that they have no known competing financial interests or personal relationships which have or could be perceived to have influenced the work reported in this article.

## Acknowledgments

All equipment provided for data collection belongs to the Neuroengineering and Neuroacoustics Research group from Tecnológico de Monterrey. Funding for this project was provided to DMZ by Consejo Nacional de Ciencia y Tecnología (CONACYT) (Grant number: 1007725) and Tecnológico de Monterrey by granting a full tuition scholarship to DMZ. Center for Neuroplasticity and Pain (CNAP) is supported by the Danish National Research Foundation (DNRF121).

## Data Availability

- [Chronic Neuropathic Pain: EEG data in eyes open and eyes closed with questionnaire reports (Original data)](https://data.mendeley.com/datasets/yj52xrfgtz/1) (Mendeley Data).

- [Chronic Neuropathic Pain: EEG data in eyes open (5 min) and eyes closed (5 min) with questionnaire reports (Original data)](https://data.mendeley.com/datasets/yj52xrfgtz/4) (Mendeley Data).


## References

- 1.van Hecke O., Austin S.K., Khan R.A., Smith B.H., Torrance N. Neuropathic pain in the general population: a systematic review of epidemiological studies. Pain. 2014 doi: 10.1016/j.pain.2013.11.013. \[ [DOI](https://doi.org/10.1016/j.pain.2013.11.013)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/24291734/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Pain&title=Neuropathic%20pain%20in%20the%20general%20population:%20a%20systematic%20review%20of%20epidemiological%20studies&author=O.%20van%20Hecke&author=S.K.%20Austin&author=R.A.%20Khan&author=B.H.%20Smith&author=N.%20Torrance&publication_year=2014&pmid=24291734&doi=10.1016/j.pain.2013.11.013&)\]
- 2.Bhatia A., Brennan L., Abrahams M., Gilder F. Chronic pain in children in the UK: a survey of pain clinicians and general practitioners. Paediatr. Anaesth. 2008 doi: 10.1111/j.1460-9592.2008.02710.x. \[ [DOI](https://doi.org/10.1111/j.1460-9592.2008.02710.x)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/18673319/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Paediatr.%20Anaesth.&title=Chronic%20pain%20in%20children%20in%20the%20UK:%20a%20survey%20of%20pain%20clinicians%20and%20general%20practitioners&author=A.%20Bhatia&author=L.%20Brennan&author=M.%20Abrahams&author=F.%20Gilder&publication_year=2008&pmid=18673319&doi=10.1111/j.1460-9592.2008.02710.x&)\]
- 3.Scholz J., Finnerup N.B., Attal N., Aziz Q., Baron R., Bennett M.I., Benoliel R., Cohen M., Cruccu G., Davis K.D., Evers S., First M., Giamberardino M.A., Hansson P., Kaasa S., Korwisi B., Kosek E., Lavand'homme P., Nicholas M., Nurmikko T., Perrot S., Raja S.N., Rice A.S.C., Rowbotham M.C., Schug S., Simpson D.M., Smith B.H., Svensson P., Vlaeyen J.W.S., Wang S.-J., Barke A., Rief W., Treede R.-D. The IASP classification of chronic pain for ICD-11: chronic neuropathic pain. Pain. 2019;160:53–59. doi: 10.1097/j.pain.0000000000001365. \[ [DOI](https://doi.org/10.1097/j.pain.0000000000001365)\] \[ [PMC free article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6310153/)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/30586071/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Pain&title=The%20IASP%20classification%20of%20chronic%20pain%20for%20ICD-11:%20chronic%20neuropathic%20pain&author=J.%20Scholz&author=N.B.%20Finnerup&author=N.%20Attal&author=Q.%20Aziz&author=R.%20Baron&volume=160&publication_year=2019&pages=53-59&pmid=30586071&doi=10.1097/j.pain.0000000000001365&)\]
- 4.Shaygan M., Böger A., Kröner-Herwig B. Clinical features of chronic pain with neuropathic characteristics: a symptom-based assessment using the Pain DETECT questionnaire. Eur. J. Pain. 2013 doi: 10.1002/j.1532-2149.2013.00322.x. n/a-n/a. \[ [DOI](https://doi.org/10.1002/j.1532-2149.2013.00322.x)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/23649845/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Eur.%20J.%20Pain&title=Clinical%20features%20of%20chronic%20pain%20with%20neuropathic%20characteristics:%20a%20symptom-based%20assessment%20using%20the%20Pain%20DETECT%20questionnaire&author=M.%20Shaygan&author=A.%20B%C3%B6ger&author=B.%20Kr%C3%B6ner-Herwig&publication_year=2013&pmid=23649845&doi=10.1002/j.1532-2149.2013.00322.x&)\]
- 5.P.T.H.V.D, Zebhauser P.M. Resting-state electroencephalography and magnetoencephalography as biomarkers of chronic pain: a systematic review. Pain. 2022 doi: 10.1097/j.pain.0000000000002825. \[ [DOI](https://doi.org/10.1097/j.pain.0000000000002825)\] \[ [PMC free article](https://pmc.ncbi.nlm.nih.gov/articles/PMC10184564/)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/36409624/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Pain&title=Resting-state%20electroencephalography%20and%20magnetoencephalography%20as%20biomarkers%20of%20chronic%20pain:%20a%20systematic%20review&author=%20P.T.H.V.D&author=P.M.%20Zebhauser&publication_year=2022&pmid=36409624&doi=10.1097/j.pain.0000000000002825&)\]
- 6.Zolezzi D.M., Alonso-Valerdi L.M., Ibarra-Zarate D.I. EEG frequency band analysis in chronic neuropathic pain: a linear and nonlinear approach to classify pain severity. Comput. Methods Programs Biomed. 2023;230 doi: 10.1016/j.cmpb.2023.107349. \[ [DOI](https://doi.org/10.1016/j.cmpb.2023.107349)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/36689806/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Comput.%20Methods%20Programs%20Biomed.&title=EEG%20frequency%20band%20analysis%20in%20chronic%20neuropathic%20pain:%20a%20linear%20and%20nonlinear%20approach%20to%20classify%20pain%20severity&author=D.M.%20Zolezzi&author=L.M.%20Alonso-Valerdi&author=D.I.%20Ibarra-Zarate&volume=230&publication_year=2023&pmid=36689806&doi=10.1016/j.cmpb.2023.107349&)\]
- 7.Alois Schlögl, Biosig - a free and open source software library for biomedical signal processing, [https://Biosig.Sourceforge.Net/Download.Html](https://biosig.sourceforge.net/Download.Html). (2003). \[ [DOI](https://doi.org/10.1155/2011/935364)\] \[ [PMC free article](https://pmc.ncbi.nlm.nih.gov/articles/PMC3061298/)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/21437227/)\]
- 8.Schlögl A. GDF - a general dataformat for biosignals V2.51. 2006.  \[ [DOI](https://doi.org/10.48550/arXiv.cs/0608052)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?title=GDF%20-%20a%20general%20dataformat%20for%20biosignals%20V2.51&author=A.%20Schl%C3%B6gl&publication_year=2006&)\]
- 9.de Andrés J., Pérez-Cajaraville J., Lopez-Alarcón M.D., López-Millán J.M., Margarit C., Rodrigo-Royo M.D., Franco-Gay M.L., Abejón D., Ruiz M.A., López-Gomez V., Pérez M. Cultural adaptation and validation of the painDETECT scale into Spanish. Clin. J. Pain. 2012;28:243–253. doi: 10.1097/AJP.0b013e31822bb35b. \[ [DOI](https://doi.org/10.1097/AJP.0b013e31822bb35b)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/21926908/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Clin.%20J.%20Pain&title=Cultural%20adaptation%20and%20validation%20of%20the%20painDETECT%20scale%20into%20Spanish&author=J.%20de%20Andr%C3%A9s&author=J.%20P%C3%A9rez-Cajaraville&author=M.D.%20Lopez-Alarc%C3%B3n&author=J.M.%20L%C3%B3pez-Mill%C3%A1n&author=C.%20Margarit&volume=28&publication_year=2012&pages=243-253&pmid=21926908&doi=10.1097/AJP.0b013e31822bb35b&)\]
- 10.Badia X., Muriel C., Gracia A., Manuel Núñez-Olarte J., Perulero N., Gálvez R., Carulla J., Cleeland C.S. Validación española del cuestionario Brief Pain Inventory en pacientes con dolor de causa neoplásica. Med. Clin. (Barc) 2003;120:52–59. doi: 10.1016/S0025-7753(03)73601-X. \[ [DOI](https://doi.org/10.1016/S0025-7753(03)73601-X)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/12570914/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Med.%20Clin.%20(Barc)&title=Validaci%C3%B3n%20espa%C3%B1ola%20del%20cuestionario%20Brief%20Pain%20Inventory%20en%20pacientes%20con%20dolor%20de%20causa%20neopl%C3%A1sica&author=X.%20Badia&author=C.%20Muriel&author=A.%20Gracia&author=J.%20Manuel%20N%C3%BA%C3%B1ez-Olarte&author=N.%20Perulero&volume=120&publication_year=2003&pages=52-59&pmid=12570914&doi=10.1016/S0025-7753(03)73601-X&)\]
- 11.Erdemoglu Â.K., Koc R. Brief pain inventory score identifying and discriminating neuropathic and nociceptive pain. Acta Neurol. Scand. 2013 doi: 10.1111/ane.12131. n/a-n/a. \[ [DOI](https://doi.org/10.1111/ane.12131)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/23594114/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Acta%20Neurol.%20Scand.&title=Brief%20pain%20inventory%20score%20identifying%20and%20discriminating%20neuropathic%20and%20nociceptive%20pain&author=%C3%82.K.%20Erdemoglu&author=R.%20Koc&publication_year=2013&pmid=23594114&doi=10.1111/ane.12131&)\]
- 12.Freynhagen R., Baron R., Gockel U., Tölle T.R. painDETECT : a new screening questionnaire to identify neuropathic components in patients with back pain. Curr. Med. Res. Opin. 2006;22:1911–1920. doi: 10.1185/030079906x132488. \[ [DOI](https://doi.org/10.1185/030079906x132488)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/17022849/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Curr.%20Med.%20Res.%20Opin.&title=painDETECT%C2%A0:%20a%20new%20screening%20questionnaire%20to%20identify%20neuropathic%20components%20in%20patients%20with%20back%20pain&author=R.%20Freynhagen&author=R.%20Baron&author=U.%20Gockel&author=T.R.%20T%C3%B6lle&volume=22&publication_year=2006&pages=1911-1920&pmid=17022849&doi=10.1185/030079906x132488&)\]
- 13.Inc. The MathWorks, Statistics and machine learning toolbox, (2022).
- 14.Vanneste S., To W.T., de Ridder D. Tinnitus and neuropathic pain share a common neural substrate in the form of specific brain connectivity and microstate profiles. Prog. Neuropsychopharmacol. Biol. Psychiatry. 2019;88:388–400. doi: 10.1016/j.pnpbp.2018.08.015. \[ [DOI](https://doi.org/10.1016/j.pnpbp.2018.08.015)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/30142355/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Prog.%20Neuropsychopharmacol.%20Biol.%20Psychiatry&title=Tinnitus%20and%20neuropathic%20pain%20share%20a%20common%20neural%20substrate%20in%20the%20form%20of%20specific%20brain%20connectivity%20and%20microstate%20profiles&author=S.%20Vanneste&author=W.T.%20To&author=D.%20de%20Ridder&volume=88&publication_year=2019&pages=388-400&pmid=30142355&doi=10.1016/j.pnpbp.2018.08.015&)\]
- 15.Alonso-Valerdi L.M., González-Salazar J.I., Ibarra-Zarate D.I. Neuropsychological monitoring of current acoustic therapies as alternative treatment of chronic tinnitus. Am. J. Otolaryngol. 2021;42 doi: 10.1016/j.amjoto.2021.103109. \[ [DOI](https://doi.org/10.1016/j.amjoto.2021.103109)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/34175772/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Am.%20J.%20Otolaryngol.&title=Neuropsychological%20monitoring%20of%20current%20acoustic%20therapies%20as%20alternative%20treatment%20of%20chronic%20tinnitus&author=L.M.%20Alonso-Valerdi&author=J.I.%20Gonz%C3%A1lez-Salazar&author=D.I.%20Ibarra-Zarate&volume=42&publication_year=2021&pmid=34175772&doi=10.1016/j.amjoto.2021.103109&)\]
- 16.Klimesch W. The frequency architecture of brain and brain body oscillations: an analysis. Eur. J. Neurosci. 2018 doi: 10.1111/ejn.14192. \[ [DOI](https://doi.org/10.1111/ejn.14192)\] \[ [PMC free article](https://pmc.ncbi.nlm.nih.gov/articles/PMC6668003/)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/30281858/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Eur.%20J.%20Neurosci.&title=The%20frequency%20architecture%20of%20brain%20and%20brain%20body%20oscillations:%20an%20analysis&author=W.%20Klimesch&publication_year=2018&pmid=30281858&doi=10.1111/ejn.14192&)\]
- 17.Wieneke G.H., Deinema C.H.A., Spoelstra P., Van Leeuwen W.S., Versteeg H. Normative spectral data on alpha rhythm in male adults. Electroencephalogr. Clin. Neurophysiol. 1980 doi: 10.1016/0013-4694(80)90404-6. \[ [DOI](https://doi.org/10.1016/0013-4694(80)90404-6)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/6158443/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Electroencephalogr.%20Clin.%20Neurophysiol.&title=Normative%20spectral%20data%20on%20alpha%20rhythm%20in%20male%20adults&author=G.H.%20Wieneke&author=C.H.A.%20Deinema&author=P.%20Spoelstra&author=W.S.%20Van%20Leeuwen&author=H.%20Versteeg&publication_year=1980&pmid=6158443&doi=10.1016/0013-4694(80)90404-6&)\]
- 18.Sarnthein J., Jeanmonod D. High thalamocortical theta coherence in patients with neurogenic pain. Neuroimage. 2008 doi: 10.1016/j.neuroimage.2007.10.019. \[ [DOI](https://doi.org/10.1016/j.neuroimage.2007.10.019)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/18060808/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Neuroimage&title=High%20thalamocortical%20theta%20coherence%20in%20patients%20with%20neurogenic%20pain&author=J.%20Sarnthein&author=D.%20Jeanmonod&publication_year=2008&pmid=18060808&doi=10.1016/j.neuroimage.2007.10.019&)\]
- 19.Zolezzi D.M., Alonso-Valerdi L.M., Ibarra-Zarate D.I. Chronic neuropathic pain is more than a perception: systems and methods for an integral characterization. Neurosci. Biobehav. Rev. 2022;136 doi: 10.1016/j.neubiorev.2022.104599. \[ [DOI](https://doi.org/10.1016/j.neubiorev.2022.104599)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/35271915/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Neurosci.%20Biobehav.%20Rev.&title=Chronic%20neuropathic%20pain%20is%20more%20than%20a%20perception:%20systems%20and%20methods%20for%20an%20integral%20characterization&author=D.M.%20Zolezzi&author=L.M.%20Alonso-Valerdi&author=D.I.%20Ibarra-Zarate&volume=136&publication_year=2022&pmid=35271915&doi=10.1016/j.neubiorev.2022.104599&)\]
- 20.Backonja M.M., Serra J. Pharmacologic management part 1: better-studied neuropathic pain diseases. Pain Med. 2004 doi: 10.1111/j.1526-4637.2004.04020.x. \[ [DOI](https://doi.org/10.1111/j.1526-4637.2004.04020.x)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/14996229/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Pain%20Med.&title=Pharmacologic%20management%20part%201:%20better-studied%20neuropathic%20pain%20diseases&author=M.M.%20Backonja&author=J.%20Serra&publication_year=2004&pmid=14996229&doi=10.1111/j.1526-4637.2004.04020.x&)\]
- 21.Veldhuijzen D.S., Kenemans J.L., Van Wijck A.J.M., Olivier B., Kalkman C.J., Volkerts E.R. Acute and subchronic effects of amitriptyline on processing capacity in neuropathic pain patients using visual event-related potentials: Preliminary findings. Psychopharmacology (Berl.) 2006 doi: 10.1007/s00213-005-0204-3. \[ [DOI](https://doi.org/10.1007/s00213-005-0204-3)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/16292592/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Psychopharmacology%20(Berl.)&title=Acute%20and%20subchronic%20effects%20of%20amitriptyline%20on%20processing%20capacity%20in%20neuropathic%20pain%20patients%20using%20visual%20event-related%20potentials:%20Preliminary%20findings&author=D.S.%20Veldhuijzen&author=J.L.%20Kenemans&author=A.J.M.%20Van%20Wijck&author=B.%20Olivier&author=C.J.%20Kalkman&publication_year=2006&pmid=16292592&doi=10.1007/s00213-005-0204-3&)\]
- 22.Sarnthein J., Stern J., Aufenberg C., Rousson V., Jeanmonod D. Increased EEG power and slowed dominant frequency in patients with neurogenic pain. Brain. 2006 doi: 10.1093/brain/awh631. \[ [DOI](https://doi.org/10.1093/brain/awh631)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/16183660/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?journal=Brain&title=Increased%20EEG%20power%20and%20slowed%20dominant%20frequency%20in%20patients%20with%20neurogenic%20pain&author=J.%20Sarnthein&author=J.%20Stern&author=C.%20Aufenberg&author=V.%20Rousson&author=D.%20Jeanmonod&publication_year=2006&pmid=16183660&doi=10.1093/brain/awh631&)\]
- 23.Zolezzi D.M., Alonso-Valerdi L.M., Ibarra-Zarate D.I., Naal-Ruiz N.E., Zolezzi D.M., Alonso-Valerdi L.M., Naal-Ruiz N.E., Ibarra-Zarate D.I. 2021 43rd Annual International Conference of the IEEE Engineering in Medicine & Biology Society (EMBC), IEEE. 2021. Identification of neuropathic pain severity based on linear and non-linear EEG features; pp. 169–173. \[ [DOI](https://doi.org/10.1109/EMBC46164.2021.9630101)\] \[ [PubMed](https://pubmed.ncbi.nlm.nih.gov/34891264/)\] \[ [Google Scholar](https://scholar.google.com/scholar_lookup?title=2021%2043rd%20Annual%20International%20Conference%20of%20the%20IEEE%20Engineering%20in%20Medicine%20&%20Biology%20Society%20(EMBC),%20IEEE&author=D.M.%20Zolezzi&author=L.M.%20Alonso-Valerdi&author=D.I.%20Ibarra-Zarate&author=N.E.%20Naal-Ruiz&author=D.M.%20Zolezzi&publication_year=2021&)\]
- 24.Y. Renard, F. Lotte, G. Gibert, M. Congedo, E. Maby, V. Delannoy, O. Bertrand, A. Lécuyer, “OpenViBE: An Open-SourceSoftware Platform to Design, Test and Use Brain-Computer Interfaces in Real and Virtual Environments”, Presence :teleoperators and virtual environments, vol. 19, no 1, 2010

## Associated Data

_This section collects any data citations, data availability statements, or supplementary materials included in this article._

### Data Availability Statement

- [Chronic Neuropathic Pain: EEG data in eyes open and eyes closed with questionnaire reports (Original data)](https://data.mendeley.com/datasets/yj52xrfgtz/1) (Mendeley Data).

- [Chronic Neuropathic Pain: EEG data in eyes open (5 min) and eyes closed (5 min) with questionnaire reports (Original data)](https://data.mendeley.com/datasets/yj52xrfgtz/4) (Mendeley Data).


![Close](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/usa-icons/close.svg)

## ACTIONS

- [![View on publisher site icon](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/usa-icons/launch.svg)View on publisher site](https://doi.org/10.1016/j.dib.2023.109060)
- [![Download PDF icon](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/usa-icons/file_download.svg)PDF (961.0 KB)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10050461/pdf/main.pdf)
- ![Cite icon](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/usa-icons/format_quote.svg)Cite
- ![Collections icon](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/action-bookmark-full.svg)![Collections icon](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/action-bookmark-empty.svg)Collections
- ![Permalink icon](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/usa-icons/share.svg)Permalink




## PERMALINK



![Copy icon](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/usa-icons/content_copy.svg)Copy


## RESOURCES

### Similar articles

### Cited by other articles

### Links to NCBI Databases

Back to Top![back to top icon](https://cdn.ncbi.nlm.nih.gov/pmc/pd-medc-pmc-cloudpmc-viewer/production/9b1d9d87/var/data/static/img/usa-icons/arrow_upward.svg

---

Repository record: https://data.mendeley.com/datasets/yj52xrfgtz/4 (Mendeley Data). Its metadata identifies raw electrophysiological signals, electrode-position text, and questionnaire/demographic spreadsheets; it is the publicly linked dataset analyzed by Ryu et al. https://www.nature.com/articles/s41598-024-71219-3
