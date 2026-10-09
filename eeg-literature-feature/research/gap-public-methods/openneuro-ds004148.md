Fetched from https://openneuro.org/datasets/ds004148/versions/1.0.1

# [EEG](https://openneuro.org/search/modality/eeg) A test-retest resting and cognitive state EEG dataset

[Files](https://openneuro.org/datasets/ds004148/versions/1.0.1)[Download](https://openneuro.org/datasets/ds004148/versions/1.0.1/download)[Metadata](https://openneuro.org/datasets/ds004148/versions/1.0.1/metadata)

Follow18

Bookmark22

### BIDS Validation

1 Warning Valid

Analyze

Clone

## README

# General information

This dataset contains resting(eyes closed, eyes open) and cognitive(subtraction, music, memory) state EEG recordings with 60 participants
during three experimental sessions together with sleep, emotion, mental health, and mind-wandering related measures

# Dataset

## Presentation

```
 The data collection was initiated in September 2019 and was terminated in April 2021. The detailed description of the dataset is currently under working
 by Yulin Wang, and will submit to Scientific Data for publication.

```

#### EEG acquisition

```
 * EEG system (Brain Products GmbH, Steing- rabenstr, Germany, 64 electrodes)
 * Sampling frequency: 500Hz
 * Impedances were kept below 5k
```

## Contact

```
 * If you have any questions or comments, please contact:
 * Xu Lei: xlei@swu.edu.cn
 * Yulin Wang: yulin.wang90.swu@gmail.com

```

Read MoreCollapse

Files: 5585 Size: 30.67GB

A test-retest resting and cognitive state EEG dataset

-   CHANGES
-   README
-   dataset\_description.json
-   .gitattributes
-   participants.json
-   participants.tsv
- derivatives

- sub-01

- sub-02

- sub-03

- sub-04

- sub-05

- sub-06

- sub-07

- sub-08

- sub-09

- sub-10

- sub-11

- sub-12

- sub-13

- sub-14

- sub-15

- sub-16

- sub-17

- sub-18

- sub-19

- sub-20

- sub-21

- sub-22

- sub-23

- sub-24

- sub-25

- sub-26

- sub-27

- sub-28

- sub-29

- sub-30

- sub-31

- sub-32

- sub-33

- sub-34

- sub-35

- sub-36

- sub-37

- sub-38

- sub-39

- sub-40

- sub-41

- sub-42

- sub-43

- sub-44

- sub-45

- sub-46

- sub-47

- sub-48

- sub-49

- sub-50

- sub-51

- sub-52

- sub-53

- sub-54

- sub-55

- sub-56

- sub-57

- sub-58

- sub-59

- sub-60


## Comments

Please sign in to contribute to the discussion.

By Yulin Wang - about 4 years ago

The detailed information about this dataset was published in Scientific Data, check it out here: https://www.nature.com/articles/s41597-022-01607-9

By Eric Laycock - about 3 years ago

What preprocessing was applied to this data? In your paper you mention ICA artifact rejection was done in the eyes open condition, but in Reproducibility of power spectrum, functional connectivity and network construction in resting-state EEG (Duan et al. 2021) it appears that FIR filtering, bad channel interpolation, bad epoch rejection and ICA artifact rejection were all done. Moreover, the PSD plot shows a notch filter applied in 49-51 Hz. Would you be able to speak to the prepocessing done on this data?

By Eric Laycock - about 3 years ago

By [MAROWA KAMAL ALDEEN](https://openneuro.org/user/0000-0002-9951-0427) [![ORCID logo](data:image/svg+xml,%3c?xml%20version=%271.0%27%20encoding=%27utf-8%27?%3e%3c!--%20Generator:%20Adobe%20Illustrator%2019.1.0,%20SVG%20Export%20Plug-In%20.%20SVG%20Version:%206.00%20Build%200)%20--%3e%3csvg%20version=%271.1%27%20id=%27Layer_1%27%20xmlns=%27http://www.w3.org/2000/svg%27%20xmlns:xlink=%27http://www.w3.org/1999/xlink%27%20x=%270px%27%20y=%270px%27%20viewBox=%270%200%20256%20256%27%20style=%27enable-background:new%200%200%20256%20256;%27%20xml:space=%27preserve%27%3e%3cstyle%20type=%27text/css%27%3e%20.st0{fill:%23A6CE39;}%20.st1{fill:%23FFFFFF;}%20%3c/style%3e%3cpath%20class=%27st0%27%20d=%27M256,128c0,70.7-57.3,128-128,128C57.3,256,0,198.7,0,128C0,57.3,57.3,0,128,0C198.7,0,256,57.3,256,128z%27/%3e%3cg%3e%3cpath%20class=%27st1%27%20d=%27M86.3,186.2H70.9V79.1h15.4v48.4V186.2z%27/%3e%3cpath%20class=%27st1%27%20d=%27M108.9,79.1h41.6c39.6,0,57,28.3,57,53.6c0,27.5-21.5,53.6-56.8,53.6h-41.8V79.1z%20M124.3,172.4h24.5%20c34.9,0,42.9-26.5,42.9-39.7c0-21.5-13.7-39.7-43.7-39.7h-23.7V172.4z%27/%3e%3cpath%20class=%27st1%27%20d=%27M88.7,56.8c0,5.5-4.5,10.1-10.1,10.1c-5.6,0-10.1-4.6-10.1-10.1c0-5.6,4.5-10.1,10.1-10.1%20C84.2,46.7,88.7,51.3,88.7,56.8z%27/%3e%3c/g%3e%3c/svg%3e)](https://orcid.org/0000-0002-9951-0427) \- 10 months ago

how to use the annex to be able to use them in my source localization, where when I download subjects is not working

By clicking "I Agree", I affirm that I have the appropriate institutional permissions to receive de-identified data for secondary data analysis, and that neither I nor my collaborators will attempt to reidentify individuals whose data are contained in downloads from OpenNeuro. Further, if for any reason the identity of participants contained in downloads from OpenNeuro become known to me I will make no effort to recontact such participants and will provide immediate notice to OpenNeuro staff.

I Agree