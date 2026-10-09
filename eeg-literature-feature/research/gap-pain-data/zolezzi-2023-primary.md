# Zolezzi et al. 2023 primary article (original analysis)

Primary source: https://www.sciencedirect.com/science/article/pii/S0169260723000160
DOI: 10.1016/j.cmpb.2023.107349. Title: “EEG frequency band analysis in chronic neuropathic pain: A linear and nonlinear approach to classify pain severity.” Authors: Daniela M. Zolezzi, Luz María Alonso-Valerdi, David I. Ibarra-Zarate. Computer Methods and Programs in Biomedicine 230 (2023), 107349.

The article reports 36 chronic neuropathic-pain participants and a separate 13-person healthy-control reference database. Pain groups (low/moderate/high) were defined from Brief Pain Inventory ratings. It extracts absolute power and approximate entropy (ApEn) in five frequency bands at 22 pain-cohort channels. The abstract/results state ApEn differed across severity groups in most bands/conditions (p <= 0.0001) and groups clustered with theta/alpha/beta ApEn. These are reported inferential/group-separation results, not externally validated accuracy or R². Article text says the 36-patient database is publicly available and refers to repository [24].

Raw EEG is described as 10 minutes per participant, eyes open/closed, 250 Hz, 0.1–100 Hz; feature estimates use one-minute segments. This design does not equate to generic five-second prediction windows. The pain EEG includes C3/Cz/C4, but also frontal/temporal/parietal/occipital channels. Dataset format/access is independently checked in mendeley-zolezzi-public-files.md.
