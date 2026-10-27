#!/usr/bin/env python3
"""Reproduce acute-pain EEG published-table validation calculations.

IMPORTANT: this does not process raw EEG. It reproduces descriptive and exploratory
statistics from the 16 rounded performance rows printed in Sun et al. (2020), Table 2.
The rows are repeated train/test configurations from only two patients.
"""
from pathlib import Path
import csv, json, numpy as np
from scipy import stats
HERE=Path(__file__).resolve().parent
with open(HERE/'published_table2_performance_rows.csv', newline='', encoding='utf-8') as f:
    rows=list(csv.DictReader(f))
ssm=np.array([float(x['ssm_unsupervised_auc']) for x in rows])
svm=np.array([float(x['svm_supervised_auc']) for x in rows])
d=ssm-svm
out={
 'analysis_status':'PUBLISHED_AGGREGATE_REANALYSIS_NOT_RAW_EEG_VALIDATION',
 'n_rows':len(rows),
 'ssm_auc':{'mean':float(ssm.mean()),'sd_sample':float(ssm.std(ddof=1)),'median':float(np.median(ssm)),'min':float(ssm.min()),'max':float(ssm.max())},
 'svm_auc':{'mean':float(svm.mean()),'sd_sample':float(svm.std(ddof=1)),'median':float(np.median(svm)),'min':float(svm.min()),'max':float(svm.max())},
 'ssm_minus_svm':{'mean':float(d.mean()),'sd_sample':float(d.std(ddof=1)),'wins':int((d>0).sum()),'losses':int((d<0).sum()),'ties':int((d==0).sum())},
 'paired_t_test':{'t':float(stats.ttest_rel(ssm,svm).statistic),'p':float(stats.ttest_rel(ssm,svm).pvalue),'status':'exploratory only; dependent rows'},
 'wilcoxon_approx':{'W':float(stats.wilcoxon(d,method='approx').statistic),'p':float(stats.wilcoxon(d,method='approx').pvalue),'status':'rounded table values; dependent rows'},
 'exact_sign_test_two_sided':{'p':float(stats.binomtest(int((d>0).sum()),int((d!=0).sum()),.5).pvalue),'status':'exploratory only'},
 'raw_trial_level_eeg_processed':False,
 'pain_intensity_formula_fit':False,
 'valid_population_inference_from_table_rows':False,
}
out_path=HERE/'acute_validation_reproduced_stats.json'
out_path.write_text(json.dumps(out,indent=2),encoding='utf-8')
print(json.dumps(out,indent=2))
