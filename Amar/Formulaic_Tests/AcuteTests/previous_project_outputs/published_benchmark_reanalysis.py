#!/usr/bin/env python3
"""Reproduce a transparent statistical reanalysis of published acute-pain EEG benchmark results.

Input is the 16 rounded performance rows reported in Table 2 of Sun et al. (2020),
not raw EEG signals. This is a reanalysis of published model metrics, not an application
or validation of the project's proposed 0-10 pain-intensity formulae. The rows are repeated
cross-modality/cross-subject configurations from only two chronic-back-pain participants;
therefore tests that treat rows as independent are exploratory and not generalizable.

Usage: python published_benchmark_reanalysis.py [--outdir DIR]
"""
from __future__ import annotations
import argparse, json, platform
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

SOURCE = "https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/"
SOURCE_OSF = "https://osf.io/bsv86/"

# Transcribed from Table 2 rows as shown in the open-access article.
# Values are rounded to two decimal places in the article, and reproduced as printed.
ROWS = [
    ("Hand #1", "Hand #1", .81, .74, .61),
    ("Hand #1", "Back #1", .61, .58, .55),
    ("Hand #1", "Hand #2", .70, .50, .50),
    ("Hand #1", "Back #2", .73, .71, .73),
    ("Back #1", "Hand #1", .80, .58, .70),
    ("Back #1", "Back #1", .65, .83, .66),
    ("Back #1", "Hand #2", .70, .46, .52),
    ("Back #1", "Back #2", .73, .45, .53),
    ("Hand #2", "Hand #1", .79, .62, .63),
    ("Hand #2", "Back #1", .60, .59, .58),
    ("Hand #2", "Hand #2", .71, .77, .67),
    ("Hand #2", "Back #2", .70, .59, .62),
    ("Back #2", "Hand #1", .80, .68, .65),
    ("Back #2", "Back #1", .61, .43, .53),
    ("Back #2", "Hand #2", .72, .57, .55),
    ("Back #2", "Back #2", .71, .73, .60),
]

PATIENT_NRS = [
    ("Subject #1", "Hand", "Painful pinprick", 1, 1, 1),
    ("Subject #1", "Hand", "Non-painful monofilament", 1, 1, 1),
    ("Subject #1", "Back", "Painful pinprick", 1.5, 1, 2),
    ("Subject #1", "Back", "Non-painful monofilament", 1, 1, 1),
    ("Subject #2", "Hand", "Painful pinprick", 3, 2, 4),
    ("Subject #2", "Hand", "Non-painful monofilament", 2, 1, 2),
    ("Subject #2", "Back", "Painful pinprick", 4, 3, 5),
    ("Subject #2", "Back", "Non-painful monofilament", 2, 2, 3),
]

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--outdir', default=str(Path(__file__).parent)); args=ap.parse_args()
    out=Path(args.outdir); out.mkdir(parents=True, exist_ok=True)
    df=pd.DataFrame(ROWS, columns=['training_configuration','testing_configuration','ssm_unsupervised_auc','svm_supervised_auc','svm_peak_accuracy'])
    df['ssm_minus_svm_auc']=df.ssm_unsupervised_auc-df.svm_supervised_auc
    df['source_url']=SOURCE
    df['data_status']='published rounded model output; not raw EEG'
    df.to_csv(out/'published_table2_performance_rows.csv', index=False)
    nrs=pd.DataFrame(PATIENT_NRS, columns=['subject','site','stimulus','nrs_median','nrs_iqr_low','nrs_iqr_high'])
    nrs['source_url']=SOURCE
    nrs['data_status']='published summary (median and IQR); not trial-level labels'
    nrs.to_csv(out/'published_patient_nrs_summary.csv', index=False)

    u=df.ssm_unsupervised_auc.to_numpy(float); s=df.svm_supervised_auc.to_numpy(float); a=df.svm_peak_accuracy.to_numpy(float); d=u-s
    def desc(x):
        return {'n_table_rows':int(len(x)), 'mean':float(np.mean(x)), 'sd_sample':float(np.std(x,ddof=1)),
                'median':float(np.median(x)), 'min':float(np.min(x)), 'max':float(np.max(x)),
                'count_gt_0_5':int(np.sum(x>.5)), 'count_eq_0_5':int(np.sum(x==.5)), 'count_lt_0_5':int(np.sum(x<.5))}
    # All row-based p-values are labelled exploratory because rows are dependent configurations.
    paired_t=stats.ttest_rel(u,s)
    w=stats.wilcoxon(u,s,alternative='two-sided',method='approx',correction=False)
    sign=stats.binomtest(int(np.sum(d>0)), int(np.sum(d!=0)), p=.5, alternative='two-sided')
    pos=stats.binomtest(int(np.sum(d>0)), int(np.sum(d!=0)), p=.5, alternative='greater')
    se=stats.sem(d); tcrit=stats.t.ppf(.975,len(d)-1); ci=(float(d.mean()-tcrit*se),float(d.mean()+tcrit*se))
    # Sensitivity: one table row removed, reported as a diagnostic only.
    loo=[]
    for i in range(len(d)):
        z=np.delete(d,i)
        loo.append({'omitted_row_1_based':i+1,'mean_difference':float(z.mean()),'ssm_wins':int(np.sum(z>0)),'svm_wins':int(np.sum(z<0))})
    pear=stats.pearsonr(u,s); spear=stats.spearmanr(u,s)
    metrics={
      'analysis_status':'REPRODUCED_AGGREGATE_TABLE_REANALYSIS_ONLY',
      'source':SOURCE,
      'dataset_source_for_first_cohort':SOURCE_OSF,
      'table':'Sun et al. (2020), Table 2; 16 rows transcribed as printed to 2 decimal places',
      'sample_context':{
         'healthy_volunteers_used_for_main_analysis':50,
         'original_public_dataset_recruited':51,
         'chronic_pain_patients_in_second_dataset':2,
         'notes':'Table 2 entries are multiple train/test modality configurations from two patients, not 16 independent people or independent experiments.'},
      'descriptive_results':{
        'ssm_unsupervised_auc':desc(u), 'svm_supervised_auc':desc(s), 'svm_peak_accuracy':desc(a),
        'paired_auc_difference_ssm_minus_svm':desc(d),
        'ssm_win_count':int(np.sum(d>0)), 'svm_win_count':int(np.sum(d<0)), 'ties':int(np.sum(d==0)),
        'mean_auc_difference_nominal_t_95ci':ci,
        'paired_standardized_effect_dz':float(d.mean()/np.std(d,ddof=1)),
        'cross-row_pearson_ssm_auc_vs_svm_auc':{'r':float(pear.statistic),'p_naive':float(pear.pvalue)},
        'cross-row_spearman_ssm_auc_vs_svm_auc':{'rho':float(spear.statistic),'p_naive':float(spear.pvalue)},
      },
      'exploratory_row_level_tests_not_valid_as_population_inference':{
        'paired_t_test_ssm_vs_svm_auc':{'t':float(paired_t.statistic),'df':int(len(d)-1),'p_two_sided':float(paired_t.pvalue)},
        'wilcoxon_signed_rank_ssm_vs_svm_auc':{'statistic':float(w.statistic),'p_two_sided_approx':float(w.pvalue),'continuity_correction':False},
        'exact_sign_test_two_sided':{'positive':int(np.sum(d>0)),'negative':int(np.sum(d<0)),'p':float(sign.pvalue)},
        'exact_sign_test_ssm_greater':{'positive':int(np.sum(d>0)),'negative':int(np.sum(d<0)),'p':float(pos.pvalue)},
        'one_sample_t_ssm_auc_vs_chance_0_5':{'t':float(stats.ttest_1samp(u,.5).statistic),'p':float(stats.ttest_1samp(u,.5).pvalue)},
        'one_sample_t_svm_auc_vs_chance_0_5':{'t':float(stats.ttest_1samp(s,.5).statistic),'p':float(stats.ttest_1samp(s,.5).pvalue)},
        'one_sample_t_svm_accuracy_vs_0_5':{'t':float(stats.ttest_1samp(a,.5).statistic),'p':float(stats.ttest_1samp(a,.5).pvalue)},
        'caveat':'These tests quantify the reported table values, but repeated train/test configurations are dependent. Treat p-values and CI as illustrative checks only; they must not be used as confirmatory claims about new subjects.'
      },
      'published_original_findings_not_recomputed':{
        'main_cohort_subjects_analysed':50,
        'group_with_clear_evoked_potentials_n':32,
        'group_without_clear_evoked_potentials_n':18,
        'ssm_auc_group_difference_rank_sum_p':1.9e-5,
        'high_vs_low_stimulus_detection_latency_median_ms':[180,202],
        'high_vs_low_latency_rank_sum_p':1.5e-5,
        'detection_latency_vs_reaction_time_spearman_rho':.29,
        'detection_latency_vs_reaction_time_p':.06,
        'cross_subject_ssm_auc_mean':.62,
        'best_single_subject_unsupervised_auc':.79,
        'best_single_subject_supervised_auc':.95,
        'warning':'These are values reported by the authors; without raw trial data, they were not independently recomputed.'
      },
      'formula_validation_status':{
        'proposed_project_0_10_pain_intensity_formulae_applied_to_raw_trials':False,
        'reason':'The local runtime cannot resolve external dataset hosts and no raw EEG/sample-level feature file is present in the project directory. The published article table provides detection metrics, not EEG predictors or individual pain-score predictions.',
        'what_this_analysis_does':'Recomputes descriptive summaries and exploratory paired tests from the published Table 2 metric rows; reports independent paper-level results separately.',
        'not_claimed':['raw EEG feature extraction','fitted coefficient estimates for the proposed 0-10 scales','new subject-level prediction scores','clinical validation']
      },
      'leave_one_row_out_sensitivity':loo,
      'runtime':{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__,'scipy':__import__('scipy').__version__}
    }
    (out/'published_benchmark_reanalysis.json').write_text(json.dumps(metrics,indent=2))
    # README results from exact printed values; caveats prominently stated.
    md=f'''# Empirical benchmark reanalysis — actual acute-pain EEG study\n\n**Status:** reproducible reanalysis of published numerical outputs, not raw EEG validation of the project's 0–10 score formulae.\n\n## Source and provenance\n\nSun et al., *Detecting acute pain signals from human EEG*, J Neurosci Methods 347 (2020), 108964. Open full text: {SOURCE}. Public first-cohort source cited by that paper: {SOURCE_OSF}.\n\nThe analysis transcribes the 16 published Table 2 rows (SSM unsupervised AUC, SVM supervised peak AUC and peak accuracy). Table values were rounded to two decimal places in the article. The same table's 16 lines are multiple train/test modality configurations arising from **two chronic-back-pain participants**, not 16 independent patients. Therefore row-based inference is exploratory only.\n\n## Results reproduced from Table 2\n\n| Metric | Mean | SD | Median | Range |\n|---|---:|---:|---:|---:|\n| SSM unsupervised AUC | {u.mean():.3f} | {u.std(ddof=1):.3f} | {np.median(u):.2f} | {u.min():.2f}–{u.max():.2f} |\n| SVM supervised peak AUC | {s.mean():.3f} | {s.std(ddof=1):.3f} | {np.median(s):.2f} | {s.min():.2f}–{s.max():.2f} |\n| SVM peak accuracy | {a.mean():.3f} | {a.std(ddof=1):.3f} | {np.median(a):.3f} | {a.min():.2f}–{a.max():.2f} |\n| Paired AUC difference (SSM−SVM) | {d.mean():.3f} | {d.std(ddof=1):.3f} | {np.median(d):.3f} | {d.min():.2f}–{d.max():.2f} |\n\nSSM had higher AUC in **{int(np.sum(d>0))}/16** rows; SVM had higher AUC in **{int(np.sum(d<0))}/16** rows. Nominal mean difference was {d.mean():.3f} AUC points, with a conventional paired-t 95% interval of [{ci[0]:.3f}, {ci[1]:.3f}]. Because the rows are dependent and only two patients underpin the table, this interval is not a valid population-level confidence interval.\n\n### Sensitivity tests on rounded row values (exploratory only)\n\n| Test | Estimate / statistic | p-value | Interpretation limit |\n|---|---:|---:|---|\n| Paired t-test, SSM vs SVM AUC | t={paired_t.statistic:.3f}, df=15 | {paired_t.pvalue:.4f} | Depends on treating configurations as paired independent units; not population-valid here |\n| Wilcoxon signed-rank, two-sided | W={w.statistic:.1f} | {w.pvalue:.4f} | Similar direction to paper's reported p=0.0113; slight difference expected from rounded table values and implementation |\n| Exact sign test, two-sided | 13 wins vs 3 losses | {sign.pvalue:.4f} | Directional consistency check; still dependent rows |\n| Pearson association of SSM vs SVM AUC across rows | r={pear.statistic:.3f} | {pear.pvalue:.4f} | Cross-row descriptive only |\n| Spearman association of SSM vs SVM AUC across rows | ρ={spear.statistic:.3f} | {spear.pvalue:.4f} | Cross-row descriptive only |\n\nThese tests do **not** show that the new pain-intensity formulas work. They show that on this paper's rounded benchmark outputs the SSM detector's AUC tends to exceed the supervised SVM AUC across the listed configurations. The publication itself reports a signed-rank p-value of 0.0113 for the comparison (source, Table 2).\n\n## Additional findings reported by the original authors (not recalculated)\n\n- The public acute-pain EEG cohort included 50 analyzed healthy participants (the source study recruited 51; the motor condition analysis used 50).\n- Among the 50, 32 were reported to have clear localized evoked potentials and 18 were not; group AUC differed at rank-sum p=1.9×10⁻⁵. This is a reminder that signal quality/ERP detectability varies appreciably across individuals.\n- High-intensity pain detection latency had a median of 180 ms versus 202 ms for low intensity, with rank-sum p=1.5×10⁻⁵. Detection latency and reaction time showed a weak, nonsignificant Spearman correlation (ρ=0.29, p=0.06).\n- Mean cross-subject SSM AUC was reported as 0.62; best single-subject SSM AUC was 0.79, while best supervised SVM AUC was 0.95. Best-case results should not be treated as typical performance.\n- In the chronic-back-pain demonstration, the authors analyzed two patients (ages 69 and 48). The table's reported NRS summaries show overlap between painful pinprick and non-painful monofilament, especially for subject 1 (hand medians both 1; back 1.5 vs 1). Thus stimulus condition is not a perfect proxy for experienced pain intensity.\n\n## Which project formulae were applied?\n\n**None of the proposed 0–10 EEG intensity formulae were fitted or applied to these records.** The paper's table contains model AUC/accuracy summaries, not individual EEG samples, spectral features, LEP amplitudes paired to each trial rating, or trial-level predicted intensities. It would be scientifically incorrect to call an AUC-to-0–10 mapping a pain score. Formula families remain unvalidated until actual trial-level EEG and labels are read.\n\n## Blocker observed and reproducibility\n\nThe runtime attempted raw-file access to `raw.githubusercontent.com`, `api.github.com`, and `openneuro.org`, but DNS name resolution failed. No raw EEG or extracted feature file was present in `/mnt/data/EEG_Pain_Scale_Project`. The script deliberately does not fabricate trials or estimate coefficients from the aggregate benchmark table.\n\nRun the aggregate reanalysis: `python published_benchmark_reanalysis.py --outdir .`\n\nRun the intended genuine-trial EEG pipeline when a real preprocessed feature CSV is available: `python eeg_pain_models.py --features /path/to/real_trial_features.csv --outdir results`\n\n## References\n\n1. Sun G, Wen Z, Ok D, et al. Detecting acute pain signals from human EEG. *Journal of Neuroscience Methods*. 2020;347:108964. {SOURCE}\n2. Tiemann L, et al. Public EEG dataset referenced by Sun et al. Open Science Framework record: {SOURCE_OSF}.\n3. Zhao X, et al. A comprehensive EEG dataset of laser-evoked potentials for pain research. *Scientific Data*. 2025;12:1536. https://pmc.ncbi.nlm.nih.gov/articles/PMC12405539/ (independent larger dataset; raw per-trial EEG/ratings are needed for the target regression analysis).\n'''
    (out/'Empirical_Benchmark_Reanalysis.md').write_text(md)
    print(json.dumps({'status':metrics['analysis_status'],'table_rows':len(df),'SSM_mean_AUC':u.mean(),'SVM_mean_AUC':s.mean(),'mean_difference':d.mean(),'paired_t_p':paired_t.pvalue,'wilcoxon_p':w.pvalue,'sign_test_p':sign.pvalue,'files':['published_table2_performance_rows.csv','published_patient_nrs_summary.csv','published_benchmark_reanalysis.json','Empirical_Benchmark_Reanalysis.md']},indent=2))

if __name__=='__main__': main()
