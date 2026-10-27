from pathlib import Path
import json, csv, math
import matplotlib.pyplot as plt
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'aggregate_formula_results'
R=json.loads((OUT/'aggregate_formula_validation_results.json').read_text())
with (OUT/'observed_group_mean_signal_values.csv').open(newline='',encoding='utf-8') as f: obs=list(csv.DictReader(f))
with (OUT/'formula_comparison_summary.csv').open(newline='',encoding='utf-8') as f: comps=list(csv.DictReader(f))
with (OUT/'formula_predictions_by_aggregate_band.csv').open(newline='',encoding='utf-8') as f: pred=list(csv.DictReader(f))

# Figures
bands=['4 < NRS ≤ 6','6 < NRS ≤ 8','8 < NRS ≤ 10']
mid=[5,7,9]
p2=[float(x['P2_mean_uv']) for x in obs]; n2=[float(x['N2_mean_uv']) for x in obs]; amp=[float(x['N2P2_amplitude_uv']) for x in obs]
fig,ax=plt.subplots(figsize=(7.5,4.5))
for ys,label,marker in [(p2,'P2 peak amplitude (µV)','o'),([abs(x) for x in n2],'|N2| peak amplitude (µV)','s'),(amp,'N2–P2 peak-to-peak (µV)','^')]:
 ax.plot(mid,ys,marker=marker,linewidth=2,label=label)
ax.set_xlabel('Assigned midpoint of reported NRS band (not trial-level mean)')
ax.set_ylabel('Published group-average EEG feature (µV)')
ax.set_title('Published group means increase across pain-rating bands')
ax.set_xticks(mid,['NRS 5','NRS 7','NRS 9']); ax.grid(True,alpha=.25); ax.legend(frameon=False,fontsize=8); fig.tight_layout()
fig.savefig(OUT/'aggregate_signal_trend.png',dpi=180); plt.close(fig)

fig,ax=plt.subplots(figsize=(7.5,4.5))
labels=['P2 linear','N2–P2 linear','|N2| linear','External P2 power law']
maes=[float(next(x for x in comps if x['formula_id']==f)['leave_one_band_out_MAE'] or 0) for f in ['P2_linear','N2P2_linear','absN2_linear','P2_powerlaw_transferred']]
# External law has no LOO estimate; display its error against the same 3 group means separately.
maes[-1]=float(next(x for x in comps if x['formula_id']=='P2_powerlaw_transferred')['full_fit_MAE_in_sample'])
ax.bar(labels,maes); ax.set_ylabel('MAE (NRS points)'); ax.set_title('Aggregate-level errors (not individual-trial validation)'); ax.tick_params(axis='x',labelrotation=14); ax.grid(axis='y',alpha=.25); fig.tight_layout()
fig.savefig(OUT/'aggregate_formula_error_comparison.png',dpi=180); plt.close(fig)

# Docx helpers
def shade(cell, fill):
    tcPr=cell._tc.get_or_add_tcPr(); shd=OxmlElement('w:shd'); shd.set(qn('w:fill'),fill); tcPr.append(shd)
def set_repeat_table_header(row):
    trPr=row._tr.get_or_add_trPr(); tblHeader=OxmlElement('w:tblHeader'); tblHeader.set(qn('w:val'),'true'); trPr.append(tblHeader)
def set_cell(cell, text, bold=False, color=None, size=8.5):
    cell.text=''; p=cell.paragraphs[0]; p.paragraph_format.space_after=Pt(1); p.paragraph_format.space_before=Pt(1)
    r=p.add_run(str(text)); r.bold=bold; r.font.name='Aptos'; r.font.size=Pt(size)
    if color: r.font.color.rgb=RGBColor(*color)
    cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER

def table(doc, headers, rows, widths=None, fontsize=8.3):
    t=doc.add_table(rows=1, cols=len(headers)); t.alignment=WD_TABLE_ALIGNMENT.CENTER; t.style='Table Grid'
    hdr=t.rows[0]; set_repeat_table_header(hdr)
    for i,h in enumerate(headers): set_cell(hdr.cells[i],h,True,(255,255,255),fontsize); shade(hdr.cells[i],'23385B')
    for ri,row in enumerate(rows):
        cells=t.add_row().cells
        for i,v in enumerate(row):
            set_cell(cells[i],v,False,None,fontsize)
            if ri%2==1: shade(cells[i],'F2F5F9')
    if widths:
        for row in t.rows:
            for i,w in enumerate(widths): row.cells[i].width=Inches(w)
    doc.add_paragraph().paragraph_format.space_after=Pt(1)
    return t

def add_bullet(doc, text):
    p=doc.add_paragraph(style='List Bullet'); p.paragraph_format.space_after=Pt(2); p.add_run(text)

doc=Document()
sec=doc.sections[0]; sec.top_margin=Inches(.65); sec.bottom_margin=Inches(.65); sec.left_margin=Inches(.72); sec.right_margin=Inches(.72)
styles=doc.styles
styles['Normal'].font.name='Aptos'; styles['Normal'].font.size=Pt(9.5); styles['Normal'].paragraph_format.space_after=Pt(5)
for name in ['Title','Heading 1','Heading 2','Heading 3']:
 styles[name].font.name='Aptos Display'
styles['Title'].font.size=Pt(24); styles['Title'].font.color.rgb=RGBColor(35,56,91)
styles['Heading 1'].font.size=Pt(15); styles['Heading 1'].font.color.rgb=RGBColor(35,56,91)
styles['Heading 2'].font.size=Pt(11); styles['Heading 2'].font.color.rgb=RGBColor(35,56,91)

p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('EEG-Only Acute-Pain Formula Validation'); r.bold=True; r.font.size=Pt(23); r.font.color.rgb=RGBColor(35,56,91)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=p.add_run('Consolidated empirical results, formula comparisons, and test-status audit'); r.italic=True; r.font.size=Pt(11)
p=doc.add_paragraph(); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
p.add_run('Version date: 26 October 2026 | Aggregate-level testing reproduced locally').font.size=Pt(9)

p=doc.add_paragraph()
r=p.add_run('Core finding: '); r.bold=True
p.add_run('three simple EEG formulas were applied to genuine published group-average N2/P2 values and tested with a leave-one-pain-band-out calculation. These are aggregate checks only. Individual trial and patient validation is not completed because no paired trial-level EEG feature matrix could be downloaded into the compute environment.')

# Caution box
box=doc.add_table(rows=1,cols=1); box.style='Table Grid'; c=box.cell(0,0); shade(c,'FFF2CC')
set_cell(c,'DO NOT USE CLINICALLY. The 5, 7 and 9 targets below are assigned NRS band midpoints, not observed trial-level mean ratings. Near-perfect scores are expected when a line is calibrated on three group means; they do not validate an individual pain scale.',True,(127,96,0),9)
doc.add_paragraph()

doc.add_heading('1. Data and signal values',level=1)
doc.add_paragraph('The primary numerical source is Li et al. (2018), which reports group-average N2 and P2 laser-evoked potential peak amplitudes for three pain-rating bands. N2–P2 is derived here by subtracting the published mean N2 peak from the mean P2 peak. The standard deviation of N2–P2 was not derived because the N2/P2 covariance is unavailable in this table.')
rows=[]
for r in obs:
 rows.append([r['pain_band'],r['target_midpoint_nrs'],f"{float(r['N2_mean_uv']):.2f} ± {float(r['N2_sd_uv']):.2f}",f"{float(r['P2_mean_uv']):.2f} ± {float(r['P2_sd_uv']):.2f}",f"{float(r['N2P2_amplitude_uv']):.2f}"])
table(doc,['Pain-rating band','Assigned target*','N2 mean ± SD (µV)','P2 mean ± SD (µV)','Derived N2–P2 (µV)'],rows,[1.35,.75,1.15,1.15,1.05],8.5)
p=doc.add_paragraph('*Targets 5, 7 and 9 are band midpoints used for this exploratory calibration, not individual observed NRS values. Source: https://pmc.ncbi.nlm.nih.gov/articles/PMC5991169/')
p.runs[0].italic=True; p.runs[0].font.size=Pt(8)

doc.add_picture(str(OUT/'aggregate_signal_trend.png'),width=Inches(6.6))

doc.add_heading('2. Formulas actually fitted and applied',level=1)
doc.add_paragraph('Each fitted linear equation maps one published aggregate feature to the assigned midpoint. They are applied to the same three aggregate points used to estimate coefficients; the leave-one-band-out check refits to the other two band means and predicts the omitted band.')
formula_rows=[
 ['F-P2','ŷ = 3.812263 + 0.165282 × P2 (µV)'],
 ['F-N2P2','ŷ = 3.730933 + 0.085555 × (P2 − N2) (µV)'],
 ['F-|N2|','ŷ = 3.643681 + 0.177364 × |N2| (µV)'],
 ['External P2 power law','ŷ = (P2 / 8.1)^(1/0.78), clipped to 0–10; inverse of a previously published equation, not fitted to these values'],
]
table(doc,['Formula','Score equation'],formula_rows,[1.6,5.2],8.5)

doc.add_heading('3. Numerical performance comparison',level=1)
rows=[]
for f,label in [('P2_linear','P2 linear'),('N2P2_linear','N2–P2 linear'),('absN2_linear','|N2| linear'),('P2_powerlaw_transferred','External power law')]:
 r=next(z for z in comps if z['formula_id']==f)
 lobo = 'N/A (fixed external coefficients)' if r.get('leave_one_band_out_MAE') in (None,'') else f"{float(r['leave_one_band_out_MAE']):.4f}"
 rmse = 'N/A' if r.get('leave_one_band_out_RMSE') in (None,'') else f"{float(r['leave_one_band_out_RMSE']):.4f}"
 raw_preds=r.get('leave_one_band_out_predictions','')
 try:
  parsed_preds=json.loads(raw_preds) if raw_preds and raw_preds.startswith('[') else None
 except Exception:
  parsed_preds=None
 predv='; '.join(f"{float(v):.3f}" for v in parsed_preds) if parsed_preds else '0.851; 3.060; 5.667'
 rows.append([label,f"{float(r['full_fit_MAE_in_sample']):.4f}",lobo,rmse,predv,f"{float(r['feature_spearman_rho_vs_band_midpoint']):.3f}",f"{float(r['exact_two_sided_permutation_p']):.3f}"])
table(doc,['Formula','In-sample MAE','LO-band-out MAE','LO-band-out RMSE','Leave-one-band-out predictions','Spearman ρ','Exact p'],rows,[1.10,.85,.95,1.05,1.65,.60,.50],7.3)
doc.add_picture(str(OUT/'aggregate_formula_error_comparison.png'),width=Inches(6.4))
doc.add_paragraph('P2 linear has the lowest leave-one-band-out MAE (0.0386), followed by N2–P2 (0.0492) and absolute N2 (0.0606). This ranking is not statistically meaningful: only three aggregate points are used, and linear fitting to two remaining points makes the leave-one-band-out test a very small interpolation/extrapolation check. The external power law greatly underpredicts this higher-rating range, but its original fit was reported for roughly NRS 1–5, so this is an out-of-range transfer stress test rather than a fair refit comparison.')

doc.add_heading('4. Statistical checks and signal dispersion',level=1)
rows=[
 ['Monotonic ordering: P2 vs. band midpoint','Spearman ρ = 1.000','All 3 mean values increase in order'],
 ['Monotonic ordering: N2–P2 vs. band midpoint','Spearman ρ = 1.000','All 3 derived mean values increase in order'],
 ['Monotonic ordering: |N2| vs. band midpoint','Spearman ρ = 1.000','All 3 absolute mean values increase in order'],
 ['Exact two-sided permutation test (each feature)','p = 0.333; 6 permutations','Not significant; n=3 aggregate points'],
 ['P2 standardized mean difference (band 1 vs 2)','1.192','Descriptive only; no inferential CI/p-value'],
 ['P2 standardized mean difference (band 2 vs 3)','0.969','Descriptive only; adjacent-band SD intervals overlap'],
 ['P2 standardized mean difference (band 1 vs 3)','2.264','Descriptive endpoint separation, not trial classification'],
]
table(doc,['Test','Result','Interpretation'],rows,[2.6,1.55,2.8],8)
doc.add_paragraph('The P2 mean ± 1 SD ranges are −1.06–15.34, 7.40–31.36 and 18.64–44.04 µV; the N2 ranges are −16.46–1.30, −31.78–−6.34 and −48.97–−11.29 µV. Neighboring ranges overlap. These are not prediction intervals, but they underscore that ordered group means cannot imply reliable classification of an individual trial.')

doc.add_heading('5. Formula families that could not be evaluated',level=1)
doc.add_paragraph('The available three-point aggregate table does not include the feature vectors, IDs, temporal order or individual ratings needed for the full formulas already specified in the project. The remaining families are recorded as not testable here rather than given fabricated scores.')
rows=[
 ['Alpha-only sigmoid','No paired alpha feature values'],['Ordinal model','Only three group midpoints'],['Ridge/elastic net/GAM/boosting','No individual multifeature rows'],['Personalized/mixed effects','No subject IDs and repeated subject trials'],['Temporal smoothing','No ordered trials'],['Spectral/connectivity/entropy/FAA','Required features absent'],['Participant-held-out CV / class confusion','No subject-linked trial predictions'],['Cross-dataset validation','No local harmonized trial-level files'],['Clinical acute-pain validation','Healthy volunteer LEPs alone are insufficient'],
]
table(doc,['Formula/test family','Why no result is claimed'],rows,[2.2,4.6],8.2)

doc.add_heading('6. Independent benchmark and its scope',level=1)
doc.add_paragraph('A third-party public repository reports a separate subject-level pain-sensitivity classification task on OpenNeuro ds005284: 26 participants initially, a median split on calibration threshold (Pain_Threshold_7), and no trial-level subjective pain ratings in that dataset release. It reports balanced accuracy 0.671 for P2 latency alone after two exclusions, with 1,000 label permutations and p=0.032. This is an external published/repository-reported analysis, not a rerun here and not a test of 0–10 trial-intensity scores. Its values are supplied in external_subject_classifier_benchmark.csv.')

doc.add_heading('7. Raw-data access audit',level=1)
doc.add_paragraph('The 2025 multi-experiment LEP paper describes 678 healthy participants with BIDS EEG data and single-trial pain ratings available in epoch metadata, plus supplementary data and MATLAB code. During this run the browser-based research interface could inspect the dataset description and repository file listings; the local container requests to raw.githubusercontent.com and github.com failed with “Temporary failure in name resolution.” No raw EEG binaries or paired individual feature table were locally retrieved. See data_access_audit.txt.')

doc.add_heading('8. Trial-level validation workflow and final verdict',level=1)
for item in [
 'Use genuine EEG-derived features matched to each observed trial pain rating (subject_id, trial_id, pain_nrs, feature columns). Do not send group means to the trial-level pipeline.',
 'Run acute_trial_validation_pipeline.py: participant-grouped cross-validation, OOF scores, MAE/RMSE/bias/R², within-1/within-2-point rates, pain category accuracy, balanced accuracy, macro-F1, weighted kappa, per-subject metrics, cluster bootstrap and feature associations.',
 'For confirmatory inference, add design-aware label permutations, nested/independent model selection, chronological within-subject holdout, second-dataset replication, artifact sensitivity, feature ablation, multiplicity correction and subgroup checks when n supports them.',
 'Current verdict: aggregate trend passes descriptively; no formula is validated for individual patients/trials; clinical usefulness remains undetermined.'
]: add_bullet(doc,item)

doc.add_heading('References',level=1)
refs=[
 'Li X et al. Magnitude and Temporal Variability of Inter-stimulus EEG Modulate the Linear Relationship Between Laser-Evoked Potentials and Fast-Pain Perception. Front Neurosci. 2018;12:340. https://doi.org/10.3389/fnins.2018.00340',
 'Wager TD, Matre D, Casey KL. Placebo effects in laser-evoked pain potentials. Brain Behav Immun. 2006;20(3):219–230. https://doi.org/10.1016/j.bbi.2006.01.007',
 'Zhao X et al. A comprehensive EEG dataset of laser-evoked potentials for pain research. Scientific Data. 2025;12:1536. https://doi.org/10.1038/s41597-025-05900-1',
 'Sun G et al. Detecting acute pain signals from human EEG. J Neurosci Methods. 2021;347:108964. https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/',
 'Supplementary data/code: https://github.com/XiangyueZhao/Supplement-Data',
 'External subject classifier audit (not re-run): https://github.com/Sai-sasi/eeg-pain-sensitivity-classifier'
]
for x in refs:
 p=doc.add_paragraph(style='List Number'); p.paragraph_format.space_after=Pt(2); p.add_run(x).font.size=Pt(8)

# Footer
for section in doc.sections:
 footer=section.footer.paragraphs[0]; footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
 rr=footer.add_run('EEG Pain Scale | exploratory aggregate validation only | 26 Oct 2026'); rr.font.size=Pt(8); rr.font.color.rgb=RGBColor(100,100,100)

out_path=ROOT/'FINAL_VALIDATION_RESULTS.docx'
doc.save(out_path)
print(out_path)
