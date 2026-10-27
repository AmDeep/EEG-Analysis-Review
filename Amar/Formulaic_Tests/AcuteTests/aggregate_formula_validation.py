#!/usr/bin/env python3
"""Reproduce formula tests using published group-average acute-pain EEG signals.

This script intentionally does NOT create synthetic subjects/trials. It fits simple
calibration formulas to three published group means, performs leave-one-pain-band-out
checks, and writes CSV/JSON outputs. This is aggregate signal-level exploration only;
it is not individual-level clinical validation.
"""
from pathlib import Path
import csv, json, itertools, platform, sys
import numpy as np
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "aggregate_formula_results"
OUT.mkdir(exist_ok=True)

# Source-reported group-average N2/P2 amplitudes (microvolts) and the rating-band
# midpoints used in the prior exploratory analysis. The midpoint is NOT an observed
# mean individual rating; it is the representative target assigned to each band.
data = [
    {"pain_band":"4 < NRS <= 6", "target_midpoint_nrs":5.0, "N2_mean_uv":-7.58, "N2_sd_uv":8.88,
     "P2_mean_uv":7.14, "P2_sd_uv":8.20, "N2P2_source":"published group mean; peak-to-peak derived"},
    {"pain_band":"6 < NRS <= 8", "target_midpoint_nrs":7.0, "N2_mean_uv":-19.06, "N2_sd_uv":12.72,
     "P2_mean_uv":19.38, "P2_sd_uv":11.98, "N2P2_source":"published group mean; peak-to-peak derived"},
    {"pain_band":"8 < NRS <= 10", "target_midpoint_nrs":9.0, "N2_mean_uv":-30.13, "N2_sd_uv":18.84,
     "P2_mean_uv":31.34, "P2_sd_uv":12.70, "N2P2_source":"published group mean; peak-to-peak derived"},
]
XFEATURES = {
    "P2_linear": lambda d: np.array([r["P2_mean_uv"] for r in d], float),
    "N2P2_linear": lambda d: np.array([r["P2_mean_uv"]-r["N2_mean_uv"] for r in d], float),
    "absN2_linear": lambda d: np.array([abs(r["N2_mean_uv"]) for r in d], float),
}
y = np.array([r["target_midpoint_nrs"] for r in data], float)
for r in data:
    r["N2P2_amplitude_uv"] = round(r["P2_mean_uv"] - r["N2_mean_uv"], 4)
    r["absN2_uv"] = abs(r["N2_mean_uv"])
    r["N2_mean_minus_1SD_uv"] = r["N2_mean_uv"] - r["N2_sd_uv"]
    r["N2_mean_plus_1SD_uv"] = r["N2_mean_uv"] + r["N2_sd_uv"]
    r["P2_mean_minus_1SD_uv"] = r["P2_mean_uv"] - r["P2_sd_uv"]
    r["P2_mean_plus_1SD_uv"] = r["P2_mean_uv"] + r["P2_sd_uv"]
    r["target_category"] = {5.0:"middle band (4 < NRS <= 6)", 7.0:"upper-middle band (6 < NRS <= 8)", 9.0:"high band (8 < NRS <= 10)"}[r["target_midpoint_nrs"]]

# Write observed data table.
with (OUT/"observed_group_mean_signal_values.csv").open("w", newline="", encoding="utf-8") as f:
    w=csv.DictWriter(f, fieldnames=list(data[0].keys())); w.writeheader(); w.writerows(data)

# Small sample exact rank permutation test: all 3! label permutations.
def exact_two_sided_spearman_p(x, y):
    obs=abs(float(spearmanr(x,y).statistic))
    perm_stats=[]
    for perm in itertools.permutations(y):
        perm_stats.append(abs(float(spearmanr(x, np.array(perm)).statistic)))
    return sum(v >= obs-1e-12 for v in perm_stats)/len(perm_stats), len(perm_stats)

summary=[]; prediction_rows=[]; fit_rows=[]; score_rows=[]
category_midpoints=np.array([5.,7.,9.])
for formula, getx in XFEATURES.items():
    x=getx(data)
    full=LinearRegression().fit(x[:,None],y)
    trainpred=np.clip(full.predict(x[:,None]),0,10)
    lobo=np.full(3,np.nan)
    loo_coef=[]
    for i in range(3):
        tr=np.arange(3)!=i
        model=LinearRegression().fit(x[tr,None],y[tr])
        lobo[i]=np.clip(model.predict(x[~tr,None])[0],0,10)
        loo_coef.append({"held_out_band":data[i]["pain_band"],"intercept":float(model.intercept_),"slope":float(model.coef_[0])})
    train_mae=float(mean_absolute_error(y,trainpred))
    train_rmse=float(mean_squared_error(y,trainpred)**0.5)
    lobo_mae=float(mean_absolute_error(y,lobo))
    lobo_rmse=float(mean_squared_error(y,lobo)**0.5)
    lobo_bias=float(np.mean(lobo-y))
    # Categories based on nearest band midpoint; these classes are coarse bands, not clinic labels.
    predicted_categories=[float(category_midpoints[np.argmin(np.abs(category_midpoints-p))]) for p in lobo]
    cat_acc=float(np.mean(np.array(predicted_categories)==y))
    rho=float(spearmanr(x,y).statistic)
    p_exact,nperm=exact_two_sided_spearman_p(x,y)
    trend=bool(np.all(np.diff(x)>0))
    summary.append({
        "formula_id":formula,"predictor_definition":{"P2_linear":"P2 amplitude, microvolts","N2P2_linear":"P2 minus N2 amplitude, microvolts","absN2_linear":"absolute N2 peak amplitude, microvolts"}[formula],
        "n_aggregate_bands":3,"full_fit_intercept":float(full.intercept_),"full_fit_slope_per_uv":float(full.coef_[0]),
        "full_fit_R2_in_sample":float(r2_score(y,trainpred)),"full_fit_MAE_in_sample":train_mae,"full_fit_RMSE_in_sample":train_rmse,
        "leave_one_band_out_predictions":lobo.tolist(),"leave_one_band_out_MAE":lobo_mae,"leave_one_band_out_RMSE":lobo_rmse,
        "leave_one_band_out_mean_bias":lobo_bias,"leave_one_band_out_category_accuracy_nearest_midpoint":cat_acc,
        "feature_spearman_rho_vs_band_midpoint":rho,"exact_two_sided_permutation_p":p_exact,"permutations":nperm,
        "feature_strictly_increases_across_group_means":trend,"interpretation":"aggregate-level only; three group means; not subject/trial validation"})
    fit_rows.append({"formula_id":formula,"intercept":float(full.intercept_),"slope_per_uv":float(full.coef_[0]),"units":"NRS midpoint per microvolt","trained_on":"same 3 published aggregate points used for in-sample metrics","not_for_clinical_use":True})
    for i,r in enumerate(data):
        prediction_rows.append({"formula_id":formula,"pain_band":r["pain_band"],"target_midpoint_nrs":y[i],"signal_value_uv":float(x[i]),"in_sample_predicted_score":float(trainpred[i]),"leave_one_band_out_predicted_score":float(lobo[i]),"absolute_lobo_error":float(abs(lobo[i]-y[i])),"target_category":r["target_category"],"predicted_category_by_nearest_midpoint":float(predicted_categories[i]),"category_correct":bool(predicted_categories[i]==y[i])})

# Published power-law formula from a separate study: P2 = 8.1 * NRS^0.78; invert to NRS.
p2=np.array([r["P2_mean_uv"] for r in data],float)
power=np.clip((p2/8.1)**(1/0.78),0,10)
power_mae=float(mean_absolute_error(y,power)); power_rmse=float(mean_squared_error(y,power)**0.5)
summary.append({"formula_id":"P2_powerlaw_transferred","predictor_definition":"inverse of a published external power law: NRS=(P2/8.1)^(1/0.78)",
 "n_aggregate_bands":3,"full_fit_intercept":None,"full_fit_slope_per_uv":None,"full_fit_R2_in_sample":float(r2_score(y,power)),
 "full_fit_MAE_in_sample":float(mean_absolute_error(y,power)),"full_fit_RMSE_in_sample":float(mean_squared_error(y,power)**0.5),
 "leave_one_band_out_predictions":None,"leave_one_band_out_MAE":None,"leave_one_band_out_RMSE":None,
 "leave_one_band_out_mean_bias":float(np.mean(power-y)),"leave_one_band_out_category_accuracy_nearest_midpoint":float(np.mean(np.array([category_midpoints[np.argmin(np.abs(category_midpoints-p))] for p in power])==y)),
 "feature_spearman_rho_vs_band_midpoint":float(spearmanr(p2,y).statistic),"exact_two_sided_permutation_p":exact_two_sided_spearman_p(p2,y)[0],"permutations":6,
 "feature_strictly_increases_across_group_means":bool(np.all(np.diff(p2)>0)),"interpretation":"cross-study transfer test; severe under-calibration in these 3 aggregate points; not fair clinical validation"})
for i,r in enumerate(data):
    cat=float(category_midpoints[np.argmin(np.abs(category_midpoints-power[i]))])
    prediction_rows.append({"formula_id":"P2_powerlaw_transferred","pain_band":r["pain_band"],"target_midpoint_nrs":y[i],"signal_value_uv":float(p2[i]),"in_sample_predicted_score":float(power[i]),"leave_one_band_out_predicted_score":"N/A (external fixed coefficients)","absolute_lobo_error":float(abs(power[i]-y[i])),"target_category":r["target_category"],"predicted_category_by_nearest_midpoint":cat,"category_correct":bool(cat==y[i])})

with (OUT/"formula_comparison_summary.csv").open("w",newline="",encoding="utf-8") as f:
    fields=list(summary[0].keys()); w=csv.DictWriter(f,fieldnames=fields); w.writeheader(); w.writerows(summary)
with (OUT/"formula_predictions_by_aggregate_band.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(prediction_rows[0].keys())); w.writeheader(); w.writerows(prediction_rows)
with (OUT/"fitted_linear_formula_coefficients.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(fit_rows[0].keys())); w.writeheader(); w.writerows(fit_rows)

# Signal SD overlap is reported only for P2, since only P2 SDs were transcribed from source.
unc=[]
for r in data:
    unc.append({"pain_band":r["pain_band"],"N2_mean_uv":r["N2_mean_uv"],"N2_SD_uv":r["N2_sd_uv"],"N2_mean_minus_1SD_uv":r["N2_mean_uv"]-r["N2_sd_uv"],"N2_mean_plus_1SD_uv":r["N2_mean_uv"]+r["N2_sd_uv"],"P2_mean_uv":r["P2_mean_uv"],"P2_SD_uv":r["P2_sd_uv"],"P2_mean_minus_1SD_uv":r["P2_mean_uv"]-r["P2_sd_uv"],"P2_mean_plus_1SD_uv":r["P2_mean_uv"]+r["P2_sd_uv"],"meaning":"descriptive mean +/- 1 SD; not a prediction interval"})
with (OUT/"signal_dispersion_check.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(unc[0].keys())); w.writeheader(); w.writerows(unc)

# Propagate reported one-SD signal ranges through the fitted aggregate equations.
# This is a sensitivity calculation, not a prediction interval or individual-level error rate.
p2_fit=LinearRegression().fit(np.array([r["P2_mean_uv"] for r in data],float)[:,None],y)
n2_feat=np.array([abs(r["N2_mean_uv"]) for r in data],float)
n2_fit=LinearRegression().fit(n2_feat[:,None],y)
score_unc=[]
for r in data:
    p2_mean=float(r["P2_mean_uv"]); p2_sd=float(r["P2_sd_uv"])
    n2_mean=float(r["N2_mean_uv"]); n2_sd=float(r["N2_sd_uv"])
    p2_center=float(np.clip(p2_fit.predict([[p2_mean]])[0],0,10))
    p2_low=float(np.clip(p2_fit.predict([[p2_mean-p2_sd]])[0],0,10))
    p2_high=float(np.clip(p2_fit.predict([[p2_mean+p2_sd]])[0],0,10))
    n2_a=float(np.clip(n2_fit.predict([[abs(n2_mean-n2_sd)]])[0],0,10))
    n2_b=float(np.clip(n2_fit.predict([[abs(n2_mean+n2_sd)]])[0],0,10))
    score_unc.append({"pain_band":r["pain_band"],"target_midpoint_nrs":r["target_midpoint_nrs"],
      "P2_center_score":p2_center,"P2_score_low_at_mean_minus_1SD":min(p2_low,p2_high),"P2_score_high_at_mean_plus_1SD":max(p2_low,p2_high),
      "P2_slope_times_reported_SD_NRS_points":abs(float(p2_fit.coef_[0]))*p2_sd,
      "absN2_center_score":float(np.clip(n2_fit.predict([[abs(n2_mean)]])[0],0,10)),
      "absN2_score_low_from_N2_mean_plusminus_1SD":min(n2_a,n2_b),"absN2_score_high_from_N2_mean_plusminus_1SD":max(n2_a,n2_b),
      "interpretation":"sensitivity range from published group SDs, not individual prediction intervals"})
with (OUT/"signal_score_uncertainty_check.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(score_unc[0].keys())); w.writeheader(); w.writerows(score_unc)

pairwise=[]
for i,j in [(0,1),(1,2),(0,2)]:
    m1,m2=float(data[i]["P2_mean_uv"]),float(data[j]["P2_mean_uv"])
    s1,s2=float(data[i]["P2_sd_uv"]),float(data[j]["P2_sd_uv"])
    rms_sd=float(np.sqrt((s1*s1+s2*s2)/2))
    pairwise.append({"band_1":data[i]["pain_band"],"band_2":data[j]["pain_band"],"P2_mean_difference_uv":m2-m1,"root_mean_average_variance_uv":rms_sd,"descriptive_standardized_mean_difference":(m2-m1)/rms_sd,"interpretation":"descriptive standardized group mean difference; no inferential interval/p-value"})
with (OUT/"pairwise_p2_standardized_mean_differences.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.DictWriter(f,fieldnames=list(pairwise[0].keys())); w.writeheader(); w.writerows(pairwise)

# Formula readiness registry: whether available published measurements permit actual testing.
not_testable = [
("alpha_sigmoid","alpha band power / within-person standardized alpha","NOT TESTABLE HERE","No alpha-feature values paired with these pain-band group means."),
("ordinal_cumulative_link","trial-level EEG predictors + individual ordinal NRS","NOT TESTABLE HERE","Three group midpoints cannot support ordinal model fitting or validation."),
("ridge_elasticnet","multivariate trial-level EEG features + individual NRS","NOT TESTABLE HERE","No paired multifeature trial-level rows; n=3 aggregates is inadequate."),
("GAM_or_gradient_boosting","trial-level features and individual NRS","NOT TESTABLE HERE","Would grossly overfit three aggregate observations."),
("within_person_calibration","multiple time-ordered trials per subject","NOT TESTABLE HERE","Group means contain no subject or trial structure."),
("temporal_EWMA","ordered trial predictions within subject","NOT TESTABLE HERE","No temporal sequence available."),
("spectral_only","alpha/beta/theta/gamma power by trial","NOT TESTABLE HERE","Spectral values absent from available signal table."),
("connectivity_entropy","connectivity and entropy metrics by trial","NOT TESTABLE HERE","Features absent; no channel time series available."),
("mixed_effects","subject ID + trial-level EEG + NRS","NOT TESTABLE HERE","No subject-linked raw rows in available source table."),
("random_forest_hgb","trial-level EEG matrix + NRS","NOT TESTABLE HERE","No training rows; no valid held-out evaluation possible."),
]
with (OUT/"formula_test_coverage.csv").open("w",newline="",encoding="utf-8") as f:
    w=csv.writer(f); w.writerow(["formula_family","required_input","status","reason"])
    w.writerows([(s[0],s[1],s[2],s[3]) for s in not_testable])

# Summary and explicit limitations.
result={
 "run_status":"COMPLETED_AGGREGATE_LEVEL_ONLY",
 "run_timestamp_utc":"2026-10-27",
 "python_version":platform.python_version(),
 "source":"Li et al. 2018, published group-average N2/P2 amplitudes by NRS band; source URL https://pmc.ncbi.nlm.nih.gov/articles/PMC7744433/",
 "observations_used":3,"observations_unit":"published pain-band group means, not individual subjects/trials",
 "target_values":[5,7,9],"target_caveat":"rating-band midpoints assigned as representative targets, not observed per-trial ratings",
 "results":summary,
 "tests_completed":["fitted simple linear calibration on all 3 aggregate points","leave-one-aggregate-band-out linear prediction","MAE/RMSE/bias/R2 on aggregate points","nearest-midpoint category agreement","Spearman rank ordering","exact two-sided rank permutation test across all 3! = 6 permutations","P2 mean +/- 1SD descriptive dispersion check","external power-law transfer check","formula input coverage audit"],
 "tests_not_possible":["participant-held-out CV","trial-level rating regression/classification","clustered confidence intervals","subject-preserving permutation test","temporal holdout","cross-dataset trial-level evaluation","calibration on unobserved trials","multiple-feature model ranking","clinical acute-pain patient test"],
 "why_not_possible":"Raw or trial-level EEG feature rows were not locally accessible. GitHub/OpenNeuro host requests from this runtime failed with DNS resolution errors; source documents provide aggregate means and not paired individual data in this run.",
 "statistical_conclusion":"Not enough independent observations for population inference. Exact Spearman permutation p=0.333 for strict monotonic ordering with n=3. The very small aggregate LOBO errors are not evidence of subject-level predictive performance.",
 "no_synthetic_data_used":True
}
(OUT/"aggregate_formula_validation_results.json").write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps({"run_status":result["run_status"],"n_aggregate_points":3,"formula_results":[{k:r[k] for k in ["formula_id","full_fit_MAE_in_sample","leave_one_band_out_MAE","leave_one_band_out_RMSE","leave_one_band_out_mean_bias","feature_spearman_rho_vs_band_midpoint","exact_two_sided_permutation_p"] if k in r} for r in summary]},indent=2))
