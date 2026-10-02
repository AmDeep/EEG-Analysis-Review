"""Stage 1 statistical test: does baseline-normalized EEG change track laser
intensity, after accounting for subject?

For every (feature, normalization) pair, fits via statsmodels MixedLM (ML, not
REML, so the two models below are comparable by likelihood-ratio test):

  linear:    Y_ij = b0 + b1*P_ij + u_i + e_ij
  quadratic: Y_ij = b0 + b1*P_ij + b2*P_ij^2 + u_i + e_ij

where i = subject (random intercept u_i), j = epoch, P = laser power.

Also reports a naive pooled OLS R^2 (ignoring subject grouping) as a purely
descriptive number alongside the mixed-model beta/p, and a subject-consistency
score: the fraction of subjects (with >= MIN_SUBJECT_POWER_LEVELS distinct
power levels) whose own per-subject slope has the same sign as the population
fixed effect.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from stage1_baseline_normalization import base_feature_names

NORMALIZATIONS = ["delta_abs", "delta_rel", "z"]
MIN_SUBJECT_POWER_LEVELS = 3
MIN_SUBJECTS = 5
MIN_TRIALS = 30


def subject_consistency(sub, beta_sign):
    slopes = []
    for _, g in sub.groupby("global_subject"):
        if g["laser_power"].nunique() < MIN_SUBJECT_POWER_LEVELS:
            continue
        slope = np.polyfit(g["laser_power"], g["y"], 1)[0]
        if slope != 0:
            slopes.append(np.sign(slope))
    if not slopes:
        return np.nan, 0
    slopes = np.array(slopes)
    return float(np.mean(slopes == beta_sign)), int(slopes.size)


def fit_one(df, feature, norm):
    col = f"{feature}_{norm}"
    sub = df[["global_subject", "laser_power", col]].rename(columns={col: "y"}).dropna()
    if sub["global_subject"].nunique() < MIN_SUBJECTS or len(sub) < MIN_TRIALS:
        return None

    sub = sub.copy()
    sub["power2"] = sub["laser_power"] ** 2

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            m_lin = smf.mixedlm("y ~ laser_power", sub, groups=sub["global_subject"]).fit(reml=False)
            m_quad = smf.mixedlm("y ~ laser_power + power2", sub, groups=sub["global_subject"]).fit(reml=False)
        except Exception:
            return None

    beta = m_lin.params.get("laser_power", np.nan)
    p_linear = m_lin.pvalues.get("laser_power", np.nan)
    if np.isnan(beta) or np.isnan(p_linear):
        return None

    lr_stat = max(0.0, 2 * (m_quad.llf - m_lin.llf))
    p_nonlinear = float(stats.chi2.sf(lr_stat, df=1))

    ols_coef = np.polyfit(sub["laser_power"], sub["y"], 1)
    pred = np.polyval(ols_coef, sub["laser_power"])
    ss_res = np.sum((sub["y"] - pred) ** 2)
    ss_tot = np.sum((sub["y"] - sub["y"].mean()) ** 2)
    r2_pooled = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    consistency, n_subj_used = subject_consistency(sub, np.sign(beta))

    return {
        "feature": feature,
        "normalization": norm,
        "direction": "up" if beta > 0 else "down",
        "linear_beta": beta,
        "r2_pooled": r2_pooled,
        "p_linear": p_linear,
        "p_nonlinear": p_nonlinear,
        "nonlinear": bool(p_nonlinear < 0.05),
        "subject_consistency": consistency,
        "n_subjects_consistency": n_subj_used,
        "n_trials": len(sub),
        "n_subjects": sub["global_subject"].nunique(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="features_normalized.csv")
    ap.add_argument("--output", default="stage1_results.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    features = base_feature_names(df.columns)
    print(f"{len(features)} base features x {len(NORMALIZATIONS)} normalizations")

    results = []
    for feature in features:
        for norm in NORMALIZATIONS:
            r = fit_one(df, feature, norm)
            if r is not None:
                results.append(r)
            else:
                print(f"  skipped {feature} ({norm}): insufficient data or model failed to fit")

    out = pd.DataFrame(results).sort_values(["feature", "normalization"])
    out.to_csv(a.output, index=False)
    print(f"\nsaved {a.output}: {out.shape}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
