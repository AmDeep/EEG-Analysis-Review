"""Stage 9: does bringing in a DIFFERENT feature set (the KNNs folder's
Cz-channel ERP extraction) explain any of the ~52% of variance Stage 3/4 left
on the table?

Stage 3's one-way ANOVA of the composite score across laser_power levels gave
eta^2=0.48 on the within-subject cohort (124 subjects, ds005293+ds005473) --
48% of variance explained by intensity, 52% left over. `../KNNs/
features_cz_erp_merged.csv` has a per-trial feature set from an independent
extraction pipeline for the SAME 10,984 trials, including things Stage 1-8's
20-feature set never computed: classic N2/P2 laser-evoked-potential
amplitude/latency (the standard single-trial pain-intensity biomarker in the
LEP literature), skewness, kurtosis, and permutation entropy.

This is an actual ANCOVA: same one-way ANOVA design as Stage 3
(score ~ C(laser_power)), plus these new covariates added to the SAME model,
fit via OLS with Type II sums of squares. If total R^2 grows past 0.48, some
of the "unexplained" variance is explained by information Stage 1-8 didn't
have; the covariates' individual partial eta^2/p-values say which ones.
"""
import argparse

import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.anova import anova_lm

NEW_COVARIATES = [
    "n2_latency_ms", "p2_latency_ms", "n2p2_peaktopeak_uv", "skewness", "kurtosis", "max_abs",
    # NOTE: permutation_entropy dropped -- it's a constant 0 for all 10,984 trials
    # in features_cz_erp_merged.csv (a bug in whatever produced that column, not
    # a real feature). n2_amplitude_uv/p2_amplitude_uv dropped too -- by definition
    # n2p2_peaktopeak_uv = p2_amplitude_uv - n2_amplitude_uv exactly, so having all
    # three is an exact linear dependency (confirmed: max abs difference ~1e-16).
    # Keeping the standard peak-to-peak amplitude, the metric actually used in the
    # LEP literature, instead of the two raw components.
]


def load_merged(scored_path, erp_path):
    scored = pd.read_csv(scored_path)
    erp = pd.read_csv(erp_path)
    erp["global_subject"] = erp["dataset"] + "_" + erp["subject"]
    erp = erp.rename(columns={"trial": "epoch"})[["global_subject", "epoch", *NEW_COVARIATES]]

    merged = scored.merge(erp, on=["global_subject", "epoch"], how="inner")
    print(f"{len(merged)} / {len(scored)} trials matched to the KNNs ERP feature file")
    return merged


def subject_level_aggregate(df, cols):
    return df.groupby(["global_subject", "laser_power"])[cols].mean().reset_index()


def run_ancova(subject_level, covariates):
    subject_level = subject_level.dropna(subset=["score", "laser_power", *covariates]).copy()

    baseline_model = smf.ols("score ~ C(laser_power)", subject_level).fit()
    baseline_r2 = baseline_model.rsquared
    print(f"=== Baseline: score ~ laser_power only (matches Stage 3's eta^2) ===")
    print(f"R^2 = {baseline_r2:.4f} (Stage 3 reported eta^2=0.48 on the same cohort/design)")

    formula = "score ~ C(laser_power) + " + " + ".join(covariates)
    full_model = smf.ols(formula, subject_level).fit()
    print(f"\n=== ANCOVA: score ~ laser_power + {len(covariates)} new covariates ===")
    print(f"R^2 = {full_model.rsquared:.4f} (delta = {full_model.rsquared - baseline_r2:+.4f} vs. baseline)")

    anova_table = anova_lm(full_model, typ=2)
    anova_table["partial_eta2"] = anova_table["sum_sq"] / (anova_table["sum_sq"] + anova_table["sum_sq"]["Residual"])
    print("\nType II ANCOVA table (partial eta^2 per term):")
    print(anova_table.to_string())

    return baseline_model, full_model, anova_table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", default="stage2_scored_trials.csv")
    ap.add_argument("--erp-features", default="../KNNs/features_cz_erp_merged.csv")
    ap.add_argument("--output-prefix", default="stage9")
    a = ap.parse_args()

    merged = load_merged(a.scored, a.erp_features)
    merged = merged.rename(columns={"neural_response_score": "score"})

    subject_level = subject_level_aggregate(merged, ["score", *NEW_COVARIATES])
    baseline_model, full_model, anova_table = run_ancova(subject_level, NEW_COVARIATES)

    anova_table.to_csv(f"{a.output_prefix}_ancova_table.csv")
    pd.DataFrame({
        "model": ["laser_power_only", "laser_power_plus_erp_covariates"],
        "r2": [baseline_model.rsquared, full_model.rsquared],
        "n_params": [baseline_model.df_model, full_model.df_model],
    }).to_csv(f"{a.output_prefix}_r2_comparison.csv", index=False)
    print(f"\nsaved {a.output_prefix}_ancova_table.csv, {a.output_prefix}_r2_comparison.csv")

    sig_covariates = anova_table.drop(index=["C(laser_power)", "Residual"], errors="ignore")
    sig_covariates = sig_covariates[sig_covariates["PR(>F)"] < 0.05].sort_values("partial_eta2", ascending=False)
    if len(sig_covariates):
        print(f"\nCovariates significant at p<0.05, ranked by partial eta^2:")
        print(sig_covariates.to_string())
    else:
        print("\nNo covariate reached p<0.05 in the ANCOVA.")


if __name__ == "__main__":
    main()
