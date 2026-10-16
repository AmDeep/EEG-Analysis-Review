#!/usr/bin/env python3
"""EEG-only neural score + cross-fitted stimulus -> EEG score -> pain associations.

Research prototype: predictive statistical mediation, NOT proof of neural causation.

Inputs: trial-level CSV produced by eeg_pain_features.py, or a compatible table.
Outputs: subject-held-out predictions, fitted EEG-only score, coefficient table,
         predictive benchmarks, descriptive cross-fitted mediation associations.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Exactly one prespecified compact representative for EACH table category.
# Category 13 is neonatal-specific and is NEVER included in adult models.
FEATURE_CATALOG = [
    (1, 'Sensorimotor alpha power (8-13 Hz)', 'delta_sensorimotor_alpha_ln_power', 'adult'),
    (2, 'Peak alpha frequency', 'delta_sensorimotor_peak_alpha_hz', 'adult_exploratory'),
    (3, 'Alpha envelope dynamics', 'delta_sensorimotor_alpha_envelope_cv', 'adult_exploratory'),
    (4, 'Frontal alpha asymmetry', 'delta_frontal_alpha_asymmetry_ln_f4_over_f3', 'adult_exploratory'),
    (5, 'Beta power (13-30 Hz)', 'delta_sensorimotor_beta_ln_power', 'adult'),
    (6, 'Theta power (4-8 Hz)', 'delta_global_theta_ln_power', 'adult'),
    (7, 'Spatially defined midline theta', 'delta_frontal_midline_theta_ln_power', 'adult_exploratory'),
    (8, 'Gamma power (>30 Hz, analyzed 30-45 Hz)', 'delta_gamma_central_gamma_ln_power', 'adult'),
    (9, 'Connectivity and phase synchronization', 'delta_central_c3_c4_alpha_pli', 'adult_exploratory'),
    (10, 'Spectral entropy', 'delta_sensorimotor_spectral_entropy', 'adult'),
    (11, 'Response variability and temporal features', 'delta_cz_response_temporal_sd_uv', 'adult_exploratory'),
    (12, 'Vertex N2-P2 amplitude', 'erp_cz_n2_to_p2_uv', 'adult_discrete_stimulus'),
    (13, 'Infant noxious-evoked EEG template', 'neonatal_noxious_template_corr', 'neonatal_template_required'),
]
FULL_ADULT_FEATURES = [c for _, _, c, kind in FEATURE_CATALOG if kind != 'neonatal_template_required']
ALL_13_FEATURES = [c for _, _, c, _ in FEATURE_CATALOG]
CORE_FEATURES = [FULL_ADULT_FEATURES[i-1] for i in (1,5,6,8,9,10,12)]
EXPLORATORY_FEATURES = [
    'delta_centro_midline_theta_ln_power',
    'delta_central_c3_c4_alpha_coherence',
    'delta_midline_fz_cz_theta_pli',
    'erp_cz_n2_latency_ms',
    'erp_cz_p2_latency_ms',
    'erp_cz_response_temporal_rms_uv',
    'erp_cz_n2_to_p2_previous5_sd_uv', # previous 5 accepted trials only; missing early trials
    'neonatal_noxious_template_projection', # only for vetted neonatal template
]
RIDGE_ALPHAS = [0.01, 0.1, 1.0, 10.0]
EPS = 1e-10


def numeric_matrix(frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    return frame.reindex(columns=names).apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def fit_preprocessing(X: np.ndarray) -> dict:
    med = np.zeros(X.shape[1])
    for j in range(X.shape[1]):
        col = X[:, j]
        finite = col[np.isfinite(col)]
        med[j] = float(np.median(finite)) if len(finite) else 0.0
    Xi = np.where(np.isfinite(X), X, med)
    mu = Xi.mean(axis=0)
    sigma = Xi.std(axis=0)
    sigma = np.where(sigma < EPS, 1.0, sigma)
    return {"median": med.tolist(), "mean": mu.tolist(), "sd": sigma.tolist()}


def transform_preprocessing(X: np.ndarray, prep: dict) -> np.ndarray:
    median = np.asarray(prep["median"])
    mu = np.asarray(prep["mean"])
    sd = np.asarray(prep["sd"])
    X = np.where(np.isfinite(X), X, median)
    return (X - mu) / sd


def design_intensity_subject(I: np.ndarray, subjects: np.ndarray) -> np.ndarray:
    """Nuisance design for within-subject stimulus-adjusted feature learning."""
    I = np.asarray(I, dtype=float)
    subjects = np.asarray(subjects).astype(str)
    u = np.unique(subjects)
    sd = max(float(I.std()), EPS)
    cols = [np.ones(len(I)), (I - I.mean()) / sd]
    cols.extend((subjects == x).astype(float) for x in u[1:])
    return np.column_stack(cols)


def centered_by_subject(a: np.ndarray, subj: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    output = a.copy()
    for s in np.unique(subj):
        ix = subj == s
        output[ix] = a[ix] - a[ix].mean(axis=0)
    return output


def ols(D: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.lstsq(D, y, rcond=None)[0]


def fit_score(frame: pd.DataFrame, features: list[str], alpha: float) -> dict:
    """Fit linear single-index S=f(EEG), conditioned on I and subject during learning.

    No stimulus or pain labels enter S at inference time.
    """
    I = frame["stimulus_intensity"].to_numpy(float)
    P = frame["pain_rating"].to_numpy(float)
    subj = frame["subject"].astype(str).to_numpy()
    pre = fit_preprocessing(numeric_matrix(frame, features))
    X = transform_preprocessing(numeric_matrix(frame, features), pre)
    # Learn the index from WITHIN-SUBJECT pain variation WITHOUT residualizing
    # stimulus intensity out of both signals. Doing so would deliberately remove
    # stimulus-related EEG variance, undermining the proposed I -> S -> P test.
    # Only subject mean differences are removed for fitting the weights.
    XR = centered_by_subject(X, subj)
    PR = centered_by_subject(P, subj)
    n = len(P)
    lhs = XR.T @ XR / n + float(alpha) * np.eye(X.shape[1])
    rhs = XR.T @ PR / n
    w = np.linalg.solve(lhs, rhs)
    raw = X @ w
    score_mu = float(raw.mean())
    score_sd = float(raw.std())
    if score_sd < EPS:
        score_sd = 1.0
    S = (raw - score_mu) / score_sd
    # Test-time pain heads are pooled rather than relying on unknown subject IDs.
    head_i = ols(np.column_stack([np.ones(n), I]), P)
    head_joint = ols(np.column_stack([np.ones(n), I, S]), P)
    head_eeg = ols(np.column_stack([np.ones(n), S]), P)
    return {
        "features": features,
        "preprocessing": pre,
        "weights": w.tolist(),
        "score_mean": score_mu,
        "score_sd": score_sd,
        "pain_head_intensity_only": head_i.tolist(),
        "pain_head_intensity_and_eeg": head_joint.tolist(),
        "pain_head_eeg_only": head_eeg.tolist(),
        "ridge_alpha": float(alpha),
        "description": "Linear EEG-only index trained against within-subject pain variation (without removing stimulus-driven EEG variance); stimulus-adjusted pain head fitted separately. Descriptive statistical mediation only.",
    }


def predict(frame: pd.DataFrame, model: dict, pain_min: float, pain_max: float) -> pd.DataFrame:
    X = numeric_matrix(frame, model["features"])
    Xt = transform_preprocessing(X, model["preprocessing"])
    z = ((Xt @ np.asarray(model["weights"])) - model["score_mean"]) / model["score_sd"]
    P_eeg = np.column_stack([np.ones(len(z)), z]) @ np.asarray(model["pain_head_eeg_only"])
    pred = pd.DataFrame({
        "neural_score_z": z,
        "pred_pain_eeg_only": np.clip(P_eeg, pain_min, pain_max),
        "score_0_100": 100.0 * np.clip(P_eeg - pain_min, 0, pain_max-pain_min) / (pain_max-pain_min),
    }, index=frame.index)
    if "stimulus_intensity" in frame:
        I = pd.to_numeric(frame["stimulus_intensity"], errors="coerce").to_numpy(float)
        head_i = np.asarray(model["pain_head_intensity_only"])
        head_j = np.asarray(model["pain_head_intensity_and_eeg"])
        pred["pred_pain_intensity_only"] = np.clip(head_i[0] + head_i[1]*I, pain_min, pain_max)
        pred["pred_pain_intensity_and_eeg"] = np.clip(head_j[0] + head_j[1]*I + head_j[2]*z, pain_min, pain_max)
    return pred


def select_alpha(frame: pd.DataFrame, features: list[str], pain_min: float, pain_max: float, seed: int) -> float:
    subj = frame["subject"].astype(str).to_numpy()
    if len(np.unique(subj)) < 4:
        return 0.1  # No meaningful subject-grouped nested CV.
    cv = GroupKFold(n_splits=min(3, len(np.unique(subj))))
    # Crucially the alpha search sees only the OUTER TRAINING subjects.
    loss = []
    for alpha in RIDGE_ALPHAS:
        errs = []
        for tr, va in cv.split(frame, groups=subj):
            a, b = frame.iloc[tr], frame.iloc[va]
            m = fit_score(a, features, alpha)
            prediction = predict(b, m, pain_min, pain_max)
            errs.extend(abs(b["pain_rating"].to_numpy(float) - prediction["pred_pain_intensity_and_eeg"].to_numpy(float)))
        loss.append(float(np.mean(errs)))
    return float(RIDGE_ALPHAS[int(np.argmin(loss))])


def evaluate(y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "MAE": float(mean_absolute_error(y, pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, pred))),
        "R2": float(r2_score(y, pred)) if np.std(y) > EPS else None,
    }


def within_subject_mediation(table: pd.DataFrame) -> dict:
    """Within-subject linear a, b, c', a*b associations on cross-fitted S.

    Descriptive associations, NOT identification of causal mediation.
    """
    subj = table["subject"].astype(str).to_numpy()
    I = centered_by_subject(table["stimulus_intensity"].to_numpy(float), subj)
    S = centered_by_subject(table["neural_score_z"].to_numpy(float), subj)
    P = centered_by_subject(table["pain_rating"].to_numpy(float), subj)
    if np.std(I) < EPS or np.std(S) < EPS:
        raise ValueError("No within-subject intensity or neural-score variation for mediation estimates.")
    a = float(ols(I[:,None], S)[0])
    c = float(ols(I[:,None], P)[0])
    c_prime, b = map(float, ols(np.column_stack([I, S]), P))
    return {"a_intensity_to_neural": a, "b_neural_to_pain_given_intensity": b,
            "c_total_intensity_to_pain": c, "c_prime_direct_association": c_prime,
            "indirect_product_a_times_b": a*b,
            "c_minus_c_prime": c-c_prime,
            "pain_rating_units_per_stimulus_unit": True}


def cluster_bootstrap_fixed_scores(table: pd.DataFrame, reps: int, seed: int) -> dict:
    """Cluster-resample subjects of fixed OOF scores; excludes score-training uncertainty."""
    rng = np.random.default_rng(seed)
    groups = [g.copy() for _,g in table.groupby("subject", sort=False)]
    all_estimates = []
    for _ in range(reps):
        picked = rng.integers(0, len(groups), size=len(groups))
        chunks = []
        for i, j in enumerate(picked):
            g = groups[j].copy()
            g["subject"] = f"boot-{i}"  # repeated draws are independent clusters
            chunks.append(g)
        try:
            all_estimates.append(within_subject_mediation(pd.concat(chunks, ignore_index=True)))
        except ValueError:
            pass
    if not all_estimates:
        return {}
    return {k: {"lower_2_5": float(np.quantile([q[k] for q in all_estimates], 0.025)),
                "upper_97_5": float(np.quantile([q[k] for q in all_estimates], 0.975))}
            for k in ["a_intensity_to_neural", "b_neural_to_pain_given_intensity",
                      "c_prime_direct_association", "indirect_product_a_times_b"]}


def analyze(csv_path: Path, out: Path, expanded: bool = False, extra: list[str] | None = None,
            exclude: list[str] | None = None, folds: int = 5, bootstraps: int = 300, seed: int = 21, pain_min: float = 0,
            pain_max: float = 10, fit_only: bool = False,
            feature_set: str = 'all-adult') -> None:
    out.mkdir(parents=True, exist_ok=True)
    df0 = pd.read_csv(csv_path)
    required = ["subject", "stimulus_intensity", "pain_rating"]
    missing = [x for x in required if x not in df0.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}. Pain labels must be trial-level actual ratings.")
    if feature_set not in {'core', 'all-adult', 'all-13'}:
        raise ValueError('feature_set must be core, all-adult, or all-13.')
    if feature_set == 'all-13':
        if 'cohort' not in df0 or not df0['cohort'].astype(str).str.lower().eq('neonatal').all():
            raise ValueError('all-13 requires exclusively neonatal cohort data with a validated external infant template. Adult studies should use all-adult (12 categories).')
    features = {'core': CORE_FEATURES, 'all-adult': FULL_ADULT_FEATURES,
                'all-13': ALL_13_FEATURES}[feature_set].copy()
    if expanded:
        # Include only neonatal projection for appropriate neonatal cohorts.
        extras = [f for f in EXPLORATORY_FEATURES
                  if (not f.startswith('neonatal_') or feature_set == 'all-13')]
        features.extend(extras)
    features.extend(extra or [])
    features = [x for x in dict.fromkeys(features) if x not in set(exclude or [])]
    if feature_set != 'all-13' and any(x.startswith('neonatal_') for x in features):
        raise ValueError('Neonatal template metrics cannot be mixed into an adult score.')
    absent = [c for c in features if c not in df0.columns]
    if absent:
        raise ValueError(f'Missing expected EEG features: {absent}. Rerun v2 preprocessing, check required electrodes, or explicitly --exclude-feature name.')
    if len(features) < 2:
        raise ValueError('At least two EEG features are required.')
    empty_features = [c for c in features if not np.isfinite(pd.to_numeric(df0[c], errors='coerce').to_numpy(float)).any()]
    if empty_features:
        raise ValueError(f'These EEG features contain NO measurements: {empty_features}. Do not silently replace an entire unavailable feature with zeros. Check electrodes, response timings, cohort, or --exclude-feature.')
    sparse_features = [c for c in features if np.mean(~np.isfinite(pd.to_numeric(df0[c], errors='coerce').to_numpy(float))) > 0.25]
    if sparse_features:
        warnings.warn(f'>25% missingness in {sparse_features}; inspect artifact QC / electrode coverage before interpreting coefficients.')
    df = df0.copy()
    for c in ["stimulus_intensity", "pain_rating"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.loc[np.isfinite(df["stimulus_intensity"]) & np.isfinite(df["pain_rating"])].copy()
    df["subject"] = df["subject"].astype(str)
    if df.empty:
        raise ValueError("No trials with both physical stimulus labels and actual pain ratings.")
    if pain_max <= pain_min or not df["pain_rating"].between(pain_min, pain_max).all():
        raise ValueError(f"Pain ratings must be within [{pain_min}, {pain_max}]. Adjust --pain-min/max to your scale.")
    # A row with all missing EEG features is not a valid EEG trial.
    useable = np.isfinite(numeric_matrix(df, features)).any(axis=1)
    df = df.loc[useable].reset_index(drop=True)
    n_subject = df["subject"].nunique()
    if not fit_only and n_subject < 3:
        raise ValueError("Subject-held-out evaluation requires >=3 subjects. Use --fit-only to fit a provisional score on 1-2 subjects (no validation/mediation claims).")
    if n_subject >= 3 and not fit_only:
        splitter = GroupKFold(n_splits=min(folds, n_subject))
        oofs = []
        fold_log = []
        for fold, (tr, te) in enumerate(splitter.split(df, groups=df["subject"].to_numpy()), start=1):
            train, test = df.iloc[tr], df.iloc[te]
            alpha = select_alpha(train, features, pain_min, pain_max, seed)
            model = fit_score(train, features, alpha)
            p = predict(test, model, pain_min, pain_max)
            r = test.copy().reset_index(drop=True)
            for col in p.columns:
                r[col] = p[col].to_numpy()
            r["cv_fold"] = fold
            oofs.append(r)
            fold_log.append({"fold": fold, "train_subjects": int(train["subject"].nunique()),
                             "test_subjects": int(test["subject"].nunique()),
                             "test_trials": len(test), "ridge_alpha": alpha})
        heldout = pd.concat(oofs, ignore_index=True)
        heldout.to_csv(out / "heldout_trial_predictions.csv", index=False)
        truth = heldout["pain_rating"].to_numpy(float)
        metrics = {
            "intensity_only": evaluate(truth, heldout["pred_pain_intensity_only"].to_numpy(float)),
            "intensity_plus_eeg": evaluate(truth, heldout["pred_pain_intensity_and_eeg"].to_numpy(float)),
            "eeg_only": evaluate(truth, heldout["pred_pain_eeg_only"].to_numpy(float)),
        }
        metrics["MAE_gain_joint_over_intensity_only"] = metrics["intensity_only"]["MAE"] - metrics["intensity_plus_eeg"]["MAE"]
        mediation = {
            "method": "within-subject demeaned linear associations using cross-fitted scores",
            "point_estimates": within_subject_mediation(heldout),
            "conditional_fixed_score_subject_bootstrap_CI": cluster_bootstrap_fixed_scores(heldout, bootstraps, seed) if bootstraps else {},
            "limitations": "Exploratory statistical mediation; scores and equations learned from outcome labels; cross-fitting limits in-sample score leakage but does not identify a causal indirect effect. Fixed-score cluster bootstrap CIs omit uncertainty from fitting EEG weights, so do not present as formal mediation significance tests. Unmeasured attention, salience, expectations, and EEG measurement limitations remain.",
        }
        (out / "mediation_oof.json").write_text(json.dumps(mediation, indent=2))
        (out / "heldout_metrics.json").write_text(json.dumps(metrics, indent=2))
        (out / "outer_cv_folds.json").write_text(json.dumps(fold_log, indent=2))
    else:
        metrics = None
        mediation = None
        warnings.warn("FIT-ONLY mode: apparent fit only; no held-out validation or mediation report.")
    alpha_final = select_alpha(df, features, pain_min, pain_max, seed) if n_subject >= 4 else 0.1
    model = fit_score(df, features, alpha_final)
    model.update({"model_kind": "EEG neural index and separate pain calibration heads",
                  "training_trials": len(df), "training_subjects": n_subject,
                  "pain_min": float(pain_min), "pain_max": float(pain_max),
                  "synthetic_warning": "If fit on synthetic data, all predictions are synthetic demonstrations only.",
                  "source_csv_name": csv_path.name})
    (out / "eeg_neural_score_model.json").write_text(json.dumps(model, indent=2))
    coeffs = pd.DataFrame({"eeg_feature": features,
                           "weight_per_standardized_feature": model["weights"],
                           "training_median": model["preprocessing"]["median"],
                           "training_mean_after_imputation": model["preprocessing"]["mean"],
                           "training_sd_after_imputation": model["preprocessing"]["sd"]})
    coeffs.to_csv(out / "feature_weights.csv", index=False)
    equation = (
        "EEG-only score (no stimulus or pain labels required at inference):\n"
        "  x_std[j] = (impute_median(x[j]) - feature_mean[j]) / feature_sd[j]\n"
        "  raw = " + " + ".join(f"({w:.6g})*z[{name}]" for name, w in zip(features, model["weights"])) + "\n"
        f"  neural_score_z = (raw - ({model['score_mean']:.6g})) / ({model['score_sd']:.6g})\n"
        "  pred_pain_from_EEG_only = clip(\n"
        f"      ({model['pain_head_eeg_only'][0]:.6g}) + ({model['pain_head_eeg_only'][1]:.6g})*neural_score_z,\n"
        f"      {pain_min}, {pain_max})\n"
        f"  score_0_100 = 100*(pred_pain_from_EEG_only - {pain_min})/({pain_max}-{pain_min})\n"
        "See feature_weights.csv for training-set imputation, means and standard deviations.\n"
        "This is a fitted predictive index, NOT a clinically validated pain biomarker.\n"
    )
    (out / "learned_equation.txt").write_text(equation)
    summary = {"trials_read": len(df0), "trials_with_labels_and_eeg": len(df),
               "subjects": n_subject, "feature_set": feature_set, "features_used": features,
               "categories_represented": [entry[1] for entry in FEATURE_CATALOG if entry[2] in features],
               "features_missing": absent,
               "final_ridge_alpha": alpha_final, "subject_held_out_validation": metrics is not None,
               "caution": "These are candidate EEG-derived pain estimates, not validated objective pain measures."}
    (out / "run_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({"summary": summary, "heldout_metrics": metrics,
                      "crossfit_indirect": (mediation["point_estimates"]["indirect_product_a_times_b"] if mediation else None)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True, help="Concatenated trial feature CSV across subjects")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--feature-set", choices=['core','all-adult','all-13'], default='all-adult',
                        help="all-adult uses categories 1-12 by default; all-13 only for template-matched neonates")
    parser.add_argument("--expanded", action="store_true", help="Include additional secondary feature choices")
    parser.add_argument("--extra-feature", action="append", default=[], help="Additional column, repeatable")
    parser.add_argument("--exclude-feature", action="append", default=[], help="Omit a default/expanded feature; repeatable")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--bootstraps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--pain-min", type=float, default=0)
    parser.add_argument("--pain-max", type=float, default=10)
    parser.add_argument("--fit-only", action="store_true", help="Fit on all data (even one subject); NO validity or mediation evaluation")
    a = parser.parse_args()
    if a.folds < 2 or a.bootstraps < 0:
        parser.error("--folds must be >=2 and --bootstraps must be >=0")
    analyze(a.csv, a.out, a.expanded, a.extra_feature, a.exclude_feature, a.folds, a.bootstraps,
            a.seed, a.pain_min, a.pain_max, a.fit_only, a.feature_set)


if __name__ == "__main__":
    main()
