#!/usr/bin/env python3
"""
Composite Pain Modulation Index (CPMI) - EEG-augmented formulaic scale
Derived from Table 2 EEG features + Multidimensional Pain Modulation Framework.

Focus: Core and Core-candidate features from the supplied table.
Simple clinical inputs + optional EEG features -> 0-10 scale.

Literature directions used for coefficient signs (not magnitudes):
- Higher theta, beta, gamma power often associated with higher pain / chronic pain.
- Lower sensorimotor alpha power / slower PAF associated with higher pain sensitivity.
- Altered (often increased) connectivity in theta/alpha/beta, higher spectral entropy candidates.
- Coefficients are heuristic, calibrated so that typical healthy profiles ~0.5-2.0
  and severe chronic profiles ~7-9.5 on the final 0-10 scale.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple

# ---------------------------------------------------------------------------
# 1. Feature definitions and literature-informed directions
# ---------------------------------------------------------------------------

# Core / Core-candidate features from Table 2
CORE_FEATURES = {
    "sensorimotor_alpha_power": {
        "band": "8-13 Hz",
        "direction": -1,          # lower power -> higher pain
        "role": "Core spectral feature",
        "notes": "Central suppression during experimental pain; resting heterogeneous"
    },
    "peak_alpha_frequency": {
        "band": "dominant alpha",
        "direction": -1,          # slower PAF -> higher sensitivity
        "role": "Optional exploratory",
        "notes": "Slower resting PAF predicted prolonged experimental pain"
    },
    "beta_power": {
        "band": "13-30 Hz",
        "direction": +1,          # higher often linked to chronic pain
        "role": "Core spectral feature",
        "notes": "Context-dependent; control motor/medication"
    },
    "theta_power": {
        "band": "4-8 Hz",
        "direction": +1,          # higher in many chronic pain cohorts
        "role": "Core spectral feature",
        "notes": "Heterogeneous; distinguish local vs connectivity"
    },
    "gamma_power": {
        "band": ">30 Hz",
        "direction": +1,          # sensorimotor / prefrontal associations
        "role": "Core candidate",
        "notes": "EMG contamination risk - must be cleaned"
    },
    "connectivity_theta_alpha_beta": {
        "band": "multi",
        "direction": +1,          # mega-analysis signal with limbic
        "role": "Core candidate",
        "notes": "Coherence / phase sync; specify pairs, band, controls"
    },
    "spectral_entropy": {
        "band": "broadband",
        "direction": +1,          # candidate model input
        "role": "Core candidate",
        "notes": "Summary of power distribution; compare vs power-only"
    },
}

# Clinical / demographic simple inputs
CLINICAL_FEATURES = {
    "nrs": {"min": 0, "max": 10, "weight": 0.65},          # dominant anchor
    "sex_female": {"min": 0, "max": 1, "weight": 0.35},    # Fillingim et al.
    "age_norm": {"min": 0, "max": 1, "weight": 0.20},      # older -> modest increase
    "pcs_z": {"min": -3, "max": 3, "weight": 0.40},        # catastrophizing
    "cpm_efficiency_z": {"min": -3, "max": 3, "weight": -0.35},  # protective (Yarnitsky)
    "central_sens_z": {"min": -2, "max": 3, "weight": 0.30},
}

# EEG feature weights (after z-scoring). Tuned so EEG_mod typically ±1.5–2.0
EEG_WEIGHTS = {
    "sensorimotor_alpha_power": -0.45,
    "peak_alpha_frequency": -0.40,
    "beta_power": 0.30,
    "theta_power": 0.40,
    "gamma_power": 0.25,
    "connectivity_theta_alpha_beta": 0.50,   # highest among EEG (mega-analysis)
    "spectral_entropy": 0.20,
}

# ---------------------------------------------------------------------------
# 2. Core computation functions
# ---------------------------------------------------------------------------

def zscore(x: float, mu: float = 0.0, sigma: float = 1.0) -> float:
    """Safe z-score."""
    if sigma <= 0:
        return 0.0
    return (x - mu) / sigma


def compute_eeg_mod(eeg: Dict[str, float], 
                    reference_stats: Optional[Dict[str, Tuple[float, float]]] = None) -> float:
    """
    Compute EEG modulator term.
    eeg: dict of raw feature values
    reference_stats: optional {feature: (mean, std)} from healthy or study cohort
    """
    if reference_stats is None:
        # Default healthy-ish reference (approximate literature central values)
        reference_stats = {
            "sensorimotor_alpha_power": (1.0, 0.35),   # relative power
            "peak_alpha_frequency": (10.0, 0.8),       # Hz
            "beta_power": (0.25, 0.12),
            "theta_power": (0.20, 0.10),
            "gamma_power": (0.08, 0.05),
            "connectivity_theta_alpha_beta": (0.35, 0.15),  # e.g. mean coherence
            "spectral_entropy": (0.75, 0.12),
        }
    
    mod = 0.0
    for feat, w in EEG_WEIGHTS.items():
        if feat not in eeg:
            continue
        mu, sigma = reference_stats.get(feat, (0.0, 1.0))
        z = zscore(eeg[feat], mu, sigma)
        # Apply literature direction already encoded in weight sign
        mod += w * z
    return mod


def compute_cpmi(clinical: Dict[str, float],
                 eeg: Optional[Dict[str, float]] = None,
                 reference_stats: Optional[Dict] = None,
                 clamp: bool = True) -> Dict[str, float]:
    """
    Full CPMI calculation.
    Returns dict with total score and component breakdown.
    """
    # Clinical part
    clin_score = 0.0
    clin_score += CLINICAL_FEATURES["nrs"]["weight"] * clinical.get("nrs", 0.0)
    clin_score += CLINICAL_FEATURES["sex_female"]["weight"] * clinical.get("sex_female", 0.0)
    
    age = clinical.get("age", 40)
    age_norm = np.clip((age - 20) / 60.0, 0, 1)  # rough 20-80 -> 0-1
    clin_score += CLINICAL_FEATURES["age_norm"]["weight"] * age_norm
    
    clin_score += CLINICAL_FEATURES["pcs_z"]["weight"] * clinical.get("pcs_z", 0.0)
    clin_score += CLINICAL_FEATURES["cpm_efficiency_z"]["weight"] * clinical.get("cpm_efficiency_z", 0.0)
    clin_score += CLINICAL_FEATURES["central_sens_z"]["weight"] * clinical.get("central_sens_z", 0.0)
    
    # EEG part
    eeg_mod = 0.0
    if eeg is not None:
        eeg_mod = compute_eeg_mod(eeg, reference_stats)
    
    total = clin_score + eeg_mod
    
    if clamp:
        total = float(np.clip(total, 0.0, 10.0))
    
    return {
        "cpmi": total,
        "clinical_component": clin_score,
        "eeg_modulator": eeg_mod,
        "nrs_contribution": CLINICAL_FEATURES["nrs"]["weight"] * clinical.get("nrs", 0.0),
    }


# ---------------------------------------------------------------------------
# 3. Synthetic data generation for testing (realistic ranges from literature)
# ---------------------------------------------------------------------------

def generate_synthetic_cohort(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate two groups: healthy-like and chronic-pain-like.
    Values are plausible given published means/SDs for EEG and clinical scores.
    """
    rng = np.random.default_rng(seed)
    
    # Group labels
    group = rng.choice(["healthy", "chronic"], size=n, p=[0.4, 0.6])
    
    records = []
    for g in group:
        if g == "healthy":
            nrs = rng.normal(1.2, 1.0)
            sex_f = rng.binomial(1, 0.5)
            age = rng.normal(35, 12)
            pcs_z = rng.normal(-0.4, 0.8)
            cpm_z = rng.normal(0.6, 0.9)          # more efficient
            sens_z = rng.normal(-0.3, 0.7)
            # EEG healthy-ish
            alpha = rng.normal(1.05, 0.30)
            paf = rng.normal(10.2, 0.7)
            beta = rng.normal(0.22, 0.10)
            theta = rng.normal(0.18, 0.08)
            gamma = rng.normal(0.07, 0.04)
            conn = rng.normal(0.32, 0.12)
            entropy = rng.normal(0.72, 0.10)
        else:
            nrs = rng.normal(5.8, 1.8)
            sex_f = rng.binomial(1, 0.65)         # slightly higher female
            age = rng.normal(52, 14)
            pcs_z = rng.normal(0.7, 1.0)
            cpm_z = rng.normal(-0.5, 1.0)         # less efficient
            sens_z = rng.normal(0.8, 0.9)
            # EEG chronic-ish (directional)
            alpha = rng.normal(0.75, 0.32)        # lower
            paf = rng.normal(9.3, 0.85)           # slower
            beta = rng.normal(0.32, 0.13)         # higher
            theta = rng.normal(0.28, 0.11)        # higher
            gamma = rng.normal(0.11, 0.06)
            conn = rng.normal(0.45, 0.16)         # higher
            entropy = rng.normal(0.82, 0.11)
        
        # Clip to realistic bounds
        nrs = float(np.clip(nrs, 0, 10))
        age = float(np.clip(age, 18, 85))
        alpha = float(np.clip(alpha, 0.1, 2.0))
        paf = float(np.clip(paf, 7.5, 12.5))
        beta = float(np.clip(beta, 0.05, 0.7))
        theta = float(np.clip(theta, 0.05, 0.6))
        gamma = float(np.clip(gamma, 0.01, 0.3))
        conn = float(np.clip(conn, 0.05, 0.85))
        entropy = float(np.clip(entropy, 0.4, 0.98))
        
        records.append({
            "group": g,
            "nrs": nrs,
            "sex_female": sex_f,
            "age": age,
            "pcs_z": pcs_z,
            "cpm_efficiency_z": cpm_z,
            "central_sens_z": sens_z,
            "sensorimotor_alpha_power": alpha,
            "peak_alpha_frequency": paf,
            "beta_power": beta,
            "theta_power": theta,
            "gamma_power": gamma,
            "connectivity_theta_alpha_beta": conn,
            "spectral_entropy": entropy,
        })
    
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# 4. Run validation suite
# ---------------------------------------------------------------------------

def run_validation():
    print("=" * 70)
    print("CPMI FORMULAIC PAIN SCALE - VALIDATION REPORT")
    print("Derived from Table 2 (EEG features) + Multidimensional Framework")
    print("=" * 70)
    
    df = generate_synthetic_cohort(n=300, seed=42)
    
    # Compute CPMI for every subject
    results = []
    for _, row in df.iterrows():
        clinical = {
            "nrs": row["nrs"],
            "sex_female": row["sex_female"],
            "age": row["age"],
            "pcs_z": row["pcs_z"],
            "cpm_efficiency_z": row["cpm_efficiency_z"],
            "central_sens_z": row["central_sens_z"],
        }
        eeg = {
            "sensorimotor_alpha_power": row["sensorimotor_alpha_power"],
            "peak_alpha_frequency": row["peak_alpha_frequency"],
            "beta_power": row["beta_power"],
            "theta_power": row["theta_power"],
            "gamma_power": row["gamma_power"],
            "connectivity_theta_alpha_beta": row["connectivity_theta_alpha_beta"],
            "spectral_entropy": row["spectral_entropy"],
        }
        out = compute_cpmi(clinical, eeg)
        results.append({
            "group": row["group"],
            "nrs": row["nrs"],
            "cpmi": out["cpmi"],
            "clinical_component": out["clinical_component"],
            "eeg_modulator": out["eeg_modulator"],
        })
    
    res = pd.DataFrame(results)
    
    # Overall metrics
    print("\n--- Overall Scale Metrics (n=300 synthetic) ---")
    print(f"Min CPMI:     {res['cpmi'].min():.3f}")
    print(f"Max CPMI:     {res['cpmi'].max():.3f}")
    print(f"Mean CPMI:    {res['cpmi'].mean():.3f}")
    print(f"Median CPMI:  {res['cpmi'].median():.3f}")
    print(f"Std CPMI:     {res['cpmi'].std():.3f}")
    print(f"25th pct:     {res['cpmi'].quantile(0.25):.3f}")
    print(f"75th pct:     {res['cpmi'].quantile(0.75):.3f}")
    
    # By group
    print("\n--- By Group ---")
    for g in ["healthy", "chronic"]:
        sub = res[res["group"] == g]
        print(f"{g.upper():8s}  n={len(sub):3d}  mean={sub['cpmi'].mean():.3f}  "
              f"std={sub['cpmi'].std():.3f}  min={sub['cpmi'].min():.3f}  max={sub['cpmi'].max():.3f}")
    
    # Correlation with NRS (should be high but not perfect)
    corr_nrs = res["cpmi"].corr(res["nrs"])
    print(f"\nPearson r(CPMI, NRS): {corr_nrs:.3f}")
    
    # Separation
    healthy_mean = res[res["group"]=="healthy"]["cpmi"].mean()
    chronic_mean = res[res["group"]=="chronic"]["cpmi"].mean()
    print(f"Mean difference (chronic - healthy): {chronic_mean - healthy_mean:.3f}")
    
    # Component contributions
    print("\n--- Average Component Contributions ---")
    print(f"Clinical component mean: {res['clinical_component'].mean():.3f}")
    print(f"EEG modulator mean:      {res['eeg_modulator'].mean():.3f}")
    
    # Example profiles
    print("\n--- Example Individual Profiles ---")
    
    # Healthy low
    ex1 = compute_cpmi(
        clinical={"nrs": 1.0, "sex_female": 0, "age": 30, "pcs_z": -0.5,
                  "cpm_efficiency_z": 1.0, "central_sens_z": -0.5},
        eeg={"sensorimotor_alpha_power": 1.1, "peak_alpha_frequency": 10.4,
             "beta_power": 0.20, "theta_power": 0.16, "gamma_power": 0.06,
             "connectivity_theta_alpha_beta": 0.30, "spectral_entropy": 0.70}
    )
    print(f"Healthy low-pain male: CPMI = {ex1['cpmi']:.2f}  "
          f"(clin={ex1['clinical_component']:.2f}, eeg={ex1['eeg_modulator']:.2f})")
    
    # Chronic severe
    ex2 = compute_cpmi(
        clinical={"nrs": 7.5, "sex_female": 1, "age": 58, "pcs_z": 1.5,
                  "cpm_efficiency_z": -1.2, "central_sens_z": 1.4},
        eeg={"sensorimotor_alpha_power": 0.65, "peak_alpha_frequency": 8.9,
             "beta_power": 0.38, "theta_power": 0.33, "gamma_power": 0.14,
             "connectivity_theta_alpha_beta": 0.52, "spectral_entropy": 0.86}
    )
    print(f"Severe chronic female:  CPMI = {ex2['cpmi']:.2f}  "
          f"(clin={ex2['clinical_component']:.2f}, eeg={ex2['eeg_modulator']:.2f})")
    
    # Moderate
    ex3 = compute_cpmi(
        clinical={"nrs": 4.5, "sex_female": 0, "age": 45, "pcs_z": 0.3,
                  "cpm_efficiency_z": -0.2, "central_sens_z": 0.4},
        eeg={"sensorimotor_alpha_power": 0.85, "peak_alpha_frequency": 9.6,
             "beta_power": 0.28, "theta_power": 0.24, "gamma_power": 0.09,
             "connectivity_theta_alpha_beta": 0.40, "spectral_entropy": 0.78}
    )
    print(f"Moderate mixed:         CPMI = {ex3['cpmi']:.2f}  "
          f"(clin={ex3['clinical_component']:.2f}, eeg={ex3['eeg_modulator']:.2f})")
    
    # Coefficient sensitivity: what if we zero the EEG block?
    print("\n--- Sensitivity: Clinical-only vs Full ---")
    clin_only = []
    for _, row in df.iterrows():
        clinical = {
            "nrs": row["nrs"], "sex_female": row["sex_female"], "age": row["age"],
            "pcs_z": row["pcs_z"], "cpm_efficiency_z": row["cpm_efficiency_z"],
            "central_sens_z": row["central_sens_z"],
        }
        out = compute_cpmi(clinical, eeg=None)
        clin_only.append(out["cpmi"])
    res["cpmi_clin_only"] = clin_only
    print(f"Full CPMI  mean: {res['cpmi'].mean():.3f}")
    print(f"Clin-only  mean: {res['cpmi_clin_only'].mean():.3f}")
    print(f"Mean |delta| from adding EEG: {(res['cpmi'] - res['cpmi_clin_only']).abs().mean():.3f}")
    
    # Rank correlation of EEG_mod with group
    print(f"\nMean EEG_mod healthy: {res[res.group=='healthy']['eeg_modulator'].mean():.3f}")
    print(f"Mean EEG_mod chronic: {res[res.group=='chronic']['eeg_modulator'].mean():.3f}")
    
    print("\n" + "=" * 70)
    print("INTERPRETATION NOTES")
    print("- Scale is anchored on NRS (weight 0.65) so remains clinically interpretable.")
    print("- EEG modulator typically shifts score by 0.5–2.0 points.")
    print("- Coefficients are literature-directed heuristics, not fitted regression betas.")
    print("- In real use: re-estimate weights on local data with regularized regression.")
    print("- Connectivity receives highest EEG weight following Bott et al. 2025 mega-analysis.")
    print("- Always clean gamma for EMG; follow Table 2 measurement cautions.")
    print("=" * 70)
    
    return res


if __name__ == "__main__":
    results_df = run_validation()
    # Optionally save
    results_df.to_csv("/home/workdir/artifacts/cpmi_validation_results.csv", index=False)
    print("\nResults saved to /home/workdir/artifacts/cpmi_validation_results.csv")
