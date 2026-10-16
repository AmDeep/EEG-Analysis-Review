#!/usr/bin/env python3
"""Generate multi-subject simulated EEG features with a KNOWN synthetic indirect pathway."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from neural_mediation import CORE_FEATURES, EXPLORATORY_FEATURES, FULL_ADULT_FEATURES


def make_demo(out: Path, subjects: int = 30, trials: int = 40, seed: int = 2026, cohort: str = "adult"):
    rng = np.random.default_rng(seed)
    rows = []
    # The synthetic structural equations are illustrative; NOT human pain physiology.
    loadings = np.array([-0.45, -0.30, 0.33, 0.55, 0.28, 0.18, 6.0])
    scales = np.array([0.20, 0.18, 0.20, 0.29, 0.12, 0.11, 5.0])
    for sub in range(subjects):
        subject = f"sub-DEMO{sub+1:03d}"
        neural_bias = rng.normal(0, 0.25)
        pain_bias = rng.normal(0, 0.5)
        for t in range(trials):
            I = float(rng.choice([2, 4, 6, 8]))
            M = 0.30*(I-5) + neural_bias + rng.normal(0, 0.65)
            feats = loadings*M + rng.normal(size=len(loadings))*scales
            # P is affected by I AND M, with independent pain noise.
            pain = np.clip(2.1 + 0.52*I + 0.90*M + pain_bias + rng.normal(0, 0.70), 0, 10)
            row = {"subject": subject, "session": "ses-01", "cohort": cohort,
                   "trial_id": f"trial-{t+1:03d}", "stimulus_intensity": I, "pain_rating": pain}
            row.update({k: float(v) for k, v in zip(CORE_FEATURES, feats)})
            # All five additional adult feature categories available as noise
            # features, to stress-test feature selection (NO observed EEG).
            row.update({k: float(rng.normal()) for k in FULL_ADULT_FEATURES if k not in CORE_FEATURES})
            row.update({k: float(rng.normal()) for k in EXPLORATORY_FEATURES if not k.startswith('neonatal_')})
            if cohort == 'neonatal':
                # Random values strictly for code tests; NOT a legitimate
                # neonatal waveform, marker, or physiological simulation.
                row['neonatal_noxious_template_corr'] = float(rng.uniform(-1, 1))
                row['neonatal_noxious_template_projection'] = float(rng.normal())
            rows.append(row)
    df = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} explicitly SYNTHETIC trials across {subjects} subjects to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("demo_input/synthetic_multisubject_features.csv"))
    parser.add_argument("--subjects", type=int, default=30)
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument('--cohort', choices=['adult', 'neonatal'], default='adult',
                        help='SYNTHETIC MODEL INPUTS ONLY; neonatal scores are random placeholders.')
    args = parser.parse_args()
    make_demo(args.out, args.subjects, args.trials, args.seed, args.cohort)
