"""Turn per-trial baseline/response feature values into baseline-normalized
change scores.

For every base feature X (e.g. alpha_power_abs) computed by extract_features.py,
adds three columns per trial:

  X_delta_abs = X_response - X_baseline                    (this trial's own baseline)
  X_delta_rel = X_delta_abs / |X_baseline|
  X_z         = X_delta_abs / sigma_baseline(subject)

sigma_baseline(subject) is the std of X_baseline pooled across ALL of that
subject's trials -- a single trial's 1s baseline window is too short to give a
stable variance estimate on its own, so Z uses the subject's baseline
variability across trials as the yardstick. Subjects with fewer than
--min-baseline-trials usable (non-NaN) baseline values for a feature get Z=NaN
for that feature rather than a noisy estimate.
"""
import argparse

import numpy as np
import pandas as pd

META_COLS = {"dataset", "global_subject", "epoch", "laser_power", "vertex_channel"}


def base_feature_names(columns):
    names = []
    for c in columns:
        if c.endswith("_baseline") and c not in META_COLS:
            base = c[: -len("_baseline")]
            if f"{base}_response" in columns:
                names.append(base)
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="features_baseline_response.csv")
    ap.add_argument("--output", default="features_normalized.csv")
    ap.add_argument("--min-baseline-trials", type=int, default=5)
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    features = base_feature_names(df.columns)
    print(f"{len(features)} features: {features}")

    n_valid_baseline = df.groupby("global_subject")[[f"{f}_baseline" for f in features]].transform(
        lambda s: s.notna().sum()
    )
    sigma_baseline = df.groupby("global_subject")[[f"{f}_baseline" for f in features]].transform("std")

    new_cols = {}
    for f in features:
        base, resp = df[f"{f}_baseline"], df[f"{f}_response"]
        delta_abs = resp - base
        new_cols[f"{f}_delta_abs"] = delta_abs
        new_cols[f"{f}_delta_rel"] = delta_abs / base.abs()

        sigma = sigma_baseline[f"{f}_baseline"]
        n_valid = n_valid_baseline[f"{f}_baseline"]
        z = delta_abs / sigma
        z[(n_valid < a.min_baseline_trials) | (sigma == 0)] = np.nan
        new_cols[f"{f}_z"] = z

    df = pd.concat([df, pd.DataFrame(new_cols)], axis=1)
    df.to_csv(a.output, index=False)
    print(f"saved {a.output}: {df.shape}")

    per_subject_n = df.groupby("global_subject")[f"{features[0]}_baseline"].apply(lambda s: s.notna().sum())
    n_low = int((per_subject_n < a.min_baseline_trials).sum())
    print(f"{per_subject_n.size} subjects; {n_low} below --min-baseline-trials={a.min_baseline_trials} "
          f"(Z=NaN for those, based on {features[0]})")


if __name__ == "__main__":
    main()
