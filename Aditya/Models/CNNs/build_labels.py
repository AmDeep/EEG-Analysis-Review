"""Create tertile labels from the fixed epoch table.

The thresholds are NOT learned globally. During CV, train_eegnet.py learns
q33/q67 from the training fold only and applies those thresholds to val/test.
This script is only for inspecting the label distribution or making a
standalone label file when a single global split is intentionally desired.
"""
import argparse
import numpy as np
import pandas as pd


def add_tertiles(df, power_col="laser_power"):
    x = pd.to_numeric(df[power_col], errors="coerce")
    if x.isna().any():
        raise ValueError(f"{x.isna().sum()} non-numeric/missing values in {power_col}")
    q33, q67 = x.quantile([1/3, 2/3]).to_numpy()
    # 0 = low, 1 = middle, 2 = high
    y = np.select([x <= q33, x <= q67], [0, 1], default=2).astype(np.int64)
    out = df.copy()
    out["label"] = y
    out["q33"] = q33
    out["q67"] = q67
    return out, float(q33), float(q67)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--power-col", default="laser_power")
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    out, q33, q67 = add_tertiles(df, a.power_col)
    out.to_csv(a.output, index=False)

    print(f"q33={q33:.6g}, q67={q67:.6g}")
    print(out["label"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
