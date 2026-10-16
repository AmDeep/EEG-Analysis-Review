#!/usr/bin/env python3
"""Apply a previously fitted EEG-only score to trial feature CSV (no labels needed)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
from neural_mediation import predict


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--model", required=True, type=Path)
    a.add_argument("--csv", required=True, type=Path)
    a.add_argument("--out", required=True, type=Path)
    args = a.parse_args()
    m = json.loads(args.model.read_text())
    df = pd.read_csv(args.csv)
    missing = [c for c in m["features"] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for this model: {missing}")
    scores = predict(df, m, m["pain_min"], m["pain_max"])
    keep = [c for c in ["subject", "session", "trial_id", "onset_s", "pain_rating", "stimulus_intensity"] if c in df.columns]
    result = pd.concat([df[keep].reset_index(drop=True), scores.reset_index(drop=True)], axis=1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print(f"Scored {len(result)} trials; output {args.out}")

if __name__ == "__main__":
    main()
