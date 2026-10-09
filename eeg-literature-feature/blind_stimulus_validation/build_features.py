"""Extract the frozen candidates from downloaded real EEG."""
import collections
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from features import FAMILIES, extract
from readers import WORKSPACE, describe_targets, heat, shock

HERE = Path(__file__).resolve().parent


def main():
    metadata, vectors, covariance = [], [], []
    names = None
    audit = []
    manifest = json.loads((HERE / "download_manifest.json").read_text())
    (HERE / "target_mapping.json").write_text(json.dumps(describe_targets(), indent=2))
    for r in manifest:
        if r["status"] != "downloaded":
            audit.append({**r, "exclusion": "download_failed"})
            continue
        reader = heat if r["dataset"].startswith("heat") else shock
        subject = Path(r["name"]).stem
        errors = collections.Counter()
        included = 0
        try:
            for row in reader(WORKSPACE / r["path"]):
                if row["error"]:
                    errors[row["error"]] += 1
                    continue
                feats, cov = extract(*row.pop("windows"))
                if names is None:
                    names = sorted(feats)
                if set(feats) != set(names):
                    raise ValueError("Feature schema changed across trials")
                row.pop("error")
                metadata.append({**row, "dataset": r["dataset"], "subject": f"{r['dataset']}:{subject}"})
                vectors.append([feats[n] for n in names])
                covariance.append(cov)
                included += 1
        except Exception as e:
            # A schema/algorithm error is fatal, not a silently dropped cohort.
            audit.append({"dataset": r["dataset"], "subject": subject, "fatal_error": repr(e)})
            (HERE / "extraction_audit.json").write_text(json.dumps(audit, indent=2))
            raise
        audit.append({"dataset": r["dataset"], "subject": subject, "included_trials": included,
                      "quality_exclusions": dict(errors), "sha256": r["sha256"]})
        print(r["dataset"], subject, "included", included, "excluded", dict(errors), flush=True)
        (HERE / "extraction_audit.json").write_text(json.dumps(audit, indent=2))
    if not vectors:
        raise ValueError("No usable independently labeled EEG trials; no fabricated scores")
    matrix = np.asarray(vectors, float)
    matrix[~np.isfinite(matrix)] = np.nan
    np.savez_compressed(HERE / "features.npz", X=matrix, names=np.array(names),
                        covariance=np.stack(covariance))
    frame = pd.DataFrame(metadata)
    frame.to_csv(HERE / "trial_metadata.csv", index=False)
    finish_reports(frame, matrix, names)


def finish_reports(frame, matrix, names):
    counts = frame.groupby(["dataset", "subject", "binary_target"]).size().rename("trials").reset_index()
    counts.to_csv(HERE / "participant_target_counts.csv", index=False)
    generated = pd.DataFrame({
        "feature": names, "family_code": [n.split("__")[1] for n in names],
        "family": [FAMILIES[n.split("__")[1]] for n in names],
        "finite_trials": np.isfinite(matrix).sum(axis=0),
        "missing_trials": np.isnan(matrix).sum(axis=0),
        "total_trials": len(frame)
    })
    generated.to_csv(HERE / "generated_feature_catalogue.csv", index=False)
    original = pd.read_csv(HERE.parent / "feature_prediction_table.csv")
    coverage = []
    for _, row in original.iterrows():
        values = row.to_dict()
        family_name = values["feature"]
        family_code = next((k for k, v in FAMILIES.items() if v == family_name), None)
        number = int((generated.family == family_name).sum())
        status = "candidate implementation tested; not an exact paper-specific algorithm replication"
        note = "Fixed common montage and short event epochs; exploratory family-level validation."
        if family_name == "High-gamma response":
            status, note = "not tested", "Common 1-40 Hz pipeline; source shock EEG was already filtered at 70 Hz. No 60-100 Hz substitute."
        elif family_name == "CVUE/Piqβ":
            status, note = "not tested", "No valid reproduction of source-specific long-record Piqβ estimator/threshold from short epochs. Envelope CV is labelled variability instead."
        elif family_code == "csp":
            number, status, note = 2, "trained inside each training fold only", "Two generalized-eigenvector log-variance features; training low/higher labels, not PCA."
        elif family_code in ("paf", "aper"):
            note = "FOOOF 1.1.1, fixed aperiodic mode; central PSD 2-40 Hz, alpha component 8-13 Hz. Missing peaks remain NaN, not forced estimates. Short windows limit spectral precision."
        elif family_code == "morph":
            note = "Prespecified extrema/latencies and six mean waveform bins; NOT Hu WF+MLRd."
        elif family_code == "pac":
            note = "Tort 18-bin MI: theta-beta and alpha-30-40 Hz. Very few cycles per epoch; not a significance-tested coupling claim."
        coverage.append({**values, "status": status, "generated_columns": number, "implementation_note": note})
    pd.DataFrame(coverage).to_csv(HERE / "feature_family_coverage.csv", index=False)
    summary = {
        "trials": len(frame), "columns_cached": len(names), "csp_columns_inside_training": 2,
        "participants": frame.groupby("dataset").subject.nunique().to_dict(),
        "trials_by_dataset": frame.groupby("dataset").size().to_dict(),
        "finite_fraction": float(np.isfinite(matrix).mean()),
        "families_tested": len({n.split('__')[1] for n in names}) + 1,
        "untested_families": ["High-gamma response", "CVUE/Piqβ"],
        "scope": "No dimensionality reduction/PCA or unsupervised clustering altered; new supervised validation only."
    }
    (HERE / "feature_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
